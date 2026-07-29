"""Provider abstraction, fallback chain, per-call cost attribution.

The loop talks to a Router, never to a provider directly. A Router holds an
ordered chain of providers; if the primary raises, the next one is tried, and
only when every provider has failed does the loop see AllProvidersFailed
(which it turns into the MODEL_ERROR stop status). Every completed call is
priced from tether.config and attributed to a trace span.

Message format (provider-neutral):
  {"role": "system"|"user"|"assistant"|"tool", "content": str,
   "tool_calls": [ToolCall, ...]?,      # assistant only
   "tool_call_id": str?, "name": str?}  # tool only
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import pricing_for
from .trace import Tracer


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ModelResponse:
    model: str
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class ProviderError(RuntimeError):
    """A single provider failed; the router may still fall back."""


class AllProvidersFailed(RuntimeError):
    """Every provider in the chain failed. The loop stops with MODEL_ERROR."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


class Provider(Protocol):
    model: str

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelResponse: ...


class MockProvider:
    """Offline, free, deterministic — and deliberately imperfect.

    It runs a fixed script: read the ticket, search the KB, categorize, then
    resolve. It always categorizes as 'access' and it never escalates. Both
    flaws are on purpose: the eval suite must catch its own fixture, and the
    escalation monitor must have something real to fire on.
    """

    model = "mock"

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelResponse:
        ticket_id = self._find_ticket_id(messages)
        turn = sum(1 for m in messages if m["role"] == "assistant")
        script: list[tuple[str, dict[str, Any], str]] = [
            ("get_ticket", {"ticket_id": ticket_id}, "Reading the ticket."),
            (
                "search_kb",
                {"query": "account access reset"},
                "Searching the knowledge base.",
            ),
            (
                "categorize_ticket",
                {"ticket_id": ticket_id, "category": "access"},
                "This looks like an access issue.",
            ),
            (
                "resolve_ticket",
                {
                    "ticket_id": ticket_id,
                    "answer": (
                        "Please reset your password from the self-service "
                        "portal at portal.example.com/reset and sign in "
                        "again. If access is still blocked after the reset, "
                        "reply to this ticket."
                    ),
                },
                "Resolving with standard access guidance.",
            ),
        ]
        name, args, text = script[min(turn, len(script) - 1)]
        input_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4
        return ModelResponse(
            model=self.model,
            text=text,
            tool_calls=[ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=name, args=args)],
            input_tokens=input_tokens,
            output_tokens=48,
        )

    @staticmethod
    def _find_ticket_id(messages: list[dict[str, Any]]) -> str:
        for m in messages:
            content = str(m.get("content", ""))
            for token in content.replace("\n", " ").split():
                cleaned = token.strip(".,:;()[]\"'")
                if cleaned.startswith("T-") and cleaned[2:].isdigit():
                    return cleaned
        return "T-0000"


class AnthropicProvider:
    """Thin adapter over the Anthropic Messages API."""

    def __init__(self, model: str = "claude-sonnet-5", max_tokens: int = 1024) -> None:
        self.model = model
        self.max_tokens = max_tokens

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelResponse:
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError(f"anthropic SDK not installed: {exc}") from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ProviderError("ANTHROPIC_API_KEY is not set")

        system, api_messages = self._to_api_messages(messages)
        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system or anthropic.NOT_GIVEN,
                messages=api_messages,
                tools=tools or anthropic.NOT_GIVEN,
            )
        except Exception as exc:
            raise ProviderError(f"{self.model}: {type(exc).__name__}: {exc}") from exc

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, args=dict(block.input))
                )
        return ModelResponse(
            model=self.model,
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    @staticmethod
    def _to_api_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        system = ""
        api_messages: list[dict[str, Any]] = []
        for m in messages:
            role = m["role"]
            if role == "system":
                system = m["content"]
            elif role == "assistant":
                content: list[dict[str, Any]] = []
                if m.get("content"):
                    content.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls", []):
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.args,
                        }
                    )
                api_messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                api_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m["tool_call_id"],
                                "content": m["content"],
                            }
                        ],
                    }
                )
            else:
                api_messages.append({"role": "user", "content": m["content"]})
        return system, api_messages


class Router:
    """Ordered fallback chain with per-call cost attribution."""

    def __init__(self, providers: list[Provider]) -> None:
        if not providers:
            raise ValueError("router needs at least one provider")
        self.providers = providers

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tracer: Tracer,
    ) -> tuple[ModelResponse, float]:
        errors: list[str] = []
        for provider in self.providers:
            with tracer.span(
                f"model:{provider.model}", "model_call", model=provider.model
            ) as span:
                try:
                    response = provider.complete(messages, tools)
                except ProviderError as exc:
                    span.metadata["error"] = str(exc)
                    errors.append(f"{provider.model}: {exc}")
                    continue
                cost = pricing_for(provider.model).cost(
                    response.input_tokens, response.output_tokens
                )
                span.cost_usd = cost
                span.metadata.update(
                    {
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "tool_calls": [tc.name for tc in response.tool_calls],
                    }
                )
                return response, cost
        raise AllProvidersFailed(errors)


def format_tool_calls_for_log(tool_calls: list[ToolCall]) -> str:
    return json.dumps(
        [{"name": tc.name, "args": tc.args} for tc in tool_calls], ensure_ascii=False
    )
