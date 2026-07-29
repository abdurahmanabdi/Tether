"""Test doubles shared across the suite."""

from __future__ import annotations

import uuid
from typing import Any

from tether.router import ModelResponse, ProviderError, ToolCall


class ScriptedProvider:
    """Replays a fixed list of (tool_name, args) turns, then repeats the last."""

    model = "mock"

    def __init__(self, script: list[tuple[str, dict[str, Any]]]) -> None:
        self.script = script

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelResponse:
        turn = sum(1 for m in messages if m["role"] == "assistant")
        name, args = self.script[min(turn, len(self.script) - 1)]
        return ModelResponse(
            model=self.model,
            text="",
            tool_calls=[
                ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=name, args=args)
            ],
            input_tokens=10,
            output_tokens=10,
        )


class FailingProvider:
    """Always raises ProviderError, as a dead primary would."""

    def __init__(self, model: str = "mock") -> None:
        self.model = model

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelResponse:
        raise ProviderError(f"{self.model} is down")


class TextProvider:
    """Returns plain text with a fixed reply and no tool calls."""

    model = "mock"

    def __init__(self, text: str) -> None:
        self.text = text

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelResponse:
        return ModelResponse(
            model=self.model, text=self.text, input_tokens=10, output_tokens=10
        )
