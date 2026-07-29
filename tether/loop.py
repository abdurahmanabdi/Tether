"""Execution core: plan -> act -> observe -> verify -> iterate -> stop.

Every exit is named (RunStatus). The verifier gates SUCCESS deterministically;
everything else is a budget or an error condition. There is no exit that means
"the loop just stopped" — if you see a status you cannot explain, that is a
bug in the loop, not in the run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .config import Budgets
from .router import AllProvidersFailed, Router
from .state import RunState
from .tools import ToolRegistry
from .trace import Tracer


class RunStatus(str, Enum):
    SUCCESS = "SUCCESS"
    MAX_ITERATIONS = "MAX_ITERATIONS"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    REPEATED_ERROR = "REPEATED_ERROR"
    MODEL_ERROR = "MODEL_ERROR"


@dataclass
class VerifierResult:
    passed: bool
    failures: list[str] = field(default_factory=list)


class ToolContractVerifier:
    """Deterministic gate on the loop's exit.

    Passes only when: every required tool has been called successfully, a
    terminal action has been taken, and the final answer is non-empty. It has
    no model in it and therefore cannot be talked out of a failure.
    """

    def __init__(self, required_tools: list[str] | None = None) -> None:
        self.required_tools = list(required_tools or [])

    def verify(self, state: RunState) -> VerifierResult:
        failures: list[str] = []
        called = set(state.tools_called_ok)
        for tool in self.required_tools:
            if tool not in called:
                failures.append(f"required tool never called successfully: {tool}")
        if state.terminal_action is None:
            failures.append("no terminal action taken")
        if not (state.final_answer or "").strip():
            failures.append("final answer is empty")
        return VerifierResult(passed=not failures, failures=failures)


@dataclass
class RunResult:
    status: RunStatus
    state: RunState
    tracer: Tracer
    stop_reason: str
    wall_clock_seconds: float

    @property
    def cost_usd(self) -> float:
        return self.state.cost_usd


class AgentLoop:
    def __init__(
        self,
        router: Router,
        registry: ToolRegistry,
        verifier: ToolContractVerifier,
        budgets: Budgets | None = None,
    ) -> None:
        self.router = router
        self.registry = registry
        self.verifier = verifier
        self.budgets = budgets or Budgets()

    def run(
        self,
        system_prompt: str,
        user_message: str,
        state: RunState | None = None,
    ) -> RunResult:
        state = state or RunState()
        tracer = Tracer()
        started = time.monotonic()

        state.append_message({"role": "system", "content": system_prompt})
        state.append_message({"role": "user", "content": user_message})

        status: RunStatus | None = None
        stop_reason = ""

        while status is None:
            elapsed = time.monotonic() - started

            if state.iterations >= self.budgets.max_iterations:
                status = RunStatus.MAX_ITERATIONS
                stop_reason = (
                    f"iteration budget reached ({self.budgets.max_iterations})"
                )
                break
            if elapsed > self.budgets.max_wall_clock_seconds:
                status = RunStatus.BUDGET_EXCEEDED
                stop_reason = (
                    f"wall clock {elapsed:.1f}s exceeded budget "
                    f"{self.budgets.max_wall_clock_seconds}s"
                )
                break
            if state.cost_usd > self.budgets.max_cost_usd:
                status = RunStatus.BUDGET_EXCEEDED
                stop_reason = (
                    f"cost ${state.cost_usd:.4f} exceeded budget "
                    f"${self.budgets.max_cost_usd:.4f}"
                )
                break

            state.iterations += 1

            try:
                response, cost = self.router.complete(
                    state.messages, self.registry.schemas(), tracer
                )
            except AllProvidersFailed as exc:
                status = RunStatus.MODEL_ERROR
                stop_reason = f"all providers failed: {exc}"
                break
            state.cost_usd += cost

            state.append_message(
                {
                    "role": "assistant",
                    "content": response.text,
                    "tool_calls": response.tool_calls,
                }
            )

            for tool_call in response.tool_calls:
                spec = self.registry.get(tool_call.name)
                with tracer.span(
                    f"tool:{tool_call.name}", "tool_call", args=tool_call.args
                ) as span:
                    result = self.registry.execute(tool_call.name, tool_call.args)
                    span.metadata["ok"] = result.ok

                if result.ok:
                    state.record_tool_success(tool_call.name)
                    if spec is not None and spec.terminal:
                        answer = None
                        if spec.answer_arg is not None:
                            answer = tool_call.args.get(spec.answer_arg)
                        state.record_terminal(tool_call.name, answer)
                    observation = result.content
                else:
                    error = result.error or "unknown error"
                    state.record_tool_error(tool_call.name, error)
                    observation = f"ERROR: {error}"
                    if (
                        state.error_signature_count(tool_call.name, error)
                        >= self.budgets.max_repeated_errors
                    ):
                        status = RunStatus.REPEATED_ERROR
                        stop_reason = (
                            f"tool {tool_call.name!r} failed identically "
                            f"{self.budgets.max_repeated_errors} times; "
                            "retrying further would spend budget on the same wall"
                        )

                state.append_message(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": observation,
                    }
                )
                if status is not None:
                    break
            if status is not None:
                break

            verdict = self.verifier.verify(state)
            if verdict.passed:
                status = RunStatus.SUCCESS
                stop_reason = "verifier passed"

        state.status = status.value
        state.record_event(
            "run_complete", {"status": status.value, "stop_reason": stop_reason}
        )
        return RunResult(
            status=status,
            state=state,
            tracer=tracer,
            stop_reason=stop_reason,
            wall_clock_seconds=time.monotonic() - started,
        )
