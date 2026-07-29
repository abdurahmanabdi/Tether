"""Deterministic graders + LLM judge.

Deterministic graders compare what the agent *did* (recorded in TicketEnv and
RunState) against ground truth. They are cheap, they run on every case, and
they cannot be argued with.

The LLM judge exists for the judgement that genuinely needs a model — "is
this resolution actually actionable" — and it runs a different prompt,
optionally on a different model, than the actor. If its output cannot be
parsed, the case FAILS. A grader that silently passes when it breaks is worse
than no grader.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from domain.env import TicketEnv
from domain.tickets import Ticket
from tether.loop import RunResult, RunStatus
from tether.prompts import load_prompt
from tether.router import Router
from tether.trace import Tracer


@dataclass
class GraderResult:
    grader: str
    passed: bool
    detail: str = ""


@dataclass
class GradingContext:
    ticket: Ticket
    env: TicketEnv
    run: RunResult
    judge_router: Router | None = None
    judge_tracer: Tracer | None = None


# --- deterministic graders ------------------------------------------------


def run_succeeded(ctx: GradingContext, **_: Any) -> GraderResult:
    ok = ctx.run.status == RunStatus.SUCCESS
    return GraderResult(
        "run_succeeded",
        ok,
        "" if ok else f"status was {ctx.run.status.value}: {ctx.run.stop_reason}",
    )


def category_is(ctx: GradingContext, expected: str, **_: Any) -> GraderResult:
    actual = ctx.env.assigned_category
    ok = actual == expected
    return GraderResult(
        "category_is",
        ok,
        "" if ok else f"expected {expected!r}, agent assigned {actual!r}",
    )


def escalated(ctx: GradingContext, **_: Any) -> GraderResult:
    actual = ctx.env.terminal_action
    ok = actual == "escalate_ticket"
    return GraderResult(
        "escalated",
        ok,
        "" if ok else f"ticket required escalation but agent took {actual!r}",
    )


def answer_contains(
    ctx: GradingContext, keywords: list[str], **_: Any
) -> GraderResult:
    answer = (ctx.run.state.final_answer or "").lower()
    missing = [k for k in keywords if k.lower() not in answer]
    ok = not missing
    return GraderResult(
        "answer_contains",
        ok,
        "" if ok else f"answer missing keyword(s): {', '.join(missing)}",
    )


def required_tools_called(
    ctx: GradingContext, tools: list[str], **_: Any
) -> GraderResult:
    called = set(ctx.run.state.tools_called_ok)
    missing = [t for t in tools if t not in called]
    ok = not missing
    return GraderResult(
        "required_tools_called",
        ok,
        "" if ok else f"never called: {', '.join(missing)}",
    )


def cost_under(ctx: GradingContext, max_usd: float, **_: Any) -> GraderResult:
    cost = ctx.run.cost_usd
    ok = cost <= max_usd
    return GraderResult(
        "cost_under",
        ok,
        "" if ok else f"run cost ${cost:.4f} > ${max_usd:.4f}",
    )


# --- LLM judge ------------------------------------------------------------


def llm_judge(
    ctx: GradingContext,
    judge_prompt_version: str = "v1",
    **_: Any,
) -> GraderResult:
    """A different prompt, and optionally a different model, than the actor.

    Fails closed: no judge router, an errored call, or unparseable output all
    fail the case rather than quietly passing it.
    """
    if ctx.judge_router is None:
        return GraderResult("llm_judge", False, "judge requested but no judge router")

    action = ctx.env.terminal_action or "(no terminal action)"
    answer = ctx.run.state.final_answer or "(empty)"
    system = load_prompt("judge", judge_prompt_version)
    user = (
        f"TICKET:\n{ctx.ticket.render()}\n\n"
        f"AGENT ACTION: {action}\n\n"
        f"AGENT FINAL ANSWER:\n{answer}"
    )
    tracer = ctx.judge_tracer or Tracer()
    try:
        response, _cost = ctx.judge_router.complete(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=[],
            tracer=tracer,
        )
    except Exception as exc:
        return GraderResult("llm_judge", False, f"judge call failed: {exc}")

    verdict = _parse_judge_output(response.text)
    if verdict is None:
        return GraderResult(
            "llm_judge",
            False,
            f"judge output unparseable (fails closed): {response.text[:200]!r}",
        )
    passed, reasoning = verdict
    return GraderResult("llm_judge", passed, reasoning)


def _parse_judge_output(text: str) -> tuple[bool, str] | None:
    """Extract {"pass": bool, "reasoning": str} or return None."""
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("pass"), bool):
        return None
    return obj["pass"], str(obj.get("reasoning", ""))


GRADERS = {
    "run_succeeded": run_succeeded,
    "category_is": category_is,
    "escalated": escalated,
    "answer_contains": answer_contains,
    "required_tools_called": required_tools_called,
    "cost_under": cost_under,
    "llm_judge": llm_judge,
}


def grade(ctx: GradingContext, grader_specs: list[dict[str, Any]]) -> list[GraderResult]:
    results: list[GraderResult] = []
    for spec in grader_specs:
        spec = dict(spec)
        grader_type = spec.pop("type", None)
        fn = GRADERS.get(grader_type)
        if fn is None:
            # An unknown grader in a suite is a config bug; fail the case
            # loudly rather than skipping the check.
            results.append(
                GraderResult(
                    str(grader_type), False, f"unknown grader type {grader_type!r}"
                )
            )
            continue
        results.append(fn(ctx, **spec))
    return results
