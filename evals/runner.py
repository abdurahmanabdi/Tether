"""Suite execution and metric aggregation.

The headline metric is cost per accepted result, not pass rate. Both are
computed and stored side by side, along with p50/p95 latency and per-grader
pass rates, so a change cannot look like a win when it is a wash.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from domain.env import TicketEnv, build_registry
from domain.tickets import TICKETS
from tether.config import Budgets
from tether.loop import AgentLoop, ToolContractVerifier
from tether.prompts import load_prompt
from tether.router import Router
from tether.state import RunState

from .graders import GradingContext, grade


@dataclass
class CaseResult:
    case_id: str
    ticket_id: str
    passed: bool
    status: str
    stop_reason: str
    cost_usd: float
    latency_seconds: float
    grader_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "ticket_id": self.ticket_id,
            "passed": self.passed,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "cost_usd": self.cost_usd,
            "latency_seconds": self.latency_seconds,
            "graders": self.grader_results,
        }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return float(
        statistics.quantiles(values, n=100, method="inclusive")[int(pct) - 1]
    )


def load_suite(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        suite = yaml.safe_load(f)
    for key in ("suite", "cases"):
        if key not in suite:
            raise ValueError(f"suite file {path} missing required key {key!r}")
    return suite


def run_suite(
    suite_path: str | Path,
    router: Router,
    prompt_version: str | None = None,
    judge_router: Router | None = None,
    label: str = "run",
    state_dir: Path | None = None,
) -> dict[str, Any]:
    suite = load_suite(suite_path)
    prompt_name = suite.get("prompt", "triage")
    version = prompt_version or suite.get("prompt_version", "v1")
    system_prompt = load_prompt(prompt_name, version)
    default_budgets = Budgets().merged(suite.get("budgets"))
    verifier_cfg = suite.get("verifier", {})

    case_results: list[CaseResult] = []
    for case in suite["cases"]:
        case_id = case["id"]
        ticket_id = case["ticket_id"]
        ticket = TICKETS.get(ticket_id)
        if ticket is None:
            raise ValueError(f"case {case_id!r} references unknown ticket {ticket_id!r}")

        env = TicketEnv()
        registry = build_registry(env)
        verifier = ToolContractVerifier(
            required_tools=case.get(
                "required_tools", verifier_cfg.get("required_tools", [])
            )
        )
        budgets = default_budgets.merged(case.get("budgets"))
        state = RunState(
            jsonl_path=(state_dir / f"{case_id}.jsonl") if state_dir else None
        )
        loop = AgentLoop(router, registry, verifier, budgets)

        started = time.monotonic()
        run = loop.run(
            system_prompt=system_prompt,
            user_message=f"Please handle ticket {ticket_id}.",
            state=state,
        )
        latency = time.monotonic() - started

        grader_specs = list(case.get("graders", []))
        if judge_router is None:
            # A disabled judge is not a broken judge: skipping is explicit
            # configuration (no --judge flag), whereas a judge that errors or
            # emits garbage fails the case inside llm_judge itself.
            grader_specs = [g for g in grader_specs if g.get("type") != "llm_judge"]
        ctx = GradingContext(
            ticket=ticket, env=env, run=run, judge_router=judge_router
        )
        grader_results = grade(ctx, grader_specs)

        case_results.append(
            CaseResult(
                case_id=case_id,
                ticket_id=ticket_id,
                passed=all(g.passed for g in grader_results),
                status=run.status.value,
                stop_reason=run.stop_reason,
                cost_usd=run.cost_usd,
                latency_seconds=latency,
                grader_results=[
                    {"grader": g.grader, "passed": g.passed, "detail": g.detail}
                    for g in grader_results
                ],
            )
        )

    return {
        "suite": suite["suite"],
        "label": label,
        "prompt": prompt_name,
        "prompt_version": version,
        "model": router.providers[0].model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cases": [c.to_dict() for c in case_results],
        "metrics": compute_metrics(case_results),
    }


def compute_metrics(case_results: list[CaseResult]) -> dict[str, Any]:
    total = len(case_results)
    passed = sum(1 for c in case_results if c.passed)
    total_cost = sum(c.cost_usd for c in case_results)
    latencies = sorted(c.latency_seconds for c in case_results)

    per_grader: dict[str, dict[str, int]] = {}
    for c in case_results:
        for g in c.grader_results:
            bucket = per_grader.setdefault(g["grader"], {"passed": 0, "total": 0})
            bucket["total"] += 1
            bucket["passed"] += int(g["passed"])

    return {
        "cases_total": total,
        "cases_passed": passed,
        "pass_rate": (passed / total) if total else 0.0,
        "total_cost_usd": total_cost,
        # The headline metric. None when nothing was accepted: infinite cost
        # per result is a fact worth surfacing, not a division to hide.
        "cost_per_accepted_usd": (total_cost / passed) if passed else None,
        "latency_p50_seconds": _percentile(latencies, 50),
        "latency_p95_seconds": _percentile(latencies, 95),
        "per_grader_pass_rate": {
            name: {
                "passed": b["passed"],
                "total": b["total"],
                "rate": b["passed"] / b["total"] if b["total"] else 0.0,
            }
            for name, b in sorted(per_grader.items())
        },
    }


def save_result(result: dict[str, Any], out_dir: str | Path = "eval_runs") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out / f"{result['suite']}_{result['label']}_{stamp}.json"
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path
