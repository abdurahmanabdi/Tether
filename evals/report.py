"""Markdown output for PR comments."""

from __future__ import annotations

from typing import Any

from .regression import RegressionReport


def _fmt_cost(value: float | None) -> str:
    return f"${value:.4f}" if value is not None else "n/a (0 accepted)"


def render_suite_report(result: dict[str, Any]) -> str:
    m = result["metrics"]
    lines = [
        f"## Eval: `{result['suite']}` — {result['label']}",
        "",
        f"Prompt `{result['prompt']}/{result['prompt_version']}` on "
        f"`{result['model']}` at {result['timestamp']}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| **Cost per accepted result** | {_fmt_cost(m['cost_per_accepted_usd'])} |",
        f"| Pass rate | {m['cases_passed']}/{m['cases_total']} "
        f"({m['pass_rate']:.0%}) |",
        f"| Total cost | {_fmt_cost(m['total_cost_usd'])} |",
        f"| Latency p50 / p95 | {m['latency_p50_seconds']:.2f}s / "
        f"{m['latency_p95_seconds']:.2f}s |",
        "",
        "### Per-grader pass rates",
        "",
        "| Grader | Passed | Rate |",
        "| --- | --- | --- |",
    ]
    for name, bucket in m["per_grader_pass_rate"].items():
        lines.append(
            f"| `{name}` | {bucket['passed']}/{bucket['total']} "
            f"| {bucket['rate']:.0%} |"
        )

    lines += ["", "### Cases", "", "| Case | Result | Status | Failing graders |",
              "| --- | --- | --- | --- |"]
    for case in result["cases"]:
        failing = ", ".join(
            f"`{g['grader']}`" for g in case["graders"] if not g["passed"]
        ) or "—"
        lines.append(
            f"| `{case['case_id']}` | {'PASS' if case['passed'] else 'FAIL'} "
            f"| {case['status']} | {failing} |"
        )
    return "\n".join(lines) + "\n"


def render_regression_report(report: RegressionReport) -> str:
    verdict = "REGRESSION" if report.is_regression else "no regression"
    lines = [
        f"## Regression check: `{report.candidate_label}` vs "
        f"`{report.baseline_label}` — **{verdict}**",
        "",
        f"Cost per accepted result: {_fmt_cost(report.baseline_cost_per_accepted)} "
        f"→ {_fmt_cost(report.candidate_cost_per_accepted)}",
        "",
    ]
    sections = [
        ("Newly broken", report.newly_broken),
        ("Missing from candidate", report.missing_in_candidate),
        ("Newly fixed", report.newly_fixed),
        ("Still broken", report.still_broken),
    ]
    for title, case_ids in sections:
        if case_ids:
            lines.append(f"**{title}:** " + ", ".join(f"`{c}`" for c in case_ids))
            lines.append("")
    return "\n".join(lines)
