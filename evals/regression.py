"""Baseline vs candidate comparison.

A regression is a case that passed in the baseline and fails in the
candidate. That rule is deliberately strict and admittedly noisy on small
suites (see the roadmap note on statistical significance); the current
position is that on a suite this size, a newly-broken case is worth a human
look every time.

Cost is compared but does not gate: a cost increase is reported, a
correctness regression fails the build.
"""

from __future__ import annotations

import glob as globlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RegressionReport:
    baseline_label: str
    candidate_label: str
    newly_broken: list[str] = field(default_factory=list)
    newly_fixed: list[str] = field(default_factory=list)
    still_broken: list[str] = field(default_factory=list)
    missing_in_candidate: list[str] = field(default_factory=list)
    baseline_cost_per_accepted: float | None = None
    candidate_cost_per_accepted: float | None = None

    @property
    def is_regression(self) -> bool:
        # A case the baseline covered but the candidate did not run is also a
        # regression: silently shrinking the suite must not look like passing.
        return bool(self.newly_broken or self.missing_in_candidate)


def load_baseline(pattern: str) -> dict[str, Any]:
    """Resolve a glob to the most recent matching baseline file."""
    matches = sorted(globlib.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no baseline file matches {pattern!r}")
    # utf-8-sig tolerates a BOM, which Windows editors like to add to JSON.
    return json.loads(Path(matches[-1]).read_text(encoding="utf-8-sig"))


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> RegressionReport:
    base_cases = {c["case_id"]: c["passed"] for c in baseline["cases"]}
    cand_cases = {c["case_id"]: c["passed"] for c in candidate["cases"]}

    report = RegressionReport(
        baseline_label=baseline.get("label", "baseline"),
        candidate_label=candidate.get("label", "candidate"),
        baseline_cost_per_accepted=baseline["metrics"].get("cost_per_accepted_usd"),
        candidate_cost_per_accepted=candidate["metrics"].get("cost_per_accepted_usd"),
    )
    for case_id, base_passed in base_cases.items():
        if case_id not in cand_cases:
            report.missing_in_candidate.append(case_id)
        elif base_passed and not cand_cases[case_id]:
            report.newly_broken.append(case_id)
        elif not base_passed and cand_cases[case_id]:
            report.newly_fixed.append(case_id)
        elif not base_passed:
            report.still_broken.append(case_id)
    return report
