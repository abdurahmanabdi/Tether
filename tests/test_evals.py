"""The eval suite's own contracts: mock pass rate, judge fails closed,
regression gating."""

from __future__ import annotations

import pytest

from evals.graders import _parse_judge_output
from evals.regression import compare
from evals.runner import run_suite
from tether.router import MockProvider, Router


@pytest.fixture(scope="module")
def mock_basic_result():
    return run_suite(
        "evals/cases/triage_basic.yaml", Router([MockProvider()]), label="mock"
    )


def test_mock_suite_reports_one_third_pass_rate(mock_basic_result):
    # The mock provider always categorizes as `access`: only the access case
    # passes. The suite catching its own fixture is the demonstration.
    metrics = mock_basic_result["metrics"]
    assert metrics["cases_total"] == 3
    assert metrics["cases_passed"] == 1
    assert metrics["pass_rate"] == pytest.approx(1 / 3)


def test_mock_failures_name_category_is_as_the_failing_grader(mock_basic_result):
    for case in mock_basic_result["cases"]:
        failing = [g["grader"] for g in case["graders"] if not g["passed"]]
        if case["case_id"] == "basic_access_lockout":
            assert failing == []
        else:
            assert failing == ["category_is"]


def test_mock_provider_fails_the_entire_escalation_suite():
    result = run_suite(
        "evals/cases/triage_escalation.yaml", Router([MockProvider()]), label="mock"
    )
    assert result["metrics"]["cases_passed"] == 0
    assert result["metrics"]["cost_per_accepted_usd"] is None
    for case in result["cases"]:
        failing = [g["grader"] for g in case["graders"] if not g["passed"]]
        assert "escalated" in failing


def test_judge_output_parsing_fails_closed():
    assert _parse_judge_output('{"pass": true, "reasoning": "fine"}') == (True, "fine")
    assert _parse_judge_output('noise {"pass": false, "reasoning": "bad"} noise') == (
        False,
        "bad",
    )
    # Everything unparseable or malformed is None, which llm_judge turns
    # into a failed case: a broken judge must not silently pass anything.
    assert _parse_judge_output("I think this looks great!") is None
    assert _parse_judge_output('{"pass": "yes"}') is None
    assert _parse_judge_output('{"verdict": true}') is None
    assert _parse_judge_output("") is None


def _fake_result(label, passed_by_case):
    cases = [
        {"case_id": cid, "passed": passed} for cid, passed in passed_by_case.items()
    ]
    return {
        "label": label,
        "cases": cases,
        "metrics": {"cost_per_accepted_usd": 0.01},
    }


def test_regression_flags_newly_broken_case():
    baseline = _fake_result("v1", {"a": True, "b": True, "c": False})
    candidate = _fake_result("v2", {"a": True, "b": False, "c": False})
    report = compare(baseline, candidate)
    assert report.is_regression
    assert report.newly_broken == ["b"]
    assert report.still_broken == ["c"]


def test_regression_treats_a_dropped_case_as_a_regression():
    baseline = _fake_result("v1", {"a": True, "b": True})
    candidate = _fake_result("v2", {"a": True})
    report = compare(baseline, candidate)
    assert report.is_regression
    assert report.missing_in_candidate == ["b"]


def test_no_regression_when_candidate_only_improves():
    baseline = _fake_result("v1", {"a": True, "b": False})
    candidate = _fake_result("v2", {"a": True, "b": True})
    report = compare(baseline, candidate)
    assert not report.is_regression
    assert report.newly_fixed == ["b"]
