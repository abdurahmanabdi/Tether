"""Run an eval suite and optionally gate against a stored baseline.

Examples:
    # offline, free, deterministic — this is what CI runs
    python scripts/run_eval.py --suite evals/cases/triage_basic.yaml --mock

    # live, against a real model and a chosen prompt version
    python scripts/run_eval.py --suite evals/cases/triage_basic.yaml \
        --prompt-version v2 --judge --label sonnet-v2

    # gate a change against a stored baseline (exit 1 on regression)
    python scripts/run_eval.py --suite evals/cases/triage_basic.yaml \
        --prompt-version v2 --baseline "eval_runs/triage_basic_sonnet-v1_*.json"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Report output is UTF-8 markdown (em dashes, arrows) meant for PR comments;
# on Windows the console's default cp1252 encoding can't print it and crashes.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from evals.regression import compare, load_baseline
from evals.report import render_regression_report, render_suite_report
from evals.runner import run_suite, save_result
from tether.router import AnthropicProvider, MockProvider, Router


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, help="path to a suite YAML")
    parser.add_argument(
        "--mock", action="store_true", help="use the offline mock provider"
    )
    parser.add_argument(
        "--model", default="claude-sonnet-5", help="primary model for live runs"
    )
    parser.add_argument(
        "--fallback-model",
        default="claude-haiku-4-5-20251001",
        help="fallback model for live runs",
    )
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument(
        "--judge",
        action="store_true",
        help="run llm_judge graders (live only; requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument("--label", default=None, help="label stored in the result file")
    parser.add_argument(
        "--baseline",
        default=None,
        help="glob for a stored baseline result; exit 1 if this run regresses it",
    )
    parser.add_argument("--out-dir", default="eval_runs")
    args = parser.parse_args()

    if args.mock:
        router = Router([MockProvider()])
        judge_router = None
        if args.judge:
            print(
                "note: --judge ignored with --mock; the judge needs a real model",
                file=sys.stderr,
            )
        label = args.label or "mock"
    else:
        router = Router(
            [AnthropicProvider(args.model), AnthropicProvider(args.fallback_model)]
        )
        # The judge deliberately runs the cheaper model with its own prompt:
        # a model reviewing its own work grades generously.
        judge_router = (
            Router([AnthropicProvider(args.fallback_model)]) if args.judge else None
        )
        label = args.label or args.model

    result = run_suite(
        args.suite,
        router=router,
        prompt_version=args.prompt_version,
        judge_router=judge_router,
        label=label,
    )
    saved_to = save_result(result, args.out_dir)

    print(render_suite_report(result))
    print(f"Result saved to {saved_to}")

    if args.baseline:
        baseline = load_baseline(args.baseline)
        regression = compare(baseline, result)
        print()
        print(render_regression_report(regression))
        if regression.is_regression:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
