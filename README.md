# Tether

A minimal agent execution harness with first-class evaluation infrastructure.

Tether runs a multi-step LLM agent against a set of tools, verifies its output
before accepting it, and stops on explicit budgets. The part worth reading is
`evals/`: a grading suite that scores runs deterministically, compares a
candidate prompt or model against a stored baseline, and fails with a non-zero
exit code when quality regresses.

## What this is and is not

**Is:** a working harness I built to study reliability and evaluation patterns
for tool-using agents — orchestration, fallback, tracing, prompt versioning,
regression detection, and production invariants, on a synthetic domain with
checkable ground truth.

**Is not:** a production system. It has never served real users, the ticket
domain is fabricated, and the cost figures in `tether/config.py` are
illustrative. Any number in this repo comes from running the suite in this repo.

That distinction is stated up front because an eval framework whose own claims
are unverifiable would be self-refuting.

## Why evaluation is the centerpiece

Repetition without a gate is not progress — it is a model agreeing with itself
more expensively. Three design choices follow from that:

1. **The verifier is not the actor.** `ToolContractVerifier` gates the loop's
   exit deterministically: required tools called, a terminal action taken, a
   non-empty answer. It cannot be talked out of a failure. Quality judgement
   that genuinely needs a model lives in `evals/graders.py::llm_judge`, which
   runs a *different* prompt and can run on a *different* model — because a
   model reviewing its own work grades generously.
2. **A broken judge fails closed.** If the judge's output cannot be parsed, the
   case fails. A grader that silently passes when it breaks is worse than no
   grader.
3. **The headline metric is cost per accepted result, not pass rate.** A suite
   that passes more cases at four times the cost has not improved anything.
   Both are reported side by side, plus p50/p95 latency and per-grader pass
   rates, so a change cannot look like a win when it is a wash.

## Architecture

```
tether/
  loop.py       execution core: plan -> act -> observe -> verify -> iterate -> stop
  router.py     provider abstraction, fallback chain, per-call cost attribution
  tools.py      registry, schema validation, retries; errors are values, not crashes
  state.py      run state persisted to JSONL outside the model
  trace.py      spans with cost and latency attribution
  config.py     budgets (every field is a stop condition) and model pricing
  prompts/      versioned templates on disk, addressed by name + version
domain/         synthetic enterprise ticket domain with known-correct answers
evals/
  graders.py    deterministic graders + LLM judge
  runner.py     suite execution, metric aggregation
  regression.py baseline vs candidate comparison
  report.py     markdown output for PR comments
  cases/        YAML suites with per-case expectations and budgets
monitors/
  invariants.py cheap assertions for sampled online traffic
```

### Stop conditions

Every exit is named. A loop whose only exit is success is a loop that can run
until the budget is gone:

| Status | Cause |
| --- | --- |
| `SUCCESS` | verifier passed |
| `MAX_ITERATIONS` | iteration budget reached |
| `BUDGET_EXCEEDED` | wall-clock or cost budget exceeded |
| `REPEATED_ERROR` | same tool error more than N times — no point retrying |
| `MODEL_ERROR` | every model in the fallback chain failed |

### Failure paths are what the tests cover

`tests/` asserts on the paths that matter: a verifier that never passes
terminates on the iteration budget; an unknown tool becomes an agent-visible
error instead of a crash; schema violations are caught before execution; a
failing tool is retried then reported; the router falls back when the primary
model raises; and a monitor catches the agent auto-answering a ticket that
required escalation.

## Running it

```bash
pip install -r requirements.txt

# offline, free, deterministic — this is what CI runs
python scripts/run_eval.py --suite evals/cases/triage_basic.yaml --mock

# live, against a real model and a chosen prompt version
cp .env.example .env   # add your ANTHROPIC_API_KEY
python scripts/run_eval.py --suite evals/cases/triage_basic.yaml \
    --prompt-version v2 --judge --label sonnet-v2

# gate a change against a stored baseline (exit 1 on regression)
python scripts/run_eval.py --suite evals/cases/triage_basic.yaml \
    --prompt-version v2 --baseline "eval_runs/triage_basic_sonnet-v1_*.json"

pytest -q
```

The mock provider is deliberately imperfect: it always categorizes tickets as
`access`, so the suite reports a 33% pass rate offline and names
`category_is` as the failing grader on the other two cases. The suite catching
its own fixture is the demonstration.

### The escalation suite is the interesting one

`evals/cases/triage_escalation.yaml` holds three tickets that look answerable
but must be escalated: deletion of a former employee's mailbox, a billing
dispute, and a request for access to a colleague's private drive. An agent that
helpfully resolves these is the exact failure mode worth catching before
production, and `monitors/invariants.py` asserts against it on live runs too.

## Roadmap

- [ ] Per-case retry strategies (reflect-and-retry vs. plain retry) with an eval
      comparing them on cost per accepted result
- [ ] Sampled online monitoring: run invariants on a percentage of live traffic
      and alert on violation rate
- [ ] Multi-agent decomposition (planner + executor) measured against the
      single-agent baseline — including whether the extra tokens buy anything
- [ ] Statistical significance on regression detection: current comparison flags
      any newly-broken case, which is noisy on small suites
- [ ] Concurrency in the suite runner; currently cases run serially
