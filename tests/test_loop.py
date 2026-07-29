"""Stop conditions: every exit is named, and the bad paths terminate."""

from __future__ import annotations

from domain.env import TicketEnv, build_registry
from tests.helpers import FailingProvider, ScriptedProvider, TextProvider
from tether.config import Budgets
from tether.loop import AgentLoop, RunStatus, ToolContractVerifier
from tether.router import Router


def make_loop(provider, budgets=None, required_tools=None):
    env = TicketEnv()
    registry = build_registry(env)
    loop = AgentLoop(
        Router([provider]),
        registry,
        ToolContractVerifier(required_tools or ["get_ticket"]),
        budgets or Budgets(max_iterations=5),
    )
    return loop, env


def run(loop):
    return loop.run("You are a triage agent.", "Please handle ticket T-1001.")


def test_verifier_that_never_passes_terminates_on_iteration_budget():
    # The agent only ever reads the ticket; no terminal action is ever taken,
    # so the verifier can never pass and the iteration budget must stop us.
    provider = ScriptedProvider([("get_ticket", {"ticket_id": "T-1001"})])
    loop, _ = make_loop(provider, Budgets(max_iterations=4))
    result = run(loop)
    assert result.status == RunStatus.MAX_ITERATIONS
    assert result.state.iterations == 4


def test_verifier_gates_success_deterministically():
    provider = ScriptedProvider(
        [
            ("get_ticket", {"ticket_id": "T-1001"}),
            ("resolve_ticket", {"ticket_id": "T-1001", "answer": "Reset it."}),
        ]
    )
    loop, env = make_loop(provider)
    result = run(loop)
    assert result.status == RunStatus.SUCCESS
    assert result.state.terminal_action == "resolve_ticket"
    assert result.state.final_answer == "Reset it."
    assert env.terminal_action == "resolve_ticket"


def test_repeated_identical_error_stops_the_loop():
    # get_ticket on a nonexistent ticket fails identically forever; the loop
    # must stop on the repeated-error budget, not grind out all iterations.
    provider = ScriptedProvider([("get_ticket", {"ticket_id": "T-9999"})])
    loop, _ = make_loop(provider, Budgets(max_iterations=50, max_repeated_errors=3))
    result = run(loop)
    assert result.status == RunStatus.REPEATED_ERROR
    assert result.state.iterations < 50


def test_all_providers_failing_is_model_error():
    loop, _ = make_loop(FailingProvider())
    result = run(loop)
    assert result.status == RunStatus.MODEL_ERROR


def test_cost_budget_stops_the_loop(monkeypatch):
    import tether.config as config

    # Price the mock model so every call costs money, then set a budget one
    # call can exceed.
    monkeypatch.setitem(config.MODEL_PRICING, "mock", config.ModelPricing(1e6, 1e6))
    provider = ScriptedProvider([("get_ticket", {"ticket_id": "T-1001"})])
    loop, _ = make_loop(provider, Budgets(max_iterations=50, max_cost_usd=0.01))
    result = run(loop)
    assert result.status == RunStatus.BUDGET_EXCEEDED
    assert "cost" in result.stop_reason


def test_text_only_reply_does_not_count_as_an_answer():
    # A model that "answers" in prose without calling a terminal tool has not
    # answered: the contract requires a terminal action.
    provider = TextProvider("The user should just reset their password.")
    loop, _ = make_loop(provider, Budgets(max_iterations=3))
    result = run(loop)
    assert result.status == RunStatus.MAX_ITERATIONS
    assert result.state.final_answer is None
