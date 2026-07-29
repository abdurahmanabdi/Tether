"""The production invariants, exercised against real runs of the loop."""

from __future__ import annotations

from domain.env import TicketEnv, build_registry
from domain.tickets import TICKETS
from monitors.invariants import check_run, escalation_respected
from tests.helpers import ScriptedProvider
from tether.config import Budgets
from tether.loop import AgentLoop, ToolContractVerifier
from tether.router import MockProvider, Router


def run_ticket(provider, ticket_id):
    env = TicketEnv()
    loop = AgentLoop(
        Router([provider]),
        build_registry(env),
        ToolContractVerifier(["get_ticket"]),
        Budgets(max_iterations=6),
    )
    result = loop.run(
        "You are a triage agent.", f"Please handle ticket {ticket_id}."
    )
    return env, result


def test_monitor_catches_auto_answered_escalation_ticket():
    # T-2001 (delete a former employee's mailbox) requires escalation. The
    # mock provider helpfully resolves it — the exact failure mode the
    # monitor exists to catch.
    ticket = TICKETS["T-2001"]
    env, result = run_ticket(MockProvider(), "T-2001")

    assert env.terminal_action == "resolve_ticket"  # the agent did misbehave
    violations = check_run(ticket, env, result)
    names = [v.invariant for v in violations]
    assert "escalation_respected" in names


def test_monitor_is_silent_when_escalation_is_respected():
    ticket = TICKETS["T-2001"]
    provider = ScriptedProvider(
        [
            ("get_ticket", {"ticket_id": "T-2001"}),
            (
                "escalate_ticket",
                {"ticket_id": "T-2001", "reason": "Permanent data deletion."},
            ),
        ]
    )
    env, result = run_ticket(provider, "T-2001")
    assert escalation_respected(ticket, env, result) is None
    assert check_run(ticket, env, result) == []


def test_monitor_is_silent_on_a_routine_ticket_resolved_normally():
    ticket = TICKETS["T-1001"]
    env, result = run_ticket(MockProvider(), "T-1001")
    assert check_run(ticket, env, result) == []
