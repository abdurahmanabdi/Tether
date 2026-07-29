"""Cheap assertions for sampled online traffic.

Evals run before a change ships; invariants run after. They are deterministic
predicates over a completed run — no model calls, microseconds each — so they
can be applied to a sample of live traffic and alert on violation rate.

The one that matters most: an agent must never auto-answer a ticket that
required escalation. That is the escalation suite's check, restated as a
production assertion, because a property worth gating a PR on is worth
watching in production too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from domain.env import TicketEnv
from domain.tickets import Ticket
from tether.loop import RunResult


@dataclass
class InvariantViolation:
    invariant: str
    detail: str


def escalation_respected(
    ticket: Ticket, env: TicketEnv, run: RunResult
) -> InvariantViolation | None:
    """A ticket that requires escalation must never be auto-resolved."""
    if ticket.requires_escalation and env.terminal_action == "resolve_ticket":
        return InvariantViolation(
            "escalation_respected",
            f"ticket {ticket.ticket_id} requires escalation but was "
            f"auto-resolved with: {env.resolution_answer!r}",
        )
    return None


def terminal_action_is_exclusive(
    ticket: Ticket, env: TicketEnv, run: RunResult
) -> InvariantViolation | None:
    """Exactly one terminal outcome: a run cannot both resolve and escalate."""
    if env.resolution_answer is not None and env.escalation_reason is not None:
        return InvariantViolation(
            "terminal_action_is_exclusive",
            f"ticket {ticket.ticket_id} was both resolved and escalated",
        )
    return None


def cost_attributed(
    ticket: Ticket, env: TicketEnv, run: RunResult
) -> InvariantViolation | None:
    """Recorded run cost must equal the sum of its trace spans."""
    span_cost = run.tracer.total_cost_usd()
    if abs(span_cost - run.cost_usd) > 1e-9:
        return InvariantViolation(
            "cost_attributed",
            f"run cost ${run.cost_usd:.6f} != sum of spans ${span_cost:.6f}; "
            "some spend is unattributed",
        )
    return None


def status_is_named(
    ticket: Ticket, env: TicketEnv, run: RunResult
) -> InvariantViolation | None:
    """Every run must end with an explicit status and stop reason."""
    if run.state.status is None or not run.stop_reason:
        return InvariantViolation(
            "status_is_named",
            f"run for {ticket.ticket_id} ended without a named stop condition",
        )
    return None


INVARIANTS: list[Callable[[Ticket, TicketEnv, RunResult], InvariantViolation | None]] = [
    escalation_respected,
    terminal_action_is_exclusive,
    cost_attributed,
    status_is_named,
]


def check_run(
    ticket: Ticket, env: TicketEnv, run: RunResult
) -> list[InvariantViolation]:
    violations = []
    for invariant in INVARIANTS:
        violation = invariant(ticket, env, run)
        if violation is not None:
            violations.append(violation)
    return violations
