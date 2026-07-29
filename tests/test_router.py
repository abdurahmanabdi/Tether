"""Fallback chain and cost attribution."""

from __future__ import annotations

import pytest

from tests.helpers import FailingProvider, ScriptedProvider
from tether.router import AllProvidersFailed, Router
from tether.trace import Tracer


def test_router_falls_back_when_primary_raises():
    primary = FailingProvider(model="mock")
    secondary = ScriptedProvider([("get_ticket", {"ticket_id": "T-1001"})])
    router = Router([primary, secondary])
    tracer = Tracer()

    response, cost = router.complete(
        [{"role": "user", "content": "hi"}], tools=[], tracer=tracer
    )

    assert response.tool_calls[0].name == "get_ticket"
    # Both attempts are traced: the failure is visible, not swallowed.
    assert len(tracer.spans) == 2
    assert "error" in tracer.spans[0].metadata
    assert tracer.spans[1].metadata["tool_calls"] == ["get_ticket"]


def test_router_raises_only_when_every_provider_fails():
    router = Router([FailingProvider("mock"), FailingProvider("mock")])
    with pytest.raises(AllProvidersFailed) as excinfo:
        router.complete([{"role": "user", "content": "hi"}], tools=[], tracer=Tracer())
    assert len(excinfo.value.errors) == 2


def test_cost_is_attributed_to_the_span_that_incurred_it(monkeypatch):
    import tether.config as config

    monkeypatch.setitem(
        config.MODEL_PRICING, "mock", config.ModelPricing(3.00, 15.00)
    )
    provider = ScriptedProvider([("get_ticket", {"ticket_id": "T-1001"})])
    tracer = Tracer()
    _, cost = Router([provider]).complete(
        [{"role": "user", "content": "hi"}], tools=[], tracer=tracer
    )
    # ScriptedProvider reports 10 input + 10 output tokens.
    expected = (10 * 3.00 + 10 * 15.00) / 1_000_000
    assert cost == pytest.approx(expected)
    assert tracer.total_cost_usd() == pytest.approx(expected)
