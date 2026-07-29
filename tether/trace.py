"""Spans with cost and latency attribution.

A trace is a flat list of spans. Every model call and every tool execution
opens a span; cost is attributed to the span that incurred it, so the answer
to "where did the money go" is a sum over spans, not a guess.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class Span:
    name: str
    kind: str  # "model_call" | "tool_call" | "run"
    started_at: float
    ended_at: float | None = None
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def latency_seconds(self) -> float:
        if self.ended_at is None:
            return 0.0
        return self.ended_at - self.started_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "name": self.name,
            "kind": self.kind,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "latency_seconds": self.latency_seconds,
            "cost_usd": self.cost_usd,
            "metadata": self.metadata,
        }


class Tracer:
    def __init__(self) -> None:
        self.spans: list[Span] = []

    @contextmanager
    def span(self, name: str, kind: str, **metadata: Any) -> Iterator[Span]:
        s = Span(name=name, kind=kind, started_at=time.time(), metadata=dict(metadata))
        self.spans.append(s)
        try:
            yield s
        finally:
            s.ended_at = time.time()

    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    def total_latency_seconds(self, kind: str | None = None) -> float:
        return sum(
            s.latency_seconds for s in self.spans if kind is None or s.kind == kind
        )

    def to_dicts(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.spans]
