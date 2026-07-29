"""Budgets and model pricing.

Every field on Budgets is a stop condition. If a field cannot stop the loop,
it does not belong here.

Pricing figures are illustrative (USD per million tokens). They exist so that
cost attribution and cost-based stop conditions are exercised end to end, not
because they are quotes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Budgets:
    """Hard limits on a single run. Each field independently stops the loop."""

    max_iterations: int = 10
    max_wall_clock_seconds: float = 120.0
    max_cost_usd: float = 1.00
    max_repeated_errors: int = 3  # same tool error signature seen this many times

    def merged(self, overrides: dict | None) -> "Budgets":
        """Return a copy with any non-None overrides applied."""
        if not overrides:
            return self
        current = {
            "max_iterations": self.max_iterations,
            "max_wall_clock_seconds": self.max_wall_clock_seconds,
            "max_cost_usd": self.max_cost_usd,
            "max_repeated_errors": self.max_repeated_errors,
        }
        for key, value in overrides.items():
            if key not in current:
                raise KeyError(f"unknown budget field: {key}")
            if value is not None:
                current[key] = value
        return Budgets(**current)


@dataclass(frozen=True)
class ModelPricing:
    """USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_mtok
            + output_tokens * self.output_per_mtok
        ) / 1_000_000


MODEL_PRICING: dict[str, ModelPricing] = {
    "claude-sonnet-5": ModelPricing(3.00, 15.00),
    "claude-haiku-4-5-20251001": ModelPricing(1.00, 5.00),
    "mock": ModelPricing(0.0, 0.0),
}


def pricing_for(model: str) -> ModelPricing:
    if model not in MODEL_PRICING:
        raise KeyError(
            f"no pricing entry for model {model!r}; add one to "
            "tether.config.MODEL_PRICING so cost attribution stays honest"
        )
    return MODEL_PRICING[model]


@dataclass
class RunConfig:
    """Everything a single run needs beyond the domain itself."""

    prompt_name: str = "triage"
    prompt_version: str = "v1"
    budgets: Budgets = field(default_factory=Budgets)
