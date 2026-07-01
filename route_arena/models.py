"""Model registry: pricing, nominal latency, and provider metadata.

Prices are USD per 1,000,000 tokens and reflect public list pricing as of
mid-2025. They are intentionally kept in one place so the cost model is easy
to audit and update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# Prompt categories the router understands. Quality is tracked per-category
# because a model that is great at code may be mediocre at summarization.
CATEGORIES: List[str] = ["code", "summarization", "reasoning", "extraction"]


@dataclass(frozen=True)
class ModelSpec:
    """Static description of a candidate model.

    Attributes:
        name: Canonical model id used everywhere (registry key, results.json).
        provider: "openai", "anthropic", or "meta" (routed via a host).
        input_price: USD per 1M input (prompt) tokens.
        output_price: USD per 1M output (completion) tokens.
        base_latency_ms: Nominal end-to-end p50 latency, used as a prior until
            real observations accumulate in the rolling latency history.
        context_window: Max context in tokens (informational).
    """

    name: str
    provider: str
    input_price: float
    output_price: float
    base_latency_ms: float
    context_window: int

    def token_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimated USD cost for a single request of the given token shape."""
        return (
            input_tokens * self.input_price + output_tokens * self.output_price
        ) / 1_000_000


# The candidate pool the router chooses between.
_SPECS: List[ModelSpec] = [
    ModelSpec("gpt-4o", "openai", 2.50, 10.00, 1400, 128_000),
    ModelSpec("gpt-4o-mini", "openai", 0.15, 0.60, 700, 128_000),
    ModelSpec("claude-sonnet", "anthropic", 3.00, 15.00, 1600, 200_000),
    ModelSpec("claude-haiku", "anthropic", 0.80, 4.00, 600, 200_000),
    ModelSpec("llama-3-70b", "meta", 0.59, 0.79, 900, 8_000),
]

MODEL_REGISTRY: Dict[str, ModelSpec] = {spec.name: spec for spec in _SPECS}


def all_models() -> List[ModelSpec]:
    """Return the candidate models in a stable order."""
    return list(_SPECS)
