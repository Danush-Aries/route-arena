"""Routing strategies and the Router that applies them.

Three strategies ship out of the box:

  * cost_optimal  - cheapest model that clears a quality floor.
  * quality_first - highest quality, ties broken by cost.
  * balanced      - weighted Pareto: pick the frontier model with the best
                    weighted blend of cost/quality/latency scores.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from route_arena.models import CATEGORIES, ModelSpec, all_models
from route_arena.router.scoring import (
    Candidate,
    LatencyHistory,
    classify_prompt,
    estimate_tokens,
    normalize_candidates,
    pareto_frontier,
)

_DEFAULT_QUALITY_PATH = Path(__file__).resolve().parent.parent / "eval" / "results.json"

# When a category has no eval score for a model, fall back to this neutral ELO.
_NEUTRAL_QUALITY = 1000.0


def load_quality_table(path: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
    """Load per-model, per-category quality (ELO) scores from results.json.

    Returns a mapping ``{model_name: {category: elo}}``. Missing files yield an
    empty table (the router then treats every model as neutral quality).
    """
    path = path or _DEFAULT_QUALITY_PATH
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return data.get("quality", data)


@dataclass
class RoutingDecision:
    """The result of routing one prompt with one strategy."""

    strategy: str
    prompt: str
    category: str
    chosen: str
    projected_cost_usd: float
    projected_latency_ms: float
    projected_quality: float
    rationale: str
    candidates: List[Candidate] = field(default_factory=list)
    frontier: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "strategy": self.strategy,
            "category": self.category,
            "chosen": self.chosen,
            "projected_cost_usd": round(self.projected_cost_usd, 6),
            "projected_latency_ms": round(self.projected_latency_ms, 1),
            "projected_quality": round(self.projected_quality, 1),
            "rationale": self.rationale,
            "frontier": self.frontier,
        }


# A strategy picks one Candidate from the scored list. It may consult the
# frontier (already computed) for Pareto-aware behaviour.
Strategy = Callable[[List[Candidate], List[Candidate]], Candidate]


def _cost_optimal(cands: List[Candidate], frontier: List[Candidate]) -> Candidate:
    """Cheapest model that clears a relative quality floor.

    The floor is 80% of the way from the worst to the best available quality,
    so we never route to a genuinely weak model just to save fractions of a
    cent. Among qualifying models, pick the lowest raw cost.
    """
    qualities = [c.quality for c in cands]
    lo, hi = min(qualities), max(qualities)
    floor = lo + 0.80 * (hi - lo)
    eligible = [c for c in cands if c.quality >= floor] or cands
    return min(eligible, key=lambda c: (c.cost_usd, -c.quality))


def _quality_first(cands: List[Candidate], frontier: List[Candidate]) -> Candidate:
    """Highest quality; break ties by cheaper cost then lower latency."""
    return max(cands, key=lambda c: (c.quality, -c.cost_usd, -c.latency_ms))


# Weights for the balanced blend (sum to 1.0). Quality is weighted highest,
# then cost, then latency — a defensible default for general workloads.
_BALANCED_WEIGHTS = {"quality": 0.5, "cost": 0.3, "latency": 0.2}


def _weighted_score(c: Candidate) -> float:
    return (
        _BALANCED_WEIGHTS["quality"] * c.quality_score
        + _BALANCED_WEIGHTS["cost"] * c.cost_score
        + _BALANCED_WEIGHTS["latency"] * c.latency_score
    )


def _balanced(cands: List[Candidate], frontier: List[Candidate]) -> Candidate:
    """Weighted Pareto: choose the frontier model with the best weighted blend.

    Restricting the choice to the Pareto frontier guarantees we never pick a
    model that is dominated (strictly worse on every axis) by another.
    """
    pool = frontier or cands
    return max(pool, key=_weighted_score)


STRATEGIES: Dict[str, Strategy] = {
    "cost_optimal": _cost_optimal,
    "quality_first": _quality_first,
    "balanced": _balanced,
}


class Router:
    """Scores candidate models for a prompt and applies a routing strategy."""

    def __init__(
        self,
        quality_table: Optional[Dict[str, Dict[str, float]]] = None,
        latency_history: Optional[LatencyHistory] = None,
        models: Optional[List[ModelSpec]] = None,
    ) -> None:
        self.quality_table = (
            quality_table if quality_table is not None else load_quality_table()
        )
        self.latency = latency_history or LatencyHistory()
        self.models = models or all_models()

    # -- quality lookup ---------------------------------------------------- #
    def quality_for(self, model: str, category: str) -> float:
        return self.quality_table.get(model, {}).get(category, _NEUTRAL_QUALITY)

    # -- candidate construction ------------------------------------------- #
    def build_candidates(self, prompt: str, category: str) -> List[Candidate]:
        input_tokens, output_tokens = estimate_tokens(prompt, category)
        candidates: List[Candidate] = []
        for spec in self.models:
            candidates.append(
                Candidate(
                    spec=spec,
                    cost_usd=spec.token_cost(input_tokens, output_tokens),
                    latency_ms=self.latency.p50(spec),
                    quality=self.quality_for(spec.name, category),
                )
            )
        normalize_candidates(candidates)
        return candidates

    # -- routing ----------------------------------------------------------- #
    def route(self, prompt: str, strategy: str = "balanced") -> RoutingDecision:
        if strategy not in STRATEGIES:
            raise ValueError(
                f"unknown strategy '{strategy}'. options: {sorted(STRATEGIES)}"
            )
        category = classify_prompt(prompt)
        candidates = self.build_candidates(prompt, category)
        frontier = pareto_frontier(candidates)
        chosen = STRATEGIES[strategy](candidates, frontier)
        rationale = self._explain(strategy, chosen, candidates, frontier)
        return RoutingDecision(
            strategy=strategy,
            prompt=prompt,
            category=category,
            chosen=chosen.name,
            projected_cost_usd=chosen.cost_usd,
            projected_latency_ms=chosen.latency_ms,
            projected_quality=chosen.quality,
            rationale=rationale,
            candidates=candidates,
            frontier=[c.name for c in frontier],
        )

    @staticmethod
    def _explain(
        strategy: str,
        chosen: Candidate,
        cands: List[Candidate],
        frontier: List[Candidate],
    ) -> str:
        cheapest = min(cands, key=lambda c: c.cost_usd).name
        best_q = max(cands, key=lambda c: c.quality).name
        on_frontier = chosen.name in {c.name for c in frontier}
        if strategy == "cost_optimal":
            return (
                f"cheapest model clearing the quality floor "
                f"(quality={chosen.quality:.0f}, ${chosen.cost_usd:.5f}); "
                f"absolute cheapest was {cheapest}."
            )
        if strategy == "quality_first":
            return (
                f"top quality={chosen.quality:.0f} (best available is {best_q}); "
                f"cost ${chosen.cost_usd:.5f}."
            )
        return (
            f"best weighted blend on the Pareto frontier "
            f"(quality={chosen.quality:.0f}, ${chosen.cost_usd:.5f}, "
            f"{chosen.latency_ms:.0f}ms); on_frontier={on_frontier}."
        )
