"""Live A/B arena: route one prompt with two strategies, actually invoke the
chosen models, and compare projected vs realized cost/latency/quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from route_arena.eval.prompts import EVAL_SET
from route_arena.eval.scorer import score_response
from route_arena.providers import Completion, complete, is_mock_mode
from route_arena.router.policy import Router, RoutingDecision

_REFERENCE_BY_PROMPT = {item.prompt: item.reference for item in EVAL_SET}


@dataclass
class ArenaSide:
    """One competitor in the A/B: a strategy, its routing decision, and the
    realized completion from invoking the chosen model."""

    decision: RoutingDecision
    completion: Completion
    realized_quality: Optional[float]  # None when no reference is available


@dataclass
class ArenaResult:
    prompt: str
    category: str
    mode: str  # "mock" or "live"
    sides: Dict[str, ArenaSide]
    verdict: str


def _realized_quality(prompt: str, text: str) -> Optional[float]:
    ref = _REFERENCE_BY_PROMPT.get(prompt)
    if ref is None:
        return None
    return score_response(text, ref)


def run_arena(
    prompt: str,
    strategy_a: str = "cost_optimal",
    strategy_b: str = "quality_first",
    router: Optional[Router] = None,
) -> ArenaResult:
    """Route `prompt` under two strategies, invoke each chosen model, compare."""
    router = router or Router()
    sides: Dict[str, ArenaSide] = {}

    for label, strategy in (("A", strategy_a), ("B", strategy_b)):
        decision = router.route(prompt, strategy=strategy)
        completion = complete(decision.chosen, prompt, category=decision.category)
        # Feed the observed latency back so future routing adapts.
        router.latency.record(decision.chosen, completion.latency_ms)
        sides[label] = ArenaSide(
            decision=decision,
            completion=completion,
            realized_quality=_realized_quality(prompt, completion.text),
        )

    category = sides["A"].decision.category
    verdict = _verdict(sides)
    return ArenaResult(
        prompt=prompt,
        category=category,
        mode="mock" if is_mock_mode() else "live",
        sides=sides,
        verdict=verdict,
    )


def _verdict(sides: Dict[str, ArenaSide]) -> str:
    """Summarize the tradeoff: who is cheaper, who is higher quality, and the
    quality-per-dollar efficiency call."""
    a, b = sides["A"], sides["B"]
    a_cost = a.decision.projected_cost_usd
    b_cost = b.decision.projected_cost_usd
    a_q = a.decision.projected_quality
    b_q = b.decision.projected_quality

    cheaper = "A" if a_cost <= b_cost else "B"
    better = "A" if a_q >= b_q else "B"

    if cheaper == better:
        return (
            f"Side {cheaper} ({sides[cheaper].decision.chosen}) wins outright: "
            f"cheaper AND higher projected quality."
        )

    # Classic tradeoff: quantify how much extra you pay per quality point.
    hi, lo = (a, b) if a_q >= b_q else (b, a)
    hi_label = "A" if hi is a else "B"
    lo_label = "B" if hi is a else "A"
    dq = hi.decision.projected_quality - lo.decision.projected_quality
    dcost = hi.decision.projected_cost_usd - lo.decision.projected_cost_usd
    if dq <= 0:
        return f"Side {lo_label} is the efficient pick (equal quality, lower cost)."
    per_point = dcost / dq
    return (
        f"Tradeoff: Side {hi_label} ({hi.decision.chosen}) buys +{dq:.0f} quality "
        f"for +${dcost:.5f} (${per_point:.6f}/point). "
        f"Side {lo_label} ({lo.decision.chosen}) is the budget pick."
    )
