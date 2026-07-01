"""Scoring primitives: prompt classification, token/cost/latency estimation,
quality lookup, normalization, and Pareto-frontier computation.

Everything here is pure and deterministic so it is trivially testable.
"""

from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Sequence

from route_arena.models import CATEGORIES, ModelSpec

# --------------------------------------------------------------------------- #
# Prompt classification
# --------------------------------------------------------------------------- #

# Keyword signals per category. Deliberately simple and inspectable — a real
# system would swap this for an embedding classifier, but the router contract
# (prompt -> category) stays identical.
_CATEGORY_KEYWORDS: Dict[str, Sequence[str]] = {
    "code": (
        "code", "function", "python", "javascript", "bug", "class", "algorithm",
        "implement", "compile", "refactor", "api", "regex", "sql", "linked list",
        "array", "recursion", "def ", "return", "loop",
    ),
    "summarization": (
        "summarize", "summary", "tl;dr", "tldr", "condense", "shorten",
        "key points", "abstract", "digest", "brief",
    ),
    "reasoning": (
        "why", "explain", "reason", "prove", "logic", "if ", "therefore",
        "deduce", "step by step", "solve", "calculate", "how many", "puzzle",
    ),
    "extraction": (
        "extract", "list all", "find all", "parse", "pull out", "identify the",
        "entities", "json", "fields", "table", "structured",
    ),
}


def classify_prompt(prompt: str) -> str:
    """Classify a prompt into one of CATEGORIES via keyword scoring.

    Falls back to "reasoning" (the most general bucket) when no signal fires.
    """
    text = prompt.lower()
    scores: Dict[str, int] = {c: 0 for c in CATEGORIES}
    for category, keywords in _CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[category] += 1
    best = max(scores, key=lambda c: scores[c])
    return best if scores[best] > 0 else "reasoning"


# Expected completion length (in tokens) per category. Cost/latency estimates
# depend on how much output a category typically demands.
_EXPECTED_OUTPUT_TOKENS: Dict[str, int] = {
    "code": 350,
    "summarization": 180,
    "reasoning": 400,
    "extraction": 150,
}


def estimate_tokens(prompt: str, category: str) -> tuple[int, int]:
    """Estimate (input_tokens, output_tokens) for a prompt.

    Uses the common ~4-characters-per-token heuristic for the input and a
    per-category prior for the output length.
    """
    input_tokens = max(1, math.ceil(len(prompt) / 4))
    output_tokens = _EXPECTED_OUTPUT_TOKENS.get(category, 250)
    return input_tokens, output_tokens


# --------------------------------------------------------------------------- #
# Rolling latency history
# --------------------------------------------------------------------------- #


@dataclass
class LatencyHistory:
    """Per-model rolling window of observed latencies, exposing a p50.

    Until a model has observations, `p50` returns the model's `base_latency_ms`
    prior. This is what lets the router adapt to live conditions.
    """

    window: int = 50
    _samples: Dict[str, Deque[float]] = field(default_factory=dict)

    def record(self, model: str, latency_ms: float) -> None:
        buf = self._samples.setdefault(model, deque(maxlen=self.window))
        buf.append(latency_ms)

    def p50(self, spec: ModelSpec) -> float:
        buf = self._samples.get(spec.name)
        if not buf:
            return spec.base_latency_ms
        ordered = sorted(buf)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2

    def count(self, model: str) -> int:
        return len(self._samples.get(model, ()))


# --------------------------------------------------------------------------- #
# Candidate scoring
# --------------------------------------------------------------------------- #


@dataclass
class Candidate:
    """A model evaluated against a specific prompt.

    Raw objectives:
        cost_usd: estimated dollars for the request (lower is better).
        latency_ms: rolling p50 latency (lower is better).
        quality: category ELO from the offline eval (higher is better).

    Normalized-to-[0,1]-higher-is-better scores are filled in by
    `normalize_candidates` and consumed by the strategies.
    """

    spec: ModelSpec
    cost_usd: float
    latency_ms: float
    quality: float
    cost_score: float = 0.0
    latency_score: float = 0.0
    quality_score: float = 0.0

    @property
    def name(self) -> str:
        return self.spec.name


def _minmax(values: Sequence[float], higher_is_better: bool) -> List[float]:
    """Min-max normalize to [0,1]. When higher_is_better is False the scale is
    inverted so that a *lower* raw value yields a *higher* score.

    A degenerate range (all equal) maps everything to 1.0.
    """
    lo, hi = min(values), max(values)
    if math.isclose(hi, lo):
        return [1.0 for _ in values]
    span = hi - lo
    out = []
    for v in values:
        frac = (v - lo) / span
        out.append(frac if higher_is_better else 1.0 - frac)
    return out


def normalize_candidates(candidates: List[Candidate]) -> None:
    """Populate cost/latency/quality scores in place (all higher-is-better)."""
    cost_scores = _minmax([c.cost_usd for c in candidates], higher_is_better=False)
    lat_scores = _minmax([c.latency_ms for c in candidates], higher_is_better=False)
    qual_scores = _minmax([c.quality for c in candidates], higher_is_better=True)
    for cand, cs, ls, qs in zip(candidates, cost_scores, lat_scores, qual_scores):
        cand.cost_score = cs
        cand.latency_score = ls
        cand.quality_score = qs


# --------------------------------------------------------------------------- #
# Pareto frontier
# --------------------------------------------------------------------------- #


def _dominates(a: Candidate, b: Candidate) -> bool:
    """True if `a` Pareto-dominates `b`.

    Objectives: maximize quality, minimize cost, minimize latency. `a`
    dominates `b` if it is at least as good on every objective and strictly
    better on at least one.
    """
    not_worse = (
        a.quality >= b.quality
        and a.cost_usd <= b.cost_usd
        and a.latency_ms <= b.latency_ms
    )
    strictly_better = (
        a.quality > b.quality
        or a.cost_usd < b.cost_usd
        or a.latency_ms < b.latency_ms
    )
    return not_worse and strictly_better


def pareto_frontier(candidates: List[Candidate]) -> List[Candidate]:
    """Return the non-dominated candidates (the cost/quality/latency frontier).

    Order is preserved from the input list.
    """
    frontier: List[Candidate] = []
    for cand in candidates:
        if any(_dominates(other, cand) for other in candidates if other is not cand):
            continue
        frontier.append(cand)
    return frontier
