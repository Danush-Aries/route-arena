"""Tests for Pareto dominance and frontier computation."""

from __future__ import annotations

from route_arena.models import ModelSpec
from route_arena.router.scoring import Candidate, _dominates, pareto_frontier


def _c(name, cost, latency, quality):
    spec = ModelSpec(name, "test", cost, cost, latency, 1000)
    return Candidate(spec=spec, cost_usd=cost, latency_ms=latency, quality=quality)


def test_dominates_strictly_better_on_all():
    a = _c("a", 1.0, 100.0, 1500.0)
    b = _c("b", 2.0, 200.0, 1400.0)
    assert _dominates(a, b)
    assert not _dominates(b, a)


def test_no_dominance_on_tradeoff():
    cheap = _c("cheap", 1.0, 100.0, 1200.0)   # cheaper, faster, worse quality
    strong = _c("strong", 5.0, 500.0, 1600.0)  # pricier, slower, better quality
    assert not _dominates(cheap, strong)
    assert not _dominates(strong, cheap)


def test_equal_candidates_do_not_dominate():
    a = _c("a", 1.0, 100.0, 1500.0)
    b = _c("b", 1.0, 100.0, 1500.0)
    assert not _dominates(a, b)
    assert not _dominates(b, a)


def test_frontier_excludes_dominated():
    best = _c("best", 1.0, 100.0, 1600.0)
    dominated = _c("dominated", 2.0, 200.0, 1500.0)  # worse on every axis
    frontier = pareto_frontier([best, dominated])
    names = {c.name for c in frontier}
    assert "best" in names
    assert "dominated" not in names


def test_frontier_keeps_all_when_no_domination():
    cheap = _c("cheap", 1.0, 100.0, 1200.0)
    balanced = _c("balanced", 3.0, 300.0, 1450.0)
    strong = _c("strong", 5.0, 500.0, 1600.0)
    frontier = pareto_frontier([cheap, balanced, strong])
    assert len(frontier) == 3


def test_frontier_preserves_input_order():
    strong = _c("strong", 5.0, 500.0, 1600.0)
    cheap = _c("cheap", 1.0, 100.0, 1200.0)
    frontier = pareto_frontier([strong, cheap])
    assert [c.name for c in frontier] == ["strong", "cheap"]
