"""Tests for the three routing strategies via the Router."""

from __future__ import annotations

import pytest

from route_arena.models import CATEGORIES, all_models
from route_arena.router.policy import STRATEGIES, Router, load_quality_table


@pytest.fixture
def router():
    # Use the committed offline quality table for deterministic behavior.
    return Router(quality_table=load_quality_table())


def test_all_three_strategies_registered():
    assert set(STRATEGIES) == {"cost_optimal", "quality_first", "balanced"}


def test_quality_first_picks_highest_quality(router):
    prompt = "Explain why the sky is blue, step by step."
    category = router.route(prompt, "balanced").category
    decision = router.route(prompt, "quality_first")
    best_q = max(router.quality_for(s.name, category) for s in all_models())
    assert decision.projected_quality == best_q


def test_cost_optimal_not_more_expensive_than_quality_first(router):
    prompt = "Extract the email from: reach me at a@b.com please."
    cheap = router.route(prompt, "cost_optimal")
    premium = router.route(prompt, "quality_first")
    assert cheap.projected_cost_usd <= premium.projected_cost_usd


def test_cost_optimal_clears_quality_floor(router):
    prompt = "Write a python function to sort a list."
    category = router.route(prompt, "balanced").category
    qualities = [router.quality_for(s.name, category) for s in all_models()]
    lo, hi = min(qualities), max(qualities)
    floor = lo + 0.80 * (hi - lo)
    decision = router.route(prompt, "cost_optimal")
    assert decision.projected_quality >= floor


def test_balanced_picks_from_frontier(router):
    prompt = "Summarize the quarterly earnings report in one sentence."
    decision = router.route(prompt, "balanced")
    assert decision.chosen in decision.frontier


def test_unknown_strategy_raises(router):
    with pytest.raises(ValueError):
        router.route("hello", strategy="nonsense")


def test_decision_as_dict_is_serializable(router):
    d = router.route("Write a function to add two numbers.", "balanced").as_dict()
    assert d["strategy"] == "balanced"
    assert "chosen" in d and "rationale" in d
    assert isinstance(d["frontier"], list)


def test_route_classifies_all_categories(router):
    prompts = {
        "code": "Write a python function to reverse a string.",
        "summarization": "Summarize this paragraph briefly.",
        "reasoning": "If x is 2 and y is 3, explain why x plus y is 5.",
        "extraction": "Extract all numbers from the sentence.",
    }
    for expected, prompt in prompts.items():
        assert router.route(prompt, "balanced").category == expected
