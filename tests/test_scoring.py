"""Tests for prompt classification, token estimation, and normalization."""

from __future__ import annotations

import math

from route_arena.models import all_models
from route_arena.router.scoring import (
    Candidate,
    LatencyHistory,
    _minmax,
    classify_prompt,
    estimate_tokens,
    normalize_candidates,
)


def test_classify_code_prompt():
    assert classify_prompt("Write a python function to reverse a linked list.") == "code"


def test_classify_summarization_prompt():
    assert classify_prompt("Summarize this article in two sentences.") == "summarization"


def test_classify_extraction_prompt():
    assert classify_prompt("Extract all emails and list all names from the text.") == "extraction"


def test_classify_falls_back_to_reasoning():
    # A prompt with no category keywords defaults to the general bucket.
    assert classify_prompt("qwrpz mmbl zzz") == "reasoning"


def test_estimate_tokens_scales_with_length():
    short_in, short_out = estimate_tokens("hi", "code")
    long_in, _ = estimate_tokens("x" * 400, "code")
    assert short_in >= 1
    assert long_in > short_in
    # Output prior is category-driven, not prompt-length driven.
    assert short_out == 350


def test_minmax_lower_is_better_inverts():
    # Lowest raw value should map to the highest score when lower is better.
    scores = _minmax([1.0, 2.0, 3.0], higher_is_better=False)
    assert scores[0] == 1.0
    assert scores[-1] == 0.0


def test_minmax_degenerate_range_all_ones():
    assert _minmax([5.0, 5.0, 5.0], higher_is_better=True) == [1.0, 1.0, 1.0]


def test_normalize_candidates_populates_scores():
    specs = all_models()
    cands = [
        Candidate(spec=s, cost_usd=s.input_price, latency_ms=s.base_latency_ms, quality=1000 + i)
        for i, s in enumerate(specs)
    ]
    normalize_candidates(cands)
    for c in cands:
        assert 0.0 <= c.cost_score <= 1.0
        assert 0.0 <= c.latency_score <= 1.0
        assert 0.0 <= c.quality_score <= 1.0


def test_latency_history_prior_then_observed():
    hist = LatencyHistory(window=10)
    spec = all_models()[0]
    # Before any observation, the model's base latency prior is returned.
    assert hist.p50(spec) == spec.base_latency_ms
    for _ in range(5):
        hist.record(spec.name, 100.0)
    assert hist.p50(spec) == 100.0
    assert hist.count(spec.name) == 5
