"""Tests for the eval scorer and the deterministic eval harness."""

from __future__ import annotations

from route_arena.eval.harness import run_eval, to_elo
from route_arena.eval.scorer import (
    exact_or_contains,
    length_sanity,
    score_response,
    token_f1,
)


def test_exact_match_scores_one():
    assert exact_or_contains("hello world", "hello world") == 1.0


def test_containment_scores_partial():
    assert exact_or_contains("the answer is hello world today", "hello world") == 0.7


def test_no_match_scores_zero():
    assert exact_or_contains("completely different", "hello world") == 0.0


def test_token_f1_identical_is_one():
    assert token_f1("a b c", "a b c") == 1.0


def test_token_f1_disjoint_is_zero():
    assert token_f1("a b c", "x y z") == 0.0


def test_length_sanity_penalizes_empty():
    assert length_sanity("", "reference text here") == 0.0


def test_score_response_bounded_unit_interval():
    s = score_response("some prediction text", "some reference text")
    assert 0.0 <= s <= 1.0


def test_score_perfect_response_high():
    ref = "Mitochondria produce most of a cell's ATP energy."
    assert score_response(ref, ref) > 0.9


def test_to_elo_is_monotonic():
    assert to_elo(1.0) > to_elo(0.5) > to_elo(0.0)
    assert to_elo(0.0) == 1000.0


def test_run_eval_is_deterministic():
    a = run_eval()
    b = run_eval()
    assert a.quality == b.quality
    assert a.raw_scores == b.raw_scores


def test_run_eval_covers_all_models_and_categories():
    report = run_eval()
    assert set(report.quality) == {
        "gpt-4o", "gpt-4o-mini", "claude-sonnet", "claude-haiku", "llama-3-70b"
    }
    for cats in report.quality.values():
        assert set(cats) == {"code", "summarization", "reasoning", "extraction"}


def test_stronger_model_scores_higher_on_reasoning():
    # claude-sonnet is the strongest reasoning model in the skill profile;
    # the eval should measure that back out above a weak model.
    q = run_eval().quality
    assert q["claude-sonnet"]["reasoning"] > q["gpt-4o-mini"]["reasoning"]
