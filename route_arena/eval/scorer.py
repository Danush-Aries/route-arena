"""Heuristic response scorer.

Blends three signals into a [0,1] quality score:

  * exact / containment match (normalized) - rewards getting the answer right.
  * token-overlap F1 (a ROUGE-1-style measure) - rewards content overlap.
  * length sanity - lightly penalizes wildly off-length answers.

No external dependencies; deterministic and fast.
"""

from __future__ import annotations

import re
from typing import List

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def token_f1(prediction: str, reference: str) -> float:
    """ROUGE-1-style token overlap F1 between prediction and reference."""
    pred = _tokens(prediction)
    ref = _tokens(reference)
    if not pred or not ref:
        return 0.0
    # Multiset intersection count.
    ref_counts: dict[str, int] = {}
    for t in ref:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    overlap = 0
    for t in pred:
        if ref_counts.get(t, 0) > 0:
            overlap += 1
            ref_counts[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def exact_or_contains(prediction: str, reference: str) -> float:
    """1.0 for exact match, 0.7 if the reference is contained in the prediction,
    else 0.0. Whitespace/case-insensitive.
    """
    p, r = _normalize(prediction), _normalize(reference)
    if p == r:
        return 1.0
    if r and r in p:
        return 0.7
    return 0.0


def length_sanity(prediction: str, reference: str) -> float:
    """1.0 when prediction length is within 3x of the reference, decaying to a
    floor of 0.5 for very long or empty answers. Keeps a verbose model from
    scoring identically to a crisp one on overlap alone.
    """
    lp, lr = len(_tokens(prediction)), max(1, len(_tokens(reference)))
    ratio = lp / lr
    if lp == 0:
        return 0.0
    if 0.33 <= ratio <= 3.0:
        return 1.0
    return 0.5


def score_response(prediction: str, reference: str) -> float:
    """Combine the signals into a single [0,1] quality score."""
    match = exact_or_contains(prediction, reference)
    overlap = token_f1(prediction, reference)
    sanity = length_sanity(prediction, reference)
    # Overlap carries the most weight; exact match is a strong bonus; sanity is
    # a gentle multiplier expressed as a weighted term.
    raw = 0.55 * overlap + 0.35 * match + 0.10 * sanity
    return max(0.0, min(1.0, raw))
