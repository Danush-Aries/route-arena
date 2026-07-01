"""Offline (mock) model backend.

Real API keys are optional. In mock mode we synthesize a deterministic model
response by degrading the reference answer according to a per-model, per-category
"skill" profile. This keeps the whole project runnable with zero setup while
still producing *differentiated* quality scores across models (a stronger model
reproduces the reference more faithfully).

The skill profiles below are the only place where per-model competence is
encoded; the eval harness measures it back out through the scorer, and the
router consumes those measured scores. This mirrors how a real deployment would
work with the mock simply standing in for live model calls.
"""

from __future__ import annotations

import hashlib
import random
from typing import Dict

# skill[model][category] in [0,1]: probability a given reference token survives
# the mock model's (imperfect) reproduction. Loosely reflects public leaderboard
# intuition: frontier models score high everywhere; small/cheap models trade
# quality for price.
SKILL: Dict[str, Dict[str, float]] = {
    "gpt-4o":        {"code": 0.92, "summarization": 0.90, "reasoning": 0.91, "extraction": 0.93},
    "claude-sonnet": {"code": 0.93, "summarization": 0.89, "reasoning": 0.94, "extraction": 0.90},
    "gpt-4o-mini":   {"code": 0.74, "summarization": 0.80, "reasoning": 0.70, "extraction": 0.83},
    "claude-haiku":  {"code": 0.78, "summarization": 0.83, "reasoning": 0.75, "extraction": 0.85},
    "llama-3-70b":   {"code": 0.80, "summarization": 0.82, "reasoning": 0.79, "extraction": 0.81},
}

_DEFAULT_SKILL = 0.75


def _seed(model: str, prompt: str) -> int:
    digest = hashlib.sha256(f"{model}::{prompt}".encode()).hexdigest()
    return int(digest[:8], 16)


def mock_response(model: str, category: str, prompt: str, reference: str) -> str:
    """Deterministically synthesize a mock model answer for `prompt`.

    Higher-skill models keep more of the reference tokens (and, above a high
    skill bar, may reproduce it verbatim), yielding higher scorer output.
    """
    skill = SKILL.get(model, {}).get(category, _DEFAULT_SKILL)
    rng = random.Random(_seed(model, prompt))

    # Strong models occasionally nail the answer exactly.
    if skill >= 0.90 and rng.random() < (skill - 0.5):
        return reference

    tokens = reference.split()
    kept = [t for t in tokens if rng.random() < skill]
    if not kept:  # never return empty; keep at least the first token
        kept = tokens[:1]

    # Lower-skill models sometimes pad with filler, hurting length sanity.
    if skill < 0.80 and rng.random() < 0.4:
        kept.append("(approximately)")
    return " ".join(kept)


# --------------------------------------------------------------------------- #
# Canned responses for the arena when a prompt is NOT in the eval set.
# --------------------------------------------------------------------------- #

# A tiny library of plausible canned answers keyed by category, so `arena` can
# show a response even for free-form prompts in offline mode.
_CANNED: Dict[str, str] = {
    "code": (
        "def solution(*args, **kwargs):\n"
        "    # offline mock: representative implementation\n"
        "    ...\n"
        "    return result"
    ),
    "summarization": "Offline mock summary: the key point of the input, condensed.",
    "reasoning": "Offline mock: step-by-step reasoning leading to the final answer.",
    "extraction": "Offline mock: the requested field extracted from the input.",
}


def canned_response(model: str, category: str, prompt: str) -> str:
    """Return a canned, model-tagged response for an arbitrary prompt."""
    body = _CANNED.get(category, "Offline mock response.")
    return f"[{model} - offline mock]\n{body}"
