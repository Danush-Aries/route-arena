"""Offline evaluation harness: a labeled prompt set, a scorer, and the runner
that produces the committed per-model quality table (results.json).
"""

from route_arena.eval.prompts import EVAL_SET, EvalItem
from route_arena.eval.scorer import score_response

__all__ = ["EVAL_SET", "EvalItem", "score_response"]
