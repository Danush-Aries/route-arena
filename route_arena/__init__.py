"""route-arena: a cost/quality/latency LLM router with live A/B testing.

Given a prompt, route-arena picks the optimal model across a Pareto frontier
of estimated cost, rolling latency, and offline-measured quality.
"""

__version__ = "0.1.0"

from route_arena.models import MODEL_REGISTRY, ModelSpec
from route_arena.router.policy import Router, RoutingDecision
from route_arena.router.scoring import classify_prompt

__all__ = [
    "MODEL_REGISTRY",
    "ModelSpec",
    "Router",
    "RoutingDecision",
    "classify_prompt",
    "__version__",
]
