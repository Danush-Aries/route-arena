"""Model invocation layer: real providers when keys are present, mock otherwise.

Mock mode is the default so the whole project runs with zero setup. Set
ANTHROPIC_API_KEY or OPENAI_API_KEY to route the matching provider's models to
the live API (the SDKs are optional extras and imported lazily).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

from route_arena.eval.prompts import EVAL_SET
from route_arena.mock import canned_response, mock_response
from route_arena.models import MODEL_REGISTRY

# Map our canonical names to concrete provider model ids for live calls.
_LIVE_MODEL_IDS = {
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "claude-sonnet": "claude-3-5-sonnet-latest",
    "claude-haiku": "claude-3-5-haiku-latest",
    # llama-3-70b has no first-party key path here; always served via mock.
}

_REFERENCE_BY_PROMPT = {item.prompt: item.reference for item in EVAL_SET}
_CATEGORY_BY_PROMPT = {item.prompt: item.category for item in EVAL_SET}


@dataclass
class Completion:
    model: str
    text: str
    latency_ms: float
    mode: str  # "live" or "mock"


def _live_available(provider: str) -> bool:
    if provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    if provider == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    return False


def is_mock_mode() -> bool:
    """True when no provider key is set (the default, zero-setup path)."""
    return not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))


def complete(model: str, prompt: str, category: Optional[str] = None) -> Completion:
    """Return a completion for `prompt` from `model`.

    Uses the live provider when its API key is set and an SDK is importable;
    otherwise deterministically mocks the response.
    """
    spec = MODEL_REGISTRY[model]
    category = category or _CATEGORY_BY_PROMPT.get(prompt, "reasoning")

    if _live_available(spec.provider) and model in _LIVE_MODEL_IDS:
        try:
            return _complete_live(spec.provider, model, prompt)
        except Exception:
            # Fall back to mock on any live failure so a demo never crashes.
            pass

    start = time.perf_counter()
    reference = _REFERENCE_BY_PROMPT.get(prompt)
    if reference is not None:
        text = mock_response(model, category, prompt, reference)
    else:
        text = canned_response(model, category, prompt)
    elapsed = (time.perf_counter() - start) * 1000
    # Report the model's nominal latency prior in mock mode (the tiny real
    # elapsed time is not meaningful for offline synthesis).
    return Completion(model=model, text=text, latency_ms=spec.base_latency_ms + elapsed, mode="mock")


def _complete_live(provider: str, model: str, prompt: str) -> Completion:
    live_id = _LIVE_MODEL_IDS[model]
    start = time.perf_counter()
    if provider == "openai":
        from openai import OpenAI  # lazy import; optional dependency

        client = OpenAI()
        resp = client.chat.completions.create(
            model=live_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        text = resp.choices[0].message.content or ""
    elif provider == "anthropic":
        import anthropic  # lazy import; optional dependency

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=live_id,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"no live path for provider {provider}")
    elapsed = (time.perf_counter() - start) * 1000
    return Completion(model=model, text=text, latency_ms=elapsed, mode="live")
