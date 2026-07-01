"""Eval runner: score every model on the labeled set and emit results.json.

The output feeds the router's quality table. Quality scores are stored as an
ELO-style number (1000 + 600 * mean_score) which reads naturally on a
leaderboard while remaining a monotonic transform of the raw [0,1] score.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from route_arena.eval.prompts import EVAL_SET
from route_arena.eval.scorer import score_response
from route_arena.mock import mock_response
from route_arena.models import CATEGORIES, all_models

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"

_ELO_BASE = 1000.0
_ELO_SPAN = 600.0


def to_elo(mean_score: float) -> float:
    return round(_ELO_BASE + _ELO_SPAN * mean_score, 1)


@dataclass
class EvalReport:
    quality: Dict[str, Dict[str, float]]  # model -> category -> elo
    raw_scores: Dict[str, Dict[str, float]]  # model -> category -> mean [0,1]

    def to_json(self) -> str:
        return json.dumps(
            {"quality": self.quality, "raw_scores": self.raw_scores},
            indent=2,
            sort_keys=True,
        )


def run_eval() -> EvalReport:
    """Run the offline eval for every model/category and aggregate scores."""
    quality: Dict[str, Dict[str, float]] = {}
    raw: Dict[str, Dict[str, float]] = {}

    for spec in all_models():
        quality[spec.name] = {}
        raw[spec.name] = {}
        for category in CATEGORIES:
            items = [it for it in EVAL_SET if it.category == category]
            scores: List[float] = []
            for item in items:
                prediction = mock_response(
                    spec.name, category, item.prompt, item.reference
                )
                scores.append(score_response(prediction, item.reference))
            mean = statistics.mean(scores) if scores else 0.0
            raw[spec.name][category] = round(mean, 4)
            quality[spec.name][category] = to_elo(mean)

    return EvalReport(quality=quality, raw_scores=raw)


def write_results(report: EvalReport, path: Path = RESULTS_PATH) -> Path:
    path.write_text(report.to_json() + "\n")
    return path


if __name__ == "__main__":  # regenerate the committed results table
    report = run_eval()
    out = write_results(report)
    print(f"wrote {out}")
