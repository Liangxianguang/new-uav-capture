"""Small dependency-free monotone calibration for uncertainty scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class MonotoneRiskCalibration:
    """Piecewise-linear non-decreasing empirical risk map."""

    score_knots: tuple[float, ...]
    risk_values: tuple[float, ...]
    label: str = "failure"

    def __post_init__(self) -> None:
        if not self.score_knots or len(self.score_knots) != len(self.risk_values):
            raise ValueError("calibration knots and risk values must be non-empty and have equal length")
        scores = np.asarray(self.score_knots, dtype=np.float64)
        risks = np.asarray(self.risk_values, dtype=np.float64)
        if not np.isfinite(scores).all() or not np.isfinite(risks).all():
            raise ValueError("calibration knots and risk values must be finite")
        if np.any(np.diff(scores) < 0.0) or np.any(risks < 0.0) or np.any(risks > 1.0):
            raise ValueError("calibration scores must be sorted and risks must lie in [0, 1]")
        if np.any(np.diff(risks) < -1.0e-12):
            raise ValueError("risk values must be non-decreasing")

    @classmethod
    def fit(cls, scores: np.ndarray, labels: np.ndarray, *, label: str = "failure") -> "MonotoneRiskCalibration":
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        labels = np.asarray(labels, dtype=np.float64).reshape(-1)
        if scores.size == 0 or labels.shape != scores.shape:
            raise ValueError("scores and labels must be non-empty and have equal shape")
        if not np.isfinite(scores).all() or not np.isfinite(labels).all():
            raise ValueError("scores and labels must be finite")
        if np.any((labels < 0.0) | (labels > 1.0)):
            raise ValueError("labels must lie in [0, 1]")

        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        sorted_labels = labels[order]
        unique_scores, inverse = np.unique(sorted_scores, return_inverse=True)
        counts = np.bincount(inverse).astype(np.float64)
        sums = np.bincount(inverse, weights=sorted_labels).astype(np.float64)
        means = sums / counts

        # Pool adjacent violators: merge blocks until empirical risk is
        # non-decreasing with score.
        blocks: list[dict[str, float]] = []
        for index, (score, count, total) in enumerate(zip(unique_scores, counts, sums)):
            blocks.append({"start": float(index), "end": float(index), "count": float(count), "total": float(total)})
            while len(blocks) >= 2:
                previous, current = blocks[-2], blocks[-1]
                previous_mean = previous["total"] / previous["count"]
                current_mean = current["total"] / current["count"]
                if previous_mean <= current_mean + 1.0e-15:
                    break
                merged = {
                    "start": previous["start"],
                    "end": current["end"],
                    "count": previous["count"] + current["count"],
                    "total": previous["total"] + current["total"],
                }
                blocks[-2:] = [merged]
        fitted = np.empty(unique_scores.size, dtype=np.float64)
        for block in blocks:
            start, end = int(block["start"]), int(block["end"])
            fitted[start : end + 1] = block["total"] / block["count"]
        return cls(tuple(float(value) for value in unique_scores), tuple(float(value) for value in fitted), label=label)

    def predict(self, score: float | np.ndarray) -> np.ndarray:
        values = np.asarray(score, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("score must be finite")
        return np.interp(values, np.asarray(self.score_knots), np.asarray(self.risk_values))

    def threshold_for_risk(self, target_risk: float) -> float:
        target = float(target_risk)
        if not np.isfinite(target) or not 0.0 <= target <= 1.0:
            raise ValueError("target_risk must lie in [0, 1]")
        indices = np.flatnonzero(np.asarray(self.risk_values) >= target)
        return float(self.score_knots[indices[0]]) if indices.size else float(self.score_knots[-1])

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score_knots": list(self.score_knots),
            "risk_values": list(self.risk_values),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "MonotoneRiskCalibration":
        return cls(
            score_knots=tuple(float(value) for value in mapping["score_knots"]),
            risk_values=tuple(float(value) for value in mapping["risk_values"]),
            label=str(mapping.get("label", "failure")),
        )


__all__ = ["MonotoneRiskCalibration"]
