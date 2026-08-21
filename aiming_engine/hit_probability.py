"""Estimates probability of hit in [0, 1] from independent per-factor scores.

v1 keeps this deliberately simple (per implementation policy): each factor
is mapped to a [0, 1] score with a monotonic function, then combined as a
weighted average using weights from YAML. Nothing here depends on the
projectile model, so swapping in a fancier probability model later (e.g.
one that consumes dispersion from a ballistic covariance) only touches
this file.
"""

from __future__ import annotations

import numpy as np

from .config import HitProbabilityConfig


class HitProbability:
    def __init__(self, config: HitProbabilityConfig):
        self._weights = config.weights
        self._distance_scale_m = config.distance_scale_m
        self._velocity_scale_mps = config.velocity_scale_mps
        self._aim_error_scale_mrad = config.aim_error_scale_mrad

    def estimate(
        self,
        tracking_confidence: float,
        follower_stable: bool,
        aim_error_mrad: float,
        distance_m: float,
        target_speed: float,
    ) -> float:
        scores = {
            "tracking_confidence": float(np.clip(tracking_confidence, 0.0, 1.0)),
            "follower_stability": 1.0 if follower_stable else 0.0,
            "aim_error": float(np.exp(-max(aim_error_mrad, 0.0) / self._aim_error_scale_mrad)),
            "distance": float(np.exp(-max(distance_m, 0.0) / self._distance_scale_m)),
            "velocity": float(np.exp(-max(target_speed, 0.0) / self._velocity_scale_mps)),
        }

        total_weight = sum(self._weights.values())
        weighted_sum = sum(self._weights[name] * scores[name] for name in scores)
        probability = weighted_sum / total_weight if total_weight > 0 else 0.0

        return float(np.clip(probability, 0.0, 1.0))
