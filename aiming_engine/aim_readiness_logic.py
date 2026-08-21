"""Aim-readiness gate: is the current aim good enough to trust right now?

This does NOT authorize any action — there is no fire command anywhere in
this framework. It only computes the `aim_ready` boolean shown on the HUD
(e.g. driving a green/red reticle) so the operator knows when a shot would
be well-aimed; whether and when to actually fire is always the operator's
decision. Deliberately a pure boolean AND of independent checks (no
weighting/scoring) so it stays simple and auditable.
"""

from __future__ import annotations

from .config import AimReadinessConfig


class AimReadinessLogic:
    def __init__(self, config: AimReadinessConfig):
        self._config = config

    def evaluate(
        self,
        tracking_stable: bool,
        follower_stable: bool,
        hit_probability: float,
        distance_m: float,
        aim_error_mrad: float,
    ) -> bool:
        return (
            tracking_stable
            and follower_stable
            and hit_probability > self._config.hit_probability_threshold
            and self._config.min_valid_range_m <= distance_m <= self._config.max_valid_range_m
            and aim_error_mrad < self._config.max_aim_error_mrad
        )
