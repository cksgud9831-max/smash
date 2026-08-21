"""Defines the hit equation: ProjectilePosition(t, az, el) - TargetPosition(t) = 0.

This module only *represents* the equation as a residual function of
x = [time_of_flight, azimuth, elevation]; it does not solve it (see
`newton_solver.py`). Composing `TargetState.predict()` with
`ProjectileModel.state_at()` here — rather than pre-baking a single lead
point — is what makes flight time and target motion solve simultaneously.
"""

from __future__ import annotations

import numpy as np

from .projectile_model import ProjectileModel
from .target_state import TargetState
from .types import Vector3


class HitEquation:
    def __init__(
        self,
        target_state: TargetState,
        projectile_model: ProjectileModel,
        launch_point: Vector3,
        reference_time: float,
    ):
        self._target_state = target_state
        self._projectile_model = projectile_model
        self._launch_point = launch_point
        self._reference_time = reference_time

    def residual(self, x: np.ndarray) -> np.ndarray:
        """x = [time_of_flight, azimuth, elevation] -> R^3 residual vector."""

        t, azimuth, elevation = x
        t_eval = max(t, 0.0)  # tolerate Newton overshoot into negative flight time

        projectile_pos, _ = self._projectile_model.state_at(t_eval, self._launch_point, azimuth, elevation)
        target_pos = self._target_state.predict(self._reference_time + t_eval).position

        return projectile_pos.to_array() - target_pos.to_array()
