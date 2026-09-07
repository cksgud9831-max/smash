"""Wires LeadPrediction + HitEquation + NewtonSolver + ProjectileModel +
CoordinateTransform into a single per-frame aim solve.

World-frame [t, azimuth, elevation] is what NewtonSolver actually solves
for (see hit_equation.py); this module is where that gets converted into
scope-relative azimuth/elevation for the HUD reticle, and where the
previous frame's solution is kept as a warm start so a smoothly-moving
target only needs 1-2 Newton iterations per frame instead of a cold solve.

The result is advisory only (a HUD reticle position) — nothing here ever
issues an action; the operator always decides when to fire.
"""

from __future__ import annotations

import numpy as np

from .coordinate_transform import CoordinateTransform
from .hit_equation import HitEquation
from .lead_prediction import LeadPrediction
from .newton_solver import NewtonSolver
from .projectile_model import ProjectileModel, azimuth_elevation_to_direction
from .target_state import TargetState
from .types import AimSolution, TargetVector, Vector3


class AimSolver:
    def __init__(
        self,
        projectile_model: ProjectileModel,
        newton_solver: NewtonSolver,
        lead_prediction: LeadPrediction,
        coordinate_transform: CoordinateTransform,
        muzzle_velocity: float,
        warm_start_enabled: bool,
    ):
        self._projectile_model = projectile_model
        self._newton_solver = newton_solver
        self._lead_prediction = lead_prediction
        self._coordinate_transform = coordinate_transform
        self._muzzle_velocity = muzzle_velocity
        self._warm_start_enabled = warm_start_enabled
        self._previous_solution: np.ndarray | None = None

    def reset_warm_start(self) -> None:
        self._previous_solution = None

    def solve(
        self,
        target_state: TargetState,
        target_snapshot: TargetVector,
        launch_point_world: Vector3,
        reference_time: float,
        extrinsic: np.ndarray,
    ) -> AimSolution:
        equation = HitEquation(target_state, self._projectile_model, launch_point_world, reference_time)
        naive_guess = self._lead_prediction.initial_guess(target_snapshot, launch_point_world, self._muzzle_velocity)

        if self._warm_start_enabled and self._previous_solution is not None:
            x0 = self._previous_solution.copy()
        else:
            x0 = np.array([naive_guess.t0, naive_guess.azimuth0, naive_guess.elevation0])

        result = self._newton_solver.solve(equation, x0)
        # Only keep warm start solution if it is mathematically valid and reasonably bounded.
        if result.converged or (np.all(np.isfinite(result.x)) and result.residual_norm < 50.0):
            self._previous_solution = result.x.copy()
        else:
            self._previous_solution = None

        t, azimuth_world, elevation_world = result.x
        t_eval = max(t, 0.0)
        lead_point_world = target_state.predict(reference_time + t_eval).position

        solved_direction_world = azimuth_elevation_to_direction(azimuth_world, elevation_world)
        solved_direction_scope = self._coordinate_transform.world_to_scope_direction(
            solved_direction_world, extrinsic
        )
        azimuth_scope, elevation_scope = _direction_to_azimuth_elevation(solved_direction_scope)

        naive_direction_world = azimuth_elevation_to_direction(naive_guess.azimuth0, naive_guess.elevation0)
        naive_direction_scope = self._coordinate_transform.world_to_scope_direction(
            naive_direction_world, extrinsic
        )
        naive_azimuth_scope, naive_elevation_scope = _direction_to_azimuth_elevation(naive_direction_scope)

        ballistic_offset = (azimuth_scope - naive_azimuth_scope, elevation_scope - naive_elevation_scope)

        return AimSolution(
            lead_point_world=lead_point_world,
            azimuth=azimuth_scope,
            elevation=elevation_scope,
            ballistic_offset=ballistic_offset,
            time_of_flight=t_eval,
            solver_converged=result.converged,
            solver_iterations=result.iterations,
            residual_norm_m=result.residual_norm,
        )


def _direction_to_azimuth_elevation(direction: np.ndarray) -> tuple[float, float]:
    horizontal_range = float(np.hypot(direction[0], direction[1]))
    azimuth = float(np.arctan2(direction[1], direction[0]))
    elevation = float(np.arctan2(direction[2], horizontal_range))
    return azimuth, elevation
