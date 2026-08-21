"""Fast initial-guess generator for the Newton solver.

This does NOT compute the final firing solution — `HitEquation` +
`NewtonSolver` do that, by solving flight time and target motion together.
`LeadPrediction` only needs to get the solver started close enough to the
true root that it converges in a couple of iterations, which is what makes
per-frame real-time solving affordable (see NewtonSolver warm-start note
in AimSolver).

Guess procedure (single first-order pass, no iteration):
    1. t0 = straight-line range to the target's *current* position / muzzle velocity.
    2. Predict where the target will be at `t0` and aim a straight line at
       that point (ignoring gravity) for az0/el0.
"""

from __future__ import annotations

import numpy as np

from .types import InitialGuess, TargetVector, Vector3


class LeadPrediction:
    def initial_guess(self, target: TargetVector, launch_point: Vector3, muzzle_velocity: float) -> InitialGuess:
        if muzzle_velocity <= 0:
            raise ValueError(f"muzzle_velocity must be positive, got {muzzle_velocity}")

        range0 = (target.position - launch_point).norm()
        t0 = range0 / muzzle_velocity

        lead_point = target.predict_position(target.timestamp + t0)
        direction = (lead_point - launch_point).to_array()

        horizontal_range = float(np.hypot(direction[0], direction[1]))
        azimuth0 = float(np.arctan2(direction[1], direction[0]))
        elevation0 = float(np.arctan2(direction[2], horizontal_range))

        return InitialGuess(t0=t0, azimuth0=azimuth0, elevation0=elevation0, naive_lead_point=lead_point)
