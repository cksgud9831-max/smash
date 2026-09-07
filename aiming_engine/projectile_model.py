"""Projectile flight physics: integrates launch state forward under a
pluggable list of ForceModel plugins (see `forces.py`).

v1 ships with gravity only. RK4 is exact for constant-acceleration motion
(gravity-only), so the configured `integrator_step` doesn't affect accuracy
for v1 — it only matters once a nonlinear force (drag) is added, at which
point it becomes an accuracy/latency tradeoff knob.

Optimization (v2): When only GravityForce is present the closed-form solution
  p(t) = p0 + v0*t + 0.5*g*t²
is used instead of RK4, reducing per-call cost from O(n_steps*4) to O(1).
Any nonlinear force (drag, wind …) automatically falls back to RK4.
"""

from __future__ import annotations

import numpy as np

from .config import ProjectileConfig
from .forces import ForceModel, GravityForce, build_forces
from .types import Vector3


def azimuth_elevation_to_direction(azimuth: float, elevation: float) -> np.ndarray:
    """World-frame unit vector for a given azimuth (from +X, CCW in XY) and elevation (from XY plane, +Z up)."""

    return np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )


class ProjectileModel:
    def __init__(self, config: ProjectileConfig, forces: list[ForceModel] | None = None):
        self._muzzle_velocity = config.muzzle_velocity
        self._integrator_step = config.integrator_step
        self._forces = forces if forces is not None else build_forces(config)

        # Pre-detect gravity-only case for O(1) analytic shortcut.
        # Falls back to RK4 automatically when any nonlinear force (drag, wind…) is present.
        self._gravity_vec: np.ndarray | None = None
        if len(self._forces) == 1 and isinstance(self._forces[0], GravityForce):
            self._gravity_vec = self._forces[0]._accel.copy()  # shape (3,), e.g. [0, 0, -9.81]

    def launch_velocity(self, azimuth: float, elevation: float) -> np.ndarray:
        return self._muzzle_velocity * azimuth_elevation_to_direction(azimuth, elevation)

    def _derivative(self, state: np.ndarray, t: float) -> np.ndarray:
        position, velocity = state[:3], state[3:]
        total_accel = np.zeros(3)
        for force in self._forces:
            total_accel += force.acceleration(position, velocity, t)
        return np.concatenate([velocity, total_accel])

    def _rk4_step(self, state: np.ndarray, t: float, dt: float) -> np.ndarray:
        k1 = self._derivative(state, t)
        k2 = self._derivative(state + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self._derivative(state + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self._derivative(state + dt * k3, t + dt)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def state_at(
        self, t: float, launch_point: Vector3, azimuth: float, elevation: float
    ) -> tuple[Vector3, Vector3]:
        """Integrate the projectile from launch to time t. Returns (position, velocity).

        When only gravity is present, uses the closed-form solution
        p(t) = p0 + v0*t + 0.5*g*t^2  (identical result to RK4, O(1) cost).
        Falls back to RK4 for any nonlinear force combination.
        """

        if t < 0:
            raise ValueError(f"time of flight must be non-negative, got {t}")

        v0 = self.launch_velocity(azimuth, elevation)

        if t == 0.0:
            return launch_point, Vector3.from_array(v0)

        # ── Analytic shortcut (gravity-only) ──────────────────────────────────
        if self._gravity_vec is not None:
            p0 = launch_point.to_array()
            pos = p0 + v0 * t + 0.5 * self._gravity_vec * (t * t)
            vel = v0 + self._gravity_vec * t
            return Vector3.from_array(pos), Vector3.from_array(vel)

        # ── RK4 fallback (drag, wind, or any nonlinear force) ────────────────
        state = np.concatenate([launch_point.to_array(), v0])
        n_steps = max(1, round(t / self._integrator_step))
        dt = t / n_steps
        current_t = 0.0
        for _ in range(n_steps):
            state = self._rk4_step(state, current_t, dt)
            current_t += dt

        return Vector3.from_array(state[:3]), Vector3.from_array(state[3:])
