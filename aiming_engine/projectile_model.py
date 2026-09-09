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


try:
    import numba
    _jit = numba.jit(nopython=True, cache=True)
except ImportError:
    def _jit(func):
        return func

from .forces import ForceModel, GravityForce, DragForce, build_forces, _numba_drag_acceleration


@_jit
def _numba_derivative(
    state: np.ndarray,
    t: float,
    gravity: np.ndarray,
    has_drag: bool,
    mach_points: np.ndarray,
    cd_points: np.ndarray,
    speed_of_sound: float,
    air_density: float,
    area_over_mass: float,
) -> np.ndarray:
    velocity = state[3:]
    total_accel = gravity.copy()
    if has_drag:
        total_accel += _numba_drag_acceleration(
            velocity,
            mach_points,
            cd_points,
            speed_of_sound,
            air_density,
            area_over_mass,
        )
    return np.concatenate((velocity, total_accel))


@_jit
def _numba_rk4_step(
    state: np.ndarray,
    t: float,
    dt: float,
    gravity: np.ndarray,
    has_drag: bool,
    mach_points: np.ndarray,
    cd_points: np.ndarray,
    speed_of_sound: float,
    air_density: float,
    area_over_mass: float,
) -> np.ndarray:
    k1 = _numba_derivative(state, t, gravity, has_drag, mach_points, cd_points, speed_of_sound, air_density, area_over_mass)
    k2 = _numba_derivative(state + 0.5 * dt * k1, t + 0.5 * dt, gravity, has_drag, mach_points, cd_points, speed_of_sound, air_density, area_over_mass)
    k3 = _numba_derivative(state + 0.5 * dt * k2, t + 0.5 * dt, gravity, has_drag, mach_points, cd_points, speed_of_sound, air_density, area_over_mass)
    k4 = _numba_derivative(state + dt * k3, t + dt, gravity, has_drag, mach_points, cd_points, speed_of_sound, air_density, area_over_mass)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@_jit
def _numba_rk4_integrate(
    state0: np.ndarray,
    t: float,
    integrator_step: float,
    gravity: np.ndarray,
    has_drag: bool,
    mach_points: np.ndarray,
    cd_points: np.ndarray,
    speed_of_sound: float,
    air_density: float,
    area_over_mass: float,
) -> np.ndarray:
    state = state0.copy()
    n_steps = max(1, int(round(t / integrator_step)))
    dt = t / n_steps
    current_t = 0.0
    for _ in range(n_steps):
        state = _numba_rk4_step(
            state,
            current_t,
            dt,
            gravity,
            has_drag,
            mach_points,
            cd_points,
            speed_of_sound,
            air_density,
            area_over_mass,
        )
        current_t += dt
    return state


class ProjectileModel:
    def __init__(self, config: ProjectileConfig, forces: list[ForceModel] | None = None):
        self._muzzle_velocity = config.muzzle_velocity
        self._integrator_step = config.integrator_step
        self._forces = forces if forces is not None else build_forces(config)

        # Extract gravity force and match unit test expectations
        gravity_force = next((f for f in self._forces if isinstance(f, GravityForce)), None)
        has_other_forces = any(not isinstance(f, GravityForce) for f in self._forces)

        if has_other_forces:
            self._gravity_vec = None
        else:
            if gravity_force is not None:
                self._gravity_vec = gravity_force._accel.copy()
            else:
                self._gravity_vec = np.zeros(3)

        # Set up strictly-typed parameters for Numba JIT integration
        if gravity_force is not None:
            self._numba_gravity = gravity_force._accel.copy()
        else:
            g_val = config.gravity if "gravity" in config.forces else 0.0
            self._numba_gravity = np.array([0.0, 0.0, -g_val])

        self._has_drag = False
        self._mach_points = np.zeros(1)
        self._cd_points = np.zeros(1)
        self._area_over_mass = 0.0
        self._air_density = 0.0
        self._speed_of_sound = 0.0

        for force in self._forces:
            if isinstance(force, DragForce):
                self._has_drag = True
                self._mach_points = force._mach_points
                self._cd_points = force._cd_points
                self._area_over_mass = force._area_over_mass
                self._air_density = force._air_density
                self._speed_of_sound = force._speed_of_sound

        # Pre-detect gravity-only case for O(1) analytic shortcut.
        self._is_gravity_only = len(self._forces) == 1 and isinstance(self._forces[0], GravityForce)

    def launch_velocity(self, azimuth: float, elevation: float) -> np.ndarray:
        return self._muzzle_velocity * azimuth_elevation_to_direction(azimuth, elevation)

    def state_at(
        self, t: float, launch_point: Vector3, azimuth: float, elevation: float
    ) -> tuple[Vector3, Vector3]:
        """Integrate the projectile from launch to time t. Returns (position, velocity)."""

        if t < 0:
            raise ValueError(f"time of flight must be non-negative, got {t}")

        v0 = self.launch_velocity(azimuth, elevation)

        if t == 0.0:
            return launch_point, Vector3.from_array(v0)

        # ── Analytic shortcut (gravity-only) ──────────────────────────────────
        if self._is_gravity_only:
            p0 = launch_point.to_array()
            pos = p0 + v0 * t + 0.5 * self._gravity_vec * (t * t)
            vel = v0 + self._gravity_vec * t
            return Vector3.from_array(pos), Vector3.from_array(vel)

        # ── RK4 fallback (Numba JIT accelerated) ─────────────────────────────
        state0 = np.concatenate([launch_point.to_array(), v0])
        state = _numba_rk4_integrate(
            state0,
            t,
            self._integrator_step,
            self._numba_gravity,
            self._has_drag,
            self._mach_points,
            self._cd_points,
            self._speed_of_sound,
            self._air_density,
            self._area_over_mass,
        )


        return Vector3.from_array(state[:3]), Vector3.from_array(state[3:])

