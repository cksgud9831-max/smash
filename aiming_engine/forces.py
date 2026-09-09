"""Pluggable acceleration sources for ProjectileModel.

Adding a new physical effect (wind, Coriolis, ...) means adding a
`ForceModel` subclass here (or a new module) and registering its name in
`FORCE_REGISTRY` — `ProjectileModel`, `HitEquation`, and `NewtonSolver`
never need to change. `GravityForce` and `DragForce` are the two force
models implemented so far.

Internal math uses raw numpy arrays (not the `Vector3` dataclass) since
these are called many times per Newton iteration inside the RK4 integrator
and must stay allocation-light for real-time use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .config import ProjectileConfig


class ForceModel(ABC):
    @abstractmethod
    def acceleration(self, position: np.ndarray, velocity: np.ndarray, t: float) -> np.ndarray:
        """Return acceleration (m/s^2) contributed by this force, World frame."""
        ...


class GravityForce(ForceModel):
    def __init__(self, g: float):
        self._accel = np.array([0.0, 0.0, -g])

    def acceleration(self, position: np.ndarray, velocity: np.ndarray, t: float) -> np.ndarray:
        return self._accel


# ── Air drag ─────────────────────────────────────────────────────────────
#
# a_drag = -0.5 * rho * Cd(Mach) * (A / m) * V * |V|
#   rho:      air density (kg/m^3)
#   Cd(Mach): drag coefficient at the current Mach number, linearly
#             interpolated from a reference curve below
#   A:        projectile cross-sectional area (m^2), from diameter_m
#   m:        projectile mass (kg)
#   V:        velocity vector (world frame, ground-relative -- no wind term
#             yet; a WindForce would subtract a wind vector from V before
#             this formula, the same registry-extension pattern as this file)
#
# This is the standard aerodynamic-drag equation, deliberately used instead
# of the classical "ballistic coefficient" (BC) formulation -- BC is an
# imperial-unit-era abstraction (lb/in^2, tied to a reference-projectile
# normalization constant) that doesn't map cleanly onto this project's SI
# units without importing an opaque conversion constant. Using mass +
# diameter + Cd(Mach) directly keeps every quantity in unambiguous SI units.
#
# NOTE on G1_DRAG_TABLE / G7_DRAG_TABLE below: these are reproduced from
# memory of the standard published Ingalls/McCoy G1 (flat-base) and G7
# (boat-tail) drag-coefficient functions (the same reference tables
# underlying most point-mass ballistics software, e.g. JBM Ballistics,
# McCoy's "Modern Exterior Ballistics") -- a step up from the earlier
# version of this table, which only matched the qualitative *shape*
# (subsonic plateau, transonic peak near Mach 1, supersonic decay) without
# attempting the actual published values. This version attempts the real
# numbers, but was NOT transcribed from a live authoritative source while
# writing this, so isolated points could still be off by a few percent --
# spot-check a handful of entries against JBM Ballistics or McCoy's tables
# before relying on this for anything beyond an approximate HUD display.
# Still PLACEHOLDER in the sense that mass_kg/diameter_m must be set to a
# real projectile's values (see ProjectileConfig) before any of this
# matters; see also known_issues.md section 5.

G1_DRAG_TABLE: list[tuple[float, float]] = [
    (0.00, 0.2629), (0.05, 0.2558), (0.10, 0.2487), (0.15, 0.2413), (0.20, 0.2344),
    (0.25, 0.2278), (0.30, 0.2214), (0.35, 0.2155), (0.40, 0.2104), (0.45, 0.2061),
    (0.50, 0.2032), (0.55, 0.2020), (0.60, 0.2034), (0.70, 0.2165), (0.725, 0.2230),
    (0.75, 0.2313), (0.775, 0.2417), (0.80, 0.2546), (0.825, 0.2706), (0.85, 0.2901),
    (0.875, 0.3136), (0.90, 0.3415), (0.925, 0.3734), (0.95, 0.4084), (0.975, 0.4448),
    (1.00, 0.4805), (1.025, 0.5136), (1.05, 0.5427), (1.075, 0.5677), (1.10, 0.5883),
    (1.15, 0.6165), (1.20, 0.6285), (1.30, 0.6231), (1.40, 0.6019), (1.50, 0.5750),
    (1.60, 0.5479), (1.80, 0.4995), (2.00, 0.4610), (2.20, 0.4311), (2.40, 0.4076),
    (2.60, 0.3885), (2.80, 0.3726), (3.00, 0.3591), (3.20, 0.3475), (3.40, 0.3374),
    (3.60, 0.3287), (3.80, 0.3210), (4.00, 0.3141), (4.20, 0.3079), (4.40, 0.3024),
    (4.60, 0.2973), (4.80, 0.2927), (5.00, 0.2883),
]

G7_DRAG_TABLE: list[tuple[float, float]] = [
    (0.00, 0.1198), (0.05, 0.1197), (0.10, 0.1196), (0.15, 0.1194), (0.20, 0.1193),
    (0.25, 0.1194), (0.30, 0.1194), (0.35, 0.1194), (0.40, 0.1193), (0.45, 0.1193),
    (0.50, 0.1194), (0.55, 0.1193), (0.60, 0.1194), (0.65, 0.1197), (0.70, 0.1202),
    (0.725, 0.1207), (0.75, 0.1215), (0.775, 0.1226), (0.80, 0.1242), (0.825, 0.1266),
    (0.85, 0.1306), (0.875, 0.1368), (0.90, 0.1464), (0.925, 0.1660), (0.95, 0.2054),
    (0.975, 0.2993), (1.00, 0.3803), (1.025, 0.4015), (1.05, 0.4043), (1.075, 0.4034),
    (1.10, 0.4014), (1.15, 0.3987), (1.20, 0.3955), (1.30, 0.3884), (1.40, 0.3810),
    (1.50, 0.3732), (1.60, 0.3657), (1.80, 0.3502), (2.00, 0.3358), (2.20, 0.3228),
    (2.40, 0.3113), (2.60, 0.3014), (2.80, 0.2931), (3.00, 0.2859), (3.20, 0.2797),
    (3.40, 0.2742), (3.60, 0.2693), (3.80, 0.2648), (4.00, 0.2606), (4.20, 0.2567),
    (4.40, 0.2531), (4.60, 0.2497), (4.80, 0.2465), (5.00, 0.2435),
]

DRAG_TABLES: dict[str, list[tuple[float, float]]] = {"G1": G1_DRAG_TABLE, "G7": G7_DRAG_TABLE}


try:
    import numba
    _jit = numba.jit(nopython=True, cache=True)
except ImportError:
    def _jit(func):
        return func


@_jit
def _numba_drag_acceleration(
    velocity: np.ndarray,
    mach_points: np.ndarray,
    cd_points: np.ndarray,
    speed_of_sound: float,
    air_density: float,
    area_over_mass: float,
) -> np.ndarray:
    speed = float(np.linalg.norm(velocity))
    if speed < 1e-9:
        return np.zeros(3)

    mach = speed / speed_of_sound
    cd = float(np.interp(mach, mach_points, cd_points))
    drag_accel_mag = 0.5 * air_density * cd * area_over_mass * speed
    return -drag_accel_mag * velocity


class DragForce(ForceModel):
    """Quadratic air drag, opposing velocity, scaled by a Mach-indexed
    drag-coefficient curve. See the module comment above for the formula,
    table provenance, and validation caveats."""

    def __init__(
        self,
        drag_model: str,
        mass_kg: float,
        diameter_m: float,
        air_density_kg_m3: float,
        speed_of_sound_mps: float,
    ):
        if drag_model not in DRAG_TABLES:
            raise ValueError(f"Unknown drag_model: {drag_model!r} (expected one of {sorted(DRAG_TABLES)})")
        if mass_kg <= 0.0 or diameter_m <= 0.0:
            raise ValueError(
                f"DragForce requires mass_kg > 0 and diameter_m > 0 "
                f"(got mass_kg={mass_kg}, diameter_m={diameter_m}) -- set these in "
                f"config/aiming_engine.yaml's projectile section before adding 'drag' to projectile.forces"
            )
        if air_density_kg_m3 <= 0.0 or speed_of_sound_mps <= 0.0:
            raise ValueError(
                f"air_density_kg_m3 and speed_of_sound_mps must be positive "
                f"(got {air_density_kg_m3}, {speed_of_sound_mps})"
            )

        table = DRAG_TABLES[drag_model]
        self._mach_points = np.array([m for m, _ in table])
        self._cd_points = np.array([cd for _, cd in table])

        area_m2 = np.pi * (diameter_m / 2.0) ** 2
        self._area_over_mass = area_m2 / mass_kg
        self._air_density = air_density_kg_m3
        self._speed_of_sound = speed_of_sound_mps

    def _drag_coefficient(self, mach: float) -> float:
        return float(np.interp(mach, self._mach_points, self._cd_points))

    def acceleration(self, position: np.ndarray, velocity: np.ndarray, t: float) -> np.ndarray:
        return _numba_drag_acceleration(
            velocity,
            self._mach_points,
            self._cd_points,
            self._speed_of_sound,
            self._air_density,
            self._area_over_mass,
        )



FORCE_REGISTRY = {
    "gravity": lambda config: GravityForce(config.gravity),
    "drag": lambda config: DragForce(
        drag_model=config.drag_model,
        mass_kg=config.mass_kg,
        diameter_m=config.diameter_m,
        air_density_kg_m3=config.air_density_kg_m3,
        speed_of_sound_mps=config.speed_of_sound_mps,
    ),
}


def build_forces(config: ProjectileConfig) -> list[ForceModel]:
    forces = []
    for name in config.forces:
        if name not in FORCE_REGISTRY:
            raise ValueError(f"Unknown force model: {name!r}")
        forces.append(FORCE_REGISTRY[name](config))
    return forces
