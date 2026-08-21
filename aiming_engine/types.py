"""Shared data types passed between AI Smart Scope Aiming Engine modules.

Internal math (residuals, Jacobians, integration) operates on plain
numpy arrays for speed; these dataclasses are the boundary types
modules exchange with each other and with the outside world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Vector3:
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @staticmethod
    def from_array(arr: np.ndarray) -> "Vector3":
        return Vector3(float(arr[0]), float(arr[1]), float(arr[2]))

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3.from_array(self.to_array() + other.to_array())

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3.from_array(self.to_array() - other.to_array())

    def norm(self) -> float:
        return float(np.linalg.norm(self.to_array()))


@dataclass
class TrackerFrame:
    """One frame of bridge-layer tracker output."""

    timestamp: float
    bbox: tuple[float, float, float, float]
    center_px: tuple[float, float]
    velocity_px: tuple[float, float]
    tracking_confidence: float
    follower_state: str
    camera_intrinsics: np.ndarray  # 3x3
    camera_extrinsics: np.ndarray  # 4x4, camera/scope platform pose in world
    laser_range_m: Optional[float]


@dataclass
class TargetVector:
    """Estimated target motion state, always in World frame."""

    position: Vector3
    velocity: Vector3
    acceleration: Vector3
    timestamp: float

    def predict_position(self, t_future: float) -> Vector3:
        dt = t_future - self.timestamp
        p = self.position.to_array()
        v = self.velocity.to_array()
        a = self.acceleration.to_array()
        return Vector3.from_array(p + v * dt + 0.5 * a * dt * dt)


@dataclass
class InitialGuess:
    """Warm-start seed for the Newton solver, from LeadPrediction."""

    t0: float
    azimuth0: float
    elevation0: float
    naive_lead_point: Vector3


@dataclass
class SolverResult:
    converged: bool
    iterations: int
    x: np.ndarray  # [t, azimuth, elevation]
    residual_norm: float


@dataclass
class AimSolution:
    """The current aim recommendation shown to the operator via HUD.

    This is advisory only — it never triggers an action on its own. The
    human operator reads this (azimuth/elevation/lead point) off the HUD
    and decides whether and when to actually fire.
    """

    lead_point_world: Vector3
    azimuth: float  # scope-frame radians
    elevation: float  # scope-frame radians
    ballistic_offset: tuple[float, float]  # (d_az, d_el) scope-frame rad, vs naive line-of-sight
    time_of_flight: float
    solver_converged: bool
    solver_iterations: int
    residual_norm_m: float


class AimState(Enum):
    """SEARCH -> TRACK -> AIM -> READY. There is no firing state here —
    READY means "aim solution is stable enough to trust," not "fire now."
    That decision is always the operator's."""

    SEARCH = "SEARCH"
    TRACK = "TRACK"
    AIM = "AIM"
    READY = "READY"


@dataclass
class AimAssistOutput:
    """Per-frame output of AimingManager.update() — everything the HUD
    needs to render for the operator. Advisory only; there is no fire
    command anywhere in this framework."""

    state: AimState
    aim_solution: Optional[AimSolution]
    hit_probability: float
    aim_ready: bool
    debug: dict = field(default_factory=dict)
