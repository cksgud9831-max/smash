"""Target motion estimation and future-position prediction, in World frame.

`MotionEstimator` is a swappable strategy so a future Kalman/constant-
acceleration estimator can replace `ConstantVelocityEstimator` without
`TargetState`'s public interface (or any of its callers) changing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from .config import TargetStateConfig
from .types import TargetVector, Vector3


class MotionEstimator(ABC):
    @abstractmethod
    def update(self, position: Vector3, t: float) -> None: ...

    @abstractmethod
    def predict(self, t_future: float) -> TargetVector: ...


class ConstantVelocityEstimator(MotionEstimator):
    """Fits a linear position-vs-time model over a short history buffer.

    Acceleration is always reported as zero — that is what "constant
    velocity" means for this estimator; a future constant-acceleration
    or Kalman estimator fills that field with a real estimate.
    """

    def __init__(self, history_length: int, min_dt: float):
        self._history_length = history_length
        self._min_dt = min_dt
        self._times: deque[float] = deque(maxlen=history_length)
        self._positions: deque[np.ndarray] = deque(maxlen=history_length)

    def update(self, position: Vector3, t: float) -> None:
        if self._times and (t - self._times[-1]) < self._min_dt:
            return  # duplicate/out-of-order timestamp, ignore to avoid degenerate fits
        self._times.append(t)
        self._positions.append(position.to_array())

    def predict(self, t_future: float) -> TargetVector:
        if not self._times:
            raise RuntimeError("predict() called before any update()")
        last_t = self._times[-1]
        last_p = self._positions[-1]

        dt = t_future - last_t

        if len(self._times) < 2:
            velocity = np.zeros(3)
            position_future = last_p
        else:
            ts = np.array(self._times) - last_t  # ref at most recent sample
            ps = np.array(self._positions)
            # Vectorised least-squares: solve for [slope, intercept] across
            # x/y/z simultaneously — equivalent to 3x np.polyfit but ~3x faster.
            A = np.column_stack([ts, np.ones(len(ts))])
            coeffs, *_ = np.linalg.lstsq(A, ps, rcond=None)
            # Use the OLS fitted position at t=last_t (coeffs[1]) instead of
            # the raw single measurement last_p to gain full noise reduction.
            velocity = coeffs[0]
            position_fitted_last = coeffs[1]
            position_future = position_fitted_last + velocity * dt

        return TargetVector(
            position=Vector3.from_array(position_future),
            velocity=Vector3.from_array(velocity),
            acceleration=Vector3(0.0, 0.0, 0.0),
            timestamp=t_future,
        )


class ConstantAccelerationKalmanEstimator(MotionEstimator):
    """9-state 3D Constant Acceleration Kalman Filter (CA KF).

    Tracks 3D position, 3D velocity, and 3D acceleration in World frame.
    State vector x = [px, py, pz, vx, vy, vz, ax, ay, az]^T
    """

    def __init__(self, min_dt: float = 1e-3, process_noise_std: float = 1.0, measurement_noise_std: float = 0.1):
        self._min_dt = min_dt
        self._process_noise_std = process_noise_std
        self._measurement_noise_std = measurement_noise_std

        self._x: np.ndarray | None = None
        self._P: np.ndarray | None = None
        self._last_t: float | None = None

        self._H = np.zeros((3, 9))
        self._H[0:3, 0:3] = np.eye(3)

    def _state_transition(self, dt: float) -> np.ndarray:
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = np.eye(3) * (0.5 * dt**2)
        F[3:6, 6:9] = np.eye(3) * dt
        return F

    def _process_noise(self, dt: float) -> np.ndarray:
        q = self._process_noise_std**2
        G = np.zeros((9, 3))
        G[0:3, :] = np.eye(3) * (0.5 * dt**2)
        G[3:6, :] = np.eye(3) * dt
        G[6:9, :] = np.eye(3)
        return G @ G.T * q

    def update(self, position: Vector3, t: float) -> None:
        z = position.to_array()
        if self._last_t is None or self._x is None or self._P is None:
            self._x = np.zeros(9)
            self._x[0:3] = z
            self._P = np.eye(9) * 1.0
            self._last_t = t
            return

        dt = t - self._last_t
        if dt < self._min_dt:
            return

        F = self._state_transition(dt)
        Q = self._process_noise(dt)

        x_pred = F @ self._x
        P_pred = F @ self._P @ F.T + Q

        R = np.eye(3) * (self._measurement_noise_std**2)
        y = z - self._H @ x_pred
        S = self._H @ P_pred @ self._H.T + R
        K = P_pred @ self._H.T @ np.linalg.inv(S)

        self._x = x_pred + K @ y
        self._P = (np.eye(9) - K @ self._H) @ P_pred
        self._last_t = t

    def predict(self, t_future: float) -> TargetVector:
        if self._last_t is None or self._x is None:
            raise RuntimeError("predict() called before any update()")

        dt = t_future - self._last_t
        F_future = self._state_transition(dt)
        x_future = F_future @ self._x

        return TargetVector(
            position=Vector3.from_array(x_future[0:3]),
            velocity=Vector3.from_array(x_future[3:6]),
            acceleration=Vector3.from_array(x_future[6:9]),
            timestamp=t_future,
        )


def build_estimator(config: TargetStateConfig) -> MotionEstimator:
    if config.estimator == "constant_velocity":
        return ConstantVelocityEstimator(config.history_length, config.min_dt)
    if config.estimator == "constant_acceleration":
        return ConstantAccelerationKalmanEstimator(
            min_dt=config.min_dt,
            process_noise_std=config.process_noise_std,
            measurement_noise_std=config.measurement_noise_std,
        )
    raise ValueError(f"Unknown target_state.estimator: {config.estimator!r}")


class TargetState:
    """Owns the single source of truth for target position/velocity/acceleration."""

    def __init__(self, config: TargetStateConfig, estimator: MotionEstimator | None = None):
        self._estimator = estimator if estimator is not None else build_estimator(config)

    def update(self, position_world: Vector3, timestamp: float) -> TargetVector:
        self._estimator.update(position_world, timestamp)
        return self._estimator.predict(timestamp)

    def predict(self, t_future: float) -> TargetVector:
        return self._estimator.predict(t_future)
