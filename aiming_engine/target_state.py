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

    A2 (로드맵 2단계 선결과제) 관련 메모: known_issues.md 1.5절은 이미 한 번
    필터링된 트래커 위치를 여기서 다시 미분하는 "이중 미분" 구조가 노이즈를
    증폭시킨다고 지적했고, 대안으로 bridge가 제공하는 velocity_px를 이 추정기의
    초기 속도로 반영하는 방안을 제시했다. 실제 코드를 확인한 결과 이 방안은
    지금 그대로 적용할 수 없다: bridge/frame_builder.py는 production 경로에서
    velocity_px를 항상 (0.0, 0.0) 플레이스홀더로만 채우고 있고(OpticalFlowTracker가
    SmartTracker의 CA KF와 달리 bbox 속도를 자체 추정하지 않기 때문),
    scripts/06_gazebo_interactive_shooter.py 쪽은 반대로 단일 프레임 차분으로
    직접 velocity_px를 계산해 넘기므로 오히려 이 OLS 추정보다 노이즈가 더 크다.
    두 경로 모두에서 "이미 스무딩된 외부 속도 신호를 반영"한다는 원래 전제가
    성립하지 않는다.

    그래서 대신 적용한 개선은 velocity_smoothing_alpha를 통한 지수이동평균(EMA)
    스무딩이다. 매 update() 이후 처음 호출되는 predict()에서만 새 OLS 속도를
    계산해 이전 스무딩값과 지수가중 평균을 취하고, 같은 last_t에 대한 이후
    predict() 재호출(뉴턴솔버가 t_of_flight를 여러 번 시도할 때 등)에는 그
    캐시된 값을 그대로 재사용한다. 그렇지 않으면 프레임당 여러 번 호출되는
    predict()가 스무딩 상태를 중복 갱신해버린다. alpha=1.0(기본값)은 기존
    Level 1/Level 2 검증에 쓰인 것과 완전히 동일한 무보정 OLS 동작이므로 값을
    바꾸지 않는 한 기존 검증 수치에 회귀가 없다.
    """

    def __init__(self, history_length: int, min_dt: float, velocity_smoothing_alpha: float = 1.0):
        self._history_length = history_length
        self._min_dt = min_dt
        self._velocity_smoothing_alpha = velocity_smoothing_alpha
        self._times: deque[float] = deque(maxlen=history_length)
        self._positions: deque[np.ndarray] = deque(maxlen=history_length)
        self._smoothed_velocity: np.ndarray | None = None
        self._smoothed_at_t: float | None = None
        self._fitted_last_position: np.ndarray | None = None

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
            if self._smoothed_at_t == last_t and self._smoothed_velocity is not None:
                # 같은 측정 세대(last_t 불변)에 대한 재호출이므로 새로 피팅하지 않고
                # 캐시된 값을 재사용해 스무딩 상태가 중복 갱신되지 않게 한다.
                velocity = self._smoothed_velocity
                position_fitted_last = self._fitted_last_position
            else:
                ts = np.array(self._times) - last_t  # ref at most recent sample
                ps = np.array(self._positions)
                # Vectorised least-squares: solve for [slope, intercept] across
                # x/y/z simultaneously — equivalent to 3x np.polyfit but ~3x faster.
                A = np.column_stack([ts, np.ones(len(ts))])
                coeffs, *_ = np.linalg.lstsq(A, ps, rcond=None)
                raw_velocity = coeffs[0]
                position_fitted_last = coeffs[1]

                alpha = self._velocity_smoothing_alpha
                if self._smoothed_velocity is None or alpha >= 1.0:
                    velocity = raw_velocity
                else:
                    velocity = alpha * raw_velocity + (1.0 - alpha) * self._smoothed_velocity

                self._smoothed_velocity = velocity
                self._smoothed_at_t = last_t
                self._fitted_last_position = position_fitted_last

            # Use the OLS fitted position at t=last_t instead of the raw single
            # measurement last_p to gain full noise reduction.
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
        return ConstantVelocityEstimator(
            config.history_length, config.min_dt, config.velocity_smoothing_alpha
        )
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
