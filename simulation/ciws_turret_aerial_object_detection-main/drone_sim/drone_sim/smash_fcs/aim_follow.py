"""ROS-independent conversion from an FCS aim-track message to turret goals.

This is a simulator-only operator-assist loop.  It points the Gazebo pan/tilt
stand at the advisory lead direction while leaving trigger/fire decisions in
the existing fire-control path.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import numpy as np

from .turret_kinematics import solve_pan_tilt_for_point, wrap_to_pi


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _unwrap_near(angle: float, reference: float) -> float:
    return reference + (angle - reference + math.pi) % (2.0 * math.pi) - math.pi


def goal_within_acquisition_envelope(
    goal: tuple[float, float], anchor: tuple[float, float], max_offset_rad: float
) -> bool:
    """Keep unconfirmed auto-aim inside the camera's initial search window."""

    pan_offset = abs(_unwrap_near(float(goal[0]), float(anchor[0])) - float(anchor[0]))
    tilt_offset = abs(float(goal[1]) - float(anchor[1]))
    return max(pan_offset, tilt_offset) <= max(0.0, float(max_offset_rad))


def cue_goal_for_point(
    point_base: Sequence[float],
    *,
    reference_pan: float,
    tilt_min: float,
    tilt_max: float,
) -> Optional[tuple[float, float]]:
    """Joint goal that points the muzzle at an externally cued base-frame point.

    Used when the contact appears away from the crosshair (a new target after a
    kill): the initial-acquisition gate only admits candidates near the image
    centre, so the stand is slewed there first instead of waiting for the
    target to wander into the gate.  Pan is unwrapped next to ``reference_pan``.
    """

    try:
        x, y, z = (float(v) for v in point_base)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x, y, z)) or math.hypot(x, y) < 1e-6:
        return None
    pan, tilt = solve_pan_tilt_for_point(np.array([x, y, z], dtype=np.float64), pan_seed=reference_pan)
    return _unwrap_near(pan, reference_pan), _clamp(tilt, tilt_min, tilt_max)


def goal_from_aim_track(
    track: Mapping[str, object],
    *,
    reference_pan: float,
    reference_tilt: float | None = None,
    tilt_min: float,
    tilt_max: float,
) -> Optional[tuple[float, float]]:
    """Return a continuous ``(pan, tilt)`` goal for a trusted lead direction.

    HOLD/LOST data is deliberately rejected: the command node then keeps its
    last goal instead of steering from a predicted or invalid measurement.
    """

    if track.get("reliability") != "STABLE":
        return None
    direction = track.get("lead_dir_base")
    if not isinstance(direction, Sequence) or isinstance(direction, (str, bytes)) or len(direction) != 3:
        return None
    try:
        x, y, z = (float(value) for value in direction)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, z)):
        return None
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-9:
        return None

    lead_pan = math.atan2(y, x)
    lead_tilt = math.atan2(z, math.hypot(x, y))

    # 절대 base 방향은 영상 timestamp의 TF가 없어 최신 TF로 fallback했을 때
    # 카메라 이동량을 중복 포함할 수 있다. 메시지의 bore와 lead는 같은 pose로
    # 계산됐으므로 둘의 상대각은 이 지연에 불변이다. 그 상대 보정만 현재 관절
    # 목표에 더하면 고속 처리에서 나타난 양의 피드백 runaway를 막는다.
    bore = track.get("bore_dir_base")
    if (
        reference_tilt is not None
        and isinstance(bore, Sequence)
        and not isinstance(bore, (str, bytes))
        and len(bore) == 3
    ):
        try:
            bx, by, bz = (float(value) for value in bore)
        except (TypeError, ValueError):
            bx = by = bz = float("nan")
        if all(math.isfinite(value) for value in (bx, by, bz)) and math.sqrt(bx * bx + by * by + bz * bz) >= 1e-9:
            bore_pan = math.atan2(by, bx)
            bore_tilt = math.atan2(bz, math.hypot(bx, by))
            pan = float(reference_pan) + _unwrap_near(lead_pan, bore_pan) - bore_pan
            tilt = float(reference_tilt) + lead_tilt - bore_tilt
        else:
            pan, tilt = _unwrap_near(lead_pan, float(reference_pan)), lead_tilt
    else:
        pan, tilt = _unwrap_near(lead_pan, float(reference_pan)), lead_tilt
    return float(pan), float(_clamp(tilt, tilt_min, tilt_max))


class AimTrackFollower:
    """Convert trusted aim-track samples into latency-compensated joint goals.

    ``stamp`` is simulation time, so the angular velocity and lead interval
    stay correct even when Gazebo runs slower than wall clock. A trust loss or
    manual override resets the differentiator to prevent stale extrapolation.
    """

    def __init__(
        self,
        *,
        lead_time_s: float,
        max_rate_rad_s: float,
        tilt_min: float,
        tilt_max: float,
        max_goal_jump_rad: float = math.inf,
        tilt_lead_time_s: float | None = None,
        follow_mode: str = "lead",
        offset_smoothing: float = 0.0,
        rate_smoothing: float = 0.0,
    ) -> None:
        self._lead_time_s = max(0.0, float(lead_time_s))
        # 틸트 리드는 별도로 줄 수 있다. 수평 기동 표적에서는 틸트 각속도가 거의
        # 순수 잡음이라 팬과 같은 외삽을 걸면 오히려 포가 위아래로 떤다.
        self._tilt_lead_time_s = (
            self._lead_time_s if tilt_lead_time_s is None else max(0.0, float(tilt_lead_time_s))
        )
        self._max_rate = max(0.0, float(max_rate_rad_s))
        # "lead": 리드점(기본). "bbox": 팬 목표 = bbox 중심 방향 + 완만하게 평활한 (리드 - bbox) 각도 차.
        # 리드점은 속도 추정 잡음에 따라 크게 흔들리므로, 위치는 bbox 에서 얻고 리드 성분만 EMA 로 눌러 쓴다.
        self._follow_mode = str(follow_mode)
        self._offset_smoothing = _clamp(float(offset_smoothing), 0.0, 0.99)
        self._rate_smoothing = _clamp(float(rate_smoothing), 0.0, 0.99)
        self._offset_pan: float | None = None
        self._rate_pan_ema: float | None = None
        self._prev: tuple[float, float] | None = None
        self._prev_time: float | None = None
        self._tilt_min = float(tilt_min)
        self._tilt_max = float(tilt_max)
        self._max_goal_jump = max(0.0, float(max_goal_jump_rad))
        #: 직전 update 에서 쓴 (팬, 틸트) 목표각 속도 [rad/s]. rate_valid 가 False 면 0.
        self.rate: tuple[float, float] = (0.0, 0.0)
        self.rate_valid = False
        #: 계측용 누적 카운터. 동작에는 쓰지 않는다.
        self.diag = {"updates": 0, "no_goal": 0, "raw_clipped": 0, "final_clipped": 0}

    def reset(self) -> None:
        self._prev = None
        self._prev_time = None
        self._offset_pan = None
        self._rate_pan_ema = None
        self.rate = (0.0, 0.0)
        self.rate_valid = False

    def _extrapolate(self, goal: tuple[float, float], stamp: float) -> tuple[float, float]:
        """직전 목표와의 차분으로 각속도를 구해 lead_time 만큼 앞당긴다."""

        pan, tilt = goal
        rate_pan = rate_tilt = 0.0
        self.rate_valid = False
        if self._prev is not None and self._prev_time is not None:
            dt = stamp - self._prev_time
            if dt > 1e-6:
                rate_pan = wrap_to_pi(pan - self._prev[0]) / dt
                rate_tilt = (tilt - self._prev[1]) / dt
                # 재포착/보정 불연속은 물리 각속도가 아니다. 그 축은 외삽하지 않는다.
                if abs(rate_pan) > self._max_rate:
                    rate_pan = 0.0
                if abs(rate_tilt) > self._max_rate:
                    rate_tilt = 0.0
                if self._rate_smoothing > 0.0:
                    if self._rate_pan_ema is not None:
                        rate_pan = self._rate_smoothing * self._rate_pan_ema + (1.0 - self._rate_smoothing) * rate_pan
                    self._rate_pan_ema = rate_pan
                self.rate_valid = True
        self._prev = (pan, tilt)
        self._prev_time = stamp
        self.rate = (rate_pan, rate_tilt)
        return (
            pan + rate_pan * self._lead_time_s,
            tilt + rate_tilt * self._tilt_lead_time_s,
        )

    def _limit_goal_step(
        self,
        goal: tuple[float, float],
        *,
        reference_pan: float,
        reference_tilt: float | None,
    ) -> tuple[tuple[float, float], bool]:
        """Bound one detector correction without latching behind the target.

        Rejecting an over-limit correction leaves the old joint goal in place.
        Once a real moving target gets farther than the limit, every later
        correction is then rejected too, so the turret can never catch up.
        Saturating each axis keeps the same one-frame safety bound while making
        a persistent, coherent target reachable over successive detections.
        """

        pan = _unwrap_near(float(goal[0]), float(reference_pan))
        tilt = float(goal[1])
        if not math.isfinite(self._max_goal_jump):
            return (pan, tilt), False
        pan_delta = pan - float(reference_pan)
        limited_pan_delta = _clamp(pan_delta, -self._max_goal_jump, self._max_goal_jump)
        limited_tilt = tilt
        tilt_limited = False
        if reference_tilt is not None:
            tilt_delta = tilt - float(reference_tilt)
            limited_tilt_delta = _clamp(tilt_delta, -self._max_goal_jump, self._max_goal_jump)
            limited_tilt = float(reference_tilt) + limited_tilt_delta
            tilt_limited = limited_tilt_delta != tilt_delta
        return (
            (float(reference_pan) + limited_pan_delta, float(limited_tilt)),
            limited_pan_delta != pan_delta or tilt_limited,
        )

    def update(
        self,
        track: Mapping[str, object],
        *,
        reference_pan: float,
        reference_tilt: float | None = None,
        jump_reference_tilt: float | None = None,
    ) -> Optional[tuple[float, float]]:
        goal = goal_from_aim_track(
            track,
            reference_pan=reference_pan,
            reference_tilt=reference_tilt,
            tilt_min=self._tilt_min,
            tilt_max=self._tilt_max,
        )
        self.diag["updates"] += 1
        if goal is None:
            self.diag["no_goal"] += 1
            self.reset()
            return None
        if self._follow_mode == "bbox":
            bbox_dir = track.get("bbox_dir_base")
            if bbox_dir is not None:
                bbox_goal = goal_from_aim_track(
                    {**track, "lead_dir_base": bbox_dir},
                    reference_pan=reference_pan,
                    reference_tilt=reference_tilt,
                    tilt_min=self._tilt_min,
                    tilt_max=self._tilt_max,
                )
                if bbox_goal is not None:
                    offset = wrap_to_pi(goal[0] - bbox_goal[0])
                    if self._offset_pan is not None and self._offset_smoothing > 0.0:
                        offset = self._offset_smoothing * self._offset_pan + (1.0 - self._offset_smoothing) * offset
                    self._offset_pan = offset
                    goal = (bbox_goal[0] + offset, goal[1])
        # 한 프레임의 false-positive가 잠깐 STABLE이 되더라도 협각 카메라
        # 밖까지 포탑을 던지지 않는다. 정상 추종의 프레임 간 변화량은 이
        # 한계보다 훨씬 작고, 큰 재지향은 사수의 수동 slew가 담당한다.
        tilt_reference = reference_tilt if jump_reference_tilt is None else jump_reference_tilt
        goal, raw_limited = self._limit_goal_step(
            goal, reference_pan=reference_pan, reference_tilt=tilt_reference
        )
        if raw_limited:
            self.diag["raw_clipped"] += 1
            # Do not derive angular velocity from a clipped discontinuity.
            self.reset()
            return goal
        try:
            stamp = float(track.get("stamp"))
        except (TypeError, ValueError):
            self.reset()
            return goal
        if not math.isfinite(stamp):
            self.reset()
            return goal

        pan, tilt = self._extrapolate(goal, stamp)
        pan = _unwrap_near(pan, float(reference_pan))
        tilt = _clamp(tilt, self._tilt_min, self._tilt_max)
        # 원시 측정은 정상 범위여도 수치미분 feedforward가 추적 잡음을
        # 큰 외삽으로 바꿀 수 있다. 최종 명령도 같은 FOV 안전 한계를
        # 만족해야 하며, 넘으면 이 샘플의 미분 이력을 폐기한다.
        limited_goal, final_limited = self._limit_goal_step(
            (pan, tilt), reference_pan=reference_pan, reference_tilt=tilt_reference
        )
        if final_limited:
            self.diag["final_clipped"] += 1
            self.reset()
        return limited_goal


class GoalRamp:
    """Keep the pan/tilt goal moving between sparse aim-track updates.

    The follower only accepts detector-refreshed samples (about 3 Hz).  Holding
    the goal in between makes the stand move in pulses: it jumps by one sample
    interval of target motion, then waits.  This continues the last measured
    angular rate from the sample's own simulation time, so the stand tracks a
    steady target smoothly and the next sample only has to fix a small residual.

    ``smoothing`` is the EMA weight of the previous rate (0 = raw).  A failed
    rate (trust loss, clipped jump) zeroes the history.  The ramp is capped at
    ``horizon_s``: a stalled feed never drives the stand indefinitely.  Tilt is
    ramped only when ``tilt_ramp`` is set, because horizontal targets have no
    real vertical rate and the measured one is noise.
    """

    def __init__(
        self,
        *,
        horizon_s: float,
        smoothing: float,
        tilt_min: float,
        tilt_max: float,
        tilt_ramp: bool = False,
    ) -> None:
        self._horizon = max(0.0, float(horizon_s))
        self._smoothing = _clamp(float(smoothing), 0.0, 0.99)
        self._tilt_min = float(tilt_min)
        self._tilt_max = float(tilt_max)
        self._tilt_ramp = bool(tilt_ramp)
        self._anchor: tuple[float, float] | None = None
        self._rate = (0.0, 0.0)
        self._time = 0.0

    def clear(self) -> None:
        self._anchor = None
        self._rate = (0.0, 0.0)

    def set(
        self,
        goal: tuple[float, float],
        rate: tuple[float, float],
        rate_valid: bool,
        now: float,
    ) -> None:
        if self._horizon <= 0.0:
            return
        if rate_valid and self._anchor is not None:
            alpha = self._smoothing
            pan_rate = alpha * self._rate[0] + (1.0 - alpha) * rate[0]
            tilt_rate = alpha * self._rate[1] + (1.0 - alpha) * rate[1]
        elif rate_valid:
            pan_rate, tilt_rate = rate
        else:
            pan_rate = tilt_rate = 0.0
        if not self._tilt_ramp:
            tilt_rate = 0.0
        self._anchor = (float(goal[0]), float(goal[1]))
        self._rate = (float(pan_rate), float(tilt_rate))
        self._time = float(now)

    def at(self, now: float) -> Optional[tuple[float, float]]:
        if self._anchor is None:
            return None
        age = _clamp(float(now) - self._time, 0.0, self._horizon)
        return (
            self._anchor[0] + self._rate[0] * age,
            _clamp(self._anchor[1] + self._rate[1] * age, self._tilt_min, self._tilt_max),
        )
