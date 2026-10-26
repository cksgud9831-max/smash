#!/usr/bin/env python3
"""Gazebo 포탑 구동 전용 노드.

영상 추론과 같은 executor에서 관절 명령 타이머를 돌리면 YOLO 콜백 동안 명령
발행이 멈춘다. 이 노드는 시뮬레이션 구동만 별도 프로세스로 분리해, SMASH
탐지/추적/조준 알고리즘의 처리 속도와 무관하게 일정한 주기로 Gazebo 관절 명령을
보낸다.

포탑은 FCS 가 발행하는 리드 방향(/smash_fcs/aim_track)을 따라간다. 선회 명령
토픽(slew_topic)은 표적을 놓쳤을 때의 탐지 자세 복귀(뷰어 R 키)와 측정 도구의
초기 선회에 쓴다. 큐 토픽(cue_topic)은 새 표적이 화면 중심 밖에서 나타났을 때(격추 뒤
재출현) base 프레임 좌표 [x, y, z] 를 받아 그쪽으로 포를 미리 돌려 놓는다.
"""

import json
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

from drone_sim.smash_fcs.aim_follow import (
    AimTrackFollower,
    GoalRamp,
    cue_goal_for_point,
    goal_within_acquisition_envelope,
)


PAN_DITHER_RAD = 5.0e-4


def clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def rate_limit(goal: float, current: float, max_rate: float, dt: float) -> float:
    step = max(0.0, max_rate) * max(0.0, dt)
    return current + clamp(goal - current, -step, step)


class TurretCommandNode(Node):
    """FCS 리드 방향 추종과 선회 명령을 Gazebo Pan/Tilt 명령으로 변환한다."""

    def __init__(self) -> None:
        super().__init__("turret_command")

        self.declare_parameter("slew_topic", "/smash_fcs/slew_cmd")
        self.declare_parameter("cue_topic", "/smash_fcs/cue_cmd")
        self.declare_parameter("aim_track_topic", "/smash_fcs/aim_track")
        self.declare_parameter("fcs_state_topic", "/smash_fcs/state")
        self.declare_parameter("command_topic", "/turret_controller/commands")
        self.declare_parameter("command_rate_hz", 60.0)
        self.declare_parameter("pan_max_rate_rad_s", 3.0)
        self.declare_parameter("tilt_max_rate_rad_s", 2.5)
        self.declare_parameter("tilt_min_rad", 0.0)
        self.declare_parameter("tilt_max_rad", 1.4835298641951802)
        self.declare_parameter("search_pan_rad", 0.0)
        self.declare_parameter("search_tilt_rad", 0.38)
        self.declare_parameter("pan_dither_rad", PAN_DITHER_RAD)
        self.declare_parameter("slew_override_s", 0.75)
        self.declare_parameter("aim_feedforward_s", 0.19)
        self.declare_parameter("aim_feedforward_max_rate_rad_s", 0.5)
        self.declare_parameter("aim_tilt_feedforward_s", 0.0)
        self.declare_parameter("aim_extrapolate_s", 1.0)
        self.declare_parameter("aim_rate_smoothing", 0.5)
        self.declare_parameter("aim_max_goal_jump_rad", 0.012)
        self.declare_parameter("aim_acquire_envelope_rad", 0.08)
        self.declare_parameter("aim_ready_confirm_s", 1.0)
        self.declare_parameter("aim_require_detector_refresh", True)
        # 추종 입력. "lead"(기본) = 리드점, "bbox" = bbox 중심 + 평활한 리드 성분(시험용 옵션).
        self.declare_parameter("aim_follow_mode", "lead")
        self.declare_parameter("aim_offset_smoothing", 0.8)
        self.declare_parameter("aim_follower_rate_smoothing", 0.0)

        self._pan_rate = float(self.get_parameter("pan_max_rate_rad_s").value)
        self._tilt_rate = float(self.get_parameter("tilt_max_rate_rad_s").value)
        self._tilt_min = float(self.get_parameter("tilt_min_rad").value)
        self._tilt_max = float(self.get_parameter("tilt_max_rad").value)
        self._pan_goal = float(self.get_parameter("search_pan_rad").value)
        self._tilt_goal = clamp(
            float(self.get_parameter("search_tilt_rad").value), self._tilt_min, self._tilt_max
        )
        self._pan_command = self._pan_goal
        self._tilt_command = self._tilt_goal
        self._dither_amplitude = abs(float(self.get_parameter("pan_dither_rad").value))
        self._dither = self._dither_amplitude
        self._slew_override_s = max(0.0, float(self.get_parameter("slew_override_s").value))
        self._slew_override_until = 0.0
        self._aim_feedforward_s = max(0.0, float(self.get_parameter("aim_feedforward_s").value))
        self._aim_follower = AimTrackFollower(
            lead_time_s=self._aim_feedforward_s,
            max_rate_rad_s=float(self.get_parameter("aim_feedforward_max_rate_rad_s").value),
            tilt_min=self._tilt_min,
            tilt_max=self._tilt_max,
            max_goal_jump_rad=float(self.get_parameter("aim_max_goal_jump_rad").value),
            tilt_lead_time_s=float(self.get_parameter("aim_tilt_feedforward_s").value),
            follow_mode=str(self.get_parameter("aim_follow_mode").value),
            offset_smoothing=float(self.get_parameter("aim_offset_smoothing").value),
            rate_smoothing=float(self.get_parameter("aim_follower_rate_smoothing").value),
        )
        # 검출기 갱신 프레임(약 3 Hz) 사이에도 마지막 측정 각속도로 목표를 이어 움직여,
        # 포가 갱신마다 한 번에 튀었다 멈추는 계단 동작을 없앤다. 0 이면 끈다.
        self._aim_ramp = GoalRamp(
            horizon_s=float(self.get_parameter("aim_extrapolate_s").value),
            smoothing=float(self.get_parameter("aim_rate_smoothing").value),
            tilt_min=self._tilt_min,
            tilt_max=self._tilt_max,
            tilt_ramp=float(self.get_parameter("aim_tilt_feedforward_s").value) > 0.0,
        )
        self._aim_acquire_envelope = max(
            0.0, float(self.get_parameter("aim_acquire_envelope_rad").value)
        )
        self._aim_ready_confirm_s = max(
            0.0, float(self.get_parameter("aim_ready_confirm_s").value)
        )
        self._aim_require_detector_refresh = bool(
            self.get_parameter("aim_require_detector_refresh").value
        )
        self._acquisition_anchor = (self._pan_goal, self._tilt_goal)
        self._aim_lock_confirmed = False
        self._ready_since = None
        self._last_tick = time.monotonic()
        # 계측: 추종기가 목표를 받아들이는지, 한도에 걸리는지 1 Hz 로 기록한다(동작에는 쓰지 않는다).
        self._diag = {"msgs": 0, "slew_skip": 0, "no_refresh": 0, "envelope_reject": 0, "accepted": 0}
        self._diag_last_log = 0.0

        self._publisher = self.create_publisher(
            Float64MultiArray, str(self.get_parameter("command_topic").value), 10
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("slew_topic").value),
            self._on_slew,
            200,
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("cue_topic").value),
            self._on_cue,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("aim_track_topic").value),
            self._on_aim_track,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("fcs_state_topic").value),
            self._on_fcs_state,
            10,
        )
        rate = max(1.0, float(self.get_parameter("command_rate_hz").value))
        self.create_timer(1.0 / rate, self._publish_command)
        self.create_timer(1.0, self._log_diag)
        self.get_logger().info(
            f"Gazebo 포탑 독립 제어 시작: {rate:.1f} Hz, "
            f"PAN {math.degrees(self._pan_goal):.1f}deg, "
            f"TILT {math.degrees(self._tilt_goal):.1f}deg, "
            f"피드포워드 팬 {self._aim_feedforward_s:.2f}s 틸트 "
            f"{float(self.get_parameter('aim_tilt_feedforward_s').value):.2f}s, "
            f"갱신 사이 연속 이동 {float(self.get_parameter('aim_extrapolate_s').value):.1f}s, "
            f"목표 점프 한도 {float(self.get_parameter('aim_max_goal_jump_rad').value):.4f} rad, "
            f"추종 입력 {self.get_parameter('aim_follow_mode').value}"
        )

    def _on_slew(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < 2:
            return
        d_pan, d_tilt = float(msg.data[0]), float(msg.data[1])
        if not math.isfinite(d_pan) or not math.isfinite(d_tilt):
            return
        self._begin_slew()
        if abs(d_pan - 999.0) < 1e-3:
            self._pan_goal, self._tilt_goal = 0.0, 0.0
            self.get_logger().info("수평 정렬 프리셋: PAN 0.0deg, TILT 0.0deg")
        elif abs(d_pan - 888.0) < 1e-3:
            self._pan_goal, self._tilt_goal = 0.0, 0.38
            self.get_logger().info("대공 경계 프리셋: PAN 0.0deg, TILT 21.8deg")
        else:
            self._pan_goal += d_pan
            self._tilt_goal = clamp(self._tilt_goal + d_tilt, self._tilt_min, self._tilt_max)
        self._end_slew()

    def _begin_slew(self) -> None:
        # 선회 명령은 잠시 aim_track 추종보다 우선한다. 이전 추종 상태는 새 방향과 무관하다.
        self._slew_override_until = time.monotonic() + self._slew_override_s
        self._aim_follower.reset()
        self._aim_ramp.clear()

    def _end_slew(self) -> None:
        # 새 자세를 최초 포착 창의 기준으로 삼고, 이전 표적의 격발 확인은 무효로 한다.
        self._acquisition_anchor = (self._pan_goal, self._tilt_goal)
        self._aim_lock_confirmed = False
        self._ready_since = None

    def _on_cue(self, msg: Float64MultiArray) -> None:
        """새 접촉 위치(base 프레임 [x, y, z], m)로 포를 미리 돌린다.

        최초 포착 게이트는 화면 중심 근처 후보만 받으므로, 중심 밖(또는 화면 밖)에
        나타난 표적은 포가 그쪽을 보기 전에는 영원히 포착되지 않는다. 실제 대공 체계가
        레이더 지시로 광학 센서를 먼저 돌려 놓는 것과 같은 동작이다.
        """

        goal = cue_goal_for_point(
            msg.data[:3],
            reference_pan=self._pan_goal,
            tilt_min=self._tilt_min,
            tilt_max=self._tilt_max,
        ) if len(msg.data) >= 3 else None
        if goal is None:
            return
        self._begin_slew()
        self._pan_goal, self._tilt_goal = goal
        self._end_slew()
        self.get_logger().info(
            f"신규 접촉 지시: PAN {math.degrees(goal[0]):.1f}deg, TILT {math.degrees(goal[1]):.1f}deg"
        )

    def _on_fcs_state(self, msg: String) -> None:
        if self._aim_lock_confirmed:
            return
        if msg.data == "READY":
            now = time.monotonic()
            self._ready_since = self._ready_since or now
            if now - self._ready_since >= self._aim_ready_confirm_s:
                self._aim_lock_confirmed = True
        else:
            self._ready_since = None

    def _on_aim_track(self, msg: String) -> None:
        """Close the simulator pan/tilt loop on the advisory lead direction.

        A slew command temporarily wins, and an invalid/non-STABLE track never
        moves the stand.  This changes only simulated pointing; firing stays with
        the FCS READY gate.
        """

        self._diag["msgs"] += 1
        if time.monotonic() < self._slew_override_until:
            self._diag["slew_skip"] += 1
            return
        try:
            track = json.loads(msg.data)
        except (TypeError, ValueError):
            return
        if not isinstance(track, dict):
            return
        if self._aim_require_detector_refresh and not bool(track.get("detector_refresh", False)):
            self._diag["no_refresh"] += 1
            return
        goal = self._aim_follower.update(
            track,
            # YOLO 보정 프레임의 base 절대 방향만 사용한다. LK-only 프레임은
            # 위에서 거르므로 배경 고착 예측을 관절 목표에 누적하지 않는다.
            reference_pan=self._pan_goal,
            reference_tilt=None,
            jump_reference_tilt=self._tilt_goal,
        )
        if goal is not None and not self._aim_lock_confirmed and not goal_within_acquisition_envelope(
            goal, self._acquisition_anchor, self._aim_acquire_envelope
        ):
            self._diag["envelope_reject"] += 1
            self._aim_follower.reset()
            self._aim_ramp.clear()
            return
        if goal is None:
            self._aim_ramp.clear()
            return
        self._diag["accepted"] += 1
        self._pan_goal, self._tilt_goal = goal
        self._aim_ramp.set(goal, self._aim_follower.rate, self._aim_follower.rate_valid, self._sim_now())

    def _log_diag(self) -> None:
        d, f = self._diag, self._aim_follower.diag
        self.get_logger().info(
            f"[AIMDIAG] sim={self._sim_now():.2f} msgs={d['msgs']} slew_skip={d['slew_skip']} "
            f"no_refresh={d['no_refresh']} upd={f['updates']} no_goal={f['no_goal']} "
            f"raw_clip={f['raw_clipped']} final_clip={f['final_clipped']} "
            f"env_rej={d['envelope_reject']} accepted={d['accepted']} lock={int(self._aim_lock_confirmed)} "
            f"goal_pan={self._pan_goal:.4f} cmd_pan={self._pan_command:.4f}"
        )

    def _sim_now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _publish_command(self) -> None:
        ramped = self._aim_ramp.at(self._sim_now())
        if ramped is not None:
            self._pan_goal, self._tilt_goal = ramped
        now = time.monotonic()
        dt = min(0.1, max(1e-4, now - self._last_tick))
        self._last_tick = now
        self._pan_command = rate_limit(self._pan_goal, self._pan_command, self._pan_rate, dt)
        self._tilt_command = clamp(
            rate_limit(self._tilt_goal, self._tilt_command, self._tilt_rate, dt),
            self._tilt_min,
            self._tilt_max,
        )

        # gz_ros2_control/DART에서 pan 오차가 정확히 0이면 tilt가 멎는 현상을 피한다.
        # 진폭은 0.4 rad / 640 px 기준 1픽셀보다 작다.
        self._dither = -self._dither if self._dither_amplitude > 0.0 else 0.0
        msg = Float64MultiArray()
        msg.data = [self._pan_command + self._dither, self._tilt_command]
        self._publisher.publish(msg)


def main() -> None:
    rclpy.init()
    node = TurretCommandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
