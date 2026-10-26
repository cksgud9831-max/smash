#!/usr/bin/env python3
"""operator_follow.py — 사수 에뮬레이션: 포탑이 표적 실측 위치를 따라가게 한다.

실제 조준경에서는 사수가 표적 UAV 를 화면에 넣고 따라간다. 시뮬레이션에는 사수가 없으므로, 이 노드가
Gazebo 가 알려 주는 표적 실측 위치(ground truth)로 포를 돌려 표적을 화면 중심 부근에 둔다.

  - 실측 위치는 포의 *지향*에만 쓴다. 탐지·추적·거리·속도 추정·조준점·격발 판단은 영상 기반 FCS 가 그대로 한다.
  - 포 제어는 turret_command 의 cue 토픽(/smash_fcs/cue_cmd, base 프레임 [x, y, z])으로 보낸다. turret_command 는
    cue 가 계속 들어오는 동안 FCS 의 aim_track 추종을 쓰지 않는다(선회 명령이 우선하는 기존 동작).
  - lead_s 만큼 앞서 보는 것은 숙련 사수가 표적 앞을 겨냥하는 것에 해당한다. 0 이면 표적 위치 그대로.
  - fcs_lead_follow 가 켜져 있으면 사수가 FCS 가 화면에 표시한 리드점(HUD 의 LEAD 마름모)에 조준선을 올려 둔다:
    표적 방향에 "FCS 리드점 - FCS 가 본 표적(bbox) 방향" 각도를 더한다. 격발은 조준선이 FCS 리드점에 정렬됐을 때만
    나가므로, 이렇게 해야 표적이 어느 방향으로 움직여도 격발 조건이 같다. 리드점 계산은 FCS 가 그대로 한다.
  - 표적이 격추되어 떨어지는 동안(destroyed)에는 멈추고, 재출현(respawn)하면 다시 따라간다.
"""

import json
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

from .smash_fcs.operator_aim import LeadPredictor, lead_offset, rotate_aim, shift_left, world_to_base


class OperatorFollow(Node):
    def __init__(self) -> None:
        super().__init__("operator_follow")
        self.declare_parameter("ground_truth_pose_topic", "/model/drone/pose")
        self.declare_parameter("cue_topic", "/smash_fcs/cue_cmd")
        self.declare_parameter("target_event_topic", "/smash_fcs/target_event")
        self.declare_parameter("turret_world_translation_m", [0.0, 0.0, 0.05])
        self.declare_parameter("turret_world_yaw_rad", 0.0)
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("lead_s", 0.16)
        self.declare_parameter("velocity_smoothing", 0.6)
        self.declare_parameter("lateral_offset_m", 0.21)
        self.declare_parameter("fcs_lead_follow", True)
        self.declare_parameter("aim_track_topic", "/smash_fcs/aim_track")
        self.declare_parameter("lead_smoothing", 0.5)
        self.declare_parameter("lead_timeout_s", 0.4)
        self.declare_parameter("lead_max_pan_rad", 0.04)
        self.declare_parameter("lead_max_tilt_rad", 0.03)

        self._translation = [float(v) for v in self.get_parameter("turret_world_translation_m").value]
        self._yaw = float(self.get_parameter("turret_world_yaw_rad").value)
        self._lead_s = max(0.0, float(self.get_parameter("lead_s").value))
        self._lateral = float(self.get_parameter("lateral_offset_m").value)
        self._predictor = LeadPredictor(float(self.get_parameter("velocity_smoothing").value))
        self._down = False
        self._follow_lead = bool(self.get_parameter("fcs_lead_follow").value)
        self._lead_smooth = min(0.95, max(0.0, float(self.get_parameter("lead_smoothing").value)))
        self._lead_timeout = float(self.get_parameter("lead_timeout_s").value)
        self._lead_max = (float(self.get_parameter("lead_max_pan_rad").value),
                          float(self.get_parameter("lead_max_tilt_rad").value))
        self._dlead = (0.0, 0.0)      # FCS 리드점이 표적 방향에서 벗어난 각도(평활)
        self._dlead_time = None

        self._pub = self.create_publisher(Float64MultiArray, str(self.get_parameter("cue_topic").value), 10)
        self.create_subscription(PoseStamped, str(self.get_parameter("ground_truth_pose_topic").value),
                                 self._on_pose, 10)
        self.create_subscription(String, str(self.get_parameter("target_event_topic").value),
                                 self._on_event, 10)
        if self._follow_lead:
            self.create_subscription(String, str(self.get_parameter("aim_track_topic").value),
                                     self._on_aim_track, 10)
        rate = max(1.0, float(self.get_parameter("rate_hz").value))
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f"사수 에뮬레이션 시작: 표적 실측 위치로 포 지향 ({rate:.0f} Hz, 앞서 보기 {self._lead_s:.2f} s)")

    def _on_event(self, msg: String) -> None:
        try:
            kind = json.loads(msg.data).get("event")
        except (TypeError, ValueError, AttributeError):
            return
        if kind == "destroyed":
            self._down = True
        elif kind == "respawn":
            self._down = False
            self._predictor.reset()  # 새 자리에서 속도 이력을 다시 쌓는다

    def _on_aim_track(self, msg: String) -> None:
        try:
            track = json.loads(msg.data)
        except (TypeError, ValueError):
            return
        if not isinstance(track, dict):
            return
        off = lead_offset(track.get("lead_dir_base"), track.get("bbox_dir_base"))
        if off is None:
            return
        d_pan = max(-self._lead_max[0], min(self._lead_max[0], off[0]))
        d_tilt = max(-self._lead_max[1], min(self._lead_max[1], off[1]))
        a = self._lead_smooth
        self._dlead = (a * self._dlead[0] + (1.0 - a) * d_pan, a * self._dlead[1] + (1.0 - a) * d_tilt)
        self._dlead_time = self.get_clock().now().nanoseconds * 1e-9

    def _on_pose(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        self._predictor.update([p.x, p.y, p.z], self.get_clock().now().nanoseconds * 1e-9)

    def _tick(self) -> None:
        aim = None if self._down else self._predictor.aim_point(self._lead_s)
        if aim is None:
            return
        base = world_to_base(aim, self._translation, self._yaw)
        if self._follow_lead:
            now = self.get_clock().now().nanoseconds * 1e-9
            if self._dlead_time is not None and now - self._dlead_time <= self._lead_timeout:
                base = rotate_aim(base, self._dlead[0], self._dlead[1])
            else:  # 추적이 끊겼으면 리드점 보정을 천천히 푼다
                self._dlead = (self._dlead[0] * 0.8, self._dlead[1] * 0.8)
                if abs(self._dlead[0]) > 1e-5 or abs(self._dlead[1]) > 1e-5:
                    base = rotate_aim(base, self._dlead[0], self._dlead[1])
        base = shift_left(base, self._lateral)
        if not np.all(np.isfinite(base)) or math.hypot(base[0], base[1]) < 1e-6:
            return
        msg = Float64MultiArray()
        msg.data = [float(v) for v in base]
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OperatorFollow()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
