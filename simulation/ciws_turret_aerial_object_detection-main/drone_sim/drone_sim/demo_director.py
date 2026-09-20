#!/usr/bin/env python3
"""demo_director.py — 시연 연출: 명중하면 표적을 떨어뜨리고, 잠시 뒤 다시 띄운다.

흐름
    ALIVE   -- /smash_fcs/hit_result 에 HIT --> FALLING
    FALLING -- 지면 도달 ----------------------> DOWN
    DOWN    -- respawn_delay_s 경과 -----------> ALIVE (다음 호버 위치에 재출현)

표적은 kinematic 모델이라 물리로 떨어지지 않는다. 그래서 /world/<world>/set_pose
서비스(UserCommands 시스템, ros_gz_bridge 로 연결)로 자유낙하와 회전을 직접 그린다.
낙하는 약 1~2 초라 서비스 호출 부하는 짧게만 든다.

격추/재출현은 /smash_fcs/target_event(JSON 문자열)로 알린다. FCS 노드는 이를 받아
낙하 중에는 격발을 멈추고, 격추 배너와 누적 격추 수를 HUD 에 띄우며, 재출현 시
추적을 초기화해 새로 포착하게 한다.

호버 위치는 respawn_positions([x,y,z, x,y,z, ...] world 좌표)를 차례로 돈다.
비워 두면 처음 떠 있던 자리로 돌아온다. 로드맵 1단계(정지 호버링)에 맞춰
재출현 후에는 다시 정지 호버링이다.
"""

import json
import math

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from std_msgs.msg import String

G = 9.81


class DemoDirector(Node):
    def __init__(self) -> None:
        super().__init__("demo_director")
        self.declare_parameter("world", "turret_world")
        self.declare_parameter("model", "drone")
        self.declare_parameter("ground_truth_pose_topic", "/model/drone/pose")
        self.declare_parameter("respawn_delay_s", 2.0)  # sim 시각. 전체 장면 RTF 가 0.5 안팎이라 실제로는 약 4초
        self.declare_parameter("respawn_positions", [0.0])  # 비우는 대신 [0.0] 이면 "처음 자리"
        self.declare_parameter("ground_z", 0.12)

        self._world = str(self.get_parameter("world").value)
        self._model = str(self.get_parameter("model").value)
        self._respawn_delay = float(self.get_parameter("respawn_delay_s").value)
        self._ground_z = float(self.get_parameter("ground_z").value)
        flat = [float(v) for v in self.get_parameter("respawn_positions").value]
        self._positions = [tuple(flat[i:i + 3]) for i in range(0, len(flat) - 2, 3)]
        self._next_pos = 0

        self._home = None  # 처음 관측된 호버 위치
        self._pose = None  # 최근 실측 위치 (x, y, z)
        self._state = "ALIVE"
        self._t0 = 0.0
        self._fall_start = None
        self._kills = 0

        self._cli = self.create_client(SetEntityPose, f"/world/{self._world}/set_pose")
        self._event_pub = self.create_publisher(String, "/smash_fcs/target_event", 10)
        self.create_subscription(String, "/smash_fcs/hit_result", self._on_hit, 50)
        self.create_subscription(
            PoseStamped, str(self.get_parameter("ground_truth_pose_topic").value), self._on_pose, 10
        )
        self.create_timer(1.0 / 30.0, self._tick)
        self.get_logger().info("시연 연출 노드 기동: 명중 시 격추 연출 + 재출현")

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_pose(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        self._pose = (p.x, p.y, p.z)
        if self._home is None and self._state == "ALIVE":
            self._home = self._pose
            self.get_logger().info("호버 위치 기록: ({:.1f}, {:.1f}, {:.1f})".format(*self._home))

    def _on_hit(self, msg: String) -> None:
        if self._state != "ALIVE" or self._pose is None:
            return
        try:
            verdict = json.loads(msg.data).get("verdict")
        except ValueError:
            return
        if verdict != "HIT":
            return
        self._state = "FALLING"
        self._fall_start = self._pose
        self._t0 = self._now()
        self._kills += 1
        self._publish({"event": "destroyed", "kills": self._kills})
        self.get_logger().info(f"격추 #{self._kills} — 낙하 연출 시작")

    def _publish(self, payload: dict) -> None:
        m = String()
        m.data = json.dumps(payload)
        self._event_pub.publish(m)

    def _set_pose(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0) -> None:
        if not self._cli.service_is_ready():
            return
        req = SetEntityPose.Request()
        req.entity = Entity(name=self._model, type=Entity.MODEL)
        p = Pose()
        p.position.x, p.position.y, p.position.z = float(x), float(y), float(z)
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        p.orientation.w = cr * cp * cy + sr * sp * sy
        p.orientation.x = sr * cp * cy - cr * sp * sy
        p.orientation.y = cr * sp * cy + sr * cp * sy
        p.orientation.z = cr * cp * sy - sr * sp * cy
        req.pose = p
        self._cli.call_async(req)

    def _tick(self) -> None:
        t = self._now() - self._t0
        if self._state == "FALLING":
            x0, y0, z0 = self._fall_start
            z = z0 - 0.5 * G * t * t
            # 맞은 방향으로 살짝 밀리며 뒤집힌다
            x = x0 + 0.6 * t
            if z <= self._ground_z:
                z = self._ground_z
                self._state = "DOWN"
                self._t0 = self._now()
            self._set_pose(x, y0, z, roll=4.0 * t, pitch=2.5 * t)
        elif self._state == "DOWN" and t >= self._respawn_delay:
            if self._positions:
                pos = self._positions[self._next_pos % len(self._positions)]
                self._next_pos += 1
            else:
                pos = self._home or self._fall_start
            self._set_pose(*pos)
            self._state = "ALIVE"
            self._publish({"event": "respawn", "position": list(pos)})
            self.get_logger().info("재출현: ({:.1f}, {:.1f}, {:.1f})".format(*pos))


def main() -> None:
    rclpy.init()
    node = DemoDirector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
