"""가제보 3D 타겟 드론 안정 호버링(정지 체공) 드라이버

기능:
1. 전방 35m, 상공 3.5m 위치에 타겟 드론을 안정적으로 고정 체공(Hovering)
2. 초기 영점 정렬, 탐지 신뢰도 및 정지 표적 사격 격추 판정을 완벽하게 검증
"""

from __future__ import annotations

import math
import time

from gz.msgs.boolean_pb2 import Boolean
from gz.msgs.pose_pb2 import Pose
from gz.transport import Node

WORLD_NAME = "aiming_test"
FPS = 60.0
DT = 1.0 / FPS

# 사격 플랫폼 기본 자세 (원점 고정 및 미세 안정 상태)
PLATFORM_POSITION = (0.0, 0.0, 1.2)

# 드론 고정 호버링 위치 (전방 35m 정중앙, 상공 3.5m)
HOVER_X = 35.0
HOVER_Y = 0.0
HOVER_Z = 3.5


def euler_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return x, y, z, w


def set_pose(node: Node, name: str, position, orientation_xyzw) -> bool:
    req = Pose()
    req.name = name
    req.position.x, req.position.y, req.position.z = position
    req.orientation.x, req.orientation.y, req.orientation.z, req.orientation.w = orientation_xyzw
    success, response = node.request(f"/world/{WORLD_NAME}/set_pose", req, Pose, Boolean, 1000)
    return bool(success and response.data)


def main() -> None:
    node = Node()
    print("=======================================================")
    print(f" Gazebo 드론 안정 호버링(고정 체공) 시작: ({HOVER_X}m, {HOVER_Y}m, {HOVER_Z}m)")
    print("=======================================================")

    t0 = time.time()
    next_step = t0

    while True:
        now = time.time()
        t = now - t0

        # 1. 사격 플랫폼 안정 자세
        p_quat = euler_to_quat_xyzw(0.0, 0.0, 0.0)
        set_pose(node, "platform", PLATFORM_POSITION, p_quat)

        # 2. 타겟 드론 고정 호버링 (자연스러운 미세 기류 요동 2cm만 반영)
        target_x = HOVER_X
        target_y = HOVER_Y + 0.02 * math.sin(1.2 * t)
        target_z = HOVER_Z + 0.02 * math.cos(1.0 * t)
        t_quat = euler_to_quat_xyzw(0.0, 0.0, 0.0)
        
        set_pose(node, "target", (target_x, target_y, target_z), t_quat)

        next_step += DT
        sleep_dur = next_step - time.time()
        if sleep_dur > 0:
            time.sleep(sleep_dur)
        else:
            next_step = time.time()


if __name__ == "__main__":
    main()
