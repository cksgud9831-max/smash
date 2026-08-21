"""Live-viewing companion to 01_gazebo_bridge_node.py.

01_gazebo_bridge_node.py starts its OWN headless server, runs one fixed
300-frame pass, and exits -- built for batch data generation, nothing to
look at. This script instead assumes a GUI server is ALREADY running
(started separately, e.g. via gazebo/run_gazebo_gui.bat) and just connects
to it as a transport client, looping the same platform-shake + target-motion
profile indefinitely so a person watching the GUI window can see it move.

Not used by the data pipeline (01/02/03) -- this is purely for visual
sanity-checking that the world/set_pose/IMU wiring actually works, by eye.
Run gazebo/run_gazebo_gui.bat first, THEN this script (or
gazebo/run_live_demo_drive.bat) in a second terminal.
"""

from __future__ import annotations

import math
import time

from gz.msgs.boolean_pb2 import Boolean
from gz.msgs.pose_pb2 import Pose
from gz.transport import Node

WORLD_NAME = "aiming_test"
FPS = 20.0
LOOP_PERIOD_S = 20.0  # target sweeps out and resets every 20s so the demo runs forever

PLATFORM_AMP_DEG = (2.0, 2.0, 2.0)
PLATFORM_FREQ_HZ = (2.0, 2.0, 2.0)
PLATFORM_POSITION = (0.0, 0.0, 0.0)

TARGET_P0 = (75.0, 0.0, 3.0)
TARGET_V = (0.0, 8.0, 0.0)


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
    print("Connecting to a running GUI server (start gazebo/run_gazebo_gui.bat first if you haven't)...")
    print("Driving platform shake + target motion, looping forever. Ctrl+C to stop.")

    dt = 1.0 / FPS
    t = 0.0
    while True:
        t_local = t % LOOP_PERIOD_S

        roll = math.radians(PLATFORM_AMP_DEG[0]) * math.sin(2 * math.pi * PLATFORM_FREQ_HZ[0] * t)
        pitch = math.radians(PLATFORM_AMP_DEG[1]) * math.sin(2 * math.pi * PLATFORM_FREQ_HZ[1] * t)
        yaw = math.radians(PLATFORM_AMP_DEG[2]) * math.sin(2 * math.pi * PLATFORM_FREQ_HZ[2] * t)
        platform_quat = euler_to_quat_xyzw(roll, pitch, yaw)

        target_pos = (
            TARGET_P0[0] + TARGET_V[0] * t_local,
            TARGET_P0[1] + TARGET_V[1] * t_local,
            TARGET_P0[2] + TARGET_V[2] * t_local,
        )

        set_pose(node, "platform", PLATFORM_POSITION, platform_quat)
        set_pose(node, "target", target_pos, (0.0, 0.0, 0.0, 1.0))

        time.sleep(dt)
        t += dt


if __name__ == "__main__":
    main()
