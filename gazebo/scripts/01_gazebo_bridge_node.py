"""Gazebo-side data generator for the Level 2 integration test.

Runs INSIDE the Ubuntu-24.04 WSL instance (needs the `gz` Python bindings).
Starts the aiming_test.sdf world headlessly, drives the `platform` and
`target` models over simulated time via repeated calls to the world's
`/world/<world>/set_pose` service (gz.msgs.Pose -> gz.msgs.Boolean, as
confirmed via `gz service -i -s /world/aiming_test/set_pose`), subscribes
to the platform's real simulated IMU (`/platform/imu`, gz.msgs.IMU) to
capture its orientation reading at each step, and records everything to
gazebo_result.json.

The commanded platform attitude and target trajectory reuse the same
closed-form kinematics already validated in the headless Monte Carlo
oracle simulator (fixed sinusoidal platform shake + constant-velocity
target), so the commanded motion remains exact, independently-known
ground truth -- Gazebo's IMU sensor and pose-publishing pipeline are the
thing actually being exercised here, not trusted as ground truth.

Frame count/fps are chosen to line up 1:1 by index with the tracking_result
frames produced from test/visible.mp4 in step 2 (02_e2e_simulation_runner.py),
so the combiner doesn't need to interpolate between timestamps.
"""

from __future__ import annotations

import atexit
import json
import math
import subprocess
import threading
import time

from gz.msgs.boolean_pb2 import Boolean
from gz.msgs.imu_pb2 import IMU
from gz.msgs.pose_pb2 import Pose
from gz.transport import Node

WORLD_NAME = "aiming_test"
WORLD_SDF = "/mnt/c/Users/cksgu/Desktop/Aiming/gazebo/worlds/aiming_test.sdf"
OUTPUT_PATH = "/mnt/c/Users/cksgu/Desktop/Aiming/gazebo/gazebo_result.json"

FPS = 20.0
N_FRAMES = 300  # ~15s of simulated time at real_time_factor=1.0; extend later if needed

# Fixed (non-random) platform "shake" profile -- matches the sinusoidal
# TruePlatform model already validated in the Monte Carlo simulator, with
# reproducible parameters instead of randomized per-run ones.
PLATFORM_AMP_DEG = (2.0, 2.0, 2.0)  # roll, pitch, yaw amplitude
PLATFORM_FREQ_HZ = (2.0, 2.0, 2.0)
PLATFORM_POSITION = (0.0, 0.0, 0.0)  # matches aiming_engine's launch-point-at-origin convention

# Constant-velocity target trajectory, within the 50-100m envelope already validated.
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


class ImuListener:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: IMU | None = None

    def callback(self, msg: IMU) -> None:
        with self._lock:
            self._latest = msg

    def latest(self) -> IMU | None:
        with self._lock:
            return self._latest


def start_server() -> subprocess.Popen:
    proc = subprocess.Popen(
        ["gz", "sim", "-s", "-r", "--headless-rendering", WORLD_SDF],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(proc.terminate)
    time.sleep(5.0)  # matches empirically-observed startup time
    return proc


def set_pose(node: Node, name: str, position: tuple[float, float, float], orientation_xyzw: tuple[float, float, float, float]) -> bool:
    req = Pose()
    req.name = name
    req.position.x, req.position.y, req.position.z = position
    req.orientation.x, req.orientation.y, req.orientation.z, req.orientation.w = orientation_xyzw
    success, response = node.request(f"/world/{WORLD_NAME}/set_pose", req, Pose, Boolean, 1000)
    return bool(success and response.data)


def main() -> None:
    server = start_server()

    node = Node()
    listener = ImuListener()
    node.subscribe(IMU, "/platform/imu", listener.callback)
    time.sleep(1.0)  # let the subscription establish before the drive loop starts

    rows = []
    dt = 1.0 / FPS
    for i in range(N_FRAMES):
        t = i * dt

        roll = math.radians(PLATFORM_AMP_DEG[0]) * math.sin(2 * math.pi * PLATFORM_FREQ_HZ[0] * t)
        pitch = math.radians(PLATFORM_AMP_DEG[1]) * math.sin(2 * math.pi * PLATFORM_FREQ_HZ[1] * t)
        yaw = math.radians(PLATFORM_AMP_DEG[2]) * math.sin(2 * math.pi * PLATFORM_FREQ_HZ[2] * t)
        platform_quat = euler_to_quat_xyzw(roll, pitch, yaw)

        target_pos = (
            TARGET_P0[0] + TARGET_V[0] * t,
            TARGET_P0[1] + TARGET_V[1] * t,
            TARGET_P0[2] + TARGET_V[2] * t,
        )

        set_pose(node, "platform", PLATFORM_POSITION, platform_quat)
        set_pose(node, "target", target_pos, (0.0, 0.0, 0.0, 1.0))

        time.sleep(dt)  # let the sim step forward (real_time_factor=1.0) and the IMU republish

        imu_msg = listener.latest()
        imu_quat = None
        if imu_msg is not None:
            imu_quat = (imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z, imu_msg.orientation.w)

        range_m = math.dist(target_pos, PLATFORM_POSITION)

        rows.append(
            {
                "frame": i,
                "t": t,
                "platform_true_rpy_deg": [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)],
                "platform_position": list(PLATFORM_POSITION),
                "imu_orientation_xyzw": list(imu_quat) if imu_quat else None,
                "target_position": list(target_pos),
                "range_m": range_m,
            }
        )

        if i % 47 == 0:  # deliberately not a multiple of the shake period's sample count (10 @ 20fps/2Hz), so the log doesn't alias onto zero-crossings
            print(f"frame {i}/{N_FRAMES} t={t:.2f}s platform_rpy_deg={rows[-1]['platform_true_rpy_deg']} imu_quat={imu_quat}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {len(rows)} rows to {OUTPUT_PATH}")

    server.terminate()


if __name__ == "__main__":
    main()
