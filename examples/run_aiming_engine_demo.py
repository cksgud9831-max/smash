"""Synthetic real-time smoke test for the AI Smart Scope Aiming Engine.

Feeds AimingManager a sequence of synthetic TrackerFrame objects
representing a target on a straight-line approach, using the default
config/aiming_engine.yaml thresholds, and reports per-frame state/latency.

This does not exercise SmartTracker or any detector — it is a standalone
check that the Aiming Engine itself behaves sensibly and stays within a
real-time latency budget, independent of the rest of the pipeline. It is
an aim-assist demo: it never fires anything, it only prints what a HUD
would show the operator each frame.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "smash_core"))

from aiming_engine.aiming_manager import AimingManager
from aiming_engine.types import TrackerFrame

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
PIXEL = (320.0, 240.0)  # principal point -> straight down the camera boresight

# Identity: scope/camera platform sits at the world origin with its
# boresight (scope +X, per coordinate_transform.py's FLU convention)
# aligned with world +X — a level, straight-ahead engagement.
EXTRINSIC = np.eye(4)

INITIAL_RANGE_M = 250.0
CLOSING_SPEED_MPS = 15.0
FPS = 30.0
DURATION_S = 3.0


def make_frame(index: int) -> TrackerFrame:
    dt = 1.0 / FPS
    t = index * dt
    range_m = INITIAL_RANGE_M - CLOSING_SPEED_MPS * t
    return TrackerFrame(
        timestamp=t,
        bbox=(300.0, 220.0, 340.0, 260.0),
        center_px=PIXEL,
        velocity_px=(0.0, 0.0),
        tracking_confidence=0.95,
        follower_state="stable",
        camera_intrinsics=K,
        camera_extrinsics=EXTRINSIC,
        laser_range_m=range_m,
    )


def main() -> None:
    manager = AimingManager()  # loads config/aiming_engine.yaml
    num_frames = int(DURATION_S * FPS)
    latencies_ms = []

    print(f"{'frame':>5} {'t(s)':>6} {'state':>9} {'range(m)':>9} {'aim_err(mrad)':>13} "
          f"{'hit_p':>6} {'aim_ready':>9} {'iters':>5} {'latency(ms)':>11}")

    for i in range(num_frames):
        frame = make_frame(i)

        start = time.perf_counter()
        output = manager.update(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)

        aim = output.aim_solution
        print(
            f"{i:>5} {frame.timestamp:>6.2f} {output.state.value:>9} "
            f"{output.debug['distance_m']:>9.1f} {output.debug['aim_error_mrad']:>13.3f} "
            f"{output.hit_probability:>6.2f} {str(output.aim_ready):>9} {aim.solver_iterations:>5} "
            f"{elapsed_ms:>11.3f}"
        )

    print()
    print(f"per-frame latency (ms): mean={statistics.mean(latencies_ms):.3f} "
          f"median={statistics.median(latencies_ms):.3f} max={max(latencies_ms):.3f}")


if __name__ == "__main__":
    main()
