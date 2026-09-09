"""End-to-end bridge pipeline demo: video file/webcam -> OpticalFlowTracker ->
TrackerFrameBuilder -> AimingManager -> per-frame HUD line.

Uses MockRangeSensor/MockPoseSource by default (config/bridge.yaml) since
real laser/gimbal hardware isn't wired up yet -- this exercises the full
software path (tracker, laser-parallax correction, confidence mapping, aim
solver) end-to-end on the desktop, ready to swap in real sensor drivers
behind the same RangeSensor/PoseSource interfaces later.

Example:
    python examples/run_bridge_pipeline_demo.py --source path/to/video.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "smash_core"))

from aiming_engine import AimingManager
from bridge.config import load_bridge_config
from bridge.frame_builder import TrackerFrameBuilder
from bridge.pose_source import build_pose_source
from bridge.range_sensor import build_range_sensor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="0 for webcam, or path to a video file")
    parser.add_argument("--bridge-config", type=str, default=None, help="defaults to config/bridge.yaml")
    parser.add_argument("--aiming-config", type=str, default=None, help="defaults to config/aiming_engine.yaml")
    parser.add_argument("--fps", type=float, default=30.0, help="assumed source frame rate, for timestamps")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    bridge_config = load_bridge_config(args.bridge_config)
    range_sensor = build_range_sensor(bridge_config)
    pose_source = build_pose_source(bridge_config)
    frame_builder = TrackerFrameBuilder(bridge_config, range_sensor, pose_source)
    manager = AimingManager(config_path=args.aiming_config)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Error: could not open video source {args.source}")
        range_sensor.close()
        pose_source.close()
        return

    print(
        f"{'frame':>5} {'t(s)':>7} {'state':>9} {'range(m)':>9} "
        f"{'aim_err(mrad)':>13} {'hit_p':>6} {'aim_ready':>9}"
    )

    frame_idx = 0
    wall_start = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream.")
                break
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break

            timestamp = frame_idx / args.fps
            tracker_frame = frame_builder.build(frame, timestamp)

            if tracker_frame is None:
                print(f"{frame_idx:>5} {timestamp:>7.2f}   (no target / no range this frame)")
            else:
                output = manager.update(tracker_frame)
                print(
                    f"{frame_idx:>5} {timestamp:>7.2f} {output.state.value:>9} "
                    f"{output.debug['distance_m']:>9.1f} {output.debug['aim_error_mrad']:>13.3f} "
                    f"{output.hit_probability:>6.2f} {str(output.aim_ready):>9}"
                )

            frame_idx += 1
    finally:
        cap.release()
        range_sensor.close()
        pose_source.close()

    elapsed = time.time() - wall_start
    print(f"\nProcessed {frame_idx} frames in {elapsed:.1f}s ({frame_idx / max(elapsed, 1e-6):.1f} fps)")


if __name__ == "__main__":
    main()
