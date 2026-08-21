"""Render the aim-assist pipeline's output visually onto a copy of the
input video, for human sanity-checking against a recorded clip (no live
camera/laser/IMU required -- mirrors run_bridge_pipeline_demo.py's Mock
sensor setup).

Draws, per frame:
    - tracked bbox (from OpticalFlowTracker)
    - ground-truth bbox, if --gt is given (Anti-UAV format: {"exist": [...],
      "gt_rect": [[x,y,w,h], ...]})
    - the raw boresight point (center_px, no lead compensation)
    - the aim point: AimSolution.lead_point_world projected back to image
      pixels -- this is what the HUD reticle would actually show the
      operator (target position compensated for time-of-flight lead), NOT
      just where the target currently is on screen
    - a HUD text block: frame/time, state, hit_probability, aim_ready,
      distance

The aim-point reticle is green when aim_ready is True, otherwise white/red
by state -- matching aiming_engine's own "green reticle = HUD may show
locked" semantics. This script only ever draws advisory overlay text; there
is no fire command anywhere in this pipeline.

Example:
    python examples/render_aim_overlay.py --source test/visible.mp4 \
        --bridge-config config/bridge.test_video.yaml \
        --gt test/visible.json --output test/visible_aim_overlay.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aiming_engine import AimingManager
from aiming_engine.config import load_config
from aiming_engine.coordinate_transform import CoordinateTransform
from aiming_engine.types import AimState, Vector3
from bridge.config import load_bridge_config
from bridge.frame_builder import TrackerFrameBuilder
from bridge.pose_source import build_pose_source
from bridge.range_sensor import build_range_sensor

_STATE_COLOR = {
    AimState.SEARCH: (128, 128, 128),  # gray
    AimState.TRACK: (0, 200, 255),  # amber
    AimState.AIM: (0, 128, 255),  # orange
    AimState.READY: (0, 255, 0),  # green
}


def project_world_point(ct: CoordinateTransform, point_world: Vector3, extrinsic: np.ndarray, intrinsics: np.ndarray):
    """World -> Scope -> Camera -> pixel. Returns None if the point is
    behind the camera (can't be drawn)."""

    p_scope = ct.world_to_scope(point_world, extrinsic)
    p_cam = ct.scope_to_camera(p_scope).to_array()
    if p_cam[2] <= 1e-6:
        return None
    pixel = intrinsics @ (p_cam / p_cam[2])
    return float(pixel[0]), float(pixel[1])


def draw_crosshair(img, center, color, size=14, thickness=2):
    x, y = int(round(center[0])), int(round(center[1]))
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)
    cv2.circle(img, (x, y), size // 2, color, thickness)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="path to a video file")
    parser.add_argument("--output", type=str, required=True, help="path to write the annotated video to")
    parser.add_argument("--gt", type=str, default=None, help="optional Anti-UAV format ground-truth JSON")
    parser.add_argument("--bridge-config", type=str, default=None, help="defaults to config/bridge.yaml")
    parser.add_argument("--aiming-config", type=str, default=None, help="defaults to config/aiming_engine.yaml")
    parser.add_argument("--fps", type=float, default=None, help="defaults to the source video's own fps")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    bridge_config = load_bridge_config(args.bridge_config)
    aiming_config = load_config(args.aiming_config)
    coordinate_transform = CoordinateTransform(aiming_config.scope)

    range_sensor = build_range_sensor(bridge_config)
    pose_source = build_pose_source(bridge_config)
    frame_builder = TrackerFrameBuilder(bridge_config, range_sensor, pose_source)
    manager = AimingManager(config=aiming_config)

    gt = None
    if args.gt:
        gt = json.load(open(args.gt, encoding="utf-8"))

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: could not open video source {args.source}")
        range_sensor.close()
        pose_source.close()
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = args.fps if args.fps is not None else src_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (width, height))

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break

            timestamp = frame_idx / fps
            vis = frame.copy()

            if gt is not None and frame_idx < len(gt["exist"]) and gt["exist"][frame_idx]:
                gx, gy, gw, gh = gt["gt_rect"][frame_idx]
                cv2.rectangle(vis, (int(gx), int(gy)), (int(gx + gw), int(gy + gh)), (200, 200, 200), 1)
                cv2.putText(vis, "GT", (int(gx), int(gy) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            tracker_frame = frame_builder.build(frame, timestamp)

            hud_lines = [f"frame {frame_idx}  t={timestamp:6.2f}s"]

            if tracker_frame is None:
                hud_lines.append("NO TARGET / NO RANGE")
            else:
                x1, y1, x2, y2 = tracker_frame.bbox
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                draw_crosshair(vis, tracker_frame.center_px, (0, 255, 255), size=8, thickness=1)

                output = manager.update(tracker_frame)
                state_color = _STATE_COLOR[output.state]

                aim_pixel = project_world_point(
                    coordinate_transform,
                    output.aim_solution.lead_point_world,
                    tracker_frame.camera_extrinsics,
                    tracker_frame.camera_intrinsics,
                )
                reticle_color = (0, 255, 0) if output.aim_ready else state_color
                if aim_pixel is not None:
                    draw_crosshair(vis, aim_pixel, reticle_color, size=18, thickness=2)

                hud_lines += [
                    f"state={output.state.value}  aim_ready={output.aim_ready}",
                    f"hit_p={output.hit_probability:.2f}  dist={output.debug['distance_m']:.1f}m",
                    f"aim_err={output.debug['aim_error_mrad']:.2f}mrad  ttof={output.aim_solution.time_of_flight:.3f}s",
                ]

                cv2.putText(
                    vis, output.state.value, (int(x1), int(y2) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2,
                )

            # Bottom-left, not top-left: this test clip already has its own
            # baked-in telemetry caption in the top corners (visible in the
            # source recording), so drawing there would overlap it.
            hud_top = height - 20 * len(hud_lines) - 10
            for i, line in enumerate(hud_lines):
                cv2.putText(vis, line, (10, hud_top + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            writer.write(vis)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
        range_sensor.close()
        pose_source.close()

    print(f"Wrote {frame_idx} frames to {args.output}")


if __name__ == "__main__":
    main()
