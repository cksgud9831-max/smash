"""Level 2 integration test: real video tracking (test/visible.mp4) + Gazebo
IMU/range (gazebo_result.json, from 01_gazebo_bridge_node.py) -> Bridge ->
AimingManager -> performance.json.

Runs on Windows (native venv, no WSL needed) -- OpticalFlowTracker only
needs ultralytics/opencv, and gazebo_result.json is already-produced JSON
sitting on the shared filesystem.

Design (matches the previously agreed 3-source split, see known_issues.md):
  - 2D tracking (bbox/confidence) comes from the real video via the same
    OpticalFlowTracker used in production (bridge/optical_flow_tracker.py).
  - Platform attitude (camera_extrinsics) comes from Gazebo's simulated IMU,
    NOT test/visible.mp4 (that video has no real IMU channel) and NOT Mock
    identity -- this is the whole point of the Level 2 test.
  - Range (laser_range_m) comes from Gazebo's ground-truth model poses,
    standing in for a real TF02-Pro reading -- fed in as the direct
    world-frame platform-to-target distance (no LaserAligner parallax
    correction: that correction models a physical mount offset between a
    real laser axis and the camera boresight, which doesn't apply to an
    idealized Euclidean distance from a simulator).

There is no 3D ground truth for the *video's* real target motion (it's a
real recording, not motion-captured), so this run cannot validate
tracking-to-world reconstruction end-to-end. What it CAN validate: that the
pipeline runs end-to-end on non-Mock IMU/range inputs, and that
AimingManager's internally re-derived distance_m stays consistent with the
Gazebo ground-truth range_m that was fed in as laser_range_m (a coherence
check on image_to_world's extrinsic/range handling, not an accuracy claim
against reality).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aiming_engine.aiming_manager import AimingManager  # noqa: E402
from aiming_engine.types import TrackerFrame  # noqa: E402
from bridge.config import load_bridge_config  # noqa: E402
from bridge.confidence import compute_follower_state, compute_tracking_confidence  # noqa: E402
from bridge.optical_flow_tracker import OpticalFlowTracker  # noqa: E402

GAZEBO_RESULT_PATH = PROJECT_ROOT / "gazebo" / "gazebo_result.json"
VIDEO_PATH = PROJECT_ROOT / "test" / "visible.mp4"
GT_PATH = PROJECT_ROOT / "test" / "visible.json"
BRIDGE_CONFIG_PATH = PROJECT_ROOT / "config" / "bridge.test_video.yaml"
AIMING_CONFIG_PATH = PROJECT_ROOT / "config" / "aiming_engine.yaml"
OUTPUT_PATH = PROJECT_ROOT / "gazebo" / "performance.json"


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    with open(GAZEBO_RESULT_PATH) as f:
        gazebo_rows = json.load(f)
    with open(GT_PATH) as f:
        gt = json.load(f)
    gt_rects = gt["gt_rect"]
    gt_exist = gt["exist"]

    n_frames = len(gazebo_rows)
    print(f"Running {n_frames} frames (matched to gazebo_result.json length)")

    bridge_config = load_bridge_config(BRIDGE_CONFIG_PATH)
    tracker = OpticalFlowTracker(
        model_path=str(PROJECT_ROOT / bridge_config.tracker.model_path),
        device=bridge_config.tracker.device,
        conf_thres=bridge_config.tracker.conf_thres,
    )
    intrinsics = bridge_config.camera.intrinsics_matrix()

    aiming_manager = AimingManager(config_path=AIMING_CONFIG_PATH)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {VIDEO_PATH}")

    per_frame = []
    for i in range(n_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            print(f"Video ended early at frame {i}")
            break

        gz = gazebo_rows[i]
        timestamp = gz["t"]

        result = tracker.update(frame_bgr)

        row = {"frame": i, "t": timestamp, "gazebo_range_m": gz["range_m"]}

        if result is None:
            row.update({"tracked": False})
            per_frame.append(row)
            continue

        x, y, w, h = result.bbox
        bbox = (float(x), float(y), float(x + w), float(y + h))
        center_px = (float(x + w / 2.0), float(y + h / 2.0))
        tracking_confidence = compute_tracking_confidence(result.score, result.is_recovery_event)
        follower_state = compute_follower_state(result.is_recovery_event)

        gt_bbox_xywh = gt_rects[i] if i < len(gt_rects) else None
        frame_iou = None
        if gt_bbox_xywh is not None and (i >= len(gt_exist) or gt_exist[i]):
            frame_iou = iou((x, y, w, h), tuple(gt_bbox_xywh))
            row["gt_bbox_xywh"] = gt_bbox_xywh
        row["bbox_xywh"] = [float(x), float(y), float(w), float(h)]

        imu_quat = gz["imu_orientation_xyzw"]
        extrinsic = np.eye(4, dtype=np.float64)
        if imu_quat is not None:
            extrinsic[:3, :3] = Rotation.from_quat(imu_quat).as_matrix()
        extrinsic[:3, 3] = gz["platform_position"]

        tracker_frame = TrackerFrame(
            timestamp=timestamp,
            bbox=bbox,
            center_px=center_px,
            velocity_px=(0.0, 0.0),
            tracking_confidence=tracking_confidence,
            follower_state=follower_state,
            camera_intrinsics=intrinsics,
            camera_extrinsics=extrinsic,
            laser_range_m=float(gz["range_m"]),
        )

        output = aiming_manager.update(tracker_frame)

        row.update(
            {
                "tracked": True,
                "used_yolo": result.used_yolo,
                "is_recovery_event": result.is_recovery_event,
                "tracking_confidence": tracking_confidence,
                "iou_vs_gt": frame_iou,
                "state": output.state.value,
                "aim_ready": output.aim_ready,
                "hit_probability": output.hit_probability,
                "solver_converged": output.aim_solution.solver_converged if output.aim_solution else None,
                "aim_error_mrad": output.debug["aim_error_mrad"],
                "distance_m": output.debug["distance_m"],
                "distance_vs_gazebo_range_diff_m": output.debug["distance_m"] - gz["range_m"],
            }
        )
        per_frame.append(row)

        if i % 47 == 0:
            print(
                f"frame {i}/{n_frames} state={output.state.value} "
                f"aim_ready={output.aim_ready} dist={output.debug['distance_m']:.1f}m "
                f"gazebo_range={gz['range_m']:.1f}m iou={frame_iou}"
            )

    cap.release()

    tracked_rows = [r for r in per_frame if r["tracked"]]
    iou_values = [r["iou_vs_gt"] for r in tracked_rows if r["iou_vs_gt"] is not None]
    aim_ready_rows = [r for r in tracked_rows if r["aim_ready"]]
    converged_rows = [r for r in tracked_rows if r.get("solver_converged")]
    dist_diffs = [abs(r["distance_vs_gazebo_range_diff_m"]) for r in tracked_rows if r.get("distance_m") is not None]

    summary = {
        "n_frames_requested": n_frames,
        "n_frames_processed": len(per_frame),
        "n_frames_tracked": len(tracked_rows),
        "tracked_rate": len(tracked_rows) / len(per_frame) if per_frame else 0.0,
        "mean_iou_vs_gt": float(np.mean(iou_values)) if iou_values else None,
        "median_iou_vs_gt": float(np.median(iou_values)) if iou_values else None,
        "n_recovery_events": sum(1 for r in tracked_rows if r.get("is_recovery_event")),
        "aim_ready_rate_of_tracked": len(aim_ready_rows) / len(tracked_rows) if tracked_rows else 0.0,
        "solver_converged_rate_of_tracked": len(converged_rows) / len(tracked_rows) if tracked_rows else 0.0,
        "mean_abs_distance_vs_gazebo_range_diff_m": float(np.mean(dist_diffs)) if dist_diffs else None,
        "max_abs_distance_vs_gazebo_range_diff_m": float(np.max(dist_diffs)) if dist_diffs else None,
        "note": (
            "distance_vs_gazebo_range_diff is a self-consistency coherence check "
            "(AimingManager's internally re-derived distance_m vs. the Gazebo "
            "ground-truth range_m fed in as laser_range_m), not an accuracy claim "
            "against reality -- the video's actual 3D target motion has no ground "
            "truth. iou_vs_gt is the only metric checked against independent "
            "(human-annotated) ground truth."
        ),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump({"summary": summary, "per_frame": per_frame}, f, indent=2)

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nwrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
