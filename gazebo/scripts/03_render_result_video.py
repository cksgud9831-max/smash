"""Renders gazebo/performance.json (from 02_e2e_simulation_runner.py) as an
annotated MP4 overlaid on test/visible.mp4, so the Level 2 integration test
result can be checked by eye instead of only read as numbers.

Overlay per frame:
  - green box: OpticalFlowTracker's tracked bbox
  - yellow box: human-annotated ground-truth bbox (test/visible.json)
  - reticle at tracked bbox center: green when aim_ready, gray otherwise
  - text panel: AimState, distance_m (from AimingManager) vs Gazebo's
    ground-truth range_m, aim_error_mrad, hit_probability, IoU vs GT
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VIDEO_PATH = PROJECT_ROOT / "test" / "visible.mp4"
PERFORMANCE_PATH = PROJECT_ROOT / "gazebo" / "performance.json"
OUTPUT_PATH = PROJECT_ROOT / "gazebo" / "level2_result.mp4"

STATE_COLOR = {
    "SEARCH": (80, 80, 220),   # red-ish (BGR)
    "TRACK": (60, 200, 220),   # yellow-ish
    "AIM": (60, 220, 160),     # yellow-green
    "READY": (80, 220, 80),    # green
}


def draw_box(img, xywh, color, thickness=2):
    x, y, w, h = xywh
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)


def main() -> None:
    with open(PERFORMANCE_PATH) as f:
        data = json.load(f)
    per_frame = data["per_frame"]
    summary = data["summary"]

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (width, height))

    for row in per_frame:
        ok, frame = cap.read()
        if not ok:
            break

        if not row.get("tracked"):
            cv2.putText(frame, "NOT TRACKED", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            writer.write(frame)
            continue

        if "gt_bbox_xywh" in row:
            draw_box(frame, row["gt_bbox_xywh"], (0, 220, 220), thickness=1)  # yellow, thin
        if "bbox_xywh" in row:
            draw_box(frame, row["bbox_xywh"], (80, 220, 80), thickness=2)  # green

            x, y, w, h = row["bbox_xywh"]
            cx, cy = int(x + w / 2), int(y + h / 2)
            reticle_color = (80, 220, 80) if row.get("aim_ready") else (140, 140, 140)
            cv2.drawMarker(frame, (cx, cy), reticle_color, markerType=cv2.MARKER_CROSS, markerSize=28, thickness=2)
            cv2.circle(frame, (cx, cy), 22, reticle_color, 2)

        state = row.get("state", "?")
        state_color = STATE_COLOR.get(state, (255, 255, 255))
        panel_lines = [
            f"frame {row['frame']}  t={row['t']:.2f}s  state={state}",
            f"aim_ready={row.get('aim_ready')}  hit_prob={row.get('hit_probability', 0.0):.2f}",
            f"distance_m={row.get('distance_m', 0.0):.1f}  gazebo_range_m={row['gazebo_range_m']:.1f}"
            f"  diff={row.get('distance_vs_gazebo_range_diff_m', 0.0):+.2f}m",
            f"aim_error_mrad={row.get('aim_error_mrad', 0.0):.2f}  iou_vs_gt={row.get('iou_vs_gt')}",
        ]
        y0 = height - 20 - 26 * len(panel_lines)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, y0 - 14), (620, height - 10), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        for i, line in enumerate(panel_lines):
            cv2.putText(
                frame, line, (20, y0 + 20 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
            )
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), state_color, 6)

        writer.write(frame)

    cap.release()
    writer.release()

    print(f"wrote {OUTPUT_PATH}")
    print(
        f"summary: tracked_rate={summary['tracked_rate']:.2f} mean_iou={summary['mean_iou_vs_gt']:.3f} "
        f"aim_ready_rate={summary['aim_ready_rate_of_tracked']:.2f}"
    )


if __name__ == "__main__":
    main()
