#!/usr/bin/env python3
"""Monte Carlo sensitivity of the unchanged YOLO+LK+reliability pipeline to UAV pixels.

The source must be a clean raw Gazebo sequence with a clearly visible UAV. Each
trial composites that real rendered target onto its inpainted sky background at
a controlled long-side pixel size, then applies randomized residual motion,
brightness, blur, sensor noise and JPEG compression. The production tracker and
reliability engine are called without parameter changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np

from bridge.optical_flow_tracker import OpticalFlowTracker
from jun_reliability.engine import TrackingReliabilityEngine


def load_frames(path: str, limit: int = 240) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) < 20:
        raise RuntimeError(f"source sequence too short: {len(frames)} frames")
    return frames


def locate_target(model, frame: np.ndarray, device: str) -> tuple[int, int, int, int]:
    result = model.predict(frame, imgsz=960, conf=0.25, device=device, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError("could not locate UAV in source frame")
    i = int(result.boxes.conf.argmax().item())
    x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy().astype(int)
    return x1, y1, x2, y2


def source_asset(frame: np.ndarray, box: tuple[int, int, int, int]):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    margin = max(8, int(max(x2 - x1, y2 - y1) * 0.25))
    ax1, ay1 = max(0, x1 - margin), max(0, y1 - margin)
    ax2, ay2 = min(w, x2 + margin), min(h, y2 + margin)
    patch = frame[ay1:ay2, ax1:ax2].copy()
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    # UAV is substantially darker than the Gazebo sky. Otsu plus central connected
    # components retains rotors/body while rejecting most cloud texture.
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    keep = np.zeros_like(dark)
    cx, cy = (x1 + x2) / 2 - ax1, (y1 + y2) / 2 - ay1
    for idx in range(1, n):
        sx, sy, sw, sh, area = stats[idx]
        if area >= 2 and sx - 3 <= cx <= sx + sw + 3 and sy - 3 <= cy <= sy + sh + 3:
            keep[labels == idx] = 255
    if cv2.countNonZero(keep) < 8:
        keep = dark
    keep = cv2.dilate(keep, np.ones((3, 3), np.uint8), iterations=1)
    keep = cv2.GaussianBlur(keep, (5, 5), 0)

    remove = np.zeros((h, w), np.uint8)
    remove[ay1:ay2, ax1:ax2] = (keep > 8).astype(np.uint8) * 255
    background = cv2.inpaint(frame, remove, 7, cv2.INPAINT_TELEA)
    return patch, keep, background


def render_sequence(asset, target_px: int, rng: random.Random, frame_count: int) -> list[np.ndarray]:
    patch, mask, background = asset
    ph, pw = patch.shape[:2]
    object_points = cv2.findNonZero((mask > 32).astype(np.uint8))
    if object_points is None:
        raise RuntimeError("empty UAV segmentation mask")
    _mx, _my, object_w, object_h = cv2.boundingRect(object_points)
    scale = target_px / max(object_w, object_h, 1)
    nw, nh = max(3, int(pw * scale)), max(3, int(ph * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    obj = cv2.resize(patch, (nw, nh), interpolation=interpolation)
    alpha = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    if rng.random() < 0.5:
        obj, alpha = cv2.flip(obj, 1), cv2.flip(alpha, 1)

    h, w = background.shape[:2]
    x0 = rng.uniform(w * 0.35, w * 0.65)
    y0 = rng.uniform(h * 0.35, h * 0.65)
    speed = rng.uniform(0.15, 2.5)
    angle = rng.uniform(0, 2 * math.pi)
    vx, vy = speed * math.cos(angle), speed * math.sin(angle)
    brightness = rng.uniform(0.82, 1.18)
    noise_sigma = rng.uniform(0.0, 5.0)
    blur_sigma = rng.uniform(0.0, 1.25)
    jpeg_quality = rng.randint(65, 100)
    frames = []
    for i in range(frame_count):
        canvas = background.astype(np.float32) * brightness
        x = int(x0 + vx * i - nw / 2)
        y = int(y0 + vy * i - nh / 2)
        x = min(max(0, x), w - nw)
        y = min(max(0, y), h - nh)
        a = alpha[..., None]
        canvas[y:y + nh, x:x + nw] = obj * a + canvas[y:y + nh, x:x + nw] * (1 - a)
        if blur_sigma > 0.15:
            canvas = cv2.GaussianBlur(canvas, (0, 0), blur_sigma)
        if noise_sigma:
            noise = np.random.default_rng(rng.randrange(2**32)).normal(0, noise_sigma, canvas.shape)
            canvas += noise
        frame = np.clip(canvas, 0, 255).astype(np.uint8)
        ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if ok:
            frame = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        frames.append(frame)
    return frames


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    d = 1 + z * z / total
    center = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return center - half, center + half


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", required=True)
    ap.add_argument("--trials-per-bin", type=int, default=100)
    ap.add_argument("--bins", default="15,20,25,30,35,40,50,60,70,80")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260925)
    # 0 = 추적기 내장 기본값(이전 시험과 동일). 시뮬 노드의 detector_feature_quality_level /
    # detector_feature_block_size 와 같은 값을 주면 현재 운용 설정으로 시험한다.
    ap.add_argument("--feature-quality-level", type=float, default=0.0)
    ap.add_argument("--feature-block-size", type=int, default=0)
    args = ap.parse_args()

    bins = [int(x) for x in args.bins.split(",")]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = load_frames(args.source)
    tracker = OpticalFlowTracker(
        args.model, device=args.device, conf_thres=0.25,
        feature_quality_level=args.feature_quality_level,
        feature_block_size=args.feature_block_size,
    )
    source_box = locate_target(tracker._model, frames[len(frames) // 2], args.device)
    asset = source_asset(frames[len(frames) // 2], source_box)
    rng = random.Random(args.seed)
    rows = []
    started = time.time()

    for target_px in bins:
        for repetition in range(args.trials_per_bin):
            tracker.reset()
            reliability = TrackingReliabilityEngine(tracker)
            sequence = render_sequence(asset, target_px, rng, args.frames)
            present = stable = recovery = yolo = 0
            first_detection = first_stable = None
            latencies = []
            for index, frame in enumerate(sequence):
                result = reliability.update(frame, index / 15.0)
                obs = result.input
                present += int(obs.result_present)
                recovery += int(obs.is_recovery_event)
                yolo += int(obs.used_yolo)
                latencies.append(obs.update_latency_ms)
                if obs.result_present and first_detection is None:
                    first_detection = index
                if result.state.value == "STABLE":
                    stable += 1
                    if first_stable is None:
                        first_stable = index
            detected = first_detection is not None
            stable_reached = first_stable is not None
            # In the production integration, a converged advisory aim point is accepted
            # only while reliability is STABLE. This metric is therefore the optical
            # gate for end-to-end aim-point availability; Gazebo cross-validation remains separate.
            rows.append({
                "target_px": target_px,
                "trial": repetition,
                "detected": int(detected),
                "stable_reached": int(stable_reached),
                "aimpoint_gate_passed": int(stable_reached),
                "detection_frame": first_detection if detected else "",
                "stable_frame": first_stable if stable_reached else "",
                "track_coverage": present / args.frames,
                "stable_fraction": stable / args.frames,
                "recovery_fraction": recovery / args.frames,
                "yolo_fraction": yolo / args.frames,
                "latency_median_ms": float(np.median(latencies)),
            })
        print(f"{target_px}px: {args.trials_per_bin} trials complete", flush=True)

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for target_px in bins:
        group = [r for r in rows if r["target_px"] == target_px]
        n = len(group)
        s = sum(r["aimpoint_gate_passed"] for r in group)
        lo, hi = wilson(s, n)
        summary.append({
            "target_px": target_px,
            "trials": n,
            "detection_rate": sum(r["detected"] for r in group) / n,
            "stable_rate": sum(r["stable_reached"] for r in group) / n,
            "aimpoint_gate_rate": s / n,
            "aimpoint_gate_ci95": [lo, hi],
            "track_coverage_median": float(np.median([r["track_coverage"] for r in group])),
            "stable_fraction_median": float(np.median([r["stable_fraction"] for r in group])),
            "recovery_fraction_median": float(np.median([r["recovery_fraction"] for r in group])),
            "latency_median_ms": float(np.median([r["latency_median_ms"] for r in group])),
        })
    payload = {
        "method": "Gazebo-rendered UAV compositing; production YOLO+LK+reliability unchanged",
        "source": str(Path(args.source).resolve()),
        "model": str(Path(args.model).resolve()),
        "seed": args.seed,
        "frames_per_trial": args.frames,
        "feature_quality_level": args.feature_quality_level,
        "feature_block_size": args.feature_block_size,
        "fps": 15,
        "total_trials": len(rows),
        "elapsed_s": time.time() - started,
        "source_bbox_xyxy": [int(v) for v in source_box],
        "summary": summary,
    }
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
