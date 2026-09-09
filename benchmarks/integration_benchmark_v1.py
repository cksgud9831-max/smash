"""Anti-UAV integrated pipeline benchmark v1.

This is an evaluation wrapper only.  It does not alter tracker, reliability,
aiming, or decision thresholds.  Run from the repository root; use ``--help``
for the dataset and output arguments.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.simulation_orchestration import evaluate_simulated_readiness

FAILURE_IOU_THRESHOLD = 0.10  # Existing v20 evaluator definition.
PRECISION_IOU_THRESHOLD = 0.50
SUCCESS_THRESHOLDS = np.linspace(0.0, 1.0, 21)
DEFAULT_OUTPUT = ROOT / "runs" / "eval" / "integration_benchmark_v1"


@dataclass(frozen=True)
class SequenceSpec:
    name: str
    video_path: Path
    gt_path: Path


class _ResultReplayTracker:
    """Lets TrackerFrameBuilder consume a result without rerunning tracking."""

    def __init__(self) -> None:
        self.result = None

    def update(self, frame_bgr):
        return self.result


def parse_tegrastats_line(line: str) -> dict[str, float | None]:
    """Parse common Jetson tegrastats fields without assuming one JetPack layout."""
    ram = re.search(r"\bRAM\s+(\d+(?:\.\d+)?)/\d+(?:\.\d+)?MB", line)
    gpu = re.search(r"\bGR3D_FREQ\s+(\d+(?:\.\d+)?)%", line)
    power = re.search(r"\bVDD_IN\s+(\d+(?:\.\d+)?)mW", line)
    cpu_block = re.search(r"\bCPU\s+\[([^]]+)]", line)
    cpu_values = (
        [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)%@", cpu_block.group(1))]
        if cpu_block
        else []
    )
    temperatures = [float(value) for value in re.findall(r"\b\w+@(\d+(?:\.\d+)?)C", line)]
    return {
        "cpu_percent": float(np.mean(cpu_values)) if cpu_values else None,
        "gpu_percent": float(gpu.group(1)) if gpu else None,
        "ram_mb": float(ram.group(1)) if ram else None,
        "power_w": float(power.group(1)) / 1000.0 if power else None,
        "temperature_c": max(temperatures) if temperatures else None,
    }


class _TegrastatsSampler:
    """Best-effort one-second Jetson sampler; unavailable elsewhere."""

    def __init__(self, interval_ms: int = 1000) -> None:
        self.interval_ms = interval_ms
        self.samples: list[dict[str, float | None]] = []
        self.raw_lines: list[str] = []
        self._process = None
        self._thread = None

    def start(self) -> None:
        try:
            self._process = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:
            return
        self._thread = threading.Thread(target=self._read, name="benchmark-tegrastats", daemon=True)
        self._thread.start()

    def _read(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            clean = line.strip()
            if clean:
                self.raw_lines.append(clean)
                self.samples.append(parse_tegrastats_line(clean))

    def stop(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
        try:
            self._process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=3)
        if self._thread is not None:
            self._thread.join(timeout=3)


def bbox_iou(bbox_a, bbox_b) -> float:
    """XYWH IoU copied from the existing v20 evaluator."""
    if bbox_a is None or bbox_b is None:
        return 0.0
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(aw * ah + bw * bh - intersection, 1.0)
    return float(intersection / union)


def tracking_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compute the v20 tracking metrics over target-present GT frames."""
    evaluated = [float(row["iou"]) for row in rows if row["gt_present"]]
    precision = (
        float(np.mean([value >= PRECISION_IOU_THRESHOLD for value in evaluated]))
        if evaluated
        else None
    )
    success_auc = (
        float(np.mean([np.mean([value >= threshold for value in evaluated]) for threshold in SUCCESS_THRESHOLDS]))
        if evaluated
        else None
    )
    failures = sum(value < FAILURE_IOU_THRESHOLD for value in evaluated)
    return {
        "evaluated_frames": len(evaluated),
        "mean_iou": float(np.mean(evaluated)) if evaluated else None,
        "precision_iou_threshold": PRECISION_IOU_THRESHOLD,
        "tracking_precision": precision,
        "success_auc_thresholds": [float(value) for value in SUCCESS_THRESHOLDS],
        "success_auc": success_auc,
        "failure_iou_threshold": FAILURE_IOU_THRESHOLD,
        "failure_count": failures,
        "id_switch": None,
        "id_switch_reason": "N/A: single-target tracker has no identity association",
    }


def latency_summary(values: Iterable[float | None]) -> dict[str, float | None]:
    array = np.asarray([float(value) for value in values if value is not None], dtype=float)
    if not len(array):
        return {key: None for key in ("mean", "p50", "p95", "p99", "max")}
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def reliability_statistics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    counts = Counter(str(row["reliability_state"]) for row in rows)
    entries = Counter()
    exits = Counter()
    transitions: list[dict[str, Any]] = []
    for previous, current in zip(rows, rows[1:]):
        before, after = previous["reliability_state"], current["reliability_state"]
        if before != after:
            entries[after] += 1
            exits[before] += 1
            transitions.append(
                {"frame_index": current["frame_index"], "from": before, "to": after}
            )
    states = {}
    for state in ("STABLE", "HOLD", "LOST"):
        states[state] = {
            "frame_count": counts[state],
            "ratio": counts[state] / total if total else 0.0,
            "entry_count": entries[state],
            "exit_count": exits[state],
        }
    return {"states": states, "transitions": transitions}


def false_stable_statistics(rows: Sequence[dict[str, Any]], fps: float) -> dict[str, Any]:
    indices = [
        index
        for index, row in enumerate(rows)
        if not row["gt_present"] and row["reliability_state"] == "STABLE"
    ]
    runs: list[list[int]] = []
    for index in indices:
        if not runs or index != runs[-1][-1] + 1:
            runs.append([index])
        else:
            runs[-1].append(index)
    first = indices[0] if indices else None
    recovery = None
    if first is not None:
        recovery = next(
            (rows[index]["frame_index"] for index in range(first + 1, len(rows)) if rows[index]["reliability_state"] != "STABLE"),
            None,
        )
    longest = max((len(run) for run in runs), default=0)
    return {
        "applicable": any(not row["gt_present"] for row in rows),
        "frame_count": len(indices),
        "longest_duration_frames": longest,
        "longest_duration_seconds": longest / fps,
        "total_duration_seconds": len(indices) / fps,
        "first_frame": rows[first]["frame_index"] if first is not None else None,
        "state_recovery_frame": recovery,
    }


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        return result.stdout.strip() or result.stderr.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment_report(model_path: Path, config_path: Path) -> dict[str, Any]:
    git_commit = _command_output(["git", "rev-parse", "HEAD"])
    jetson_model_path = Path("/proc/device-tree/model")
    jetpack = _command_output(["dpkg-query", "-W", "nvidia-jetpack"])
    try:
        import cv2

        opencv_version = cv2.__version__
    except ImportError:
        opencv_version = None
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_branch": _command_output(["git", "branch", "--show-current"]),
        "git_commit": git_commit,
        "git_status": _command_output(["git", "status", "--short"]) or "clean",
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "jetson_model": jetson_model_path.read_text(errors="replace").strip("\x00\n") if jetson_model_path.is_file() else None,
            "jetpack": jetpack,
        },
        "python": sys.version,
        "packages": {
            "torch": _version("torch"),
            "ultralytics": _version("ultralytics"),
            "tensorrt": _version("tensorrt"),
            "opencv": opencv_version,
            "numpy": _version("numpy"),
        },
        "cuda": _command_output(["nvcc", "--version"]),
        "model_path": str(model_path.resolve()),
        "config_path": str(config_path.resolve()),
    }


def _load_gt(path: Path) -> tuple[list[Any], list[bool]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rectangles = data.get("gt_rect")
    if rectangles is None:
        raise ValueError(f"GT JSON has no gt_rect: {path}")
    raw_exist = data.get("exist")
    exists = [True] * len(rectangles) if raw_exist is None else [bool(value) for value in raw_exist]
    if len(exists) < len(rectangles):
        raise ValueError(f"GT exist array is shorter than gt_rect: {path}")
    return rectangles, exists


def _reliability_input(result, frame, timestamp: float, tracking_ms: float):
    from jun_reliability.types import ReliabilityInput

    height, width = frame.shape[:2]
    return ReliabilityInput(
        timestamp=timestamp,
        bbox=None if result is None else tuple(float(v) for v in result.bbox),
        yolo_score=None if result is None else float(result.score),
        used_yolo=False if result is None else bool(result.used_yolo),
        is_recovery_event=False if result is None else bool(result.is_recovery_event),
        result_present=result is not None,
        frame_width=int(width),
        frame_height=int(height),
        update_latency_ms=tracking_ms,
    )


def _write_csv(
    path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        columns = list(rows[0]) if rows else list(fieldnames or ())
        writer = csv.DictWriter(stream, fieldnames=columns)
        if columns:
            writer.writeheader()
        if rows:
            writer.writerows(rows)


def _resource_snapshot() -> dict[str, float | None]:
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return {
            "cpu_percent": float(process.cpu_percent(interval=None)),
            "ram_mb": float(process.memory_info().rss / (1024 * 1024)),
            "gpu_percent": None,
            "power_w": None,
            "temperature_c": None,
        }
    except (ImportError, OSError):
        return {key: None for key in ("cpu_percent", "ram_mb", "gpu_percent", "power_w", "temperature_c")}


def run_once(
    spec: SequenceSpec,
    config_path: Path,
    aiming_config_path: Path | None,
    mode: str,
    warmup_frames: int,
    max_frames: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import cv2

    from aiming_engine import AimingManager
    from bridge.config import load_bridge_config
    from bridge.final_tracker import FinalTracker
    from bridge.frame_builder import TrackerFrameBuilder
    from bridge.pose_source import build_pose_source
    from bridge.range_sensor import build_range_sensor
    from jun_reliability.pipeline import ReliabilityPipeline

    config = load_bridge_config(config_path)
    if config.tracker.backend != "final_v20":
        raise ValueError("integration benchmark v1 requires tracker.backend=final_v20")
    configured_model = Path(config.tracker.model_path)
    model_path = configured_model if configured_model.is_absolute() else ROOT / configured_model
    tracker = FinalTracker(str(model_path), device=config.tracker.device)
    reliability = ReliabilityPipeline()
    replay = _ResultReplayTracker()
    range_sensor = build_range_sensor(config)
    pose_source = build_pose_source(config)
    builder = TrackerFrameBuilder(config, range_sensor, pose_source, tracker=replay)
    manager = AimingManager(config_path=aiming_config_path)
    capture = cv2.VideoCapture(str(spec.video_path))
    if not capture.isOpened():
        tracker.close()
        builder.close()
        range_sensor.close()
        pose_source.close()
        raise RuntimeError(f"OpenCV could not open video: {spec.video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        tracker.close()
        builder.close()
        range_sensor.close()
        pose_source.close()
        raise ValueError(f"video reports invalid FPS: {fps}")
    try:
        gt_rect, gt_exists = _load_gt(spec.gt_path)
    except Exception:
        capture.release()
        tracker.close()
        builder.close()
        range_sensor.close()
        pose_source.close()
        raise
    rows: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    benchmark_start = None
    input_rate_misses = 0
    resource_samples: list[dict[str, float | None]] = []
    tegrastats = _TegrastatsSampler(interval_ms=1000)
    tegrastats.start()
    benchmark_end = None
    try:
        for source_index in range(total_video_frames):
            ok, frame = capture.read()
            if not ok:
                break
            if source_index < warmup_frames:
                tracker.update(frame)
                continue
            if max_frames is not None and len(rows) >= max_frames:
                break
            if benchmark_start is None:
                benchmark_start = time.perf_counter()
            if mode == "realtime":
                deadline = benchmark_start + len(rows) / fps
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                elif len(rows):
                    input_rate_misses += 1

            timestamp = source_index / fps
            e2e_start = time.perf_counter()
            started = time.perf_counter()
            result = tracker.update(frame)
            tracking_ms = (time.perf_counter() - started) * 1000

            started = time.perf_counter()
            reliability_result = reliability.update(_reliability_input(result, frame, timestamp, tracking_ms))
            reliability_ms = (time.perf_counter() - started) * 1000

            replay.result = result
            started = time.perf_counter()
            tracker_frame = builder.build(frame, timestamp)
            output = manager.update(tracker_frame) if tracker_frame is not None else None
            simulated_decision = evaluate_simulated_readiness(
                raw_aim_ready=False if output is None else output.aim_ready,
                reliability_state=reliability_result.state,
                tracker_frame_present=tracker_frame is not None,
            )
            aiming_decision_ms = (time.perf_counter() - started) * 1000
            e2e_ms = (time.perf_counter() - e2e_start) * 1000

            gt_present = source_index < len(gt_exists) and gt_exists[source_index]
            gt_bbox = gt_rect[source_index] if gt_present and source_index < len(gt_rect) else None
            pred_bbox = None if result is None else tuple(float(v) for v in result.bbox)
            iou = bbox_iou(pred_bbox, gt_bbox) if gt_present else None
            state = reliability_result.state.value
            rows.append(
                {
                    "frame_index": source_index,
                    "timestamp": timestamp,
                    "gt_present": gt_present,
                    "gt_bbox": json.dumps(gt_bbox) if gt_bbox is not None else "",
                    "pred_bbox": json.dumps(pred_bbox) if pred_bbox is not None else "",
                    "iou": iou,
                    "tracker_state": "TRACK" if result is not None else "SEARCH",
                    "reliability_state": simulated_decision.reliability_state.value,
                    "used_yolo": False if result is None else bool(result.used_yolo),
                    "is_recovery_event": False if result is None else bool(result.is_recovery_event),
                    "tracking_score": None if result is None else float(result.score),
                    "tracking_latency_ms": tracking_ms,
                    "reliability_latency_ms": reliability_ms,
                    "aiming_latency_ms": aiming_decision_ms,
                    "decision_latency_ms": None,
                    "e2e_latency_ms": e2e_ms,
                    "raw_aim_ready": simulated_decision.raw_aim_ready,
                    "simulated_ready": simulated_decision.simulated_ready,
                    "gate_reason": simulated_decision.gate_reason,
                    "decision": simulated_decision.simulated_ready,
                }
            )
            if len(rows) == 1 or len(rows) % max(1, round(fps)) == 0:
                snapshot = _resource_snapshot()
                if any(value is not None for value in snapshot.values()):
                    resource_samples.append(snapshot)
        benchmark_end = time.perf_counter()
    finally:
        tegrastats.stop()
        capture.release()
        tracker.close()
        builder.close()
        range_sensor.close()
        pose_source.close()

    elapsed = (benchmark_end or time.perf_counter()) - (benchmark_start or wall_start)
    reliability_stats = reliability_statistics(rows)
    report = {
        "sequence": spec.name,
        "video_path": str(spec.video_path.resolve()),
        "gt_path": str(spec.gt_path.resolve()),
        "total_frames": total_video_frames,
        "processed_frames": len(rows),
        "video_fps": fps,
        "target_present_frames": sum(bool(row["gt_present"]) for row in rows),
        "target_absent_frames": sum(not bool(row["gt_present"]) for row in rows),
        "warmup": {"enabled": warmup_frames > 0, "frames": warmup_frames},
        "benchmark_mode": mode,
        "wall_clock_seconds": elapsed,
        "overall_fps": len(rows) / elapsed if elapsed > 0 else None,
        "search_fps": _state_fps(rows, "SEARCH"),
        "track_fps": _state_fps(rows, "TRACK"),
        "realtime_deadline_misses": input_rate_misses,
        "latency_ms": {
            "tracking": latency_summary(row["tracking_latency_ms"] for row in rows),
            "reliability": latency_summary(row["reliability_latency_ms"] for row in rows),
            "aiming_and_decision": latency_summary(row["aiming_latency_ms"] for row in rows),
            "decision": latency_summary([]),
            "end_to_end": latency_summary(row["e2e_latency_ms"] for row in rows),
        },
        "comparison_A_B_C": _comparison(rows),
        "tracking_metrics": tracking_metrics(rows),
        "reliability_statistics": reliability_stats,
        "false_stable": false_stable_statistics(rows, fps),
        "decision_metrics": _decision_metrics(rows),
        "reacquisition": {
            "applicable": False,
            "reason": "TRACK->LOST->SEARCH->REACQUIRE transition not implemented in tracker",
        },
        "resource_usage": _summarize_resources(
            tegrastats.samples if tegrastats.samples else resource_samples,
            source="tegrastats" if tegrastats.samples else "psutil" if resource_samples else "unavailable",
            raw_lines=tegrastats.raw_lines,
        ),
        "limitations": [
            "Decision is internal to AimingManager.update; aiming and decision latency are combined.",
            "ID switch is N/A for a single-target tracker without identity association.",
            "GPU/power/temperature are N/A unless supplied by a platform sampler; psutil samples process CPU/RAM only.",
        ],
    }
    _sanity_check(report, rows)
    return report, rows


def _state_fps(rows: Sequence[dict[str, Any]], state: str) -> float | None:
    latencies = [row["e2e_latency_ms"] for row in rows if row["tracker_state"] == state]
    return 1000.0 * len(latencies) / sum(latencies) if latencies and sum(latencies) > 0 else None


def _comparison(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tracker = [row["tracking_latency_ms"] for row in rows]
    reliability = [row["reliability_latency_ms"] for row in rows]
    aiming = [row["aiming_latency_ms"] for row in rows]
    return {
        "A_detection_tracking": latency_summary(tracker),
        "B_plus_reliability": latency_summary(a + b for a, b in zip(tracker, reliability)),
        "C_plus_aiming_decision": latency_summary(a + b + c for a, b, c in zip(tracker, reliability, aiming)),
    }


def _decision_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    raw_count = sum(bool(row["raw_aim_ready"]) for row in rows)
    simulated_count = sum(bool(row["simulated_ready"]) for row in rows)
    return {
        "raw_aim_ready_count": raw_count,
        "raw_aim_ready_ratio": raw_count / total if total else 0.0,
        "simulated_ready_count": simulated_count,
        "simulated_ready_ratio": simulated_count / total if total else 0.0,
        "gate_reason_counts": dict(Counter(str(row["gate_reason"]) for row in rows)),
    }


def _summarize_resources(
    samples: Sequence[dict[str, float | None]], *, source: str, raw_lines: Sequence[str]
) -> dict[str, Any]:
    return {
        "sampling_interval_seconds": 1.0,
        "sample_count": len(samples),
        "source": source,
        **{key: latency_summary(sample[key] for sample in samples) for key in ("cpu_percent", "ram_mb", "gpu_percent", "power_w", "temperature_c")},
        "raw_tegrastats": list(raw_lines),
        "note": "tegrastats is preferred on Jetson; fallback is process CPU/RAM via psutil when installed.",
    }


def _sanity_check(report: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    assert report["processed_frames"] <= report["total_frames"]
    assert report["overall_fps"] is None or report["overall_fps"] > 0
    for row in rows:
        assert row["iou"] is None or 0.0 <= row["iou"] <= 1.0
        assert row["e2e_latency_ms"] >= 0
        assert row["iou"] is None if not row["gt_present"] else True
    metrics = report["tracking_metrics"]
    for key in ("mean_iou", "tracking_precision", "success_auc"):
        assert metrics[key] is None or 0.0 <= metrics[key] <= 1.0
    state_total = sum(value["frame_count"] for value in report["reliability_statistics"]["states"].values())
    assert state_total == report["processed_frames"]


def _parse_sequence(value: str) -> SequenceSpec:
    parts = value.split(",", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("sequence must be NAME,VIDEO_PATH,GT_JSON_PATH")
    return SequenceSpec(parts[0], Path(parts[1]), Path(parts[2]))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", action="append", type=_parse_sequence, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "bridge.final_v20.yaml")
    parser.add_argument("--aiming-config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("throughput", "realtime"), default="throughput")
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.repeats <= 0 or args.warmup_frames < 0 or (args.max_frames is not None and args.max_frames <= 0):
        raise SystemExit("repeats must be positive, warmup non-negative, and max-frames positive")
    if not args.config.is_file():
        raise SystemExit(f"Bridge config file not found: {args.config}")
    if args.aiming_config is not None and not args.aiming_config.is_file():
        raise SystemExit(f"Aiming config file not found: {args.aiming_config}")
    for spec in args.sequence:
        for path, label in ((spec.video_path, "video"), (spec.gt_path, "GT")):
            if not path.is_file():
                raise SystemExit(f"{label} file not found: {path}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    configured_model = Path(config["tracker"]["model_path"])
    model_path = configured_model if configured_model.is_absolute() else ROOT / configured_model
    if not model_path.is_file():
        raise SystemExit(f"Tracker model/engine file not found: {model_path}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = environment_report(model_path, args.config)
    (output / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    snapshot = {
        "bridge": config,
        "aiming": yaml.safe_load((args.aiming_config or ROOT / "config" / "aiming_engine.yaml").read_text(encoding="utf-8")),
    }
    (output / "config_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    reports = []
    comparison_rows = []
    for spec in args.sequence:
        sequence_dir = output / spec.name
        sequence_dir.mkdir(parents=True, exist_ok=True)
        run_reports = []
        for run_index in range(1, args.repeats + 1):
            report, rows = run_once(spec, args.config, args.aiming_config, args.mode, args.warmup_frames, args.max_frames)
            report.update({"git_commit": env["git_commit"], "timestamp": datetime.now(timezone.utc).isoformat(), "hardware": env["hardware"], "model_path": str(model_path), "config_path": str(args.config.resolve())})
            (sequence_dir / f"run_{run_index:02d}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            if run_index == 1:
                _write_csv(sequence_dir / "frame_metrics.csv", rows)
                _write_csv(
                    sequence_dir / "state_transitions.csv",
                    report["reliability_statistics"]["transitions"],
                    fieldnames=("frame_index", "from", "to"),
                )
            run_reports.append(report)
            reports.append(report)
            for name, latency in report["comparison_A_B_C"].items():
                comparison_rows.append({"sequence": spec.name, "run": run_index, "configuration": name, **latency})
        summary = _repeat_summary(run_reports)
        (sequence_dir / "summary.txt").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_dir = output / "summary"
    summary_dir.mkdir(exist_ok=True)
    benchmark_summary = {"git_commit": env["git_commit"], "runs": reports, "repeat_summary": _repeat_summary(reports)}
    (summary_dir / "benchmark_summary.json").write_text(json.dumps(benchmark_summary, indent=2), encoding="utf-8")
    _write_csv(summary_dir / "benchmark_summary.csv", [_flat_summary(report) for report in reports])
    _write_csv(summary_dir / "comparison_A_B_C.csv", comparison_rows)
    (summary_dir / "summary.txt").write_text(json.dumps(benchmark_summary["repeat_summary"], indent=2), encoding="utf-8")
    print(f"Benchmark results written to {output}")
    return 0


def _repeat_summary(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, values in {
        "fps": [report["overall_fps"] for report in reports],
        "mean_iou": [report["tracking_metrics"]["mean_iou"] for report in reports],
        "precision": [report["tracking_metrics"]["tracking_precision"] for report in reports],
        "success_auc": [report["tracking_metrics"]["success_auc"] for report in reports],
    }.items():
        numeric = [float(value) for value in values if value is not None]
        result[key] = {
            "min": min(numeric) if numeric else None,
            "mean": statistics.mean(numeric) if numeric else None,
            "max": max(numeric) if numeric else None,
            "std": statistics.pstdev(numeric) if numeric else None,
        }
    return result


def _flat_summary(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report["tracking_metrics"]
    return {
        "sequence": report["sequence"],
        "benchmark_mode": report["benchmark_mode"],
        "processed_frames": report["processed_frames"],
        "fps": report["overall_fps"],
        "e2e_mean_ms": report["latency_ms"]["end_to_end"]["mean"],
        "e2e_p95_ms": report["latency_ms"]["end_to_end"]["p95"],
        "mean_iou": metrics["mean_iou"],
        "precision": metrics["tracking_precision"],
        "success_auc": metrics["success_auc"],
        "failure_count": metrics["failure_count"],
        "raw_aim_ready_ratio": report["decision_metrics"]["raw_aim_ready_ratio"],
        "simulated_ready_ratio": report["decision_metrics"]["simulated_ready_ratio"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
