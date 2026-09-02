"""Runtime smoke test for the existing Bridge + reliability integration.

This runner exercises the real ``OpticalFlowTracker`` and
``TrackerFrameBuilder`` through ``BridgeReliabilityAdapter``.  It does not
invoke ``AimingManager`` and does not translate reliability state into any
control permission.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .bridge_adapter import BridgeReliabilityAdapter


@dataclass(frozen=True, slots=True)
class IntegrationSmokeSummary:
    processed_frames: int
    tracker_calls: int
    tracker_frame_count: int
    state_counts: dict[str, int]
    transitions: tuple[str, ...]
    average_update_latency_ms: float | None
    maximum_update_latency_ms: float | None
    stale_suspect_count: int
    stale_confirmed_count: int


class _CountingTracker:
    """Transparent public-API proxy used only to verify update call count."""

    def __init__(self, tracker: Any) -> None:
        self._tracker = tracker
        self.calls = 0

    def update(self, frame_bgr: Any) -> Any | None:
        self.calls += 1
        return self._tracker.update(frame_bgr)


def _parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=repo_root
        / "detector+tracker"
        / "ga_results"
        / "yolo11s_ga_final-3"
        / "weights"
        / "best.pt",
    )
    parser.add_argument("--config", type=Path, default=repo_root / "config" / "bridge.test_video.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--csv", type=Path)
    return parser


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {resolved}")
    return resolved


def _format_optional(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def run_smoke(
    *,
    video_path: Path,
    model_path: Path,
    config_path: Path,
    device: int | str,
    conf_thres: float,
    max_frames: int,
    csv_path: Path | None = None,
) -> IntegrationSmokeSummary:
    """Run a bounded real-tracker integration smoke test."""

    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if not math.isfinite(conf_thres) or not 0.0 <= conf_thres <= 1.0:
        raise ValueError("conf_thres must be finite and between 0 and 1")

    video = _require_file(video_path, "video")
    model = _require_file(model_path, "model")
    config_file = _require_file(config_path, "Bridge config")

    # Heavy/runtime dependencies stay lazy so --help and module syntax checks
    # still work in environments without the production CV stack.
    import cv2

    from bridge.config import load_bridge_config
    from bridge.optical_flow_tracker import OpticalFlowTracker
    from bridge.pose_source import build_pose_source
    from bridge.range_sensor import build_range_sensor

    bridge_config = load_bridge_config(config_file)
    range_sensor = build_range_sensor(bridge_config)
    pose_source = build_pose_source(bridge_config)
    tracker = _CountingTracker(
        OpticalFlowTracker(
            model_path=str(model),
            device=device,
            conf_thres=conf_thres,
        )
    )
    integration = BridgeReliabilityAdapter.create(
        tracker=tracker,
        bridge_config=bridge_config,
        range_sensor=range_sensor,
        pose_source=pose_source,
    )

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        range_sensor.close()
        pose_source.close()
        raise RuntimeError(f"OpenCV could not open video: {video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0.0:
        capture.release()
        range_sensor.close()
        pose_source.close()
        raise ValueError(f"video reports invalid FPS: {fps!r}")

    rows: list[dict[str, object]] = []
    state_counts = {"STABLE": 0, "HOLD": 0, "LOST": 0}
    transitions: list[str] = []
    latencies: list[float] = []
    previous_state = integration.reliability_state

    try:
        for frame_index in range(max_frames):
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            combined = integration.update(frame, timestamp)
            reliability = combined.reliability_result
            current_state = combined.reliability_state
            latency = reliability.input.update_latency_ms
            latencies.append(latency)
            state_counts[current_state.value] += 1

            changed = current_state is not previous_state
            if changed:
                transition = f"{previous_state.value} -> {current_state.value}"
                transitions.append(transition)
                prefix = f"Frame {frame_index:03d} | STATE CHANGE: {transition}"
            else:
                prefix = (
                    f"Frame {frame_index:03d} | Tracker={combined.tracker_frame is not None}"
                    f" | State={current_state.value}"
                )
            print(
                f"{prefix} | Result={reliability.input.result_present}"
                f" | Q={_format_optional(combined.overall_quality_score)}"
                f" | Gate={combined.gate_passed}"
                f" | Reason={combined.transition_reason}"
                f" | Latency={latency:.3f}ms"
            )
            rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "tracker_frame_present": combined.tracker_frame is not None,
                    "result_present": reliability.input.result_present,
                    "reliability_state": current_state.value,
                    "overall_quality_score": combined.overall_quality_score,
                    "gate_passed": combined.gate_passed,
                    "transition_reason": combined.transition_reason,
                    "tracker_update_latency_ms": latency,
                    "used_yolo": reliability.metrics.used_yolo,
                    "is_recovery_event": reliability.metrics.is_recovery_event,
                    "yolo_score": reliability.metrics.yolo_score,
                    "frames_since_last_yolo": reliability.metrics.frames_since_last_yolo,
                    "yolo_score_age_sec": reliability.metrics.yolo_score_age_sec,
                    "center_displacement_norm": reliability.metrics.center_displacement_norm,
                    "bbox_area_change_ratio": reliability.metrics.bbox_area_change_ratio,
                    "bbox_aspect_change_ratio": reliability.metrics.bbox_aspect_change_ratio,
                    "visibility_ratio": reliability.metrics.visibility_ratio,
                    "observation_score": reliability.evidence.observation.score,
                    "consistency_score": reliability.evidence.consistency.score,
                    "freshness_score": reliability.evidence.freshness.score,
                    "runtime_score": reliability.evidence.runtime.score,
                    "evidence_fusion_score": reliability.evidence.fusion_score,
                    "evidence_reasons": "|".join(reliability.evidence.reasons),
                    "temporal_deviation_ratio": reliability.stale.temporal_deviation_ratio,
                    "trusted_quality_baseline": reliability.stale.trusted_quality_baseline,
                    "quality_flatness": reliability.stale.quality_flatness,
                    "quality_range": reliability.stale.quality_range,
                    "quality_std": reliability.stale.quality_std,
                    "bbox_center_span_norm": reliability.stale.bbox_center_span_norm,
                    "bbox_area_span_ratio": reliability.stale.bbox_area_span_ratio,
                    "bbox_aspect_span_ratio": reliability.stale.bbox_aspect_span_ratio,
                    "fresh_yolo_confirmation": reliability.stale.fresh_yolo_confirmation,
                    "no_fresh_confirmation": reliability.stale.no_fresh_confirmation,
                    "stale_suspect": reliability.stale.suspect,
                    "stale_confirmed": reliability.stale.confirmed,
                    "target_evidence_state": reliability.stale.evidence_state.value,
                    "stale_score": reliability.stale.score,
                    "stale_reasons": "|".join(reliability.stale.reasons),
                    "stale_suspect_frames": reliability.stale.suspect_frames,
                    "stale_confirmed_frames": reliability.stale.stale_frames,
                    "stale_suspect_duration_sec": reliability.stale.suspect_duration_sec,
                    "stale_confirmed_duration_sec": reliability.stale.stale_duration_sec,
                    "consecutive_missing_frames": reliability.metrics.consecutive_missing_frames,
                    "effective_tracking_present": reliability.stale.effective_tracking_present,
                    "gate_reasons": "|".join(reliability.gate.reasons),
                    "fresh_observation_candidate": reliability.temporal.fresh_observation_candidate,
                    "fresh_reacquisition_candidate": reliability.temporal.fresh_reacquisition_candidate,
                    "observation_sufficient": reliability.evidence.observation.sufficient,
                    "consistency_sufficient": reliability.evidence.consistency.sufficient,
                    "freshness_sufficient": reliability.evidence.freshness.sufficient,
                    "observation_reasons": "|".join(reliability.evidence.observation.reasons),
                    "consistency_reasons": "|".join(reliability.evidence.consistency.reasons),
                    "freshness_reasons": "|".join(reliability.evidence.freshness.reasons),
                    "runtime_healthy": not reliability.evidence.runtime.latency_anomaly,
                    "reacquisition_block_reasons": "|".join(
                        reliability.temporal.reacquisition_block_reasons
                    ),
                }
            )
            previous_state = current_state
    finally:
        capture.release()
        range_sensor.close()
        pose_source.close()

    processed = len(rows)
    if tracker.calls != processed:
        raise RuntimeError(
            "real tracker call-count invariant failed: "
            f"processed_frames={processed}, tracker_calls={tracker.calls}"
        )

    if csv_path is not None:
        output = csv_path.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [
                "frame_index", "timestamp", "tracker_frame_present", "result_present",
                "reliability_state", "overall_quality_score", "gate_passed",
                "transition_reason", "tracker_update_latency_ms",
                "used_yolo", "is_recovery_event", "yolo_score",
                "frames_since_last_yolo", "yolo_score_age_sec",
                "center_displacement_norm", "bbox_area_change_ratio",
                "bbox_aspect_change_ratio", "visibility_ratio",
                "observation_score", "consistency_score", "freshness_score",
                "runtime_score", "evidence_fusion_score", "evidence_reasons",
                "temporal_deviation_ratio", "trusted_quality_baseline",
                "quality_flatness", "quality_range", "quality_std", "bbox_center_span_norm",
                "bbox_area_span_ratio", "bbox_aspect_span_ratio",
                "fresh_yolo_confirmation", "no_fresh_confirmation",
                "stale_suspect", "stale_confirmed", "target_evidence_state", "stale_score",
                "stale_reasons", "stale_suspect_frames",
                "stale_confirmed_frames", "stale_suspect_duration_sec",
                "stale_confirmed_duration_sec", "consecutive_missing_frames",
                "effective_tracking_present", "gate_reasons",
                "fresh_observation_candidate", "fresh_reacquisition_candidate",
                "observation_sufficient", "consistency_sufficient",
                "freshness_sufficient", "observation_reasons",
                "consistency_reasons", "freshness_reasons", "runtime_healthy",
                "reacquisition_block_reasons",
            ])
            writer.writeheader()
            writer.writerows(rows)

    return IntegrationSmokeSummary(
        processed_frames=processed,
        tracker_calls=tracker.calls,
        tracker_frame_count=sum(bool(row["tracker_frame_present"]) for row in rows),
        state_counts=state_counts,
        transitions=tuple(transitions),
        average_update_latency_ms=(sum(latencies) / len(latencies)) if latencies else None,
        maximum_update_latency_ms=max(latencies) if latencies else None,
        stale_suspect_count=sum(bool(row["stale_suspect"]) for row in rows),
        stale_confirmed_count=sum(bool(row["stale_confirmed"]) for row in rows),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run_smoke(
            video_path=args.video,
            model_path=args.model,
            config_path=args.config,
            device=args.device,
            conf_thres=args.conf_thres,
            max_frames=args.max_frames,
            csv_path=args.csv,
        )
    except Exception as exc:
        print(f"Integration smoke test failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("Integration smoke test summary")
    print(f"  processed_frames={summary.processed_frames}")
    print(f"  tracker_calls={summary.tracker_calls}")
    print(f"  tracker_frame_count={summary.tracker_frame_count}")
    print(f"  state_counts={summary.state_counts}")
    print(f"  transitions={list(summary.transitions)}")
    print(f"  average_update_latency_ms={_format_optional(summary.average_update_latency_ms)}")
    print(f"  maximum_update_latency_ms={_format_optional(summary.maximum_update_latency_ms)}")
    print(f"  stale_suspect_count={summary.stale_suspect_count}")
    print(f"  stale_confirmed_count={summary.stale_confirmed_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
