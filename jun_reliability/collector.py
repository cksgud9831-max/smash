"""Offline raw ``ReliabilityInput`` collection from recorded video.

This tool wraps the existing public tracker API through
``TrackingReliabilityAdapter``.  It does not run the reliability decision
pipeline, calibrate thresholds, or connect to aiming or control systems.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Protocol

from .adapter import TrackingReliabilityAdapter
from .replay import write_inputs
from .types import ReliabilityInput


class CaptureLike(Protocol):
    def read(self) -> tuple[bool, Any]:
        ...


@dataclass(frozen=True, slots=True)
class CollectionSummary:
    total_frames: int
    result_present_frames: int
    missing_result_frames: int
    used_yolo_frames: int
    recovery_event_frames: int
    valid_yolo_score_frames: int
    average_update_latency_ms: float | None
    max_update_latency_ms: float | None


@dataclass(frozen=True, slots=True)
class CollectionRun:
    inputs: tuple[ReliabilityInput, ...]
    summary: CollectionSummary


class ReliabilityInputCollector:
    """Collect adapter outputs in capture order using video-relative time."""

    def __init__(self, adapter: TrackingReliabilityAdapter) -> None:
        self._adapter = adapter

    def collect_capture(
        self,
        capture: CaptureLike,
        fps: float,
        max_frames: int | None = None,
    ) -> CollectionRun:
        validated_fps = _validate_fps(fps)
        if max_frames is not None and max_frames < 0:
            raise ValueError("max_frames must be non-negative or None")

        collected: list[ReliabilityInput] = []
        frame_index = 0
        while max_frames is None or frame_index < max_frames:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            timestamp = frame_index / validated_fps
            collected.append(self._adapter.update(frame_bgr, timestamp))
            frame_index += 1

        inputs = tuple(collected)
        return CollectionRun(inputs=inputs, summary=summarize_collection(inputs))


def collect_video(
    video_path: str | Path,
    output_path: str | Path,
    model_path: str | Path,
    device: int | str = 0,
    conf_thres: float = 0.25,
    fps_override: float | None = None,
    max_frames: int | None = None,
) -> CollectionRun:
    """Run the existing tracker on a video and exclusively create a raw log."""

    import cv2

    from bridge.optical_flow_tracker import OpticalFlowTracker

    resolved_output = Path(output_path)
    if resolved_output.exists():
        raise FileExistsError(f"output already exists: {resolved_output}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"could not open video: {video_path}")

    try:
        source_fps = fps_override if fps_override is not None else capture.get(cv2.CAP_PROP_FPS)
        fps = _validate_fps(source_fps)
        tracker = OpticalFlowTracker(
            model_path=str(model_path),
            device=device,
            conf_thres=_validate_confidence_threshold(conf_thres),
        )
        adapter = TrackingReliabilityAdapter(tracker)
        collection = ReliabilityInputCollector(adapter).collect_capture(
            capture,
            fps=fps,
            max_frames=max_frames,
        )
    finally:
        capture.release()

    write_inputs(collection.inputs, resolved_output)
    return collection


def summarize_collection(inputs: Iterable[ReliabilityInput]) -> CollectionSummary:
    retained = tuple(inputs)
    latency_values = [item.update_latency_ms for item in retained if math.isfinite(item.update_latency_ms)]
    return CollectionSummary(
        total_frames=len(retained),
        result_present_frames=sum(item.result_present for item in retained),
        missing_result_frames=sum(not item.result_present for item in retained),
        used_yolo_frames=sum(item.used_yolo for item in retained),
        recovery_event_frames=sum(item.is_recovery_event for item in retained),
        valid_yolo_score_frames=sum(
            item.yolo_score is not None and math.isfinite(item.yolo_score) for item in retained
        ),
        average_update_latency_ms=(
            sum(latency_values) / len(latency_values) if latency_values else None
        ),
        max_update_latency_ms=max(latency_values) if latency_values else None,
    )


def _validate_fps(value: float) -> float:
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(
            "video FPS must be finite and greater than zero; provide --fps when metadata is invalid"
        )
    return converted


def _validate_confidence_threshold(value: float) -> float:
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError("conf_thres must be finite and within [0, 1]")
    return converted


def _parse_device(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect raw tracking ReliabilityInput logs")
    parser.add_argument("--video", required=True, help="recorded video path")
    parser.add_argument("--output", required=True, help="new .csv or .json output path")
    parser.add_argument("--model", required=True, help="YOLO model path for OpticalFlowTracker")
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0, cpu, cuda:0")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="explicit video-relative FPS override; required if video metadata is invalid",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    collection = collect_video(
        video_path=args.video,
        output_path=args.output,
        model_path=args.model,
        device=_parse_device(args.device),
        conf_thres=args.conf_thres,
        fps_override=args.fps,
        max_frames=args.max_frames,
    )
    print(json.dumps(asdict(collection.summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
