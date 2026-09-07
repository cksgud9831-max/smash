"""Offline replay, serialization, and summary utilities.

This module consumes recorded ``ReliabilityInput`` values and calls only the
public ``ReliabilityPipeline`` API.  It does not run a tracker, import hardware
or aiming modules, calibrate thresholds, or issue system-control commands.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .pipeline import ReliabilityFrameResult, ReliabilityPipeline
from .state_machine import TrackingReliabilityState
from .types import BBoxXYWH, ReliabilityInput


_INPUT_FIELDS = frozenset(
    {
        "timestamp",
        "bbox",
        "yolo_score",
        "used_yolo",
        "is_recovery_event",
        "result_present",
        "frame_width",
        "frame_height",
        "update_latency_ms",
    }
)

_CSV_INPUT_FIELDS = (
    "timestamp",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "yolo_score",
    "used_yolo",
    "is_recovery_event",
    "result_present",
    "frame_width",
    "frame_height",
    "update_latency_ms",
)


@dataclass(frozen=True, slots=True)
class ReplaySummary:
    frame_count: int
    stable_frame_count: int
    hold_frame_count: int
    lost_frame_count: int
    state_transition_count: int
    gate_fail_count: int
    gate_reason_counts: tuple[tuple[str, int], ...]
    valid_overall_quality_count: int


@dataclass(frozen=True, slots=True)
class ReplayRun:
    results: tuple[ReliabilityFrameResult, ...]
    summary: ReplaySummary


class ReliabilityReplayRunner:
    """Replay ordered inputs through a reset pipeline."""

    def __init__(self, pipeline: ReliabilityPipeline) -> None:
        self._pipeline = pipeline

    def run(self, inputs: Iterable[ReliabilityInput]) -> ReplayRun:
        self._pipeline.reset()
        results = tuple(self._pipeline.update(item) for item in inputs)
        return ReplayRun(results=results, summary=summarize_results(results))

    def run_file(self, input_path: str | Path, output_path: str | Path) -> ReplayRun:
        inputs = load_inputs(input_path)
        replay = self.run(inputs)
        write_results(replay.results, output_path)
        return replay


def load_inputs(path: str | Path) -> tuple[ReliabilityInput, ...]:
    """Load strict JSON or CSV input without sorting or silent correction."""

    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".json":
        return _load_json_inputs(resolved)
    if suffix == ".csv":
        return _load_csv_inputs(resolved)
    raise ValueError(f"unsupported replay input format {suffix!r}; expected .csv or .json")


def write_results(results: Iterable[ReliabilityFrameResult], path: str | Path) -> None:
    """Create a new CSV or JSON result file; never overwrite an existing file."""

    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".csv":
        content = results_to_csv(results)
    elif suffix == ".json":
        content = results_to_json(results)
    else:
        raise ValueError(f"unsupported replay output format {suffix!r}; expected .csv or .json")

    with resolved.open("x", encoding="utf-8", newline="") as output_file:
        output_file.write(content)


def write_inputs(inputs: Iterable[ReliabilityInput], path: str | Path) -> None:
    """Create a replay-compatible raw-input file without overwriting."""

    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".csv":
        content = inputs_to_csv(inputs)
    elif suffix == ".json":
        content = inputs_to_json(inputs)
    else:
        raise ValueError(f"unsupported replay input format {suffix!r}; expected .csv or .json")

    with resolved.open("x", encoding="utf-8", newline="") as output_file:
        output_file.write(content)


def inputs_to_csv(inputs: Iterable[ReliabilityInput]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_CSV_INPUT_FIELDS, lineterminator="\n")
    writer.writeheader()
    for item in inputs:
        bbox = item.bbox
        writer.writerow(
            {
                "timestamp": item.timestamp,
                "bbox_x": "" if bbox is None else bbox[0],
                "bbox_y": "" if bbox is None else bbox[1],
                "bbox_width": "" if bbox is None else bbox[2],
                "bbox_height": "" if bbox is None else bbox[3],
                "yolo_score": "" if item.yolo_score is None else item.yolo_score,
                "used_yolo": str(item.used_yolo).lower(),
                "is_recovery_event": str(item.is_recovery_event).lower(),
                "result_present": str(item.result_present).lower(),
                "frame_width": item.frame_width,
                "frame_height": item.frame_height,
                "update_latency_ms": item.update_latency_ms,
            }
        )
    return output.getvalue()


def inputs_to_json(inputs: Iterable[ReliabilityInput]) -> str:
    rows = []
    for item in inputs:
        rows.append(
            {
                "timestamp": item.timestamp,
                "bbox": None if item.bbox is None else list(item.bbox),
                "yolo_score": item.yolo_score,
                "used_yolo": item.used_yolo,
                "is_recovery_event": item.is_recovery_event,
                "result_present": item.result_present,
                "frame_width": item.frame_width,
                "frame_height": item.frame_height,
                "update_latency_ms": item.update_latency_ms,
            }
        )
    return json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def results_to_csv(results: Iterable[ReliabilityFrameResult]) -> str:
    rows = [_flatten_result(index, result, json_compatible=False) for index, result in enumerate(results)]
    output = io.StringIO(newline="")
    if not rows:
        return ""
    writer = csv.DictWriter(output, fieldnames=tuple(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def results_to_json(results: Iterable[ReliabilityFrameResult]) -> str:
    rows = [_flatten_result(index, result, json_compatible=True) for index, result in enumerate(results)]
    return json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def summarize_results(results: Iterable[ReliabilityFrameResult]) -> ReplaySummary:
    retained = tuple(results)
    state_counts = {state: 0 for state in TrackingReliabilityState}
    reason_counts: dict[str, int] = {}
    transition_count = 0
    gate_fail_count = 0
    valid_quality_count = 0

    for result in retained:
        state_counts[result.state] += 1
        transition_count += int(result.transition.changed)
        gate_fail_count += int(not result.gate.passed)
        valid_quality_count += int(result.quality.overall_quality_score is not None)
        for reason in result.gate.reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return ReplaySummary(
        frame_count=len(retained),
        stable_frame_count=state_counts[TrackingReliabilityState.STABLE],
        hold_frame_count=state_counts[TrackingReliabilityState.HOLD],
        lost_frame_count=state_counts[TrackingReliabilityState.LOST],
        state_transition_count=transition_count,
        gate_fail_count=gate_fail_count,
        gate_reason_counts=tuple(sorted(reason_counts.items())),
        valid_overall_quality_count=valid_quality_count,
    )


def _load_json_inputs(path: Path) -> tuple[ReliabilityInput, ...]:
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if isinstance(payload, dict) and set(payload) == {"inputs"}:
        payload = payload["inputs"]
    if not isinstance(payload, list):
        raise ValueError("JSON replay input must be a list or an object containing only 'inputs'")
    return tuple(_input_from_mapping(row, f"JSON item {index}") for index, row in enumerate(payload))


def _load_csv_inputs(path: Path) -> tuple[ReliabilityInput, ...]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        actual_fields = tuple(reader.fieldnames or ())
        if actual_fields != _CSV_INPUT_FIELDS:
            raise ValueError(
                "CSV replay fields must be exactly "
                f"{_CSV_INPUT_FIELDS!r}; received {actual_fields!r}"
            )
        return tuple(_input_from_csv_row(row, line_number) for line_number, row in enumerate(reader, start=2))


def _input_from_mapping(value: Any, location: str) -> ReliabilityInput:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location}: expected an object")
    keys = set(value)
    if keys != _INPUT_FIELDS:
        missing = sorted(_INPUT_FIELDS - keys)
        extra = sorted(keys - _INPUT_FIELDS)
        raise ValueError(f"{location}: schema mismatch; missing={missing}, extra={extra}")

    bbox_value = value["bbox"]
    bbox: BBoxXYWH | None
    if bbox_value is None:
        bbox = None
    elif isinstance(bbox_value, (list, tuple)) and len(bbox_value) == 4:
        bbox = tuple(_required_json_number(item, f"{location}.bbox[{index}]") for index, item in enumerate(bbox_value))  # type: ignore[assignment]
    else:
        raise ValueError(f"{location}.bbox: expected null or four numeric values")

    return ReliabilityInput(
        timestamp=_required_json_number(value["timestamp"], f"{location}.timestamp"),
        bbox=bbox,
        yolo_score=_optional_number(value["yolo_score"], f"{location}.yolo_score"),
        used_yolo=_required_bool(value["used_yolo"], f"{location}.used_yolo"),
        is_recovery_event=_required_bool(value["is_recovery_event"], f"{location}.is_recovery_event"),
        result_present=_required_bool(value["result_present"], f"{location}.result_present"),
        frame_width=_required_positive_int(value["frame_width"], f"{location}.frame_width"),
        frame_height=_required_positive_int(value["frame_height"], f"{location}.frame_height"),
        update_latency_ms=_required_json_nonnegative_number(
            value["update_latency_ms"], f"{location}.update_latency_ms"
        ),
    )


def _input_from_csv_row(row: Mapping[str, str | None], line_number: int) -> ReliabilityInput:
    location = f"CSV line {line_number}"
    if None in row:
        raise ValueError(f"{location}: row has more values than the declared schema")
    bbox_parts = [row[name] for name in ("bbox_x", "bbox_y", "bbox_width", "bbox_height")]
    if all(part == "" for part in bbox_parts):
        bbox = None
    elif any(part == "" for part in bbox_parts):
        raise ValueError(f"{location}: bbox columns must be either all blank or all populated")
    else:
        bbox = tuple(
            _required_number(part, f"{location}.{name}")
            for name, part in zip(("bbox_x", "bbox_y", "bbox_width", "bbox_height"), bbox_parts)
        )

    yolo_text = row["yolo_score"]
    mapping = {
        "timestamp": _required_number(row["timestamp"], f"{location}.timestamp"),
        "bbox": bbox,
        "yolo_score": None if yolo_text == "" else _required_number(yolo_text, f"{location}.yolo_score"),
        "used_yolo": _csv_bool(row["used_yolo"], f"{location}.used_yolo"),
        "is_recovery_event": _csv_bool(row["is_recovery_event"], f"{location}.is_recovery_event"),
        "result_present": _csv_bool(row["result_present"], f"{location}.result_present"),
        "frame_width": _csv_positive_int(row["frame_width"], f"{location}.frame_width"),
        "frame_height": _csv_positive_int(row["frame_height"], f"{location}.frame_height"),
        "update_latency_ms": _required_nonnegative_number(
            row["update_latency_ms"], f"{location}.update_latency_ms"
        ),
    }
    return ReliabilityInput(**mapping)  # type: ignore[arg-type]


def _flatten_result(index: int, result: ReliabilityFrameResult, json_compatible: bool) -> dict[str, Any]:
    item = result.input
    bbox = item.bbox
    reasons: Any = list(result.gate.reasons) if json_compatible else "|".join(result.gate.reasons)
    diagnostics: Any = list(result.gate.diagnostics) if json_compatible else "|".join(result.gate.diagnostics)
    stale_reasons: Any = list(result.stale.reasons) if json_compatible else "|".join(result.stale.reasons)
    return {
        "frame_index": index,
        "timestamp": item.timestamp,
        "bbox_x": None if bbox is None else bbox[0],
        "bbox_y": None if bbox is None else bbox[1],
        "bbox_width": None if bbox is None else bbox[2],
        "bbox_height": None if bbox is None else bbox[3],
        "yolo_score": item.yolo_score,
        "used_yolo": item.used_yolo,
        "is_recovery_event": item.is_recovery_event,
        "result_present": item.result_present,
        "frame_width": item.frame_width,
        "frame_height": item.frame_height,
        "update_latency_ms": item.update_latency_ms,
        "result_continuity": result.metrics.result_continuity,
        "consecutive_result_frames": result.metrics.consecutive_result_frames,
        "consecutive_missing_frames": result.metrics.consecutive_missing_frames,
        "tracking_duration_sec": result.metrics.tracking_duration_sec,
        "center_displacement_norm": result.metrics.center_displacement_norm,
        "bbox_area_change_ratio": result.metrics.bbox_area_change_ratio,
        "bbox_aspect_change_ratio": result.metrics.bbox_aspect_change_ratio,
        "visibility_ratio": result.metrics.visibility_ratio,
        "frames_since_last_yolo": result.metrics.frames_since_last_yolo,
        "yolo_score_age_sec": result.metrics.yolo_score_age_sec,
        "continuity_score": result.quality.continuity_score,
        "duration_score": result.quality.duration_score,
        "center_stability_score": result.quality.center_stability_score,
        "bbox_stability_score": result.quality.bbox_stability_score,
        "visibility_score": result.quality.visibility_score,
        "latency_score": result.quality.latency_score,
        "yolo_confidence_score": result.quality.yolo_confidence_score,
        "yolo_freshness_score": result.quality.yolo_freshness_score,
        "overall_quality_score": result.quality.overall_quality_score,
        "available_component_count": result.quality.available_component_count,
        "used_weight_sum": result.quality.used_weight_sum,
        "gate_passed": result.gate.passed,
        "gate_reasons": reasons,
        "gate_diagnostics": diagnostics,
        "observation_score": result.evidence.observation.score,
        "observation_sufficient": result.evidence.observation.sufficient,
        "consistency_score": result.evidence.consistency.score,
        "consistency_sufficient": result.evidence.consistency.sufficient,
        "freshness_score": result.evidence.freshness.score,
        "freshness_sufficient": result.evidence.freshness.sufficient,
        "runtime_score": result.evidence.runtime.score,
        "latency_anomaly": result.evidence.runtime.latency_anomaly,
        "evidence_fusion_score": result.evidence.fusion_score,
        "stable_eligible": result.evidence.stable_eligible,
        "temporal_deviation_ratio": result.stale.temporal_deviation_ratio,
        "trusted_quality_baseline": result.stale.trusted_quality_baseline,
        "quality_flatness": result.stale.quality_flatness,
        "quality_range": result.stale.quality_range,
        "quality_std": result.stale.quality_std,
        "bbox_center_span_norm": result.stale.bbox_center_span_norm,
        "bbox_area_span_ratio": result.stale.bbox_area_span_ratio,
        "bbox_aspect_span_ratio": result.stale.bbox_aspect_span_ratio,
        "fresh_yolo_confirmation": result.stale.fresh_yolo_confirmation,
        "no_fresh_confirmation": result.stale.no_fresh_confirmation,
        "stale_suspect": result.stale.suspect,
        "stale_confirmed": result.stale.confirmed,
        "target_evidence_state": result.stale.evidence_state.value,
        "stale_score": result.stale.score,
        "stale_reasons": stale_reasons,
        "stale_suspect_frames": result.stale.suspect_frames,
        "stale_confirmed_frames": result.stale.stale_frames,
        "stale_suspect_duration_sec": result.stale.suspect_duration_sec,
        "stale_confirmed_duration_sec": result.stale.stale_duration_sec,
        "effective_tracking_present": result.stale.effective_tracking_present,
        "trusted_history_count": result.stale.trusted_history_count,
        "stable_candidate_frames": result.temporal.stable_candidate_frames,
        "unstable_frames": result.temporal.unstable_frames,
        "missing_frames": result.temporal.missing_frames,
        "recovery_hold_remaining": result.temporal.recovery_hold_remaining,
        "stable_entry_ready": result.temporal.stable_entry_ready,
        "stable_exit_ready": result.temporal.stable_exit_ready,
        "lost_ready": result.temporal.lost_ready,
        "in_hysteresis_band": result.temporal.in_hysteresis_band,
        "untrustworthy_duration_sec": result.temporal.untrustworthy_duration_sec,
        "missing_result_lost_ready": result.temporal.missing_result_lost_ready,
        "observation_lost_ready": result.temporal.observation_lost_ready,
        "fresh_observation_candidate": result.temporal.fresh_observation_candidate,
        "fresh_reacquisition_candidate": result.temporal.fresh_reacquisition_candidate,
        "runtime_healthy": not result.evidence.runtime.latency_anomaly,
        "reacquisition_block_reasons": (
            list(result.temporal.reacquisition_block_reasons)
            if json_compatible
            else "|".join(result.temporal.reacquisition_block_reasons)
        ),
        "previous_state": result.transition.previous_state.value,
        "current_state": result.transition.current_state.value,
        "state_changed": result.transition.changed,
        "transition_reason": result.transition.reason,
    }


def _required_bool(value: Any, location: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{location}: expected boolean")
    return value


def _required_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{location}: expected a finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{location}: expected a finite number") from error
    if not math.isfinite(converted):
        raise ValueError(f"{location}: expected a finite number")
    return converted


def _required_json_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location}: expected a JSON number")
    return _required_number(value, location)


def _required_nonnegative_number(value: Any, location: str) -> float:
    converted = _required_number(value, location)
    if converted < 0.0:
        raise ValueError(f"{location}: expected a non-negative number")
    return converted


def _required_json_nonnegative_number(value: Any, location: str) -> float:
    converted = _required_json_number(value, location)
    if converted < 0.0:
        raise ValueError(f"{location}: expected a non-negative JSON number")
    return converted


def _optional_number(value: Any, location: str) -> float | None:
    return None if value is None else _required_json_number(value, location)


def _required_positive_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location}: expected a positive integer")
    return value


def _csv_positive_int(value: str | None, location: str) -> int:
    if value is None:
        raise ValueError(f"{location}: expected a positive integer")
    try:
        converted = int(value)
    except ValueError as error:
        raise ValueError(f"{location}: expected a positive integer") from error
    if str(converted) != value.strip() or converted <= 0:
        raise ValueError(f"{location}: expected a positive integer")
    return converted


def _csv_bool(value: str | None, location: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{location}: expected exactly 'true' or 'false'")
