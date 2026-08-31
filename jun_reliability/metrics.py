"""Raw measurements derived from a window of reliability inputs.

This module deliberately performs no thresholding, scoring, gating, or state
classification.  A returned value describes the retained observations only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .buffer import ReliabilityBuffer
from .types import BBoxXYWH, ReliabilityInput


@dataclass(frozen=True, slots=True)
class ReliabilityMetrics:
    """Raw metrics for the latest retained reliability input window."""

    result_continuity: float | None
    consecutive_result_frames: int
    consecutive_missing_frames: int
    tracking_duration_sec: float | None
    center_displacement_norm: float | None
    bbox_area_change_ratio: float | None
    bbox_aspect_change_ratio: float | None
    visibility_ratio: float | None
    update_latency_ms: float | None
    yolo_score: float | None
    frames_since_last_yolo: int | None
    yolo_score_age_sec: float | None
    used_yolo: bool | None
    is_recovery_event: bool | None


class ReliabilityMetricCalculator:
    """Calculate measurements without assigning reliability or quality."""

    def calculate(self, buffer: ReliabilityBuffer) -> ReliabilityMetrics:
        history = buffer.history
        latest = buffer.latest()

        if not history or latest is None:
            return ReliabilityMetrics(
                result_continuity=None,
                consecutive_result_frames=0,
                consecutive_missing_frames=0,
                tracking_duration_sec=None,
                center_displacement_norm=None,
                bbox_area_change_ratio=None,
                bbox_aspect_change_ratio=None,
                visibility_ratio=None,
                update_latency_ms=None,
                yolo_score=None,
                frames_since_last_yolo=None,
                yolo_score_age_sec=None,
                used_yolo=None,
                is_recovery_event=None,
            )

        valid_bbox_inputs = [item for item in history if _valid_bbox(item)]
        previous_bbox_input = valid_bbox_inputs[-2] if len(valid_bbox_inputs) >= 2 else None
        current_bbox_input = valid_bbox_inputs[-1] if valid_bbox_inputs else None

        center_displacement = None
        area_change = None
        aspect_change = None
        if previous_bbox_input is not None and current_bbox_input is not None:
            center_displacement = _center_displacement_norm(previous_bbox_input, current_bbox_input)
            area_change = _bbox_area_change_ratio(previous_bbox_input.bbox, current_bbox_input.bbox)
            aspect_change = _bbox_aspect_change_ratio(previous_bbox_input.bbox, current_bbox_input.bbox)

        return ReliabilityMetrics(
            result_continuity=sum(item.result_present for item in history) / len(history),
            consecutive_result_frames=buffer.consecutive_result_frames,
            consecutive_missing_frames=buffer.consecutive_missing_frames,
            tracking_duration_sec=_tracking_duration_sec(history),
            center_displacement_norm=center_displacement,
            bbox_area_change_ratio=area_change,
            bbox_aspect_change_ratio=aspect_change,
            visibility_ratio=_visibility_ratio(latest if _valid_bbox(latest) else None),
            update_latency_ms=_finite_nonnegative(latest.update_latency_ms),
            yolo_score=_finite_or_none(latest.yolo_score),
            frames_since_last_yolo=buffer.frames_since_last_yolo,
            yolo_score_age_sec=_yolo_score_age_sec(history),
            used_yolo=latest.used_yolo,
            is_recovery_event=latest.is_recovery_event,
        )


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _finite_nonnegative(value: float) -> float | None:
    converted = _finite_or_none(value)
    if converted is None or converted < 0.0:
        return None
    return converted


def _valid_bbox(item: ReliabilityInput) -> bool:
    if not item.result_present or item.bbox is None:
        return False
    x, y, width, height = item.bbox
    return all(math.isfinite(float(value)) for value in (x, y, width, height)) and width > 0.0 and height > 0.0


def _tracking_duration_sec(history: tuple[ReliabilityInput, ...]) -> float | None:
    if not history[-1].result_present:
        return None

    start_index = len(history) - 1
    while start_index > 0 and history[start_index - 1].result_present:
        start_index -= 1

    timestamps = [float(item.timestamp) for item in history[start_index:]]
    if not all(math.isfinite(timestamp) for timestamp in timestamps):
        return None
    if any(current < previous for previous, current in zip(timestamps, timestamps[1:])):
        return None
    return timestamps[-1] - timestamps[0]


def _center_displacement_norm(previous: ReliabilityInput, current: ReliabilityInput) -> float | None:
    if previous.bbox is None or current.bbox is None:
        return None
    if current.frame_width <= 0 or current.frame_height <= 0:
        return None

    previous_x, previous_y, previous_width, previous_height = previous.bbox
    current_x, current_y, current_width, current_height = current.bbox
    previous_center = (previous_x + previous_width / 2.0, previous_y + previous_height / 2.0)
    current_center = (current_x + current_width / 2.0, current_y + current_height / 2.0)
    displacement = math.hypot(
        current_center[0] - previous_center[0],
        current_center[1] - previous_center[1],
    )
    image_diagonal = math.hypot(current.frame_width, current.frame_height)
    return displacement / image_diagonal


def _bbox_area_change_ratio(previous: BBoxXYWH | None, current: BBoxXYWH | None) -> float | None:
    if previous is None or current is None:
        return None
    previous_area = previous[2] * previous[3]
    current_area = current[2] * current[3]
    if previous_area <= 0.0 or current_area <= 0.0:
        return None
    return abs(current_area - previous_area) / previous_area


def _bbox_aspect_change_ratio(previous: BBoxXYWH | None, current: BBoxXYWH | None) -> float | None:
    if previous is None or current is None:
        return None
    previous_ratio = previous[2] / previous[3]
    current_ratio = current[2] / current[3]
    if previous_ratio <= 0.0 or current_ratio <= 0.0:
        return None
    return abs(current_ratio - previous_ratio) / previous_ratio


def _visibility_ratio(item: ReliabilityInput | None) -> float | None:
    if item is None or item.bbox is None:
        return None
    if item.frame_width <= 0 or item.frame_height <= 0:
        return None

    x, y, width, height = item.bbox
    bbox_area = width * height
    if bbox_area <= 0.0:
        return None

    intersection_width = max(0.0, min(x + width, item.frame_width) - max(x, 0.0))
    intersection_height = max(0.0, min(y + height, item.frame_height) - max(y, 0.0))
    return (intersection_width * intersection_height) / bbox_area


def _yolo_score_age_sec(history: tuple[ReliabilityInput, ...]) -> float | None:
    last_yolo_index = next((index for index in range(len(history) - 1, -1, -1) if history[index].used_yolo), None)
    if last_yolo_index is None:
        return None

    relevant_history = history[last_yolo_index:]
    timestamps = [float(item.timestamp) for item in relevant_history]
    if not all(math.isfinite(timestamp) for timestamp in timestamps):
        return None
    if any(current < previous for previous, current in zip(timestamps, timestamps[1:])):
        return None

    current_timestamp = float(history[-1].timestamp)
    yolo_timestamp = float(history[last_yolo_index].timestamp)
    return current_timestamp - yolo_timestamp
