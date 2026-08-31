"""Public-interface adapter for the existing hybrid optical-flow tracker."""

from __future__ import annotations

import time
from typing import Any, Protocol

from .types import BBoxXYWH, ReliabilityInput


class TrackerLike(Protocol):
    """The public portion of ``OpticalFlowTracker`` used by this adapter."""

    def update(self, frame_bgr: Any) -> Any | None:
        ...


class TrackingReliabilityAdapter:
    """Convert an existing tracker's result into ``ReliabilityInput``."""

    def __init__(self, tracker: TrackerLike) -> None:
        self._tracker = tracker

    def update(self, frame_bgr: Any, timestamp: float) -> ReliabilityInput:
        """Run one tracker update and retain its raw public observations."""

        frame_height, frame_width = self._frame_dimensions(frame_bgr)

        started = time.perf_counter()
        result = self._tracker.update(frame_bgr)
        update_latency_ms = (time.perf_counter() - started) * 1_000.0

        if result is None:
            return ReliabilityInput(
                timestamp=float(timestamp),
                bbox=None,
                yolo_score=None,
                used_yolo=False,
                is_recovery_event=False,
                result_present=False,
                frame_width=frame_width,
                frame_height=frame_height,
                update_latency_ms=update_latency_ms,
            )

        bbox: BBoxXYWH = tuple(float(value) for value in result.bbox)  # type: ignore[assignment]
        if len(bbox) != 4:
            raise ValueError("tracker result bbox must contain exactly four values (x, y, width, height)")

        return ReliabilityInput(
            timestamp=float(timestamp),
            bbox=bbox,
            yolo_score=float(result.score),
            used_yolo=bool(result.used_yolo),
            is_recovery_event=bool(result.is_recovery_event),
            result_present=True,
            frame_width=frame_width,
            frame_height=frame_height,
            update_latency_ms=update_latency_ms,
        )

    @staticmethod
    def _frame_dimensions(frame_bgr: Any) -> tuple[int, int]:
        shape = getattr(frame_bgr, "shape", None)
        if shape is None or len(shape) < 2:
            raise ValueError("frame_bgr must expose an image-like shape (height, width, ...)")

        frame_height = int(shape[0])
        frame_width = int(shape[1])
        if frame_height <= 0 or frame_width <= 0:
            raise ValueError("frame dimensions must be positive")
        return frame_height, frame_width
