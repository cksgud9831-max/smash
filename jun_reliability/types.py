"""Data transfer objects owned by the tracking reliability module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


BBoxXYWH: TypeAlias = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ReliabilityInput:
    """Raw observations from one call to the existing tracker.

    ``result_present`` means only that the tracker returned a result. It does
    not prove that the target was observed in the current frame because the
    existing hybrid tracker may retain a previous bounding box.
    """

    timestamp: float
    bbox: BBoxXYWH | None
    yolo_score: float | None
    used_yolo: bool
    is_recovery_event: bool
    result_present: bool
    frame_width: int
    frame_height: int
    update_latency_ms: float
