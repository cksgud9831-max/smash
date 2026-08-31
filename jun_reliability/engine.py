"""Facade connecting an existing tracker to the reliability pipeline.

The resulting STABLE/HOLD/LOST value is a tracking-reliability state only.  It
is not firing, aiming, trigger, actuator, or other operational permission.
"""

from __future__ import annotations

from typing import Any

from .adapter import TrackerLike, TrackingReliabilityAdapter
from .pipeline import (
    ReliabilityFrameResult,
    ReliabilityPipeline,
    ReliabilityPipelineConfig,
)
from .state_machine import TrackingReliabilityState


class TrackingReliabilityEngine:
    """Run one existing-tracker update and one reliability-pipeline update.

    ``reset()`` resets only reliability history and state.  The existing
    ``OpticalFlowTracker`` has no public reset API, so this facade never reads
    or mutates tracker-private state.  Construct a new tracker and engine when
    a complete tracker reset is required.
    """

    def __init__(
        self,
        tracker: TrackerLike,
        pipeline_config: ReliabilityPipelineConfig | None = None,
    ) -> None:
        self._adapter = TrackingReliabilityAdapter(tracker)
        self._pipeline = ReliabilityPipeline(pipeline_config)

    @property
    def state(self) -> TrackingReliabilityState:
        return self._pipeline.state

    def update(self, frame_bgr: Any, timestamp: float) -> ReliabilityFrameResult:
        reliability_input = self._adapter.update(frame_bgr, timestamp)
        return self._pipeline.update(reliability_input)

    def reset(self) -> None:
        self._pipeline.reset()
