"""External integration of Bridge output and tracking-reliability output.

This module leaves Bridge and tracker code unchanged.  A per-frame fan-out
allows the reliability adapter to call the real tracker once and lets
``TrackerFrameBuilder`` consume the exact same public result.  Reliability
states remain diagnostic information, never operational permissions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aiming_engine.types import TrackerFrame
    from bridge.config import BridgeConfig
    from bridge.frame_builder import TrackerFrameBuilder
    from bridge.pose_source import PoseSource
    from bridge.range_sensor import RangeSensor

from .adapter import TrackerLike
from .engine import TrackingReliabilityEngine
from .pipeline import ReliabilityFrameResult, ReliabilityPipelineConfig
from .state_machine import TrackingReliabilityState


@dataclass(frozen=True, slots=True)
class BridgeReliabilityResult:
    """Bridge and reliability outputs produced from one shared tracker call."""

    timestamp: float
    tracker_frame: Optional[TrackerFrame]
    reliability_result: ReliabilityFrameResult
    reliability_state: TrackingReliabilityState
    overall_quality_score: float | None
    gate_passed: bool
    transition_reason: str


class _FanoutEndpoint:
    def __init__(self, fanout: "SingleFrameTrackerFanout", consumer: str) -> None:
        self._fanout = fanout
        self._consumer = consumer

    def update(self, frame_bgr: Any) -> Any | None:
        return self._fanout.consume(self._consumer, frame_bgr)


class SingleFrameTrackerFanout:
    """Expose two tracker-like endpoints backed by one real update per frame.

    Call ``begin(frame)`` once, consume each endpoint once with that identical
    frame object, then call ``finish()``.  ``BridgeReliabilityAdapter`` owns
    this lifecycle for normal use.
    """

    _RELIABILITY = "reliability"
    _BRIDGE = "bridge"

    def __init__(self, tracker: TrackerLike) -> None:
        self._tracker = tracker
        self._active_frame: Any | None = None
        self._result: Any | None = None
        self._tracker_called = False
        self._consumed: set[str] = set()
        self.reliability_tracker = _FanoutEndpoint(self, self._RELIABILITY)
        self.bridge_tracker = _FanoutEndpoint(self, self._BRIDGE)

    def begin(self, frame_bgr: Any) -> None:
        if self._active_frame is not None:
            raise RuntimeError("previous shared tracker frame has not been finished")
        self._active_frame = frame_bgr
        self._result = None
        self._tracker_called = False
        self._consumed.clear()

    def consume(self, consumer: str, frame_bgr: Any) -> Any | None:
        if self._active_frame is None:
            raise RuntimeError("shared tracker frame was not begun")
        if frame_bgr is not self._active_frame:
            raise ValueError("both integration paths must consume the identical frame object")
        if consumer not in (self._RELIABILITY, self._BRIDGE):
            raise ValueError(f"unknown shared tracker consumer: {consumer!r}")
        if consumer in self._consumed:
            raise RuntimeError(f"{consumer} already consumed this tracker frame")

        if not self._tracker_called:
            self._result = self._tracker.update(frame_bgr)
            self._tracker_called = True
        self._consumed.add(consumer)
        return self._result

    def finish(self) -> None:
        self._active_frame = None
        self._result = None
        self._tracker_called = False
        self._consumed.clear()

    def ensure_complete(self) -> None:
        expected = {self._RELIABILITY, self._BRIDGE}
        if not self._tracker_called or self._consumed != expected:
            raise RuntimeError(
                "shared tracker fan-out was not consumed exactly once by both integration paths"
            )


class BridgeReliabilityAdapter:
    """Produce Bridge and reliability results without a duplicate tracker call.

    The injected ``TrackerFrameBuilder`` and ``TrackingReliabilityEngine`` must
    be constructed with this adapter's fan-out endpoints.  Prefer ``create()``
    to ensure that wiring.  Reliability runs first so its existing adapter
    measures the real tracker update rather than a cached endpoint lookup.
    """

    def __init__(
        self,
        frame_builder: TrackerFrameBuilder,
        reliability_engine: TrackingReliabilityEngine,
        fanout: SingleFrameTrackerFanout,
    ) -> None:
        self._frame_builder = frame_builder
        self._reliability_engine = reliability_engine
        self._fanout = fanout

    @classmethod
    def create(
        cls,
        tracker: TrackerLike,
        bridge_config: BridgeConfig,
        range_sensor: RangeSensor,
        pose_source: PoseSource,
        pipeline_config: ReliabilityPipelineConfig | None = None,
    ) -> "BridgeReliabilityAdapter":
        """Wire existing tracker and Bridge dependencies through public APIs."""

        from bridge.frame_builder import TrackerFrameBuilder

        fanout = SingleFrameTrackerFanout(tracker)
        reliability_engine = TrackingReliabilityEngine(
            tracker=fanout.reliability_tracker,
            pipeline_config=pipeline_config,
        )
        frame_builder = TrackerFrameBuilder(
            config=bridge_config,
            range_sensor=range_sensor,
            pose_source=pose_source,
            tracker=fanout.bridge_tracker,  # type: ignore[arg-type]
        )
        return cls(frame_builder, reliability_engine, fanout)

    @property
    def reliability_state(self) -> TrackingReliabilityState:
        return self._reliability_engine.state

    def update(self, frame_bgr: Any, timestamp: float) -> BridgeReliabilityResult:
        self._fanout.begin(frame_bgr)
        try:
            reliability_result = self._reliability_engine.update(frame_bgr, timestamp)
            tracker_frame = self._frame_builder.build(frame_bgr, timestamp)
            self._fanout.ensure_complete()
        finally:
            self._fanout.finish()

        return BridgeReliabilityResult(
            timestamp=float(timestamp),
            tracker_frame=tracker_frame,
            reliability_result=reliability_result,
            reliability_state=reliability_result.state,
            overall_quality_score=reliability_result.quality.overall_quality_score,
            gate_passed=reliability_result.gate.passed,
            transition_reason=reliability_result.transition.reason,
        )

    def reset_reliability(self) -> None:
        """Reset reliability history only; tracker and Bridge remain untouched."""

        self._reliability_engine.reset()
