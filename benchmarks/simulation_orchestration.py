"""Simulation/evaluation-only reliability-to-readiness orchestration.

This module produces an offline decision metric.  It does not issue firing,
actuator, or hardware commands and does not modify tracker, reliability, or
aiming algorithms and thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from jun_reliability.bridge_adapter import BridgeReliabilityResult
from jun_reliability.state_machine import TrackingReliabilityState

if TYPE_CHECKING:
    from aiming_engine.types import AimAssistOutput, TrackerFrame


GATE_PASSED = "RELIABILITY_STABLE_AND_RAW_AIM_READY"
NO_TRACKER_FRAME = "NO_TRACKER_FRAME"
RELIABILITY_HOLD = "RELIABILITY_HOLD"
RELIABILITY_LOST = "RELIABILITY_LOST"
RAW_AIM_NOT_READY = "RAW_AIM_NOT_READY"


class _BridgeReliabilityLike(Protocol):
    def update(self, frame_bgr: Any, timestamp: float) -> BridgeReliabilityResult: ...


class _AimingManagerLike(Protocol):
    def update(self, frame: TrackerFrame) -> AimAssistOutput: ...


@dataclass(frozen=True, slots=True)
class SimulatedReadiness:
    """Auditable inputs and output of the evaluation-only readiness gate."""

    raw_aim_ready: bool
    reliability_state: TrackingReliabilityState
    simulated_ready: bool
    gate_reason: str


@dataclass(frozen=True, slots=True)
class SimulationEvaluationResult:
    """One combined adapter/aiming update plus its simulated decision."""

    bridge_result: BridgeReliabilityResult
    aiming_output: AimAssistOutput | None
    raw_aim_ready: bool
    reliability_state: TrackingReliabilityState
    simulated_ready: bool
    gate_reason: str


def evaluate_simulated_readiness(
    *,
    raw_aim_ready: bool,
    reliability_state: TrackingReliabilityState,
    tracker_frame_present: bool,
) -> SimulatedReadiness:
    """Combine existing outputs without changing either subsystem.

    Reliability reasons take precedence over raw aiming readiness so HOLD and
    LOST remain visible in evaluation logs even when raw aiming is also false.
    """

    if not tracker_frame_present:
        reason = NO_TRACKER_FRAME
    elif reliability_state is TrackingReliabilityState.HOLD:
        reason = RELIABILITY_HOLD
    elif reliability_state is TrackingReliabilityState.LOST:
        reason = RELIABILITY_LOST
    elif not raw_aim_ready:
        reason = RAW_AIM_NOT_READY
    else:
        reason = GATE_PASSED

    return SimulatedReadiness(
        raw_aim_ready=bool(raw_aim_ready),
        reliability_state=reliability_state,
        simulated_ready=(
            tracker_frame_present
            and raw_aim_ready
            and reliability_state is TrackingReliabilityState.STABLE
        ),
        gate_reason=reason,
    )


class SimulationEvaluationOrchestrator:
    """Run Bridge/Reliability and Aiming, then apply the simulation gate.

    Aiming still runs whenever a ``TrackerFrame`` exists, including HOLD and
    LOST, so its unmodified raw output remains available for comparison.
    """

    def __init__(
        self,
        bridge_reliability: _BridgeReliabilityLike,
        aiming_manager: _AimingManagerLike,
    ) -> None:
        self._bridge_reliability = bridge_reliability
        self._aiming_manager = aiming_manager

    def update(self, frame_bgr: Any, timestamp: float) -> SimulationEvaluationResult:
        bridge_result = self._bridge_reliability.update(frame_bgr, timestamp)
        aiming_output = (
            self._aiming_manager.update(bridge_result.tracker_frame)
            if bridge_result.tracker_frame is not None
            else None
        )
        readiness = evaluate_simulated_readiness(
            raw_aim_ready=False if aiming_output is None else aiming_output.aim_ready,
            reliability_state=bridge_result.reliability_state,
            tracker_frame_present=bridge_result.tracker_frame is not None,
        )
        return SimulationEvaluationResult(
            bridge_result=bridge_result,
            aiming_output=aiming_output,
            raw_aim_ready=readiness.raw_aim_ready,
            reliability_state=readiness.reliability_state,
            simulated_ready=readiness.simulated_ready,
            gate_reason=readiness.gate_reason,
        )
