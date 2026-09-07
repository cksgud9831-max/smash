from types import SimpleNamespace

from benchmarks.simulation_orchestration import (
    GATE_PASSED,
    RAW_AIM_NOT_READY,
    RELIABILITY_HOLD,
    RELIABILITY_LOST,
    SimulationEvaluationOrchestrator,
    evaluate_simulated_readiness,
)
from jun_reliability.state_machine import TrackingReliabilityState


def test_stable_allows_raw_aim_ready():
    result = evaluate_simulated_readiness(
        raw_aim_ready=True,
        reliability_state=TrackingReliabilityState.STABLE,
        tracker_frame_present=True,
    )
    assert result.raw_aim_ready is True
    assert result.reliability_state is TrackingReliabilityState.STABLE
    assert result.simulated_ready is True
    assert result.gate_reason == GATE_PASSED


def test_hold_blocks_raw_aim_ready_without_overwriting_it():
    result = evaluate_simulated_readiness(
        raw_aim_ready=True,
        reliability_state=TrackingReliabilityState.HOLD,
        tracker_frame_present=True,
    )
    assert result.raw_aim_ready is True
    assert result.reliability_state is TrackingReliabilityState.HOLD
    assert result.simulated_ready is False
    assert result.gate_reason == RELIABILITY_HOLD


def test_lost_blocks_raw_aim_ready_without_overwriting_it():
    result = evaluate_simulated_readiness(
        raw_aim_ready=True,
        reliability_state=TrackingReliabilityState.LOST,
        tracker_frame_present=True,
    )
    assert result.raw_aim_ready is True
    assert result.reliability_state is TrackingReliabilityState.LOST
    assert result.simulated_ready is False
    assert result.gate_reason == RELIABILITY_LOST


def test_stable_still_requires_raw_aim_ready():
    result = evaluate_simulated_readiness(
        raw_aim_ready=False,
        reliability_state=TrackingReliabilityState.STABLE,
        tracker_frame_present=True,
    )
    assert result.simulated_ready is False
    assert result.gate_reason == RAW_AIM_NOT_READY


class _FakeBridgeReliability:
    def __init__(self, result):
        self.result = result

    def update(self, frame_bgr, timestamp):
        return self.result


class _FakeAimingManager:
    def __init__(self, aim_ready):
        self.aim_ready = aim_ready
        self.frames = []

    def update(self, tracker_frame):
        self.frames.append(tracker_frame)
        return SimpleNamespace(aim_ready=self.aim_ready)


def test_orchestrator_returns_raw_and_simulated_outputs_separately():
    tracker_frame = object()
    bridge_result = SimpleNamespace(
        tracker_frame=tracker_frame,
        reliability_state=TrackingReliabilityState.HOLD,
    )
    aiming = _FakeAimingManager(aim_ready=True)
    orchestrator = SimulationEvaluationOrchestrator(
        _FakeBridgeReliability(bridge_result), aiming
    )

    result = orchestrator.update(object(), 1.25)

    assert aiming.frames == [tracker_frame]
    assert result.aiming_output.aim_ready is True
    assert result.raw_aim_ready is True
    assert result.simulated_ready is False
    assert result.gate_reason == RELIABILITY_HOLD
