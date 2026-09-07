"""Self-contained acceptance scenarios for the reliability state estimator.

Run with ``python -B -m jun_reliability.synthetic_validation``.  These tests
verify wiring and intended temporal behavior; they do not validate real-video
classification accuracy or calibrate experimental defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pipeline import ReliabilityFrameResult, ReliabilityPipeline
from .types import ReliabilityInput


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    name: str
    passed: bool
    transitions: tuple[str, ...]
    final_state: str


def _input(
    frame: int,
    *,
    present: bool = True,
    bbox: tuple[float, float, float, float] | None = None,
    used_yolo: bool = False,
    yolo_score: float = 0.90,
    latency_ms: float = 10.0,
) -> ReliabilityInput:
    if bbox is None and present:
        bbox = (100.0 + frame * 0.2, 100.0, 40.0, 30.0)
    return ReliabilityInput(
        timestamp=frame * 0.05,
        bbox=bbox,
        yolo_score=yolo_score if present else None,
        used_yolo=used_yolo,
        is_recovery_event=False,
        result_present=present,
        frame_width=640,
        frame_height=480,
        update_latency_ms=latency_ms,
    )


def _normal(
    pipeline: ReliabilityPipeline,
    start: int,
    count: int,
    *,
    stationary: bool = False,
    yolo_period: int = 20,
) -> list[ReliabilityFrameResult]:
    results = []
    for frame in range(start, start + count):
        bbox = (100.0, 100.0, 40.0, 30.0) if stationary else None
        results.append(
            pipeline.update(
                _input(frame, bbox=bbox, used_yolo=(frame % yolo_period == 0))
            )
        )
    return results


def _transitions(results: list[ReliabilityFrameResult]) -> tuple[str, ...]:
    return tuple(
        f"{result.transition.previous_state.value}->{result.state.value}:"
        f"{result.transition.reason}@{result.input.timestamp:.2f}s"
        for result in results
        if result.transition.changed
    )


def _result(name: str, results: list[ReliabilityFrameResult]) -> ScenarioResult:
    return ScenarioResult(name, True, _transitions(results), results[-1].state.value)


def run_all() -> tuple[ScenarioResult, ...]:
    scenarios: list[ScenarioResult] = []

    # 1. Normal tracking.
    pipeline = ReliabilityPipeline()
    assert pipeline.state.value == "LOST"
    results = _normal(pipeline, 0, 100)
    normal_pairs = [
        (result.transition.previous_state.value, result.state.value)
        for result in results
        if result.transition.changed
    ]
    assert results[-1].state.value == "STABLE"
    assert normal_pairs[:2] == [("LOST", "HOLD"), ("HOLD", "STABLE")]
    scenarios.append(_result("normal_tracking", results))

    # 2. One runtime spike cannot produce LOST.
    results = [pipeline.update(_input(100, latency_ms=300.0))]
    results += _normal(pipeline, 101, 5)
    assert all(result.state.value != "LOST" for result in results)
    scenarios.append(_result("single_runtime_spike", results))

    # 3. A real 0.5-second detector miss after explicit confirmation.
    pipeline = ReliabilityPipeline()
    _normal(pipeline, 0, 100)
    pipeline.update(_input(100, used_yolo=True))
    results = [pipeline.update(_input(frame)) for frame in range(101, 111)]
    assert all(result.state.value != "LOST" for result in results)
    scenarios.append(_result("short_detector_miss", results))

    # 4. A short occlusion/degradation reaches HOLD, then fresh recovery.
    pipeline = ReliabilityPipeline()
    _normal(pipeline, 0, 80)
    results = [
        pipeline.update(
            _input(frame, bbox=(116.0, 100.0, 40.0, 30.0), latency_ms=150.0)
        )
        for frame in range(80, 86)
    ]
    results.append(
        pipeline.update(
            _input(86, bbox=(117.5, 100.5, 41.0, 30.0), used_yolo=True)
        )
    )
    results += _normal(pipeline, 87, 35)
    assert any(result.state.value == "HOLD" for result in results)
    assert all(result.state.value != "LOST" for result in results)
    assert not any(result.stale.confirmed for result in results)
    assert results[-1].state.value == "STABLE"
    scenarios.append(_result("short_occlusion", results))

    # 5. Persistent missing preserves the V1 missing-result path to LOST.
    pipeline = ReliabilityPipeline()
    _normal(pipeline, 0, 80)
    results = [
        pipeline.update(_input(frame, present=False, bbox=None))
        for frame in range(80, 87)
    ]
    assert results[-1].state.value == "LOST"
    scenarios.append(_result("persistent_missing", results))

    # 6. Persistent stale output remains raw-present but reaches HOLD then LOST.
    pipeline = ReliabilityPipeline()
    _normal(pipeline, 0, 80)
    results = [
        pipeline.update(_input(frame, bbox=(116.0, 100.0, 40.0, 30.0)))
        for frame in range(80, 180)
    ]
    transitions = _transitions(results)
    assert any("STABLE->HOLD" in item for item in transitions)
    assert any("HOLD->LOST" in item for item in transitions)
    assert results[-1].input.result_present
    assert not results[-1].stale.effective_tracking_present
    scenarios.append(_result("persistent_stale_result", results))

    # 7. A stationary real target with periodic detector confirmation is valid.
    pipeline = ReliabilityPipeline()
    results = _normal(pipeline, 0, 140, stationary=True)
    assert results[-1].state.value == "STABLE"
    assert not any(result.stale.suspect or result.stale.confirmed for result in results)
    scenarios.append(_result("stationary_real_target", results))

    # 8. Reacquisition must traverse HOLD and revalidate before STABLE.
    pipeline = ReliabilityPipeline()
    _normal(pipeline, 0, 70)
    for frame in range(70, 77):
        pipeline.update(_input(frame, present=False, bbox=None))
    assert pipeline.state.value == "LOST"
    results = [
        pipeline.update(
            _input(frame, used_yolo=(frame in (80, 100, 120)))
        )
        for frame in range(77, 135)
    ]
    changed = [result for result in results if result.transition.changed]
    pairs = [(result.transition.previous_state.value, result.state.value) for result in changed]
    assert pairs[:2] == [("LOST", "HOLD"), ("HOLD", "STABLE")]
    assert changed[1].input.timestamp - changed[0].input.timestamp >= 0.40
    assert ("LOST", "STABLE") not in pairs
    scenarios.append(_result("lost_reacquisition", results))

    # 9. Suspect/stale samples cannot contaminate the trusted baseline.
    pipeline = ReliabilityPipeline()
    baseline_results = _normal(pipeline, 0, 80)
    baseline = baseline_results[-1].stale.trusted_quality_baseline
    history_count = baseline_results[-1].stale.trusted_history_count
    results = [
        pipeline.update(
            _input(frame, bbox=(116.0, 100.0, 40.0, 30.0), latency_ms=90.0)
        )
        for frame in range(80, 160)
    ]
    assert results[-1].stale.trusted_quality_baseline == baseline
    assert results[-1].stale.trusted_history_count == history_count
    scenarios.append(_result("trusted_history_contamination", results))

    # 10. Temporary quality degradation can HOLD, but never reaches LOST.
    pipeline = ReliabilityPipeline()
    _normal(pipeline, 0, 80)
    results = [
        pipeline.update(
            _input(frame, bbox=(300.0, 250.0, 75.0, 18.0), latency_ms=150.0)
        )
        for frame in range(80, 82)
    ]
    results += _normal(pipeline, 82, 35)
    assert all(result.state.value != "LOST" for result in results)
    assert results[-1].state.value == "STABLE"
    scenarios.append(_result("temporary_quality_degradation", results))

    # 11/A. A valid low-latency detector observation starts reacquisition.
    pipeline = ReliabilityPipeline()
    results = [pipeline.update(_input(0, used_yolo=True, latency_ms=10.0))]
    assert results[-1].state.value == "HOLD"
    assert results[-1].temporal.fresh_reacquisition_candidate
    scenarios.append(_result("valid_reacquisition_low_latency", results))

    # 12/B. Runtime latency does not invalidate an otherwise valid observation.
    pipeline = ReliabilityPipeline()
    results = [pipeline.update(_input(0, used_yolo=True, latency_ms=150.0))]
    assert not results[-1].gate.passed
    assert results[-1].evidence.runtime.latency_anomaly
    assert results[-1].temporal.fresh_observation_candidate
    assert results[-1].temporal.fresh_reacquisition_candidate
    assert "RUNTIME_ONLY_GATE_FAILURE" in results[-1].temporal.reacquisition_block_reasons
    assert results[-1].state.value == "HOLD"
    scenarios.append(_result("valid_reacquisition_high_latency", results))

    # 13/C. High latency plus invalid observation remains LOST.
    pipeline = ReliabilityPipeline()
    results = [
        pipeline.update(
            _input(0, bbox=(700.0, 500.0, 40.0, 30.0), used_yolo=True, latency_ms=150.0)
        )
    ]
    assert not results[-1].evidence.observation.sufficient
    assert not results[-1].temporal.fresh_reacquisition_candidate
    assert results[-1].state.value == "LOST"
    scenarios.append(_result("high_latency_invalid_observation", results))

    # 14/D. A high detector score cannot override invalid visibility evidence.
    pipeline = ReliabilityPipeline()
    results = [
        pipeline.update(
            _input(0, bbox=(700.0, 500.0, 40.0, 30.0), used_yolo=True, yolo_score=0.99)
        )
    ]
    assert "VISIBILITY_INVALID" in results[-1].temporal.reacquisition_block_reasons
    assert results[-1].state.value == "LOST"
    scenarios.append(_result("high_score_invalid_evidence", results))

    # 15/E. A raw tracker result without detector confirmation is insufficient.
    pipeline = ReliabilityPipeline()
    results = [pipeline.update(_input(0, used_yolo=False))]
    assert results[-1].input.result_present
    assert "NO_DETECTOR_CONFIRMATION" in results[-1].temporal.reacquisition_block_reasons
    assert results[-1].state.value == "LOST"
    scenarios.append(_result("result_without_fresh_confirmation", results))

    # 16/F. Reacquisition still passes through HOLD and temporal validation.
    pipeline = ReliabilityPipeline()
    results = [pipeline.update(_input(0, used_yolo=True, latency_ms=150.0))]
    results += _normal(pipeline, 1, 30)
    pairs = [
        (result.transition.previous_state.value, result.state.value)
        for result in results
        if result.transition.changed
    ]
    assert pairs[:2] == [("LOST", "HOLD"), ("HOLD", "STABLE")]
    assert results[-1].state.value == "STABLE"
    scenarios.append(_result("reacquisition_temporal_validation", results))

    return tuple(scenarios)


def main() -> None:
    results = run_all()
    for result in results:
        print(
            f"PASS {result.name}: final={result.final_state} "
            f"transitions={list(result.transitions)}"
        )
    print(f"PASS all {len(results)} synthetic scenarios")


if __name__ == "__main__":
    main()
