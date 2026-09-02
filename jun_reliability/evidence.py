"""Interpretable evidence axes for tracking-reliability state estimation.

Scores are diagnostics, not operational permissions or standalone state
decisions. All defaults are initial experimental defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .hard_gate import GateResult
from .metrics import ReliabilityMetrics
from .quality import ReliabilityQuality


@dataclass(frozen=True, slots=True)
class ObservationConfig:
    minimum_score: float = 0.55
    minimum_visibility: float = 0.50
    minimum_yolo_confidence: float = 0.25
    maximum_yolo_age_sec: float = 2.0


@dataclass(frozen=True, slots=True)
class ConsistencyConfig:
    minimum_score: float = 0.60


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    latency_anomaly_ms: float = 100.0


@dataclass(frozen=True, slots=True)
class EvidenceConfig:
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    consistency: ConsistencyConfig = field(default_factory=ConsistencyConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def __post_init__(self) -> None:
        bounded = {
            "observation.minimum_score": self.observation.minimum_score,
            "observation.minimum_visibility": self.observation.minimum_visibility,
            "observation.minimum_yolo_confidence": self.observation.minimum_yolo_confidence,
            "consistency.minimum_score": self.consistency.minimum_score,
        }
        for name, value in bounded.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1]")
        for name, value in {
            "observation.maximum_yolo_age_sec": self.observation.maximum_yolo_age_sec,
            "runtime.latency_anomaly_ms": self.runtime.latency_anomaly_ms,
        }.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ObservationEvidence:
    result_available: bool
    detector_used: bool
    detector_score_available: bool
    detector_age_valid: bool
    visibility_valid: bool
    recovery_active: bool
    score: float | None
    sufficient: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConsistencyEvidence:
    score: float | None
    sufficient: bool
    center_score: float | None
    bbox_score: float | None
    continuity_score: float | None
    visibility_score: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FreshnessEvidence:
    score: float | None
    sufficient: bool
    detector_confirmation_candidate: bool
    yolo_score: float | None
    age_sec: float | None
    frames_since_yolo: int | None
    reasons: tuple[str, ...]
    confirmation_block_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RuntimeEvidence:
    score: float | None
    latency_ms: float | None
    latency_anomaly: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReliabilityEvidence:
    observation: ObservationEvidence
    consistency: ConsistencyEvidence
    freshness: FreshnessEvidence
    runtime: RuntimeEvidence
    fusion_score: float | None
    stable_eligible: bool
    reasons: tuple[str, ...]


class ReliabilityEvidenceCalculator:
    def __init__(self, config: EvidenceConfig | None = None) -> None:
        self._config = config or EvidenceConfig()

    def calculate(
        self,
        metrics: ReliabilityMetrics,
        quality: ReliabilityQuality,
        gate: GateResult,
    ) -> ReliabilityEvidence:
        observation = self._observation(metrics, quality)
        consistency = self._consistency(quality)
        freshness = self._freshness(metrics, quality, observation, consistency)
        runtime = self._runtime(metrics, quality)
        fusion = _mean_available(
            (observation.score, consistency.score, freshness.score, runtime.score)
        )
        stable_eligible = (
            gate.passed
            and observation.sufficient
            and consistency.sufficient
            and freshness.sufficient
            and not observation.recovery_active
        )
        reasons = tuple(
            reason
            for axis in (observation, consistency, freshness, runtime)
            for reason in axis.reasons
        )
        return ReliabilityEvidence(
            observation=observation,
            consistency=consistency,
            freshness=freshness,
            runtime=runtime,
            fusion_score=fusion,
            stable_eligible=stable_eligible,
            reasons=reasons,
        )

    def _observation(self, metrics: ReliabilityMetrics, quality: ReliabilityQuality) -> ObservationEvidence:
        config = self._config.observation
        result_available = metrics.consecutive_result_frames > 0
        detector_score_available = _bounded(metrics.yolo_score) is not None
        age = _nonnegative(metrics.yolo_score_age_sec)
        detector_age_valid = age is not None and age <= config.maximum_yolo_age_sec
        visibility = _bounded(metrics.visibility_ratio)
        visibility_valid = visibility is not None and visibility >= config.minimum_visibility
        recovery = metrics.is_recovery_event is True
        score = _mean_available(
            (
                1.0 if result_available else 0.0,
                _bounded(metrics.yolo_score),
                quality.yolo_freshness_score,
                visibility,
            )
        )
        reasons: list[str] = []
        if not result_available:
            reasons.append("RESULT_UNAVAILABLE")
        if not detector_score_available:
            reasons.append("YOLO_SCORE_UNAVAILABLE")
        if not detector_age_valid:
            reasons.append("OBSERVATION_NOT_FRESH")
        if not visibility_valid:
            reasons.append("VISIBILITY_INSUFFICIENT")
        if recovery:
            reasons.append("RECOVERY_ACTIVE")
        sufficient = (
            result_available
            and detector_score_available
            and detector_age_valid
            and visibility_valid
            and score is not None
            and score >= config.minimum_score
            and not recovery
        )
        return ObservationEvidence(
            result_available=result_available,
            detector_used=metrics.used_yolo is True,
            detector_score_available=detector_score_available,
            detector_age_valid=detector_age_valid,
            visibility_valid=visibility_valid,
            recovery_active=recovery,
            score=score,
            sufficient=sufficient,
            reasons=tuple(reasons),
        )

    def _consistency(self, quality: ReliabilityQuality) -> ConsistencyEvidence:
        values = (
            quality.center_stability_score,
            quality.bbox_stability_score,
            quality.continuity_score,
            quality.visibility_score,
        )
        score = _mean_available(values)
        sufficient = score is not None and score >= self._config.consistency.minimum_score
        return ConsistencyEvidence(
            score=score,
            sufficient=sufficient,
            center_score=quality.center_stability_score,
            bbox_score=quality.bbox_stability_score,
            continuity_score=quality.continuity_score,
            visibility_score=quality.visibility_score,
            reasons=() if sufficient else ("CONSISTENCY_INSUFFICIENT",),
        )

    def _freshness(
        self,
        metrics: ReliabilityMetrics,
        quality: ReliabilityQuality,
        observation: ObservationEvidence,
        consistency: ConsistencyEvidence,
    ) -> FreshnessEvidence:
        config = self._config.observation
        yolo_score = _bounded(metrics.yolo_score)
        age = _nonnegative(metrics.yolo_score_age_sec)
        score = quality.yolo_freshness_score
        sufficient = age is not None and age <= config.maximum_yolo_age_sec and score is not None
        candidate = (
            metrics.used_yolo is True
            and yolo_score is not None
            and yolo_score >= config.minimum_yolo_confidence
            and sufficient
            and observation.sufficient
            and consistency.sufficient
        )
        block_reasons: list[str] = []
        if metrics.used_yolo is not True:
            block_reasons.append("NO_DETECTOR_CONFIRMATION")
        if yolo_score is None or yolo_score < config.minimum_yolo_confidence:
            block_reasons.append("LOW_DETECTOR_SCORE")
        if not sufficient:
            block_reasons.append("STALE_DETECTOR_CONFIRMATION")
        if not observation.sufficient:
            block_reasons.append("OBSERVATION_INSUFFICIENT")
        if not observation.visibility_valid:
            block_reasons.append("VISIBILITY_INVALID")
        if not consistency.sufficient:
            block_reasons.append("CONSISTENCY_INSUFFICIENT")
        reasons: list[str] = []
        if not sufficient:
            reasons.append("FRESHNESS_INSUFFICIENT")
        if metrics.used_yolo is True and not candidate:
            reasons.append("DETECTOR_CONFIRMATION_REJECTED")
        return FreshnessEvidence(
            score=score,
            sufficient=sufficient,
            detector_confirmation_candidate=candidate,
            yolo_score=yolo_score,
            age_sec=age,
            frames_since_yolo=metrics.frames_since_last_yolo,
            reasons=tuple(reasons),
            confirmation_block_reasons=tuple(block_reasons),
        )

    def _runtime(self, metrics: ReliabilityMetrics, quality: ReliabilityQuality) -> RuntimeEvidence:
        latency = _nonnegative(metrics.update_latency_ms)
        anomaly = latency is not None and latency > self._config.runtime.latency_anomaly_ms
        return RuntimeEvidence(
            score=quality.latency_score,
            latency_ms=latency,
            latency_anomaly=anomaly,
            reasons=("LATENCY_ANOMALY",) if anomaly else (),
        )


def _mean_available(values: tuple[float | None, ...]) -> float | None:
    retained = [value for value in (_bounded(item) for item in values) if value is not None]
    return sum(retained) / len(retained) if retained else None


def _bounded(value: float | None) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) and 0.0 <= converted <= 1.0 else None


def _nonnegative(value: int | float | None) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) and converted >= 0.0 else None
