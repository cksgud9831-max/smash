"""Temporal stale-track evidence from public reliability signals.

This is an adaptation of temporal-consistency-deviation and degradation-aware
memory-gating ideas to the signals available in this repository.  It is not an
exact reproduction of any paper metric: ``overall_quality_score`` is used as
a localization-quality proxy because no PSR or response map is exposed.

All defaults are initial experimental defaults, not validated thresholds.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
import statistics

from .buffer import ReliabilityBuffer
from .hard_gate import GateResult
from .metrics import ReliabilityMetrics
from .quality import ReliabilityQuality
from .evidence import ReliabilityEvidence


class TargetEvidenceState(Enum):
    """Diagnostic evidence classification; not a replacement state machine."""

    FRESH = "FRESH"
    SUSPECT = "SUSPECT"
    STALE = "STALE"


@dataclass(frozen=True, slots=True)
class StaleConfig:
    trusted_history_size: int = 30
    trusted_history_decay: float = 1.0
    temporal_deviation_threshold: float = 1.08
    flatness_window_size: int = 20
    quality_flatness_threshold: float = 0.01
    bbox_center_flatness_threshold: float = 0.003
    bbox_area_flatness_threshold: float = 0.02
    bbox_aspect_flatness_threshold: float = 0.02
    fresh_yolo_max_age_sec: float = 1.00
    fresh_yolo_min_score: float = 0.25
    suspect_min_evidence: int = 2
    suspect_duration_sec: float = 0.75
    confirm_duration_sec: float = 2.00
    latency_support_threshold_ms: float = 100.0
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.trusted_history_size <= 0 or self.flatness_window_size <= 1:
            raise ValueError("history sizes must be positive and flatness_window_size > 1")
        if not 1 <= self.suspect_min_evidence <= 3:
            raise ValueError("suspect_min_evidence must be within [1, 3]")
        for name in (
            "temporal_deviation_threshold", "quality_flatness_threshold",
            "trusted_history_decay",
            "bbox_center_flatness_threshold", "bbox_area_flatness_threshold",
            "bbox_aspect_flatness_threshold", "fresh_yolo_max_age_sec",
            "fresh_yolo_min_score", "suspect_duration_sec", "confirm_duration_sec",
            "latency_support_threshold_ms", "epsilon",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.temporal_deviation_threshold < 1.0 or self.epsilon == 0.0:
            raise ValueError("deviation threshold must be >= 1 and epsilon > 0")
        if not 0.0 <= self.fresh_yolo_min_score <= 1.0:
            raise ValueError("fresh_yolo_min_score must be within [0, 1]")
        if self.confirm_duration_sec < self.suspect_duration_sec:
            raise ValueError("confirm_duration_sec must be >= suspect_duration_sec")


@dataclass(frozen=True, slots=True)
class StaleEvidence:
    evidence_state: TargetEvidenceState
    temporal_deviation_ratio: float | None
    trusted_quality_baseline: float | None
    quality_flatness: float | None
    quality_range: float | None
    quality_std: float | None
    bbox_center_span_norm: float | None
    bbox_area_span_ratio: float | None
    bbox_aspect_span_ratio: float | None
    fresh_yolo_confirmation: bool
    no_fresh_confirmation: bool
    suspect: bool
    confirmed: bool
    suspect_frames: int
    stale_frames: int
    suspect_duration_sec: float
    stale_duration_sec: float
    score: float | None
    reasons: tuple[str, ...]
    effective_tracking_present: bool
    trusted_history_count: int


class TrustedQualityHistory:
    """Exponentially weighted history that accepts only trusted samples."""

    def __init__(self, maxlen: int, decay: float = 1.0) -> None:
        self._values: deque[float] = deque(maxlen=maxlen)
        self._decay = decay

    def append(self, value: float) -> None:
        if math.isfinite(value) and 0.0 <= value <= 1.0:
            self._values.append(float(value))

    @property
    def baseline(self) -> float | None:
        if not self._values:
            return None
        newest_first = reversed(self._values)
        weighted = [(math.exp(-self._decay * index), value) for index, value in enumerate(newest_first)]
        weight_sum = sum(weight for weight, _ in weighted)
        return sum(weight * value for weight, value in weighted) / weight_sum

    def clear(self) -> None:
        self._values.clear()

    def __len__(self) -> int:
        return len(self._values)


class StaleTrackDetector:
    """Detect persistent stale output without changing raw result presence."""

    def __init__(self, config: StaleConfig | None = None) -> None:
        self._config = config or StaleConfig()
        self._trusted = TrustedQualityHistory(
            self._config.trusted_history_size, self._config.trusted_history_decay
        )
        self._recent_quality: deque[float | None] = deque(maxlen=self._config.flatness_window_size)
        self.reset()

    def reset(self) -> None:
        self._trusted.clear()
        self._recent_quality.clear()
        self._candidate_start: float | None = None
        self._candidate_frames = 0
        self._confirmed_frames = 0
        self._last_fresh_confirmation_timestamp: float | None = None

    def update(
        self,
        buffer: ReliabilityBuffer,
        metrics: ReliabilityMetrics,
        quality: ReliabilityQuality,
        gate: GateResult,
        evidence: ReliabilityEvidence,
        current_state: object,
    ) -> StaleEvidence:
        config = self._config
        timestamp = _finite(metrics_timestamp(buffer))
        current_quality = _bounded(quality.overall_quality_score)
        self._recent_quality.append(current_quality)
        baseline = self._trusted.baseline
        deviation = None
        if baseline is not None and current_quality is not None:
            deviation = baseline / max(current_quality, config.epsilon)

        window = buffer.history[-config.flatness_window_size:]
        quality_values = tuple(self._recent_quality)
        quality_flatness = (
            _span([value for value in quality_values if value is not None])
            if len(quality_values) >= config.flatness_window_size
            and all(value is not None for value in quality_values)
            else None
        )
        quality_std = (
            statistics.pstdev(value for value in quality_values if value is not None)
            if len(quality_values) >= config.flatness_window_size
            and all(value is not None for value in quality_values)
            else None
        )
        center_span, area_span, aspect_span = _bbox_spans(window)
        plateau = (
            quality_flatness is not None
            and quality_flatness <= config.quality_flatness_threshold
            and center_span is not None and center_span <= config.bbox_center_flatness_threshold
            and area_span is not None and area_span <= config.bbox_area_flatness_threshold
            and aspect_span is not None and aspect_span <= config.bbox_aspect_flatness_threshold
        )
        deviated = deviation is not None and deviation >= config.temporal_deviation_threshold

        preliminary_problem = deviated or plateau
        freshness_blocked = plateau or (
            deviated and getattr(current_state, "value", None) != "LOST"
        )
        fresh = (
            evidence.freshness.detector_confirmation_candidate
            and metrics.yolo_score is not None
            and metrics.yolo_score >= config.fresh_yolo_min_score
            and metrics.yolo_score_age_sec is not None
            and metrics.yolo_score_age_sec <= config.fresh_yolo_max_age_sec
            and not freshness_blocked
        )
        if fresh and timestamp is not None:
            self._last_fresh_confirmation_timestamp = timestamp

        no_fresh = (
            timestamp is not None
            and self._last_fresh_confirmation_timestamp is not None
            and timestamp - self._last_fresh_confirmation_timestamp > config.fresh_yolo_max_age_sec
        )
        evidence_flags = (deviated, plateau, no_fresh)
        evidence_count = sum(evidence_flags)
        candidate = bool(buffer.latest() and buffer.latest().result_present) and evidence_count >= config.suspect_min_evidence

        if candidate and timestamp is not None:
            if self._candidate_start is None:
                self._candidate_start = timestamp
            self._candidate_frames += 1
        else:
            self._candidate_start = None
            self._candidate_frames = 0
            self._confirmed_frames = 0

        candidate_duration = _elapsed(self._candidate_start, timestamp)
        suspect = candidate and candidate_duration >= config.suspect_duration_sec
        confirmed = candidate and candidate_duration >= config.confirm_duration_sec
        if confirmed:
            self._confirmed_frames += 1

        reasons: list[str] = []
        if deviated:
            reasons.append("TEMPORAL_DEVIATION")
        if plateau:
            reasons.append("QUALITY_GEOMETRY_PLATEAU")
        if no_fresh:
            reasons.append("NO_FRESH_CONFIRMATION")
        if evidence.runtime.latency_anomaly:
            reasons.append("LATENCY_SUPPORT")

        # Degradation-aware memory gate: only an explicit fresh detector
        # confirmation may update the reference.  Merely remaining STABLE is
        # insufficient because a stale bbox can keep that V1 state while its
        # repeated quality plateau silently contaminates the baseline.
        if current_quality is not None and fresh and gate.passed:
            self._trusted.append(current_quality)

        raw_present = bool(buffer.latest() and buffer.latest().result_present)
        return StaleEvidence(
            evidence_state=(
                TargetEvidenceState.STALE
                if confirmed
                else TargetEvidenceState.SUSPECT
                if suspect
                else TargetEvidenceState.FRESH
            ),
            temporal_deviation_ratio=deviation,
            trusted_quality_baseline=baseline,
            quality_flatness=quality_flatness,
            quality_range=quality_flatness,
            quality_std=quality_std,
            bbox_center_span_norm=center_span,
            bbox_area_span_ratio=area_span,
            bbox_aspect_span_ratio=aspect_span,
            fresh_yolo_confirmation=fresh,
            no_fresh_confirmation=no_fresh,
            suspect=suspect,
            confirmed=confirmed,
            suspect_frames=self._candidate_frames if suspect else 0,
            stale_frames=self._confirmed_frames,
            suspect_duration_sec=candidate_duration if suspect else 0.0,
            stale_duration_sec=max(0.0, candidate_duration - config.confirm_duration_sec) if confirmed else 0.0,
            score=(evidence_count / 3.0) if current_quality is not None else None,
            reasons=tuple(reasons),
            effective_tracking_present=raw_present and not confirmed,
            trusted_history_count=len(self._trusted),
        )


def metrics_timestamp(buffer: ReliabilityBuffer) -> float | None:
    latest = buffer.latest()
    return None if latest is None else latest.timestamp


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _bounded(value: float | None) -> float | None:
    result = _finite(value)
    return result if result is not None and 0.0 <= result <= 1.0 else None


def _elapsed(start: float | None, current: float | None) -> float:
    if start is None or current is None or current < start:
        return 0.0
    return current - start


def _span(values: list[float]) -> float | None:
    return max(values) - min(values) if values else None


def _bbox_spans(history: tuple) -> tuple[float | None, float | None, float | None]:
    if len(history) < 2:
        return None, None, None
    centers: list[tuple[float, float]] = []
    areas: list[float] = []
    aspects: list[float] = []
    diagonal = None
    for item in history:
        if not item.result_present or item.bbox is None or item.frame_width <= 0 or item.frame_height <= 0:
            return None, None, None
        x, y, width, height = item.bbox
        if not all(math.isfinite(float(v)) for v in item.bbox) or width <= 0 or height <= 0:
            return None, None, None
        centers.append((x + width / 2.0, y + height / 2.0))
        areas.append(width * height)
        aspects.append(width / height)
        diagonal = math.hypot(item.frame_width, item.frame_height)
    if diagonal is None or diagonal <= 0:
        return None, None, None
    center_span = math.hypot(
        max(x for x, _ in centers) - min(x for x, _ in centers),
        max(y for _, y in centers) - min(y for _, y in centers),
    ) / diagonal
    return center_span, _relative_span(areas), _relative_span(aspects)


def _relative_span(values: list[float]) -> float | None:
    minimum = min(values)
    return None if minimum <= 0.0 else (max(values) - minimum) / minimum
