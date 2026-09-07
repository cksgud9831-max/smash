"""Convert raw reliability metrics into configurable quality components.

The defaults in this module are initial experimental defaults only.  They are
not validated operating thresholds and must be tuned with representative data.
This module does not implement hard gates, operational decisions, or states.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .metrics import ReliabilityMetrics


@dataclass(frozen=True, slots=True)
class QualityConfig:
    """Tunable scaling references and weights for quality calculation.

    Every numeric default is an initial experimental default, not a validated
    value for deployment.
    """

    duration_reference_sec: float = 1.0
    center_displacement_reference: float = 0.05
    bbox_area_change_reference: float = 0.50
    bbox_aspect_change_reference: float = 0.50
    latency_reference_ms: float = 50.0
    yolo_age_reference_sec: float = 1.0
    yolo_frame_reference: float = 30.0

    bbox_area_component_weight: float = 0.5
    bbox_aspect_component_weight: float = 0.5
    yolo_age_component_weight: float = 0.5
    yolo_frame_component_weight: float = 0.5

    continuity_weight: float = 1.0
    duration_weight: float = 1.0
    center_stability_weight: float = 1.0
    bbox_stability_weight: float = 1.0
    visibility_weight: float = 1.0
    latency_weight: float = 1.0
    yolo_confidence_weight: float = 1.0
    yolo_freshness_weight: float = 1.0

    def __post_init__(self) -> None:
        positive_fields = (
            "duration_reference_sec",
            "center_displacement_reference",
            "bbox_area_change_reference",
            "bbox_aspect_change_reference",
            "latency_reference_ms",
            "yolo_age_reference_sec",
            "yolo_frame_reference",
        )
        nonnegative_fields = (
            "bbox_area_component_weight",
            "bbox_aspect_component_weight",
            "yolo_age_component_weight",
            "yolo_frame_component_weight",
            "continuity_weight",
            "duration_weight",
            "center_stability_weight",
            "bbox_stability_weight",
            "visibility_weight",
            "latency_weight",
            "yolo_confidence_weight",
            "yolo_freshness_weight",
        )

        for name in positive_fields:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")
        for name in nonnegative_fields:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

        if self.bbox_area_component_weight + self.bbox_aspect_component_weight <= 0.0:
            raise ValueError("at least one bbox component weight must be positive")
        if self.yolo_age_component_weight + self.yolo_frame_component_weight <= 0.0:
            raise ValueError("at least one YOLO freshness component weight must be positive")
        if sum(self.overall_weights) <= 0.0:
            raise ValueError("at least one overall quality weight must be positive")

    @property
    def overall_weights(self) -> tuple[float, ...]:
        return (
            self.continuity_weight,
            self.duration_weight,
            self.center_stability_weight,
            self.bbox_stability_weight,
            self.visibility_weight,
            self.latency_weight,
            self.yolo_confidence_weight,
            self.yolo_freshness_weight,
        )


@dataclass(frozen=True, slots=True)
class ReliabilityQuality:
    """Individual quality components and their normalized weighted mean."""

    continuity_score: float | None
    duration_score: float | None
    center_stability_score: float | None
    bbox_stability_score: float | None
    visibility_score: float | None
    latency_score: float | None
    yolo_confidence_score: float | None
    yolo_freshness_score: float | None
    overall_quality_score: float | None
    available_component_count: int
    used_weight_sum: float
    used_yolo: bool | None
    is_recovery_event: bool | None


class ReliabilityQualityCalculator:
    """Map raw metrics to quality without making an operational decision."""

    def __init__(self, config: QualityConfig | None = None) -> None:
        self._config = config if config is not None else QualityConfig()

    @property
    def config(self) -> QualityConfig:
        return self._config

    def calculate(self, metrics: ReliabilityMetrics) -> ReliabilityQuality:
        config = self._config

        continuity = _bounded_metric(metrics.result_continuity)
        duration = _increasing_score(metrics.tracking_duration_sec, config.duration_reference_sec)
        center = _decreasing_score(
            metrics.center_displacement_norm,
            config.center_displacement_reference,
        )

        bbox_area = _decreasing_score(
            metrics.bbox_area_change_ratio,
            config.bbox_area_change_reference,
        )
        bbox_aspect = _decreasing_score(
            metrics.bbox_aspect_change_ratio,
            config.bbox_aspect_change_reference,
        )
        bbox = _available_weighted_mean(
            (
                (bbox_area, config.bbox_area_component_weight),
                (bbox_aspect, config.bbox_aspect_component_weight),
            )
        )[0]

        visibility = _bounded_metric(metrics.visibility_ratio)
        latency = _decreasing_score(metrics.update_latency_ms, config.latency_reference_ms)
        yolo_confidence = _bounded_metric(metrics.yolo_score)

        freshness_by_age = _decreasing_score(
            metrics.yolo_score_age_sec,
            config.yolo_age_reference_sec,
        )
        freshness_by_frames = _decreasing_score(
            _nonnegative_float(metrics.frames_since_last_yolo),
            config.yolo_frame_reference,
        )
        yolo_freshness = _available_weighted_mean(
            (
                (freshness_by_age, config.yolo_age_component_weight),
                (freshness_by_frames, config.yolo_frame_component_weight),
            )
        )[0]

        components = (
            (continuity, config.continuity_weight),
            (duration, config.duration_weight),
            (center, config.center_stability_weight),
            (bbox, config.bbox_stability_weight),
            (visibility, config.visibility_weight),
            (latency, config.latency_weight),
            (yolo_confidence, config.yolo_confidence_weight),
            (yolo_freshness, config.yolo_freshness_weight),
        )
        overall, used_weight_sum = _available_weighted_mean(components)

        return ReliabilityQuality(
            continuity_score=continuity,
            duration_score=duration,
            center_stability_score=center,
            bbox_stability_score=bbox,
            visibility_score=visibility,
            latency_score=latency,
            yolo_confidence_score=yolo_confidence,
            yolo_freshness_score=yolo_freshness,
            overall_quality_score=overall,
            available_component_count=sum(score is not None for score, _weight in components),
            used_weight_sum=used_weight_sum,
            used_yolo=metrics.used_yolo,
            is_recovery_event=metrics.is_recovery_event,
        )


def _clamp_unit(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _nonnegative_float(value: int | float | None) -> float | None:
    if value is None:
        return None
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        return None
    return converted


def _bounded_metric(value: float | None) -> float | None:
    converted = _nonnegative_float(value)
    return None if converted is None else _clamp_unit(converted)


def _increasing_score(value: float | None, reference: float) -> float | None:
    converted = _nonnegative_float(value)
    return None if converted is None else _clamp_unit(converted / reference)


def _decreasing_score(value: float | None, reference: float) -> float | None:
    converted = _nonnegative_float(value)
    return None if converted is None else _clamp_unit(1.0 - converted / reference)


def _available_weighted_mean(
    components: tuple[tuple[float | None, float], ...],
) -> tuple[float | None, float]:
    available = [(score, weight) for score, weight in components if score is not None and weight > 0.0]
    used_weight_sum = sum(weight for _score, weight in available)
    if used_weight_sum <= 0.0:
        return None, 0.0
    weighted_sum = sum(score * weight for score, weight in available)
    return _clamp_unit(weighted_sum / used_weight_sum), used_weight_sum
