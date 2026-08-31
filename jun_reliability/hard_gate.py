"""Minimum evidence and raw-input validity gates for reliability analysis.

The defaults are initial experimental defaults only, not validated deployment
limits.  This gate does not assign STABLE/HOLD/LOST, perform temporal
validation, or authorize/block any real system action.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .metrics import ReliabilityMetrics
from .quality import ReliabilityQuality


MISSING_OVERALL_QUALITY = "MISSING_OVERALL_QUALITY"
INSUFFICIENT_COMPONENTS = "INSUFFICIENT_COMPONENTS"
INSUFFICIENT_WEIGHT = "INSUFFICIENT_WEIGHT"
TOO_MANY_MISSING_FRAMES = "TOO_MANY_MISSING_FRAMES"
MISSING_VISIBILITY = "MISSING_VISIBILITY"
INVALID_VISIBILITY = "INVALID_VISIBILITY"
MISSING_LATENCY = "MISSING_LATENCY"
EXCESSIVE_LATENCY = "EXCESSIVE_LATENCY"
MISSING_YOLO_FRESHNESS = "MISSING_YOLO_FRESHNESS"
YOLO_TOO_STALE = "YOLO_TOO_STALE"
RECOVERY_EVENT = "RECOVERY_EVENT"


@dataclass(frozen=True, slots=True)
class GateConfig:
    """Configurable gate limits and missing-evidence policies.

    All numeric defaults are initial experimental defaults.  They must be
    replaced or tuned using representative measurements before deployment.
    """

    minimum_component_count: int = 5
    minimum_used_weight_sum: float = 5.0
    maximum_consecutive_missing_frames: int = 2
    minimum_visibility_ratio: float = 0.50
    maximum_update_latency_ms: float = 100.0
    maximum_yolo_age_sec: float = 2.0
    maximum_frames_since_last_yolo: int = 60
    require_visibility: bool = True
    require_latency: bool = True
    require_yolo_freshness: bool = True

    def __post_init__(self) -> None:
        if self.minimum_component_count < 0:
            raise ValueError("minimum_component_count must be non-negative")
        if self.maximum_consecutive_missing_frames < 0:
            raise ValueError("maximum_consecutive_missing_frames must be non-negative")
        if self.maximum_frames_since_last_yolo < 0:
            raise ValueError("maximum_frames_since_last_yolo must be non-negative")

        if not math.isfinite(self.minimum_used_weight_sum) or self.minimum_used_weight_sum < 0.0:
            raise ValueError("minimum_used_weight_sum must be finite and non-negative")
        if not math.isfinite(self.minimum_visibility_ratio) or not 0.0 <= self.minimum_visibility_ratio <= 1.0:
            raise ValueError("minimum_visibility_ratio must be finite and within [0, 1]")
        if not math.isfinite(self.maximum_update_latency_ms) or self.maximum_update_latency_ms < 0.0:
            raise ValueError("maximum_update_latency_ms must be finite and non-negative")
        if not math.isfinite(self.maximum_yolo_age_sec) or self.maximum_yolo_age_sec < 0.0:
            raise ValueError("maximum_yolo_age_sec must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class GateResult:
    """Gate outcome with failure reasons and non-blocking diagnostics."""

    passed: bool
    reasons: tuple[str, ...]
    available_component_count: int
    used_weight_sum: float
    diagnostics: tuple[str, ...] = ()


class ReliabilityHardGate:
    """Check evidence sufficiency and explicit raw-data validity limits."""

    def __init__(self, config: GateConfig | None = None) -> None:
        self._config = config if config is not None else GateConfig()

    @property
    def config(self) -> GateConfig:
        return self._config

    def evaluate(
        self,
        metrics: ReliabilityMetrics,
        quality: ReliabilityQuality,
    ) -> GateResult:
        config = self._config
        reasons: list[str] = []
        diagnostics: list[str] = []

        if quality.overall_quality_score is None:
            reasons.append(MISSING_OVERALL_QUALITY)
        if quality.available_component_count < config.minimum_component_count:
            reasons.append(INSUFFICIENT_COMPONENTS)
        if quality.used_weight_sum < config.minimum_used_weight_sum:
            reasons.append(INSUFFICIENT_WEIGHT)

        if metrics.consecutive_missing_frames > config.maximum_consecutive_missing_frames:
            reasons.append(TOO_MANY_MISSING_FRAMES)

        visibility = _finite_nonnegative(metrics.visibility_ratio)
        if visibility is None:
            if config.require_visibility:
                reasons.append(MISSING_VISIBILITY)
        elif visibility < config.minimum_visibility_ratio:
            reasons.append(INVALID_VISIBILITY)

        latency = _finite_nonnegative(metrics.update_latency_ms)
        if latency is None:
            if config.require_latency:
                reasons.append(MISSING_LATENCY)
        elif latency > config.maximum_update_latency_ms:
            reasons.append(EXCESSIVE_LATENCY)

        yolo_age = _finite_nonnegative(metrics.yolo_score_age_sec)
        yolo_frames = _finite_nonnegative(metrics.frames_since_last_yolo)
        if yolo_age is None and yolo_frames is None:
            if config.require_yolo_freshness:
                reasons.append(MISSING_YOLO_FRESHNESS)
        elif (
            (yolo_age is not None and yolo_age > config.maximum_yolo_age_sec)
            or (
                yolo_frames is not None
                and yolo_frames > config.maximum_frames_since_last_yolo
            )
        ):
            reasons.append(YOLO_TOO_STALE)

        if metrics.is_recovery_event is True or quality.is_recovery_event is True:
            diagnostics.append(RECOVERY_EVENT)

        return GateResult(
            passed=not reasons,
            reasons=tuple(reasons),
            available_component_count=quality.available_component_count,
            used_weight_sum=quality.used_weight_sum,
            diagnostics=tuple(diagnostics),
        )


def _finite_nonnegative(value: int | float | None) -> float | None:
    if value is None:
        return None
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        return None
    return converted
