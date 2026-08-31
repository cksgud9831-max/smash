"""Stateful temporal evidence accumulation for tracking reliability.

The defaults are initial experimental defaults only, not validated deployment
values.  This module produces evidence for a later state machine; it does not
define states, transitions, hysteresis state, or system authorization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .hard_gate import GateResult, RECOVERY_EVENT
from .metrics import ReliabilityMetrics
from .quality import ReliabilityQuality


@dataclass(frozen=True, slots=True)
class TemporalConfig:
    """Experimental thresholds and consecutive-frame requirements."""

    stable_enter_quality: float = 0.75
    stable_exit_quality: float = 0.60
    stable_enter_frames: int = 10
    stable_exit_frames: int = 3
    lost_missing_frames: int = 5
    recovery_hold_frames: int = 5

    def __post_init__(self) -> None:
        for name in ("stable_enter_quality", "stable_exit_quality"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1]")
        if self.stable_enter_quality <= self.stable_exit_quality:
            raise ValueError("stable_enter_quality must be greater than stable_exit_quality")

        for name in (
            "stable_enter_frames",
            "stable_exit_frames",
            "lost_missing_frames",
            "recovery_hold_frames",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")


@dataclass(frozen=True, slots=True)
class TemporalEvidence:
    """Current counters and readiness evidence, without a tracking state."""

    stable_candidate_frames: int
    unstable_frames: int
    missing_frames: int
    recovery_hold_remaining: int
    stable_entry_ready: bool
    stable_exit_ready: bool
    lost_ready: bool
    current_gate_passed: bool
    current_quality_score: float | None
    current_gate_reasons: tuple[str, ...]
    recovery_event_observed: bool
    recovery_hold_active: bool
    in_hysteresis_band: bool


class ReliabilityTemporalValidator:
    """Accumulate consecutive evidence for a future state machine."""

    def __init__(self, config: TemporalConfig | None = None) -> None:
        self._config = config if config is not None else TemporalConfig()
        self.reset()

    @property
    def config(self) -> TemporalConfig:
        return self._config

    def reset(self) -> None:
        self._stable_candidate_frames = 0
        self._unstable_frames = 0
        self._recovery_hold_remaining = 0

    def update(
        self,
        metrics: ReliabilityMetrics,
        quality: ReliabilityQuality,
        gate: GateResult,
    ) -> TemporalEvidence:
        config = self._config
        quality_score = _valid_quality_score(quality.overall_quality_score)
        recovery_event = (
            metrics.is_recovery_event is True
            or quality.is_recovery_event is True
            or RECOVERY_EVENT in gate.diagnostics
        )

        if recovery_event:
            self._recovery_hold_remaining = config.recovery_hold_frames

        recovery_hold_active = self._recovery_hold_remaining > 0
        stable_candidate = (
            gate.passed
            and quality_score is not None
            and quality_score >= config.stable_enter_quality
            and not recovery_hold_active
        )
        if stable_candidate:
            self._stable_candidate_frames += 1
        else:
            self._stable_candidate_frames = 0

        unstable_candidate = (
            not gate.passed
            or quality_score is None
            or quality_score < config.stable_exit_quality
        )
        if unstable_candidate:
            self._unstable_frames += 1
        else:
            self._unstable_frames = 0

        in_hysteresis_band = (
            gate.passed
            and quality_score is not None
            and config.stable_exit_quality <= quality_score < config.stable_enter_quality
        )

        missing_frames = max(int(metrics.consecutive_missing_frames), 0)
        evidence = TemporalEvidence(
            stable_candidate_frames=self._stable_candidate_frames,
            unstable_frames=self._unstable_frames,
            missing_frames=missing_frames,
            recovery_hold_remaining=self._recovery_hold_remaining,
            stable_entry_ready=self._stable_candidate_frames >= config.stable_enter_frames,
            stable_exit_ready=self._unstable_frames >= config.stable_exit_frames,
            lost_ready=missing_frames >= config.lost_missing_frames,
            current_gate_passed=gate.passed,
            current_quality_score=quality_score,
            current_gate_reasons=gate.reasons,
            recovery_event_observed=recovery_event,
            recovery_hold_active=recovery_hold_active,
            in_hysteresis_band=in_hysteresis_band,
        )

        # The event frame counts as the first held frame.  Thus a configured
        # value N blocks stable accumulation for exactly N update calls,
        # including the call on which the event is observed.
        if self._recovery_hold_remaining > 0:
            self._recovery_hold_remaining -= 1

        return evidence


def _valid_quality_score(value: float | None) -> float | None:
    if value is None:
        return None
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        return None
    return converted
