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
from .stale_detector import StaleEvidence
from .evidence import ReliabilityEvidence


@dataclass(frozen=True, slots=True)
class TemporalConfig:
    """Experimental thresholds and consecutive-frame requirements."""

    stable_enter_quality: float = 0.75
    stable_exit_quality: float = 0.60
    stable_enter_frames: int = 10
    stable_exit_frames: int = 3
    lost_missing_frames: int = 5
    recovery_hold_frames: int = 5
    lost_untrustworthy_duration_sec: float = 1.5

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
        if (
            not math.isfinite(self.lost_untrustworthy_duration_sec)
            or self.lost_untrustworthy_duration_sec <= 0.0
        ):
            raise ValueError("lost_untrustworthy_duration_sec must be finite and greater than zero")


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
    stale_suspect_ready: bool = False
    stale_confirmed_lost: bool = False
    fresh_target_confirmation: bool = False
    effective_tracking_present: bool = False
    untrustworthy_duration_sec: float = 0.0
    missing_result_lost_ready: bool = False
    observation_lost_ready: bool = False
    fresh_observation_candidate: bool = False
    fresh_reacquisition_candidate: bool = False
    reacquisition_block_reasons: tuple[str, ...] = ()


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
        self._untrustworthy_start_timestamp: float | None = None
        self._last_timestamp: float | None = None

    def update(
        self,
        metrics: ReliabilityMetrics,
        quality: ReliabilityQuality,
        gate: GateResult,
        stale: StaleEvidence | None = None,
        reliability_evidence: ReliabilityEvidence | None = None,
        allow_stable_candidate: bool = True,
        timestamp: float | None = None,
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
            and not (stale and (stale.suspect or stale.confirmed))
            and allow_stable_candidate
            and (reliability_evidence is None or reliability_evidence.stable_eligible)
        )
        if stable_candidate:
            self._stable_candidate_frames += 1
        else:
            self._stable_candidate_frames = 0

        unstable_candidate = (
            not gate.passed
            or quality_score is None
            or quality_score < config.stable_exit_quality
            or bool(stale and stale.suspect)
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
        effective_present = (
            stale.effective_tracking_present
            if stale is not None
            else missing_frames == 0
        )
        trustworthy = (
            effective_present
            and reliability_evidence is not None
            and reliability_evidence.observation.sufficient
            and reliability_evidence.consistency.sufficient
            and reliability_evidence.freshness.sufficient
        )
        valid_timestamp = _valid_timestamp(timestamp, self._last_timestamp)
        if trustworthy:
            self._untrustworthy_start_timestamp = None
        elif valid_timestamp is not None and self._untrustworthy_start_timestamp is None:
            self._untrustworthy_start_timestamp = valid_timestamp
        untrustworthy_duration = (
            valid_timestamp - self._untrustworthy_start_timestamp
            if valid_timestamp is not None and self._untrustworthy_start_timestamp is not None
            else 0.0
        )
        if valid_timestamp is not None:
            self._last_timestamp = valid_timestamp
        missing_lost_ready = missing_frames >= config.lost_missing_frames
        observation_lost_ready = untrustworthy_duration >= config.lost_untrustworthy_duration_sec
        stale_confirmed_ready = bool(stale and stale.confirmed)
        fresh_observation_candidate = bool(
            reliability_evidence
            and reliability_evidence.freshness.detector_confirmation_candidate
        )
        fresh_reacquisition_candidate = bool(
            stale and stale.fresh_yolo_confirmation and effective_present
        )
        reacquisition_block_reasons = list(
            reliability_evidence.freshness.confirmation_block_reasons
            if reliability_evidence is not None
            else ("NO_DETECTOR_CONFIRMATION",)
        )
        if stale_confirmed_ready:
            reacquisition_block_reasons.append("STALE_CONFIRMED")
        if (
            fresh_reacquisition_candidate
            and not gate.passed
            and gate.reasons == ("EXCESSIVE_LATENCY",)
        ):
            reacquisition_block_reasons.append("RUNTIME_ONLY_GATE_FAILURE")
        evidence = TemporalEvidence(
            stable_candidate_frames=self._stable_candidate_frames,
            unstable_frames=self._unstable_frames,
            missing_frames=missing_frames,
            recovery_hold_remaining=self._recovery_hold_remaining,
            stable_entry_ready=self._stable_candidate_frames >= config.stable_enter_frames,
            stable_exit_ready=self._unstable_frames >= config.stable_exit_frames,
            lost_ready=(missing_lost_ready or observation_lost_ready or stale_confirmed_ready),
            current_gate_passed=gate.passed,
            current_quality_score=quality_score,
            current_gate_reasons=gate.reasons,
            recovery_event_observed=recovery_event,
            recovery_hold_active=recovery_hold_active,
            in_hysteresis_band=in_hysteresis_band,
            stale_suspect_ready=bool(stale and stale.suspect),
            stale_confirmed_lost=stale_confirmed_ready,
            fresh_target_confirmation=bool(stale and stale.fresh_yolo_confirmation),
            effective_tracking_present=effective_present,
            untrustworthy_duration_sec=untrustworthy_duration,
            missing_result_lost_ready=missing_lost_ready,
            observation_lost_ready=observation_lost_ready,
            fresh_observation_candidate=fresh_observation_candidate,
            fresh_reacquisition_candidate=fresh_reacquisition_candidate,
            reacquisition_block_reasons=tuple(reacquisition_block_reasons),
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


def _valid_timestamp(value: float | None, previous: float | None) -> float | None:
    if value is None:
        return None
    converted = float(value)
    if not math.isfinite(converted) or (previous is not None and converted < previous):
        return None
    return converted
