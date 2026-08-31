"""Tracking-reliability state transitions driven by temporal evidence.

These states describe tracking reliability only.  They do not authorize or
block aiming, firing, actuator control, or any other real-system action.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .temporal import TemporalEvidence


INITIAL = "INITIAL"
TRACKER_RESULT_AVAILABLE = "TRACKER_RESULT_AVAILABLE"
STABLE_ENTRY_READY = "STABLE_ENTRY_READY"
STABLE_EXIT_READY = "STABLE_EXIT_READY"
LOST_READY = "LOST_READY"
STATE_MAINTAINED = "STATE_MAINTAINED"


class TrackingReliabilityState(Enum):
    """Reliability of the tracking stream, not an operational permission."""

    STABLE = "STABLE"
    HOLD = "HOLD"
    LOST = "LOST"


@dataclass(frozen=True, slots=True)
class StateTransition:
    """One deterministic state-machine update result."""

    previous_state: TrackingReliabilityState
    current_state: TrackingReliabilityState
    changed: bool
    reason: str
    timestamp: float | None = None


class ReliabilityStateMachine:
    """Apply transition rules to already-validated temporal evidence.

    Transition table, after the common highest-priority ``lost_ready`` rule:

    * LOST: missing_frames == 0 -> HOLD; otherwise remain LOST.
    * HOLD: stable_entry_ready -> STABLE; otherwise remain HOLD.
    * STABLE: stable_exit_ready -> HOLD; otherwise remain STABLE.

    A direct LOST -> STABLE transition is intentionally impossible.
    """

    def __init__(self) -> None:
        self._state = TrackingReliabilityState.LOST

    @property
    def state(self) -> TrackingReliabilityState:
        return self._state

    def reset(self) -> None:
        self._state = TrackingReliabilityState.LOST

    def update(
        self,
        evidence: TemporalEvidence,
        timestamp: float | None = None,
    ) -> StateTransition:
        previous = self._state
        current = previous
        transition_reason = STATE_MAINTAINED

        # Common highest priority: explicit LOST evidence overrides every
        # state-specific entry or exit readiness flag.
        if evidence.lost_ready:
            current = TrackingReliabilityState.LOST
            if current is not previous:
                transition_reason = LOST_READY
        elif previous is TrackingReliabilityState.LOST:
            if evidence.missing_frames == 0:
                current = TrackingReliabilityState.HOLD
                transition_reason = TRACKER_RESULT_AVAILABLE
        elif previous is TrackingReliabilityState.HOLD:
            if evidence.stable_entry_ready:
                current = TrackingReliabilityState.STABLE
                transition_reason = STABLE_ENTRY_READY
        elif previous is TrackingReliabilityState.STABLE:
            if evidence.stable_exit_ready:
                current = TrackingReliabilityState.HOLD
                transition_reason = STABLE_EXIT_READY

        changed = current is not previous
        self._state = current
        return StateTransition(
            previous_state=previous,
            current_state=current,
            changed=changed,
            reason=transition_reason,
            timestamp=timestamp,
        )
