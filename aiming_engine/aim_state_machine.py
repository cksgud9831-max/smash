"""SEARCH -> TRACK -> AIM -> READY aim-quality state machine.

There is no firing state here — this framework never fires anything.
READY means "the aim solution is currently stable enough to trust," which
the HUD can use to color the reticle or show a "locked" indicator; the
operator decides independently whether and when to actually fire.

Every frame must call `update()` exactly once; it is the only place state
transitions happen. `aim_ok` is precomputed by the caller (AimingManager)
from AimSolver output, so this class stays free of ballistic or
probability thresholds — only state-machine timing/counter parameters
(from `AimStateMachineConfig`) live here.
"""

from __future__ import annotations

from .config import AimStateMachineConfig
from .types import AimState


class AimStateMachine:
    def __init__(self, config: AimStateMachineConfig):
        self._config = config
        self._state = AimState.SEARCH
        self._track_stable_count = 0
        self._aim_stable_count = 0
        self._last_good_time: float | None = None

    @property
    def state(self) -> AimState:
        return self._state

    def update(self, tracking_confidence: float, follower_stable: bool, aim_ok: bool, now: float) -> AimState:
        tracking_ok = tracking_confidence >= self._config.tracking_confidence_threshold
        good_frame = tracking_ok and follower_stable
        if good_frame:
            self._last_good_time = now
        lost = self._last_good_time is None or (now - self._last_good_time) > self._config.track_lost_timeout_s

        if self._state == AimState.SEARCH:
            self._track_stable_count = self._track_stable_count + 1 if tracking_ok else 0
            if self._track_stable_count >= self._config.min_stable_frames_for_track:
                self._state = AimState.TRACK
                self._track_stable_count = 0

        elif self._state == AimState.TRACK:
            if lost:
                self._state = AimState.SEARCH
            elif follower_stable:
                self._state = AimState.AIM
                self._aim_stable_count = 0

        elif self._state == AimState.AIM:
            if lost:
                self._state = AimState.SEARCH
            else:
                self._aim_stable_count = self._aim_stable_count + 1 if aim_ok else 0
                if self._aim_stable_count >= self._config.min_stable_frames_for_aim:
                    self._state = AimState.READY
                    self._aim_stable_count = 0

        elif self._state == AimState.READY:
            if lost:
                self._state = AimState.SEARCH
            elif not aim_ok:
                self._state = AimState.AIM
                self._aim_stable_count = 0

        return self._state
