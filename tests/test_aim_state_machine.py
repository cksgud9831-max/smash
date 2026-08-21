import pytest

from aiming_engine.config import AimStateMachineConfig
from aiming_engine.aim_state_machine import AimStateMachine
from aiming_engine.types import AimState


def make_asm(**overrides) -> AimStateMachine:
    defaults = dict(
        tracking_confidence_threshold=0.6,
        min_stable_frames_for_track=3,
        min_stable_frames_for_aim=2,
        track_lost_timeout_s=0.3,
    )
    defaults.update(overrides)
    return AimStateMachine(AimStateMachineConfig(**defaults))


def test_full_cycle_search_to_track_to_aim_to_ready():
    asm = make_asm()

    assert asm.update(0.9, True, False, now=0.0) == AimState.SEARCH
    assert asm.update(0.9, True, False, now=0.1) == AimState.SEARCH
    assert asm.update(0.9, True, False, now=0.2) == AimState.TRACK
    assert asm.update(0.9, True, False, now=0.3) == AimState.AIM
    assert asm.update(0.9, True, True, now=0.4) == AimState.AIM
    assert asm.update(0.9, True, True, now=0.5) == AimState.READY


def test_ready_persists_while_aim_stays_ok():
    asm = make_asm(min_stable_frames_for_track=1, min_stable_frames_for_aim=1)
    asm.update(0.9, True, False, now=0.0)  # -> TRACK
    asm.update(0.9, True, False, now=0.1)  # -> AIM
    assert asm.update(0.9, True, True, now=0.2) == AimState.READY
    assert asm.update(0.9, True, True, now=0.3) == AimState.READY
    assert asm.update(0.9, True, True, now=0.4) == AimState.READY


def test_low_tracking_confidence_keeps_search_state():
    asm = make_asm()
    for i in range(5):
        state = asm.update(0.1, True, False, now=i * 0.1)
    assert state == AimState.SEARCH


def test_brief_confidence_dip_within_grace_period_does_not_reset_to_search():
    asm = make_asm(min_stable_frames_for_track=1, min_stable_frames_for_aim=1, track_lost_timeout_s=0.3)
    asm.update(0.9, True, False, now=0.0)  # -> TRACK
    asm.update(0.9, True, False, now=0.1)  # -> AIM
    # single bad frame, well within 0.3s grace period
    state = asm.update(0.1, True, False, now=0.2)
    assert state == AimState.AIM


def test_sustained_tracking_loss_past_grace_period_falls_back_to_search():
    asm = make_asm(min_stable_frames_for_track=1, min_stable_frames_for_aim=1, track_lost_timeout_s=0.2)
    asm.update(0.9, True, False, now=0.0)  # -> TRACK
    asm.update(0.9, True, False, now=0.1)  # -> AIM
    asm.update(0.1, True, False, now=0.2)
    state = asm.update(0.1, True, False, now=0.5)  # well past the 0.2s grace period
    assert state == AimState.SEARCH


def test_ready_degrades_to_aim_when_aim_ok_becomes_false():
    asm = make_asm(min_stable_frames_for_track=1, min_stable_frames_for_aim=1)
    asm.update(0.9, True, False, now=0.0)  # -> TRACK
    asm.update(0.9, True, False, now=0.1)  # -> AIM
    asm.update(0.9, True, True, now=0.2)  # -> READY
    state = asm.update(0.9, True, False, now=0.3)  # aim degraded
    assert state == AimState.AIM
