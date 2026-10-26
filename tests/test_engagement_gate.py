"""Moving-target automatic-fire gate tests (ROS-independent)."""

import os
import sys

import pytest

_SIM_PKG = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, "simulation",
    "ciws_turret_aerial_object_detection-main", "drone_sim",
))
sys.path.insert(0, _SIM_PKG)

from drone_sim.smash_fcs.engagement_gate import (  # noqa: E402
    SpeedCollapseDetector,
    motion_allows_fire,
    motion_solution_is_stable,
)


@pytest.mark.parametrize("state", [None, "WAIT", "WAIT LOCK", "HOLD", "DOWN"])
def test_moving_target_fire_fails_closed_until_move(state):
    assert not motion_allows_fire(True, state)


def test_moving_target_fire_opens_after_departure():
    assert motion_allows_fire(True, "MOVE")


def test_stationary_fire_keeps_previous_behavior():
    assert motion_allows_fire(False, None)


def test_motion_solution_gate_bypasses_static_acquisition():
    assert motion_solution_is_stable(
        "WAIT", None,
        max_velocity_disagreement_mps=0.75, max_fit_rmse_m=0.2, min_history_count=6,
    )


def test_motion_solution_gate_accepts_consistent_fit():
    assert motion_solution_is_stable(
        "MOVE", {"velocity_disagreement_mps": 0.7, "fit_rmse_m": 0.19, "history_count": 20},
        max_velocity_disagreement_mps=0.75, max_fit_rmse_m=0.2, min_history_count=6,
    )


@pytest.mark.parametrize("diagnostics", [
    None,
    {},
    {"velocity_disagreement_mps": 0.8, "fit_rmse_m": 0.1, "history_count": 20},
    {"velocity_disagreement_mps": 0.2, "fit_rmse_m": 0.21, "history_count": 20},
    {"velocity_disagreement_mps": 0.2, "fit_rmse_m": 0.1, "history_count": 5},
])
def test_motion_solution_gate_rejects_inconsistent_or_missing_fit(diagnostics):
    assert not motion_solution_is_stable(
        "MOVE", diagnostics,
        max_velocity_disagreement_mps=0.75, max_fit_rmse_m=0.2, min_history_count=6,
    )


_OK = {"velocity_disagreement_mps": 0.3, "fit_rmse_m": 0.1, "history_count": 20}


def _sign_gate(**lateral):
    return motion_solution_is_stable(
        "MOVE", {**_OK, **lateral},
        max_velocity_disagreement_mps=0.75, max_fit_rmse_m=0.2, min_history_count=6,
        lateral_sign_deadband_mps=0.15,
    )


def test_lateral_sign_gate_rejects_opposite_directions():
    assert not _sign_gate(lateral_velocity_long_mps=-0.5, lateral_velocity_recent_mps=0.3)
    assert not _sign_gate(lateral_velocity_long_mps=0.8, lateral_velocity_recent_mps=-0.2)


def test_lateral_sign_gate_accepts_same_direction_or_near_zero():
    assert _sign_gate(lateral_velocity_long_mps=-0.8, lateral_velocity_recent_mps=-0.7)
    assert _sign_gate(lateral_velocity_long_mps=0.8, lateral_velocity_recent_mps=-0.1)  # recent within deadband
    assert _sign_gate(lateral_velocity_long_mps=0.05, lateral_velocity_recent_mps=-0.5)  # long within deadband


def test_lateral_sign_gate_fails_closed_without_lateral_data():
    assert not _sign_gate()
    assert not _sign_gate(lateral_velocity_long_mps=None, lateral_velocity_recent_mps=0.4)


def test_lateral_sign_gate_disabled_by_default():
    assert motion_solution_is_stable(
        "MOVE", {**_OK, "lateral_velocity_long_mps": -0.8, "lateral_velocity_recent_mps": 0.8},
        max_velocity_disagreement_mps=1.0, max_fit_rmse_m=0.2, min_history_count=6,
    )


def _feed(detector, speeds, dt=0.05, t0=0.0):
    flags = []
    for i, v in enumerate(speeds):
        flags.append(detector.update(t0 + i * dt, v))
    return flags


def test_speed_collapse_flags_drop_after_steady_motion():
    d = SpeedCollapseDetector(ratio=0.5)
    flags = _feed(d, [0.7] * 30 + [0.3, 0.2])
    assert not any(flags[:30])
    assert flags[30] and flags[31]


def test_speed_collapse_ignores_steady_and_slightly_slower_speed():
    d = SpeedCollapseDetector(ratio=0.5)
    flags = _feed(d, [0.7] * 30 + [0.5, 0.45, 0.7])
    assert not any(flags)


def test_speed_collapse_needs_history_and_meaningful_reference():
    assert not any(_feed(SpeedCollapseDetector(ratio=0.5), [0.7] * 5 + [0.0]))  # < min_samples
    assert not any(_feed(SpeedCollapseDetector(ratio=0.5), [0.05] * 30 + [0.0]))  # stationary reference


def test_speed_collapse_window_forgets_old_speeds():
    d = SpeedCollapseDetector(ratio=0.5, window_s=1.0)
    flags = _feed(d, [0.8] * 10 + [0.1] * 60)  # slow for much longer than the window
    assert not flags[-1]


def test_speed_collapse_disabled_and_missing_data():
    assert not any(_feed(SpeedCollapseDetector(ratio=0.0), [0.7] * 30 + [0.0]))
    d = SpeedCollapseDetector(ratio=0.5)
    assert not d.update(0.0, None) and not d.update(0.1, float("nan"))


def test_speed_collapse_resets_on_clock_jump():
    d = SpeedCollapseDetector(ratio=0.5)
    _feed(d, [0.8] * 30, t0=100.0)
    assert not d.update(0.0, 0.1)  # time went backwards: history discarded, too few samples
