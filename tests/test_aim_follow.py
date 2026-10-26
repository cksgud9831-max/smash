"""Gazebo operator aim-assist conversion tests (ROS-independent)."""

import math
import os
import sys

import pytest

_SIM_PKG = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, "simulation",
    "ciws_turret_aerial_object_detection-main", "drone_sim",
))
sys.path.insert(0, _SIM_PKG)

from drone_sim.smash_fcs.aim_follow import (  # noqa: E402
    AimTrackFollower,
    GoalRamp,
    goal_from_aim_track,
    goal_within_acquisition_envelope,
)


def _goal(direction, reliability="STABLE", reference=0.0):
    return goal_from_aim_track(
        {"lead_dir_base": direction, "reliability": reliability},
        reference_pan=reference,
        tilt_min=0.0,
        tilt_max=1.48,
    )


def test_stable_lead_direction_becomes_pan_tilt_goal():
    pan, tilt = _goal([1.0, 1.0, 1.0])
    assert pan == pytest.approx(math.pi / 4)
    assert tilt == pytest.approx(math.atan2(1.0, math.sqrt(2.0)))


@pytest.mark.parametrize("reliability", ["HOLD", "LOST", ""])
def test_untrusted_track_does_not_move_goal(reliability):
    assert _goal([1.0, 0.0, 0.2], reliability=reliability) is None


@pytest.mark.parametrize("direction", [None, [], [0, 0, 0], [1, 2], [1, 2, float("nan")]])
def test_invalid_direction_is_rejected(direction):
    assert _goal(direction) is None


def test_pan_is_unwrapped_near_existing_continuous_joint_goal():
    pan, _ = _goal([-1.0, -0.01, 0.0], reference=math.pi - 0.01)
    assert pan > 3.0
    assert abs(pan - (math.pi + math.atan(0.01))) < 1e-3


def test_tilt_is_clamped_to_joint_limits():
    _, tilt = _goal([0.01, 0.0, 1.0])
    assert tilt == pytest.approx(1.48)


def test_relative_lead_minus_bore_cancels_stale_absolute_tf_rotation():
    track = {
        "lead_dir_base": [math.cos(0.12), math.sin(0.12), math.tan(0.36)],
        "bore_dir_base": [math.cos(0.10), math.sin(0.10), math.tan(0.34)],
        "reliability": "STABLE",
    }
    pan, tilt = goal_from_aim_track(
        track,
        reference_pan=0.0,
        reference_tilt=0.34,
        tilt_min=0.0,
        tilt_max=1.48,
    )
    assert pan == pytest.approx(0.02)
    assert tilt == pytest.approx(0.36)


def _track(pan, stamp, reliability="STABLE"):
    return {
        "lead_dir_base": [math.cos(pan), math.sin(pan), 0.0],
        "reliability": reliability,
        "stamp": stamp,
    }


def test_feedforward_removes_measured_constant_rate_delay():
    delay = 0.19
    rate = 0.024  # about 3 km/h transverse motion at 35 m
    follower = AimTrackFollower(
        lead_time_s=delay,
        max_rate_rad_s=0.5,
        tilt_min=0.0,
        tilt_max=1.48,
    )
    follower.update(_track(-rate * delay, 0.0), reference_pan=0.0)
    pan, _ = follower.update(_track(rate * (1.0 - delay), 1.0), reference_pan=0.0)
    assert pan == pytest.approx(rate, abs=1e-9)


def test_feedforward_rejects_measurement_jumps_instead_of_turning_them_into_lead():
    follower = AimTrackFollower(
        lead_time_s=0.19,
        max_rate_rad_s=0.5,
        tilt_min=0.0,
        tilt_max=1.48,
    )
    follower.update(_track(0.0, 0.0), reference_pan=0.0)
    pan, _ = follower.update(_track(1.0, 0.1), reference_pan=0.0)
    assert pan == pytest.approx(1.0)


def test_feedforward_result_is_saturated_by_final_goal_jump_guard():
    follower = AimTrackFollower(
        lead_time_s=0.19,
        max_rate_rad_s=0.5,
        tilt_min=0.0,
        tilt_max=1.48,
        max_goal_jump_rad=0.012,
    )
    follower.update(_track(0.0, 0.0), reference_pan=0.0)
    # Raw 0.01 rad is allowed, but its 0.1 rad/s derivative adds 0.019 rad.
    pan, _ = follower.update(_track(0.01, 0.1), reference_pan=0.0)
    assert pan == pytest.approx(0.012)


def test_feedforward_uses_the_new_direction_after_a_real_reversal():
    delay = 0.19
    rate = 0.024
    follower = AimTrackFollower(
        lead_time_s=delay,
        max_rate_rad_s=0.5,
        tilt_min=0.0,
        tilt_max=1.48,
    )
    follower.update(_track(rate, 0.0), reference_pan=rate)
    pan, _ = follower.update(_track(0.0, 1.0), reference_pan=0.0)
    assert pan == pytest.approx(-rate * delay)


def test_trust_loss_resets_feedforward_history():
    follower = AimTrackFollower(
        lead_time_s=0.19,
        max_rate_rad_s=0.5,
        tilt_min=0.0,
        tilt_max=1.48,
    )
    follower.update(_track(0.0, 0.0), reference_pan=0.0)
    assert follower.update(_track(0.1, 1.0, reliability="HOLD"), reference_pan=0.0) is None
    pan, _ = follower.update(_track(0.2, 2.0), reference_pan=0.0)
    assert pan == pytest.approx(0.2)


def test_missing_sim_stamp_uses_raw_goal_without_extrapolation():
    follower = AimTrackFollower(
        lead_time_s=0.19,
        max_rate_rad_s=0.5,
        tilt_min=0.0,
        tilt_max=1.48,
    )
    pan, _ = follower.update(
        {"lead_dir_base": [math.cos(0.1), math.sin(0.1), 0.0], "reliability": "STABLE"},
        reference_pan=0.0,
    )
    assert pan == pytest.approx(0.1)


def test_follower_bounds_single_goal_jump_that_would_leave_narrow_fov():
    follower = AimTrackFollower(
        lead_time_s=0.19,
        max_rate_rad_s=0.5,
        tilt_min=0.0,
        tilt_max=1.5,
        max_goal_jump_rad=0.08,
    )
    track = {
        "reliability": "STABLE",
        "stamp": 1.0,
        "lead_dir_base": [math.cos(0.106), -math.sin(0.106), math.tan(0.344)],
    }

    pan, _ = follower.update(track, reference_pan=0.0, reference_tilt=0.344)
    assert pan == pytest.approx(-0.08)


def test_follower_catches_up_after_real_target_exceeds_single_step_limit():
    follower = AimTrackFollower(
        lead_time_s=0.0,
        max_rate_rad_s=0.5,
        tilt_min=0.0,
        tilt_max=1.5,
        max_goal_jump_rad=0.012,
    )
    goal = 0.0
    track = _track(0.036, 1.0)
    for stamp in (1.0, 1.1, 1.2):
        track["stamp"] = stamp
        goal, _ = follower.update(track, reference_pan=goal)
    assert goal == pytest.approx(0.036)


def test_follower_accepts_normal_small_tracking_correction_with_jump_guard():
    follower = AimTrackFollower(
        lead_time_s=0.19,
        max_rate_rad_s=0.5,
        tilt_min=0.0,
        tilt_max=1.5,
        max_goal_jump_rad=0.08,
    )
    track = {
        "reliability": "STABLE",
        "stamp": 1.0,
        "lead_dir_base": [math.cos(0.02), math.sin(0.02), math.tan(0.35)],
    }

    goal = follower.update(track, reference_pan=0.0, reference_tilt=0.344)
    assert goal is not None
    assert goal[0] == pytest.approx(0.02)


def test_absolute_goal_path_still_checks_tilt_jump_against_joint_goal():
    follower = AimTrackFollower(
        lead_time_s=0.19,
        max_rate_rad_s=0.5,
        tilt_min=0.0,
        tilt_max=1.5,
        max_goal_jump_rad=0.02,
    )
    track = {
        "reliability": "STABLE",
        "stamp": 1.0,
        "lead_dir_base": [math.cos(0.01), math.sin(0.01), math.tan(0.38)],
    }
    pan, tilt = follower.update(
        track,
        reference_pan=0.0,
        reference_tilt=None,
        jump_reference_tilt=0.34,
    )
    assert pan == pytest.approx(0.01)
    assert tilt == pytest.approx(0.36)


def test_unconfirmed_goal_cannot_walk_out_of_acquisition_envelope():
    anchor = (0.0, 0.344)
    assert goal_within_acquisition_envelope((0.079, 0.40), anchor, 0.08)
    assert not goal_within_acquisition_envelope((0.081, 0.40), anchor, 0.08)
    assert not goal_within_acquisition_envelope((0.02, 0.425), anchor, 0.08)


def test_follower_exposes_measured_rate_and_tilt_lead_is_separate():
    follower = AimTrackFollower(
        lead_time_s=0.19,
        tilt_lead_time_s=0.0,
        max_rate_rad_s=0.5,
        tilt_min=0.0,
        tilt_max=1.48,
    )
    first = {"lead_dir_base": [math.cos(0.0), math.sin(0.0), math.tan(0.34)],
             "reliability": "STABLE", "stamp": 0.0}
    second = {"lead_dir_base": [math.cos(0.01), math.sin(0.01), math.tan(0.35)],
              "reliability": "STABLE", "stamp": 0.5}
    follower.update(first, reference_pan=0.0)
    assert not follower.rate_valid
    pan, tilt = follower.update(second, reference_pan=0.0)
    assert follower.rate_valid
    assert follower.rate[0] == pytest.approx(0.02, rel=0.02)
    assert follower.rate[1] != 0.0
    assert pan == pytest.approx(0.01 + 0.02 * 0.19, rel=0.02)
    # 틸트 리드가 0 이면 외삽 없이 측정 틸트 그대로다.
    assert tilt == pytest.approx(0.35, abs=1e-3)


def test_follower_reset_clears_exposed_rate():
    follower = AimTrackFollower(lead_time_s=0.19, max_rate_rad_s=0.5, tilt_min=0.0, tilt_max=1.48)
    follower.update(_track(0.0, 0.0), reference_pan=0.0)
    follower.update(_track(0.01, 0.5), reference_pan=0.0)
    assert follower.rate_valid
    follower.update(_track(0.02, 1.0, reliability="HOLD"), reference_pan=0.0)
    assert not follower.rate_valid
    assert follower.rate == (0.0, 0.0)


def _ramp(**kw):
    args = dict(horizon_s=1.0, smoothing=0.0, tilt_min=0.0, tilt_max=1.48)
    args.update(kw)
    return GoalRamp(**args)


def test_ramp_continues_last_rate_between_updates_and_caps_at_horizon():
    ramp = _ramp()
    assert ramp.at(0.0) is None
    ramp.set((0.10, 0.34), (0.02, 0.0), True, 10.0)
    pan, tilt = ramp.at(10.3)
    assert pan == pytest.approx(0.10 + 0.02 * 0.3)
    assert tilt == pytest.approx(0.34)
    pan, _ = ramp.at(20.0)
    assert pan == pytest.approx(0.10 + 0.02 * 1.0)  # horizon cap, not unbounded
    pan, _ = ramp.at(9.0)  # clock going backwards never extrapolates backwards
    assert pan == pytest.approx(0.10)


def test_ramp_ignores_tilt_rate_unless_enabled_and_clamps_tilt():
    ramp = _ramp()
    ramp.set((0.0, 0.34), (0.0, 0.05), True, 0.0)
    assert ramp.at(1.0)[1] == pytest.approx(0.34)
    ramp = _ramp(tilt_ramp=True)
    ramp.set((0.0, 1.47), (0.0, 0.05), True, 0.0)
    assert ramp.at(1.0)[1] == pytest.approx(1.48)


def test_ramp_smooths_rate_and_resets_on_invalid_rate_and_clear():
    ramp = _ramp(smoothing=0.5)
    ramp.set((0.0, 0.3), (0.04, 0.0), True, 0.0)
    ramp.set((0.0, 0.3), (0.00, 0.0), True, 1.0)
    assert ramp.at(2.0)[0] == pytest.approx(0.02)  # EMA of 0.04 and 0.0
    ramp.set((0.0, 0.3), (0.04, 0.0), False, 2.0)
    assert ramp.at(3.0)[0] == pytest.approx(0.0)  # invalid rate: no extrapolation
    ramp.set((0.0, 0.3), (0.04, 0.0), True, 3.0)
    ramp.clear()
    assert ramp.at(4.0) is None


def test_ramp_with_zero_horizon_is_disabled():
    ramp = _ramp(horizon_s=0.0)
    ramp.set((0.0, 0.3), (0.04, 0.0), True, 0.0)
    assert ramp.at(1.0) is None


# ── 신규 접촉 지시(큐) ────────────────────────────────────────────────

from drone_sim.smash_fcs.aim_follow import cue_goal_for_point  # noqa: E402


def _cue(point, reference_pan=0.0):
    return cue_goal_for_point(point, reference_pan=reference_pan, tilt_min=0.0, tilt_max=1.48)


def test_cue_points_left_up_for_left_high_contact():
    # base +y 가 왼쪽. 20 m 전방, 3 m 좌측, 7.4 m 위 -> pan 양수, tilt 양수
    pan, tilt = _cue([20.0, 3.0, 7.4])
    assert pan == pytest.approx(math.atan2(3.0, 20.0), abs=0.03)
    assert tilt == pytest.approx(math.atan2(7.4, math.hypot(20.0, 3.0)), abs=0.06)


def test_cue_right_contact_gives_negative_pan():
    pan, _ = _cue([15.0, -2.5, 5.6])
    assert pan < 0.0


def test_cue_pan_is_unwrapped_next_to_reference():
    # 기준이 +pi 근처일 때 -pi 쪽 해가 아니라 연속인 해를 돌려준다
    pan, _ = _cue([-10.0, -0.5, 3.0], reference_pan=3.1)
    assert abs(pan - 3.1) < 0.5


def test_cue_tilt_is_clamped_to_joint_limits():
    _, tilt = _cue([1.0, 0.0, 50.0])
    assert tilt == pytest.approx(1.48)


@pytest.mark.parametrize("point", [[0.0, 0.0, 5.0], [float("nan"), 1.0, 1.0], [1.0, 2.0]])
def test_cue_rejects_unusable_points(point):
    assert _cue(point) is None


def _dir(pan, tilt=0.3):
    c = math.cos(tilt)
    return [c * math.cos(pan), c * math.sin(pan), math.sin(tilt)]


def _bbox_track(lead_pan, bbox_pan, stamp):
    return {
        "reliability": "STABLE",
        "lead_dir_base": _dir(lead_pan),
        "bbox_dir_base": _dir(bbox_pan),
        "stamp": stamp,
    }


def _follower(mode, **kw):
    return AimTrackFollower(
        lead_time_s=0.0, max_rate_rad_s=0.5, tilt_min=0.0, tilt_max=1.48, follow_mode=mode, **kw
    )


def test_lead_mode_ignores_bbox_direction():
    pan, _ = _follower("lead").update(_bbox_track(0.02, 0.0, 0.0), reference_pan=0.0)
    assert pan == pytest.approx(0.02)


def test_bbox_mode_without_smoothing_equals_lead_goal():
    pan, _ = _follower("bbox").update(_bbox_track(0.02, 0.0, 0.0), reference_pan=0.0)
    assert pan == pytest.approx(0.02)


def test_bbox_mode_smooths_noisy_lead_offset_but_keeps_bbox_position():
    follower = _follower("bbox", offset_smoothing=0.8)
    follower.update(_bbox_track(0.010, 0.0, 0.0), reference_pan=0.0)  # offset 0.010
    pan, _ = follower.update(_bbox_track(0.060, 0.0, 1.0), reference_pan=0.0)  # lead spike
    # raw lead says 0.060; smoothed offset is 0.8*0.010 + 0.2*0.060 = 0.020 on top of the bbox
    assert pan == pytest.approx(0.020)


def test_bbox_mode_follows_bbox_position_when_lead_offset_is_steady():
    follower = _follower("bbox", offset_smoothing=0.8)
    follower.update(_bbox_track(0.005 + 0.01, 0.005, 0.0), reference_pan=0.0)
    pan, _ = follower.update(_bbox_track(0.011 + 0.01, 0.011, 1.0), reference_pan=0.0)
    assert pan == pytest.approx(0.021)


def test_bbox_mode_falls_back_to_lead_when_bbox_direction_missing():
    track = _bbox_track(0.02, 0.0, 0.0)
    del track["bbox_dir_base"]
    pan, _ = _follower("bbox", offset_smoothing=0.8).update(track, reference_pan=0.0)
    assert pan == pytest.approx(0.02)


def test_rate_smoothing_damps_a_rate_spike_and_default_is_raw():
    lead_time = 0.5
    def run(rate_smoothing):
        f = AimTrackFollower(
            lead_time_s=lead_time, max_rate_rad_s=0.5, tilt_min=0.0, tilt_max=1.48,
            rate_smoothing=rate_smoothing,
        )
        f.update(_track(0.0, 0.0), reference_pan=0.0)
        f.update(_track(0.01, 1.0), reference_pan=0.0)  # 0.01 rad/s
        pan, _ = f.update(_track(0.01 + 0.04, 2.0), reference_pan=0.0)  # 0.04 rad/s spike
        return pan - 0.05
    assert run(0.0) == pytest.approx(0.04 * lead_time)
    assert run(0.7) == pytest.approx((0.7 * 0.01 + 0.3 * 0.04) * lead_time)
