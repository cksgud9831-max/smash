import pytest

from aiming_engine.config import AimReadinessConfig
from aiming_engine.aim_readiness_logic import AimReadinessLogic


def make_readiness_logic() -> AimReadinessLogic:
    return AimReadinessLogic(
        AimReadinessConfig(
            hit_probability_threshold=0.75,
            max_aim_error_mrad=5.0,
            min_valid_range_m=5.0,
            max_valid_range_m=800.0,
        )
    )


def good_args(**overrides):
    args = dict(
        tracking_stable=True,
        follower_stable=True,
        hit_probability=0.9,
        distance_m=200.0,
        aim_error_mrad=1.0,
    )
    args.update(overrides)
    return args


@pytest.mark.parametrize(
    "overrides",
    [
        {},  # baseline: should be aim-ready
    ],
)
def test_aim_ready_when_all_conditions_met(overrides):
    logic = make_readiness_logic()
    assert logic.evaluate(**good_args(**overrides)) is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"tracking_stable": False},
        {"follower_stable": False},
        {"hit_probability": 0.75},  # boundary: not strictly greater -> not ready
        {"hit_probability": 0.5},
        {"distance_m": 4.99},  # below min
        {"distance_m": 800.01},  # above max
        {"aim_error_mrad": 5.0},  # boundary: not strictly less -> not ready
        {"aim_error_mrad": 10.0},
    ],
)
def test_not_aim_ready_when_any_condition_fails(overrides):
    logic = make_readiness_logic()
    assert logic.evaluate(**good_args(**overrides)) is False


def test_boundary_values_at_valid_range_limits_pass():
    logic = make_readiness_logic()
    assert logic.evaluate(**good_args(distance_m=5.0)) is True
    assert logic.evaluate(**good_args(distance_m=800.0)) is True
