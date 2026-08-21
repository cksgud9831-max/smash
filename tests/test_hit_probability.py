import pytest

from aiming_engine.config import HitProbabilityConfig
from aiming_engine.hit_probability import HitProbability


def make_hp(weights=None) -> HitProbability:
    return HitProbability(
        HitProbabilityConfig(
            weights=weights
            or {
                "tracking_confidence": 0.3,
                "follower_stability": 0.2,
                "aim_error": 0.3,
                "distance": 0.1,
                "velocity": 0.1,
            },
            distance_scale_m=500.0,
            velocity_scale_mps=30.0,
            aim_error_scale_mrad=5.0,
        )
    )


def test_perfect_conditions_give_near_max_probability():
    hp = make_hp()
    p = hp.estimate(tracking_confidence=1.0, follower_stable=True, aim_error_mrad=0.0, distance_m=0.0, target_speed=0.0)
    assert p == pytest.approx(1.0, abs=1e-9)


def test_probability_always_in_unit_range():
    hp = make_hp()
    p = hp.estimate(tracking_confidence=0.0, follower_stable=False, aim_error_mrad=1000.0, distance_m=5000.0, target_speed=500.0)
    assert 0.0 <= p <= 1.0


def test_probability_strictly_decreases_with_aim_error():
    hp = make_hp()
    base = dict(tracking_confidence=0.9, follower_stable=True, distance_m=200.0, target_speed=10.0)
    p_low = hp.estimate(aim_error_mrad=1.0, **base)
    p_high = hp.estimate(aim_error_mrad=10.0, **base)
    assert p_high < p_low


def test_probability_strictly_decreases_with_distance():
    hp = make_hp()
    base = dict(tracking_confidence=0.9, follower_stable=True, aim_error_mrad=1.0, target_speed=10.0)
    p_near = hp.estimate(distance_m=50.0, **base)
    p_far = hp.estimate(distance_m=600.0, **base)
    assert p_far < p_near


def test_probability_strictly_decreases_with_target_speed():
    hp = make_hp()
    base = dict(tracking_confidence=0.9, follower_stable=True, aim_error_mrad=1.0, distance_m=200.0)
    p_slow = hp.estimate(target_speed=1.0, **base)
    p_fast = hp.estimate(target_speed=50.0, **base)
    assert p_fast < p_slow


def test_probability_increases_with_tracking_confidence():
    hp = make_hp()
    base = dict(follower_stable=True, aim_error_mrad=1.0, distance_m=200.0, target_speed=10.0)
    p_low_conf = hp.estimate(tracking_confidence=0.2, **base)
    p_high_conf = hp.estimate(tracking_confidence=0.95, **base)
    assert p_high_conf > p_low_conf


def test_unstable_follower_lowers_probability():
    hp = make_hp()
    base = dict(tracking_confidence=0.9, aim_error_mrad=1.0, distance_m=200.0, target_speed=10.0)
    p_stable = hp.estimate(follower_stable=True, **base)
    p_unstable = hp.estimate(follower_stable=False, **base)
    assert p_unstable < p_stable
