import math

import pytest

from aiming_engine.lead_prediction import LeadPrediction
from aiming_engine.types import TargetVector, Vector3


def test_initial_guess_stationary_target_straight_ahead():
    target = TargetVector(
        position=Vector3(100.0, 0.0, 0.0),
        velocity=Vector3(0.0, 0.0, 0.0),
        acceleration=Vector3(0.0, 0.0, 0.0),
        timestamp=0.0,
    )
    guess = LeadPrediction().initial_guess(target, launch_point=Vector3(0.0, 0.0, 0.0), muzzle_velocity=100.0)

    assert guess.t0 == pytest.approx(1.0)
    assert guess.azimuth0 == pytest.approx(0.0, abs=1e-9)
    assert guess.elevation0 == pytest.approx(0.0, abs=1e-9)
    assert guess.naive_lead_point.to_array() == pytest.approx([100.0, 0.0, 0.0])


def test_initial_guess_leads_moving_target():
    target = TargetVector(
        position=Vector3(100.0, 0.0, 0.0),
        velocity=Vector3(0.0, 50.0, 0.0),
        acceleration=Vector3(0.0, 0.0, 0.0),
        timestamp=0.0,
    )
    guess = LeadPrediction().initial_guess(target, launch_point=Vector3(0.0, 0.0, 0.0), muzzle_velocity=100.0)

    # t0 based on current range (100/100=1s), target moves 50m in Y by then
    assert guess.t0 == pytest.approx(1.0)
    assert guess.naive_lead_point.to_array() == pytest.approx([100.0, 50.0, 0.0])
    assert guess.azimuth0 == pytest.approx(math.atan2(50.0, 100.0), abs=1e-9)


def test_initial_guess_rejects_nonpositive_muzzle_velocity():
    target = TargetVector(Vector3(1, 0, 0), Vector3(0, 0, 0), Vector3(0, 0, 0), timestamp=0.0)
    with pytest.raises(ValueError):
        LeadPrediction().initial_guess(target, launch_point=Vector3(0, 0, 0), muzzle_velocity=0.0)
