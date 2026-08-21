import pytest

from aiming_engine.config import TargetStateConfig
from aiming_engine.target_state import ConstantVelocityEstimator, TargetState
from aiming_engine.types import Vector3


def make_config(history_length=10, min_dt=1e-3) -> TargetStateConfig:
    return TargetStateConfig(estimator="constant_velocity", history_length=history_length, min_dt=min_dt)


def test_single_update_gives_zero_velocity():
    ts = TargetState(make_config())
    result = ts.update(Vector3(1.0, 2.0, 3.0), timestamp=0.0)
    assert result.velocity.to_array() == pytest.approx([0.0, 0.0, 0.0])
    assert result.position.to_array() == pytest.approx([1.0, 2.0, 3.0])


def test_constant_velocity_sequence_recovers_exact_velocity():
    ts = TargetState(make_config())
    velocity = (5.0, -2.0, 0.5)
    for i in range(6):
        t = i * 0.1
        pos = Vector3(velocity[0] * t, velocity[1] * t, velocity[2] * t)
        result = ts.update(pos, timestamp=t)

    assert result.velocity.to_array() == pytest.approx(velocity, abs=1e-9)


def test_predict_extrapolates_linearly():
    ts = TargetState(make_config())
    velocity = (10.0, 0.0, 0.0)
    for i in range(3):
        t = i * 0.1
        ts.update(Vector3(velocity[0] * t, 0.0, 0.0), timestamp=t)

    future = ts.predict(t_future=1.0)
    assert future.position.x == pytest.approx(10.0, abs=1e-6)


def test_duplicate_timestamp_is_ignored():
    ts = TargetState(make_config(min_dt=1e-3))
    ts.update(Vector3(0.0, 0.0, 0.0), timestamp=0.0)
    ts.update(Vector3(1.0, 0.0, 0.0), timestamp=0.0)  # dt=0, should be ignored
    result = ts.update(Vector3(2.0, 0.0, 0.0), timestamp=0.1)
    # only samples at t=0 (pos=0) and t=0.1 (pos=2) should count -> velocity 20 m/s
    assert result.velocity.x == pytest.approx(20.0, abs=1e-6)


def test_predict_before_update_raises():
    estimator = ConstantVelocityEstimator(history_length=5, min_dt=1e-3)
    with pytest.raises(RuntimeError):
        estimator.predict(1.0)


def test_history_length_bounds_buffer():
    ts = TargetState(make_config(history_length=3))
    for i in range(10):
        ts.update(Vector3(float(i), 0.0, 0.0), timestamp=float(i))
    estimator = ts._estimator
    assert len(estimator._times) == 3


def test_constant_acceleration_kalman_estimator():
    config = TargetStateConfig(
        estimator="constant_acceleration",
        history_length=10,
        min_dt=1e-3,
        process_noise_std=0.1,
        measurement_noise_std=0.01,
    )
    ts = TargetState(config)

    a_x = 2.0
    for i in range(20):
        t = i * 0.05
        pos = Vector3(0.5 * a_x * (t**2), 0.0, 0.0)
        result = ts.update(pos, timestamp=t)

    future = ts.predict(t_future=1.5)
    assert future.acceleration.x == pytest.approx(a_x, abs=0.5)
    assert future.position.x > result.position.x
