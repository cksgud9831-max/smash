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


def _reset_config(**kw) -> TargetStateConfig:
    return TargetStateConfig(estimator="constant_velocity", history_length=20, min_dt=1e-3, maneuver_reset=True, **kw)


def _run_reversal(config, noise=0.0, seed=0, range_x=33.0):
    """5 m/s 로 가다가 t=2 s 에 방향 전환. 전환 뒤 0.2~0.6 s 의 0.08 s 앞 예측 오차 최대치를 돌려준다."""
    import numpy as np

    rng = np.random.default_rng(seed)
    y = lambda t: 5.0 * t if t < 2.0 else 10.0 - 5.0 * (t - 2.0)  # noqa: E731
    ts = TargetState(config)
    worst = 0.0
    for i in range(60):  # 20 Hz
        t = i * 0.05
        ts.update(Vector3(range_x, y(t) + rng.normal(0, noise), 12.5), timestamp=t, observer=np.zeros(3))
        if 2.2 <= t <= 2.6:
            worst = max(worst, abs(ts.predict(t + 0.08).position.y - y(t + 0.08)))
    return worst, ts._estimator


def test_maneuver_reset_recovers_faster_after_reversal():
    base, _ = _run_reversal(make_config(history_length=20))
    reset, est = _run_reversal(_reset_config())
    assert est.maneuver_resets >= 1
    assert reset < 0.5 * base


def test_maneuver_reset_detects_slow_reversal_before_position_gate_at_35m():
    """3 km/h, ~7 Hz samples cannot quickly cross the 0.35 m angular gate."""
    import numpy as np

    speed = 3.0 / 3.6
    dt = 1.0 / 7.0
    reverse_at = 2.0
    ts = TargetState(_reset_config())

    for i in range(25):
        t = i * dt
        y = speed * t if t <= reverse_at else speed * reverse_at - speed * (t - reverse_at)
        ts.update(Vector3(33.0, y, 12.5), timestamp=t, observer=np.zeros(3))

    assert ts._estimator.maneuver_resets >= 1
    estimate = ts.predict(25 * dt).velocity.y
    assert estimate < -0.7


def test_slow_constant_velocity_noise_does_not_look_like_a_reversal():
    import numpy as np

    speed = 3.0 / 3.6
    dt = 1.0 / 7.0
    rng = np.random.default_rng(7)
    ts = TargetState(_reset_config())
    for i in range(200):
        t = i * dt
        ts.update(
            Vector3(33.0, speed * t + rng.normal(0, 0.03), 12.5 + rng.normal(0, 0.03)),
            timestamp=t,
            observer=np.zeros(3),
        )
    assert ts._estimator.maneuver_resets == 0


def test_maneuver_reset_does_not_fire_on_noisy_constant_velocity():
    import numpy as np

    ts = TargetState(_reset_config())
    rng = np.random.default_rng(1)
    for i in range(200):  # 35 m 에서 측정 잡음 3 cm (정지·등속 흔들림 실측 수준)
        t = i * 0.05
        ts.update(Vector3(33.0, 5.0 * t + rng.normal(0, 0.03), 12.5 + rng.normal(0, 0.03)), timestamp=t,
                  observer=np.zeros(3))
    assert ts._estimator.maneuver_resets == 0


def test_maneuver_gate_scales_with_range():
    import numpy as np

    # 100 m 에서는 문턱이 1.0 m 라 0.3 m 잡음에 걸리지 않지만, 10 m 에서는(0.1 m) 걸린다.
    far = TargetState(_reset_config())
    near = TargetState(_reset_config())
    rng = np.random.default_rng(2)
    for i in range(100):
        t = i * 0.05
        n = rng.normal(0, 0.3)
        far.update(Vector3(100.0, n, 0.0), timestamp=t, observer=np.zeros(3))
        near.update(Vector3(10.0, n, 0.0), timestamp=t, observer=np.zeros(3))
    assert far._estimator.maneuver_resets == 0
    assert near._estimator.maneuver_resets > 0


def test_maneuver_reset_off_by_default_matches_previous_behavior():
    a, _ = _run_reversal(make_config(history_length=20))
    b, est = _run_reversal(TargetStateConfig(estimator="constant_velocity", history_length=20, min_dt=1e-3))
    assert a == b and est.maneuver_resets == 0


def test_lateral_velocity_diagnostics_disagree_in_sign_just_after_reversal():
    """Without maneuver reset the long fit still points the old way while the recent fit has flipped."""
    import numpy as np

    speed = 0.83
    dt = 1.0 / 20.0
    reverse_at = 2.0
    ts = TargetState(make_config(history_length=20))

    for i in range(int(2.45 / dt)):
        t = i * dt
        y = speed * t if t <= reverse_at else speed * reverse_at - speed * (t - reverse_at)
        ts.update(Vector3(33.0, y, 12.5), timestamp=t, observer=np.zeros(3))
    ts.predict(2.45)
    diag = ts._estimator.diagnostics()

    assert diag["lateral_velocity_long_mps"] > 0.15
    assert diag["lateral_velocity_recent_mps"] < -0.15


def test_lateral_velocity_diagnostics_agree_on_constant_motion():
    import numpy as np

    ts = TargetState(make_config(history_length=20))
    for i in range(30):
        t = i * 0.05
        ts.update(Vector3(33.0, -0.83 * t, 12.5), timestamp=t, observer=np.zeros(3))
    ts.predict(1.5)
    diag = ts._estimator.diagnostics()

    assert diag["lateral_velocity_long_mps"] == pytest.approx(diag["lateral_velocity_recent_mps"], abs=1e-6)
    assert abs(diag["lateral_velocity_long_mps"]) == pytest.approx(0.83, abs=1e-3)
