import numpy as np
import pytest

from aiming_engine.config import ProjectileConfig, TargetStateConfig
from aiming_engine.hit_equation import HitEquation
from aiming_engine.projectile_model import ProjectileModel
from aiming_engine.target_state import TargetState
from aiming_engine.types import Vector3


def make_projectile_config(muzzle_velocity=100.0, forces=None) -> ProjectileConfig:
    return ProjectileConfig(
        muzzle_velocity=muzzle_velocity,
        forces=forces if forces is not None else ["gravity"],
        gravity=9.81,
        integrator_step=0.01,
    )


def make_target_state(position: Vector3, timestamp: float) -> TargetState:
    ts = TargetState(TargetStateConfig(estimator="constant_velocity", history_length=10, min_dt=1e-3))
    ts.update(position, timestamp)
    return ts


def test_residual_zero_for_trivial_direct_shot_no_gravity():
    # No gravity -> straight-line shot at a stationary target is an exact intercept.
    muzzle_velocity = 100.0
    model = ProjectileModel(make_projectile_config(muzzle_velocity, forces=[]))
    target = make_target_state(Vector3(100.0, 0.0, 0.0), timestamp=0.0)
    equation = HitEquation(target, model, launch_point=Vector3(0.0, 0.0, 0.0), reference_time=0.0)

    t_exact = 100.0 / muzzle_velocity
    residual = equation.residual(np.array([t_exact, 0.0, 0.0]))

    assert residual == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_residual_at_zero_time_is_launch_minus_target():
    model = ProjectileModel(make_projectile_config())
    launch = Vector3(1.0, 2.0, 3.0)
    target = make_target_state(Vector3(5.0, 5.0, 5.0), timestamp=10.0)
    equation = HitEquation(target, model, launch_point=launch, reference_time=10.0)

    residual = equation.residual(np.array([0.0, 0.4, 0.1]))

    expected = launch.to_array() - np.array([5.0, 5.0, 5.0])
    assert residual == pytest.approx(expected, abs=1e-9)


def test_residual_clamps_negative_flight_time_to_zero():
    model = ProjectileModel(make_projectile_config())
    launch = Vector3(1.0, 2.0, 3.0)
    target = make_target_state(Vector3(5.0, 5.0, 5.0), timestamp=10.0)
    equation = HitEquation(target, model, launch_point=launch, reference_time=10.0)

    r_negative = equation.residual(np.array([-1.0, 0.4, 0.1]))
    r_zero = equation.residual(np.array([0.0, 0.4, 0.1]))

    assert r_negative == pytest.approx(r_zero, abs=1e-9)


def test_residual_nonzero_for_moving_target_without_lead():
    model = ProjectileModel(make_projectile_config(forces=[]))
    ts = TargetState(TargetStateConfig(estimator="constant_velocity", history_length=10, min_dt=1e-3))
    ts.update(Vector3(100.0, 0.0, 0.0), timestamp=0.0)
    ts.update(Vector3(110.0, 0.0, 0.0), timestamp=0.1)  # moving at 100 m/s along +X
    equation = HitEquation(ts, model, launch_point=Vector3(0.0, 0.0, 0.0), reference_time=0.1)

    t_naive = 110.0 / 100.0  # aims at *current* position, ignoring lead
    residual = equation.residual(np.array([t_naive, 0.0, 0.0]))

    assert np.linalg.norm(residual) > 1.0  # target has moved on by the time the shot arrives
