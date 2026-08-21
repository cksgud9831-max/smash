import math

import numpy as np
import pytest

from aiming_engine.config import ProjectileConfig
from aiming_engine.forces import GravityForce, build_forces
from aiming_engine.projectile_model import ProjectileModel, azimuth_elevation_to_direction
from aiming_engine.types import Vector3


def make_config(muzzle_velocity=100.0, gravity=9.81, integrator_step=0.01, forces=None) -> ProjectileConfig:
    return ProjectileConfig(
        muzzle_velocity=muzzle_velocity,
        forces=forces if forces is not None else ["gravity"],
        gravity=gravity,
        integrator_step=integrator_step,
    )


def test_gravity_force_direction_and_magnitude():
    g = GravityForce(9.81)
    a = g.acceleration(np.zeros(3), np.zeros(3), 0.0)
    assert a == pytest.approx([0.0, 0.0, -9.81])


def test_build_forces_unknown_raises():
    config = make_config(forces=["unobtainium"])
    with pytest.raises(ValueError):
        build_forces(config)


def test_azimuth_elevation_direction_axes():
    d_forward = azimuth_elevation_to_direction(0.0, 0.0)
    assert d_forward == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
    d_up = azimuth_elevation_to_direction(0.0, math.pi / 2)
    assert d_up == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
    d_side = azimuth_elevation_to_direction(math.pi / 2, 0.0)
    assert d_side == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


def test_state_at_zero_time_returns_launch_state():
    model = ProjectileModel(make_config())
    launch = Vector3(1.0, 2.0, 3.0)
    pos, vel = model.state_at(0.0, launch, azimuth=0.3, elevation=0.1)
    assert pos.to_array() == pytest.approx(launch.to_array())
    assert vel.norm() == pytest.approx(100.0)


def test_state_at_negative_time_raises():
    model = ProjectileModel(make_config())
    with pytest.raises(ValueError):
        model.state_at(-1.0, Vector3(0, 0, 0), 0.0, 0.0)


@pytest.mark.parametrize("t,az,el", [(1.0, 0.0, 0.0), (2.5, 0.7, 0.3), (0.05, -1.2, 0.9)])
def test_state_at_matches_closed_form_gravity_only(t, az, el):
    v0_mag = 200.0
    g = 9.81
    model = ProjectileModel(make_config(muzzle_velocity=v0_mag, gravity=g))
    launch = Vector3(0.0, 0.0, 0.0)

    pos, vel = model.state_at(t, launch, azimuth=az, elevation=el)

    v0 = azimuth_elevation_to_direction(az, el) * v0_mag
    expected_pos = v0 * t + np.array([0.0, 0.0, -0.5 * g * t * t])
    expected_vel = v0 + np.array([0.0, 0.0, -g * t])

    assert pos.to_array() == pytest.approx(expected_pos, abs=1e-6)
    assert vel.to_array() == pytest.approx(expected_vel, abs=1e-6)
