import numpy as np
import pytest
from scipy.integrate import solve_ivp

from aiming_engine.config import ProjectileConfig
from aiming_engine.forces import DragForce, GravityForce, build_forces
from aiming_engine.projectile_model import ProjectileModel, azimuth_elevation_to_direction
from aiming_engine.types import Vector3


def make_drag_force(**overrides) -> DragForce:
    params = dict(
        drag_model="G1",
        mass_kg=0.01,  # 10 g
        diameter_m=0.0078,  # 7.8 mm
        air_density_kg_m3=1.225,
        speed_of_sound_mps=340.3,
    )
    params.update(overrides)
    return DragForce(**params)


def make_projectile_config(forces=None, **overrides) -> ProjectileConfig:
    params = dict(
        muzzle_velocity=800.0,
        forces=forces if forces is not None else ["gravity", "drag"],
        gravity=9.81,
        integrator_step=0.001,
        drag_model="G1",
        mass_kg=0.01,
        diameter_m=0.0078,
        air_density_kg_m3=1.225,
        speed_of_sound_mps=340.3,
    )
    params.update(overrides)
    return ProjectileConfig(**params)


# ── DragForce unit tests ─────────────────────────────────────────────────


def test_zero_velocity_gives_zero_drag():
    drag = make_drag_force()
    a = drag.acceleration(np.zeros(3), np.zeros(3), 0.0)
    assert a == pytest.approx([0.0, 0.0, 0.0])


def test_drag_opposes_velocity_direction():
    drag = make_drag_force()
    velocity = np.array([300.0, 0.0, 0.0])
    a = drag.acceleration(np.zeros(3), velocity, 0.0)
    assert a[0] < 0.0  # decelerates along +X
    assert a[1] == pytest.approx(0.0)
    assert a[2] == pytest.approx(0.0)


def test_drag_magnitude_grows_with_speed():
    drag = make_drag_force()
    a_slow = drag.acceleration(np.zeros(3), np.array([100.0, 0.0, 0.0]), 0.0)
    a_fast = drag.acceleration(np.zeros(3), np.array([700.0, 0.0, 0.0]), 0.0)
    assert np.linalg.norm(a_fast) > np.linalg.norm(a_slow)


def test_transonic_drag_coefficient_exceeds_subsonic_and_supersonic():
    drag = make_drag_force()
    cd_subsonic = drag._drag_coefficient(0.7)
    cd_transonic = drag._drag_coefficient(1.05)
    cd_supersonic = drag._drag_coefficient(3.0)
    assert cd_transonic > cd_subsonic
    assert cd_transonic > cd_supersonic


def test_unknown_drag_model_raises():
    with pytest.raises(ValueError):
        make_drag_force(drag_model="G_nonexistent")


@pytest.mark.parametrize("mass_kg,diameter_m", [(0.0, 0.0078), (0.01, 0.0), (-1.0, 0.0078)])
def test_non_positive_mass_or_diameter_raises(mass_kg, diameter_m):
    with pytest.raises(ValueError):
        make_drag_force(mass_kg=mass_kg, diameter_m=diameter_m)


def test_non_positive_atmosphere_raises():
    with pytest.raises(ValueError):
        make_drag_force(air_density_kg_m3=0.0)
    with pytest.raises(ValueError):
        make_drag_force(speed_of_sound_mps=0.0)


def test_build_forces_creates_gravity_and_drag():
    config = make_projectile_config(forces=["gravity", "drag"])
    forces = build_forces(config)
    assert len(forces) == 2
    assert isinstance(forces[0], GravityForce)
    assert isinstance(forces[1], DragForce)


# ── ProjectileModel integration: drag falls back to RK4 and reduces range ──


def test_projectile_model_falls_back_to_rk4_with_drag():
    model = ProjectileModel(make_projectile_config(forces=["gravity", "drag"]))
    assert model._gravity_vec is None  # analytic gravity-only shortcut must be disabled


def test_drag_reduces_range_compared_to_vacuum():
    launch = Vector3(0.0, 0.0, 0.0)
    t, az, el = 1.0, 0.0, 0.0

    vacuum_model = ProjectileModel(make_projectile_config(forces=["gravity"]))
    drag_model = ProjectileModel(make_projectile_config(forces=["gravity", "drag"]))

    pos_vacuum, _ = vacuum_model.state_at(t, launch, az, el)
    pos_drag, _ = drag_model.state_at(t, launch, az, el)

    # Drag only ever removes energy, so the drag-affected projectile can't
    # have traveled farther down-range than the vacuum trajectory.
    assert pos_drag.x < pos_vacuum.x


# ── RK4 step-size convergence, cross-checked against an independent ────────
# adaptive integrator (scipy solve_ivp/RK45) ────────────────────────────────
#
# There's no closed-form solution once drag is enabled (that's exactly why
# ProjectileModel falls back to RK4 -- see its module docstring), so unlike
# test_projectile_model.py's gravity-only tests, this can't cross-check
# against hand-derived algebra. What it CAN check: whether ProjectileModel's
# *fixed-step* RK4 at the step size actually shipped in
# config/aiming_engine.yaml (integrator_step: 0.01, whose own comment reads
# "tighten once drag is enabled" -- i.e. this was previously an open
# question, not yet verified) has actually converged, by comparing it
# against a *different* numerical method (scipy's adaptive-step RK45 at a
# tight tolerance) integrating the exact same ForceModel objects. Agreement
# between two independent numerical schemes is meaningful evidence of
# convergence; it does not by itself validate the drag physics/Cd tables
# against reality (see forces.py's G1_DRAG_TABLE/G7_DRAG_TABLE caveats).


def _solve_ivp_reference_position(config: ProjectileConfig, t: float, launch: Vector3, az: float, el: float) -> np.ndarray:
    forces = build_forces(config)
    v0 = config.muzzle_velocity * azimuth_elevation_to_direction(az, el)
    state0 = np.concatenate([launch.to_array(), v0])

    def derivative(t_: float, state: np.ndarray) -> np.ndarray:
        position, velocity = state[:3], state[3:]
        total_accel = np.zeros(3)
        for force in forces:
            total_accel += force.acceleration(position, velocity, t_)
        return np.concatenate([velocity, total_accel])

    result = solve_ivp(derivative, (0.0, t), state0, method="RK45", rtol=1e-10, atol=1e-10, dense_output=False)
    assert result.success
    return result.y[:3, -1]


@pytest.mark.parametrize("distance_hint_s", [0.1, 0.3, 0.6, 1.0])
def test_production_integrator_step_converged_for_drag(distance_hint_s):
    # integrator_step=0.01 matches config/aiming_engine.yaml's shipped default,
    # not this file's own tighter make_projectile_config() default (0.001) --
    # the point is to check what's actually configured for production.
    config = make_projectile_config(integrator_step=0.01)
    launch = Vector3(0.0, 0.0, 0.0)
    az, el = 0.0, 0.05  # slight elevation, same order of magnitude as a real gravity+drag solve

    model = ProjectileModel(config)
    pos_rk4, _ = model.state_at(distance_hint_s, launch, az, el)

    pos_reference = _solve_ivp_reference_position(config, distance_hint_s, launch, az, el)

    # muzzle_velocity=800 m/s, so distance_hint_s=1.0 corresponds to hundreds
    # of meters of travel -- 5mm absolute agreement at that scale is a tight
    # convergence bar, not a loose one.
    assert pos_rk4.to_array() == pytest.approx(pos_reference, abs=5e-3)
