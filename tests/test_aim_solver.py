import numpy as np
import pytest

from aiming_engine.aim_solver import AimSolver
from aiming_engine.config import ProjectileConfig, ScopeConfig, SolverConfig, TargetStateConfig
from aiming_engine.coordinate_transform import CoordinateTransform
from aiming_engine.lead_prediction import LeadPrediction
from aiming_engine.newton_solver import NewtonSolver
from aiming_engine.projectile_model import ProjectileModel
from aiming_engine.target_state import TargetState
from aiming_engine.types import Vector3


def build_aim_solver(muzzle_velocity=300.0, warm_start=True, forces=None, **projectile_overrides) -> AimSolver:
    projectile_config = ProjectileConfig(
        muzzle_velocity=muzzle_velocity,
        forces=forces if forces is not None else ["gravity"],
        gravity=9.81,
        integrator_step=0.01,
        **projectile_overrides,
    )
    solver_config = SolverConfig(max_iterations=30, tolerance_m=1e-3, jacobian_epsilon=1e-4, warm_start=warm_start)
    scope_config = ScopeConfig(mount_offset_m=(0.0, 0.0, 0.0), mount_rotation_rad=(0.0, 0.0, 0.0))

    return AimSolver(
        projectile_model=ProjectileModel(projectile_config),
        newton_solver=NewtonSolver(solver_config),
        lead_prediction=LeadPrediction(),
        coordinate_transform=CoordinateTransform(scope_config),
        muzzle_velocity=muzzle_velocity,
        warm_start_enabled=warm_start,
    )


def make_target_state(position: Vector3, timestamp: float, velocity: Vector3 | None = None) -> TargetState:
    ts = TargetState(TargetStateConfig(estimator="constant_velocity", history_length=10, min_dt=1e-3))
    ts.update(position, timestamp)
    if velocity is not None:
        ts.update(position + velocity, timestamp + 1.0)
    return ts


def test_stationary_target_converges_and_corrects_for_gravity():
    aim_solver = build_aim_solver()
    target_pos = Vector3(500.0, 0.0, 0.0)
    target_state = make_target_state(target_pos, timestamp=0.0)
    snapshot = target_state.predict(0.0)

    solution = aim_solver.solve(
        target_state, snapshot, launch_point_world=Vector3(0.0, 0.0, 0.0), reference_time=0.0, extrinsic=np.eye(4)
    )

    assert solution.solver_converged
    assert solution.azimuth == pytest.approx(0.0, abs=1e-3)
    assert solution.elevation > 0.0  # must aim above line-of-sight to counter gravity drop
    assert solution.ballistic_offset[1] > 0.0  # elevation correction vs naive line-of-sight
    assert solution.ballistic_offset[0] == pytest.approx(0.0, abs=1e-3)  # no azimuth correction needed


def test_ballistic_offset_grows_with_range():
    aim_solver_near = build_aim_solver()
    aim_solver_far = build_aim_solver()

    near_state = make_target_state(Vector3(300.0, 0.0, 0.0), timestamp=0.0)
    far_state = make_target_state(Vector3(700.0, 0.0, 0.0), timestamp=0.0)

    near_solution = aim_solver_near.solve(
        near_state, near_state.predict(0.0), Vector3(0.0, 0.0, 0.0), 0.0, np.eye(4)
    )
    far_solution = aim_solver_far.solve(far_state, far_state.predict(0.0), Vector3(0.0, 0.0, 0.0), 0.0, np.eye(4))

    assert far_solution.ballistic_offset[1] > near_solution.ballistic_offset[1]


def test_moving_target_lead_point_is_ahead_of_current_position():
    aim_solver = build_aim_solver()
    target_state = make_target_state(Vector3(400.0, 0.0, 0.0), timestamp=0.0, velocity=Vector3(0.0, 40.0, 0.0))
    snapshot = target_state.predict(1.0)

    solution = aim_solver.solve(
        target_state, snapshot, launch_point_world=Vector3(0.0, 0.0, 0.0), reference_time=1.0, extrinsic=np.eye(4)
    )

    assert solution.solver_converged
    assert solution.lead_point_world.y > snapshot.position.y  # leads ahead of current Y position
    assert solution.azimuth > 0.0  # must aim to the side to lead a target moving in +Y


def test_warm_start_uses_no_more_iterations_than_cold_solve_on_same_frame():
    # Realistic scenario: a target tracked at 30fps with a stable, consistent
    # velocity estimate (the case warm-starting is meant to help). Solve the
    # *same* frame once with warm start (primed by several prior frames) and
    # once cold, on otherwise-identical solvers, at the same target state.
    warm_solver = build_aim_solver(warm_start=True)
    cold_solver = build_aim_solver(warm_start=False)

    velocity = Vector3(10.0, 5.0, 0.0)
    dt = 1.0 / 30.0
    target_state = TargetState(TargetStateConfig(estimator="constant_velocity", history_length=10, min_dt=1e-3))
    t = 0.0
    position = Vector3(400.0, 0.0, 0.0)
    for _ in range(5):
        target_state.update(position, t)
        warm_solver.solve(target_state, target_state.predict(t), Vector3(0.0, 0.0, 0.0), t, np.eye(4))
        t += dt
        position = position + Vector3(velocity.x * dt, velocity.y * dt, velocity.z * dt)

    # one more consistent frame: the true root barely moves since velocity was already well-estimated
    target_state.update(position, t)
    snapshot = target_state.predict(t)

    warm_result = warm_solver.solve(target_state, snapshot, Vector3(0.0, 0.0, 0.0), t, np.eye(4))
    cold_result = cold_solver.solve(target_state, snapshot, Vector3(0.0, 0.0, 0.0), t, np.eye(4))

    assert warm_result.solver_converged and cold_result.solver_converged
    assert warm_result.solver_iterations <= cold_result.solver_iterations


def test_warm_start_disabled_still_converges():
    aim_solver = build_aim_solver(warm_start=False)
    target_state = make_target_state(Vector3(500.0, 0.0, 0.0), timestamp=0.0)
    solution = aim_solver.solve(
        target_state, target_state.predict(0.0), Vector3(0.0, 0.0, 0.0), 0.0, np.eye(4)
    )
    assert solution.solver_converged


def test_non_converged_solve_does_not_raise():
    # Regression guard: a previous fix that gates warm-start caching on
    # `result.residual_norm < 50.0` once referenced a nonexistent
    # `residual_norm_m` attribute on SolverResult, which raised
    # AttributeError on every non-converged solve -- i.e. exactly the
    # failure case this guard was meant to protect against. max_iterations=1
    # forces non-convergence from a cold start so this path is exercised
    # every run, not just when a particular scenario happens to be hard.
    projectile_config = ProjectileConfig(muzzle_velocity=800.0, forces=["gravity"], gravity=9.81, integrator_step=0.01)
    solver_config = SolverConfig(max_iterations=1, tolerance_m=1e-9, jacobian_epsilon=1e-4, warm_start=True)
    scope_config = ScopeConfig(mount_offset_m=(0.0, 0.0, 0.0), mount_rotation_rad=(0.0, 0.0, 0.0))
    aim_solver = AimSolver(
        projectile_model=ProjectileModel(projectile_config),
        newton_solver=NewtonSolver(solver_config),
        lead_prediction=LeadPrediction(),
        coordinate_transform=CoordinateTransform(scope_config),
        muzzle_velocity=800.0,
        warm_start_enabled=True,
    )
    target_state = make_target_state(Vector3(500.0, 0.0, 0.0), timestamp=0.0)
    solution = aim_solver.solve(
        target_state, target_state.predict(0.0), Vector3(0.0, 0.0, 0.0), 0.0, np.eye(4)
    )
    assert solution.solver_converged is False


@pytest.mark.parametrize("distance_m", [100.0, 300.0, 600.0])
def test_converges_with_drag_enabled_across_range(distance_m):
    # Regression guard for the RK4 fallback (no more closed-form shortcut)
    # that drag forces the solver through -- see forces.py:DragForce.
    aim_solver = build_aim_solver(
        muzzle_velocity=850.0,
        warm_start=False,
        forces=["gravity", "drag"],
        drag_model="G1",
        mass_kg=0.01,
        diameter_m=0.0078,
        air_density_kg_m3=1.225,
        speed_of_sound_mps=340.3,
    )
    target_state = make_target_state(Vector3(distance_m, 0.0, 0.0), timestamp=0.0, velocity=Vector3(0.0, 20.0, 0.0))
    solution = aim_solver.solve(
        target_state, target_state.predict(1.0), Vector3(0.0, 0.0, 0.0), reference_time=1.0, extrinsic=np.eye(4)
    )
    assert solution.solver_converged
    assert solution.elevation > 0.0  # still must aim above line-of-sight to counter gravity+drag drop
