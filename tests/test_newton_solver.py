import numpy as np
import pytest

from aiming_engine.config import ProjectileConfig, SolverConfig, TargetStateConfig
from aiming_engine.hit_equation import HitEquation
from aiming_engine.newton_solver import NewtonSolver
from aiming_engine.projectile_model import ProjectileModel
from aiming_engine.target_state import TargetState
from aiming_engine.types import Vector3


def make_solver(max_iterations=20, tolerance_m=1e-3, jacobian_epsilon=1e-4) -> NewtonSolver:
    return NewtonSolver(
        SolverConfig(
            max_iterations=max_iterations,
            tolerance_m=tolerance_m,
            jacobian_epsilon=jacobian_epsilon,
            warm_start=True,
        )
    )


class LinearEquation:
    """Simple analytic equation with known root, for solver-only tests."""

    def __init__(self, a: np.ndarray, b: np.ndarray):
        self._a = a
        self._b = b

    def residual(self, x: np.ndarray) -> np.ndarray:
        return self._a @ x - self._b


class ConstantUnsolvableEquation:
    def residual(self, x: np.ndarray) -> np.ndarray:
        return np.array([1.0, 1.0, 1.0])  # never zero, no matter what x is


def test_solver_converges_on_linear_system():
    a = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]])
    x_true = np.array([1.0, -2.0, 5.0])
    b = a @ x_true
    equation = LinearEquation(a, b)

    result = make_solver().solve(equation, x0=np.zeros(3))

    assert result.converged
    assert result.x == pytest.approx(x_true, abs=1e-6)


def test_solver_respects_max_iterations_on_unsolvable_equation():
    solver = make_solver(max_iterations=5)
    result = solver.solve(ConstantUnsolvableEquation(), x0=np.zeros(3))

    assert not result.converged
    assert result.iterations == 5


def test_solver_converges_on_real_hit_equation_stationary_target_no_gravity():
    muzzle_velocity = 150.0
    model = ProjectileModel(
        ProjectileConfig(muzzle_velocity=muzzle_velocity, forces=[], gravity=9.81, integrator_step=0.01)
    )
    ts = TargetState(TargetStateConfig(estimator="constant_velocity", history_length=10, min_dt=1e-3))
    target_pos = Vector3(80.0, 60.0, 0.0)  # distance 100
    ts.update(target_pos, timestamp=0.0)
    equation = HitEquation(ts, model, launch_point=Vector3(0.0, 0.0, 0.0), reference_time=0.0)

    x0 = np.array([100.0 / muzzle_velocity, 0.0, 0.0])  # deliberately wrong azimuth
    result = make_solver().solve(equation, x0)

    assert result.converged
    t, az, el = result.x
    assert t == pytest.approx(100.0 / muzzle_velocity, abs=1e-4)
    assert az == pytest.approx(np.arctan2(60.0, 80.0), abs=1e-4)
    assert el == pytest.approx(0.0, abs=1e-4)


def test_solver_iteration_count_is_non_negative_and_bounded():
    solver = make_solver(max_iterations=10)
    a = np.eye(3)
    equation = LinearEquation(a, np.array([1.0, 1.0, 1.0]))
    result = solver.solve(equation, x0=np.array([0.9, 0.9, 0.9]))  # already close
    assert 0 <= result.iterations <= 10
