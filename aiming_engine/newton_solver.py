"""Generic multivariate Newton-Raphson root finder.

Deliberately knows nothing about ballistics: it only calls
`equation.residual(x)`. This is what lets ProjectileModel grow new force
plugins (drag, wind, G1/G7) without ever touching this file. The Jacobian
is estimated by forward finite differences so no analytical derivative is
required from the equation being solved.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import SolverConfig
from .types import SolverResult


class Residual(Protocol):
    def residual(self, x: np.ndarray) -> np.ndarray: ...


class NewtonSolver:
    def __init__(self, config: SolverConfig):
        self._max_iterations = config.max_iterations
        self._tolerance = config.tolerance_m
        self._jacobian_epsilon = config.jacobian_epsilon

    def solve(self, equation: Residual, x0: np.ndarray, tolerance: float | None = None) -> SolverResult:
        x = np.array(x0, dtype=np.float64, copy=True)
        residual = equation.residual(x)
        tol = tolerance if tolerance is not None else self._tolerance

        for iteration in range(1, self._max_iterations + 1):
            norm = float(np.linalg.norm(residual))
            if norm < tol:
                return SolverResult(converged=True, iterations=iteration - 1, x=x, residual_norm=norm)

            jacobian = self._finite_difference_jacobian(equation, x, residual)
            try:
                delta = np.linalg.solve(jacobian, -residual)
            except np.linalg.LinAlgError:
                delta, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)

            x = x + delta
            if not np.all(np.isfinite(x)):
                break
            x[0] = max(x[0], 1e-4)  # Flight time must remain strictly positive physically
            residual = equation.residual(x)

        norm = float(np.linalg.norm(residual))
        return SolverResult(converged=norm < tol, iterations=self._max_iterations, x=x, residual_norm=norm)


    def _finite_difference_jacobian(self, equation: Residual, x: np.ndarray, r_base: np.ndarray) -> np.ndarray:
        n = len(x)
        jacobian = np.zeros((n, n))
        for j in range(n):
            x_perturbed = x.copy()
            x_perturbed[j] += self._jacobian_epsilon
            r_perturbed = equation.residual(x_perturbed)
            jacobian[:, j] = (r_perturbed - r_base) / self._jacobian_epsilon
        return jacobian
