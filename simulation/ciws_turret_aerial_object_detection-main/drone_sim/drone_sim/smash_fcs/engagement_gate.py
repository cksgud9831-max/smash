"""Pure gating rules shared by the ROS FCS integration and unit tests."""

from __future__ import annotations

import math
from collections import deque
from typing import Mapping

import numpy as np


def motion_allows_fire(require_moving_target: bool, motion_state: str | None) -> bool:
    """Return whether the target-mover state permits firing.

    Only an explicitly moving-target scenario is gated; stationary-target
    scenarios keep their prior behaviour. Unknown status fails closed until the
    mover reports ``MOVE``.
    """

    if not require_moving_target:
        return True
    return motion_state == "MOVE"


def motion_solution_is_stable(
    motion_state: str | None,
    diagnostics: Mapping[str, object] | None,
    *,
    max_velocity_disagreement_mps: float,
    max_fit_rmse_m: float,
    min_history_count: int,
    lateral_sign_deadband_mps: float | None = None,
) -> bool:
    """Fail closed on an internally inconsistent moving-target solution.

    ``lateral_sign_deadband_mps`` enables the direction-consistency check: when the
    long-window and the recent 6-frame fits both report a lateral speed above the
    deadband but with opposite signs, the solution straddles a direction change
    (2026-10-02 sim: 5/26 HIT) and is rejected. ``None`` disables the check.
    """

    if motion_state != "MOVE":
        return True
    if diagnostics is None:
        return False
    try:
        disagreement = float(diagnostics.get("velocity_disagreement_mps"))
        fit_rmse = float(diagnostics.get("fit_rmse_m"))
        history_count = int(diagnostics.get("history_count"))
    except (TypeError, ValueError):
        return False
    if not (
        math.isfinite(disagreement)
        and math.isfinite(fit_rmse)
        and history_count >= int(min_history_count)
        and disagreement <= float(max_velocity_disagreement_mps)
        and fit_rmse <= float(max_fit_rmse_m)
    ):
        return False
    if lateral_sign_deadband_mps is None:
        return True
    return _lateral_direction_consistent(diagnostics, float(lateral_sign_deadband_mps))


def _lateral_direction_consistent(diagnostics: Mapping[str, object], deadband_mps: float) -> bool:
    try:
        long_lat = float(diagnostics.get("lateral_velocity_long_mps"))
        recent_lat = float(diagnostics.get("lateral_velocity_recent_mps"))
    except (TypeError, ValueError):
        return False  # fail closed: the sign check was requested but the data is missing
    if not (math.isfinite(long_lat) and math.isfinite(recent_lat)):
        return False
    if abs(long_lat) <= deadband_mps or abs(recent_lat) <= deadband_mps:
        return True
    return (long_lat > 0.0) == (recent_lat > 0.0)


class SpeedCollapseDetector:
    """Flag a sudden drop of the estimated lateral speed against its recent history.

    A direction change makes the long-window OLS straddle the reversal, so its speed
    collapses toward zero (or flips sign) for ~0.6 s while the aim point lags the
    target by 0.5-0.7 m (2026-10-02 sim: 0/17 HIT at 0.3-0.6 s after reversal, and
    the estimator's own reversal detection did not fire). ``update`` returns True
    while the current speed is below ``ratio`` times the ``percentile`` of the speeds
    seen in the trailing ``window_s``. ``ratio <= 0`` disables the detector.
    """

    def __init__(
        self,
        ratio: float,
        window_s: float = 2.0,
        percentile: float = 75.0,
        min_samples: int = 10,
        min_reference_mps: float = 0.2,
    ):
        self._ratio = float(ratio)
        self._window_s = float(window_s)
        self._percentile = float(percentile)
        self._min_samples = int(min_samples)
        self._min_reference = float(min_reference_mps)
        self._history: deque[tuple[float, float]] = deque()

    def reset(self) -> None:
        self._history.clear()

    def update(self, stamp: float, speed_mps: float | None) -> bool:
        if self._ratio <= 0.0 or speed_mps is None or not math.isfinite(speed_mps):
            return False
        speed = abs(float(speed_mps))
        if self._history and stamp <= self._history[-1][0]:
            self._history.clear()  # clock jump / replay: do not mix timelines
        self._history.append((float(stamp), speed))
        while self._history and self._history[0][0] <= stamp - self._window_s:
            self._history.popleft()
        if len(self._history) < self._min_samples:
            return False
        reference = float(np.percentile([v for _, v in self._history], self._percentile))
        return reference >= self._min_reference and speed < self._ratio * reference
