"""Main pipeline controller: bridge-layer TrackerFrame in, AimAssistOutput out.

One `update()` call per frame. Wires every module together: CoordinateTransform
-> TargetState -> AimSolver (LeadPrediction/HitEquation/NewtonSolver/
ProjectileModel internally) -> HitProbability -> AimReadinessLogic ->
AimStateMachine -> AimAssistOutput.

This is an aim-assist system, not an autonomous weapon: every output is
advisory information for the HUD. There is no fire command anywhere in
this framework — the operator always decides whether and when to fire.
"""

from __future__ import annotations

from pathlib import Path

from .aim_readiness_logic import AimReadinessLogic
from .aim_solver import AimSolver
from .aim_state_machine import AimStateMachine
from .config import AimAssistConfig, load_config
from .coordinate_transform import CoordinateTransform
from .hit_probability import HitProbability
from .lead_prediction import LeadPrediction
from .newton_solver import NewtonSolver
from .projectile_model import ProjectileModel
from .target_state import TargetState
from .types import AimAssistOutput, AimSolution, AimState, TrackerFrame, Vector3

# The bridge layer's follower_state string convention isn't this module's to
# own (aiming_engine stays tracker-agnostic; see bridge/confidence.py for
# where "stable"/"unstable" actually gets decided). Update this single
# constant if that convention ever changes.
_STABLE_FOLLOWER_STATE = "stable"


class AimingManager:
    def __init__(self, config: AimAssistConfig | None = None, config_path: str | Path | None = None):
        self._config = config if config is not None else load_config(config_path)

        self._coordinate_transform = CoordinateTransform(self._config.scope)
        self._target_state = TargetState(self._config.target_state)

        projectile_model = ProjectileModel(self._config.projectile)
        newton_solver = NewtonSolver(self._config.solver)
        self._aim_solver = AimSolver(
            projectile_model=projectile_model,
            newton_solver=newton_solver,
            lead_prediction=LeadPrediction(),
            coordinate_transform=self._coordinate_transform,
            muzzle_velocity=self._config.projectile.muzzle_velocity,
            warm_start_enabled=self._config.solver.warm_start,
        )

        self._hit_probability = HitProbability(self._config.hit_probability)
        self._aim_readiness_logic = AimReadinessLogic(self._config.aim_readiness)
        self._aim_state_machine = AimStateMachine(self._config.aim_state_machine)

    def update(
        self,
        frame: TrackerFrame,
        *,
        measurement_admitted: bool = True,
        fire_time: float | None = None,
    ) -> AimAssistOutput:
        """Update aim assistance, optionally predicting without admitting a measurement.

        ``measurement_admitted=False`` advances the aim solution from the last
        trusted TargetState prediction without appending the current tracker
        position to either motion estimator.

        ``fire_time`` is when the round is expected to leave the muzzle. The
        frame is always some milliseconds old by then (capture -> processing ->
        trigger), and a moving target keeps moving meanwhile, so the lead is
        solved for that time rather than for the capture time. ``None`` keeps
        the previous behaviour (fire at ``frame.timestamp``).
        """

        launch_point_world = self._coordinate_transform.scope_to_world(
            Vector3(0.0, 0.0, 0.0), frame.camera_extrinsics
        )

        measurement_position = None
        if measurement_admitted:
            position_world = self._coordinate_transform.image_to_world(frame)
            measurement_position = position_world.to_array().tolist()
            target_snapshot = self._target_state.update(
                position_world, frame.timestamp, observer=launch_point_world.to_array()
            )
        else:
            target_snapshot = self._target_state.predict(frame.timestamp)

        reference_time = frame.timestamp if fire_time is None else float(fire_time)
        solve_snapshot = (
            target_snapshot if reference_time == frame.timestamp else self._target_state.predict(reference_time)
        )
        aim_solution = self._aim_solver.solve(
            self._target_state, solve_snapshot, launch_point_world, reference_time, frame.camera_extrinsics
        )

        distance_m = (target_snapshot.position - launch_point_world).norm()
        aim_error_mrad = aim_solution.residual_norm_m / max(distance_m, 1e-6) * 1000.0
        target_speed = target_snapshot.velocity.norm()
        motion_diagnostics = self._target_state.diagnostics()
        follower_stable = frame.follower_state == _STABLE_FOLLOWER_STATE
        tracking_ok = frame.tracking_confidence >= self._config.aim_state_machine.tracking_confidence_threshold
        aim_ok = aim_solution.solver_converged and aim_error_mrad < self._config.aim_readiness.max_aim_error_mrad

        hit_probability = self._hit_probability.estimate(
            tracking_confidence=frame.tracking_confidence,
            follower_stable=follower_stable,
            aim_error_mrad=aim_error_mrad,
            distance_m=distance_m,
            target_speed=target_speed,
        )

        state = self._aim_state_machine.update(
            tracking_confidence=frame.tracking_confidence,
            follower_stable=follower_stable,
            aim_ok=aim_ok,
            now=frame.timestamp,
        )

        # aim_ready is the HUD's "green reticle" signal — it requires both the
        # per-frame readiness checks AND that the state machine has actually
        # settled into a stable lock (READY), not just a single lucky frame.
        aim_ready = state == AimState.READY and self._aim_readiness_logic.evaluate(
            tracking_stable=tracking_ok,
            follower_stable=follower_stable,
            hit_probability=hit_probability,
            distance_m=distance_m,
            aim_error_mrad=aim_error_mrad,
        )

        return AimAssistOutput(
            state=state,
            aim_solution=aim_solution,
            hit_probability=hit_probability,
            aim_ready=aim_ready,
            debug={
                "distance_m": distance_m,
                "aim_error_mrad": aim_error_mrad,
                "target_speed": target_speed,
                "measurement_admitted": measurement_admitted,
                # 오프라인 추정기 재생(benchmarks/sim_measurement/replay_reversal.py)용 입력 기록.
                "measurement_position_world": measurement_position,
                "observer_world": launch_point_world.to_array().tolist(),
                **motion_diagnostics,
            },
        )

    def solve_at(self, fire_time: float, camera_extrinsics) -> AimSolution | None:
        """Re-solve the lead for a round leaving at ``fire_time`` from the current
        target estimate, without admitting a measurement or touching the state
        machine. Used at the moment of firing, which falls between frames.
        Returns ``None`` before the first measurement."""

        try:
            snapshot = self._target_state.predict(fire_time)
        except RuntimeError:
            return None
        launch_point_world = self._coordinate_transform.scope_to_world(Vector3(0.0, 0.0, 0.0), camera_extrinsics)
        return self._aim_solver.solve(self._target_state, snapshot, launch_point_world, fire_time, camera_extrinsics)

    def target_snapshot_at(self, timestamp: float):
        """Return the current motion estimate for runtime diagnostics."""

        try:
            return self._target_state.predict(float(timestamp))
        except RuntimeError:
            return None
