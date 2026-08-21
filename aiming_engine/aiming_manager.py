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
from .types import AimAssistOutput, AimState, TrackerFrame, Vector3

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

    def update(self, frame: TrackerFrame) -> AimAssistOutput:
        position_world = self._coordinate_transform.image_to_world(frame)
        target_snapshot = self._target_state.update(position_world, frame.timestamp)

        launch_point_world = self._coordinate_transform.scope_to_world(
            Vector3(0.0, 0.0, 0.0), frame.camera_extrinsics
        )

        aim_solution = self._aim_solver.solve(
            self._target_state, target_snapshot, launch_point_world, frame.timestamp, frame.camera_extrinsics
        )

        distance_m = (target_snapshot.position - launch_point_world).norm()
        aim_error_mrad = aim_solution.residual_norm_m / max(distance_m, 1e-6) * 1000.0
        target_speed = target_snapshot.velocity.norm()
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
            debug={"distance_m": distance_m, "aim_error_mrad": aim_error_mrad, "target_speed": target_speed},
        )
