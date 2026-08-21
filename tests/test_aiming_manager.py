import numpy as np
import pytest

from aiming_engine.config import (
    AimAssistConfig,
    AimReadinessConfig,
    AimStateMachineConfig,
    HitProbabilityConfig,
    ProjectileConfig,
    ScopeConfig,
    SolverConfig,
    TargetStateConfig,
)
from aiming_engine.aiming_manager import AimingManager
from aiming_engine.types import AimState, TrackerFrame

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
PIXEL = (320.0 + 160.0, 240.0)  # off-principal-point pixel -> non-degenerate bearing
_RAY_CAM = np.linalg.inv(K) @ np.array([PIXEL[0], PIXEL[1], 1.0])
UNIT_DIR = _RAY_CAM / np.linalg.norm(_RAY_CAM)  # fixed straight-line approach bearing


def make_config(**asm_overrides) -> AimAssistConfig:
    asm_defaults = dict(
        tracking_confidence_threshold=0.5,
        min_stable_frames_for_track=2,
        min_stable_frames_for_aim=2,
        track_lost_timeout_s=0.5,
    )
    asm_defaults.update(asm_overrides)

    return AimAssistConfig(
        projectile=ProjectileConfig(muzzle_velocity=300.0, forces=["gravity"], gravity=9.81, integrator_step=0.01),
        scope=ScopeConfig(mount_offset_m=(0.0, 0.0, 0.0), mount_rotation_rad=(0.0, 0.0, 0.0)),
        solver=SolverConfig(max_iterations=30, tolerance_m=1e-3, jacobian_epsilon=1e-4, warm_start=True),
        target_state=TargetStateConfig(estimator="constant_velocity", history_length=10, min_dt=1e-3),
        hit_probability=HitProbabilityConfig(
            weights={
                "tracking_confidence": 0.3,
                "follower_stability": 0.2,
                "aim_error": 0.3,
                "distance": 0.1,
                "velocity": 0.1,
            },
            distance_scale_m=500.0,
            velocity_scale_mps=50.0,
            aim_error_scale_mrad=20.0,
        ),
        aim_readiness=AimReadinessConfig(
            hit_probability_threshold=0.3, max_aim_error_mrad=50.0, min_valid_range_m=5.0, max_valid_range_m=800.0
        ),
        aim_state_machine=AimStateMachineConfig(**asm_defaults),
    )


def make_frame(index: int, initial_range=250.0, closing_speed=20.0, dt=0.05) -> TrackerFrame:
    t = index * dt
    range_m = initial_range - closing_speed * t
    return TrackerFrame(
        timestamp=t,
        bbox=(300.0, 220.0, 340.0, 260.0),
        center_px=PIXEL,
        velocity_px=(0.0, 0.0),
        tracking_confidence=0.95,
        follower_state="stable",
        camera_intrinsics=K,
        camera_extrinsics=np.eye(4),
        laser_range_m=range_m,
    )


def test_full_pipeline_reaches_ready_and_aim_ready_becomes_true():
    manager = AimingManager(config=make_config())

    outputs = [manager.update(make_frame(i)) for i in range(40)]

    ready_outputs = [o for o in outputs if o.state == AimState.READY]
    assert len(ready_outputs) >= 1, "synthetic approaching target never reached READY state"

    aim_ready_outputs = [o for o in outputs if o.aim_ready]
    assert len(aim_ready_outputs) >= 1, "aim_ready never became true for a well-aimed approaching target"
    # aim_ready should only ever coincide with a stable-enough aim (AIM or READY), never SEARCH/TRACK
    for output in aim_ready_outputs:
        assert output.state in (AimState.AIM, AimState.READY)


def test_ready_state_persists_without_interruption():
    # No FIRE/COOLDOWN cycle anymore -- once READY, a continuing
    # well-tracked approach should stay READY rather than bouncing out.
    manager = AimingManager(config=make_config())
    outputs = [manager.update(make_frame(i)) for i in range(40)]
    states = [o.state for o in outputs]

    first_ready_index = states.index(AimState.READY)
    assert all(s == AimState.READY for s in states[first_ready_index:])


def test_state_progression_reaches_ready_via_track_and_aim():
    manager = AimingManager(config=make_config())
    outputs = [manager.update(make_frame(i)) for i in range(40)]
    states = [o.state for o in outputs]

    first_ready_index = states.index(AimState.READY)
    assert AimState.AIM in states[:first_ready_index]
    assert AimState.TRACK in states[:first_ready_index]


def test_hit_probability_and_debug_are_populated():
    manager = AimingManager(config=make_config())
    output = manager.update(make_frame(5))

    assert 0.0 <= output.hit_probability <= 1.0
    assert "distance_m" in output.debug
    assert output.debug["distance_m"] > 0.0
