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


def test_rejected_measurement_uses_prediction_without_contaminating_motion_state():
    gated = AimingManager(config=make_config())
    clean = AimingManager(config=make_config())

    for index in (0, 1):
        gated.update(make_frame(index))
        clean.update(make_frame(index))

    jitter = make_frame(2, initial_range=5000.0, closing_speed=0.0)
    predicted = gated.update(jitter, measurement_admitted=False)
    assert predicted.debug["measurement_admitted"] is False

    recovered = gated.update(make_frame(3), measurement_admitted=True)
    expected = clean.update(make_frame(3), measurement_admitted=True)
    assert recovered.debug["measurement_admitted"] is True
    assert recovered.debug["target_speed"] == pytest.approx(expected.debug["target_speed"])
    assert recovered.aim_solution.lead_point_world.to_array() == pytest.approx(
        expected.aim_solution.lead_point_world.to_array()
    )


def test_fire_time_leads_moving_target_to_fire_time():
    # 접근 표적(20 m/s)을 50 ms 늦게 쏜다면, 조준점은 발사 시각 + 비행시간의 표적 위치여야 한다.
    manager = AimingManager(config=make_config())
    for i in range(10):
        manager.update(make_frame(i))
    frame = make_frame(10)
    delay = 0.05
    out_now = manager.update(frame)
    fire_time = frame.timestamp + delay
    sol_fire = manager.solve_at(fire_time, frame.camera_extrinsics)

    assert sol_fire is not None and sol_fire.solver_converged
    expected = manager._target_state.predict(fire_time + sol_fire.time_of_flight).position.to_array()
    assert np.allclose(sol_fire.lead_point_world.to_array(), expected, atol=1e-6)
    # 표적이 다가오므로 늦게 쏠수록 조준점이 가깝다. 만나는 거리 u*r/(u+v) 가 r 에 비례하고
    # r 은 지연 동안 v*delay 줄어드므로 조준점은 v*delay*u/(u+v) 만큼 당겨진다(u 탄속 300, v 20).
    gap = np.linalg.norm(out_now.aim_solution.lead_point_world.to_array()) - np.linalg.norm(expected)
    assert gap == pytest.approx(20.0 * delay * 300.0 / 320.0, rel=1e-3)


def test_update_fire_time_matches_solve_at_and_default_is_capture_time():
    manager_a = AimingManager(config=make_config())
    manager_b = AimingManager(config=make_config())
    for i in range(10):
        manager_a.update(make_frame(i))
        manager_b.update(make_frame(i))
    frame = make_frame(10)
    out_a = manager_a.update(frame, fire_time=frame.timestamp + 0.04)
    manager_b.update(frame)
    sol_b = manager_b.solve_at(frame.timestamp + 0.04, frame.camera_extrinsics)
    assert np.allclose(out_a.aim_solution.lead_point_world.to_array(), sol_b.lead_point_world.to_array(), atol=1e-6)

    manager_c = AimingManager(config=make_config())
    manager_d = AimingManager(config=make_config())
    for i in range(11):
        out_c = manager_c.update(make_frame(i))
        out_d = manager_d.update(make_frame(i), fire_time=make_frame(i).timestamp)
    assert out_c.aim_solution.azimuth == out_d.aim_solution.azimuth
    assert out_c.aim_solution.elevation == out_d.aim_solution.elevation


def test_solve_at_before_any_measurement_returns_none():
    assert AimingManager(config=make_config()).solve_at(1.0, np.eye(4)) is None
