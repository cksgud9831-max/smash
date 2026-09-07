#!/usr/bin/env python3
"""verify_stage3_fire_control.py — fire_control.py 오라클 정량 검증 [로드맵 3단계]

stage1_fcs_integration_report.md 와 같은 방법론이다: ROS 2 / Gazebo 없이,
순수 파이썬 수학 경로만 떼어내 알려진 정답(오라클)과 비교한다. 이 세션은
실제 gz 런타임을 실행할 수 없으므로, 이 스크립트가 검증하는 범위는 다음과
같이 명확히 제한된다.

    검증하는 것:
        1. FireController 가 재적분한 탄도가 ProjectileModel 의 해석해(gravity-only
           analytic parabola)와 수치오차 수준까지 일치하는가.
        2. FireController 가 재적분한 명중점이, 같은 조건에서 aiming_engine
           AimingManager 가 실제로 계산한 lead_point_world 와 일치하는가
           (조준 로직과 격발 로직이 서로 독립적으로 구현되었음에도 동일한
           물리를 풀어 같은 답에 도달하는지 교차검증).
        3. Hit/Miss 판정 경계(반경 정확히 == 명중, 살짝 초과 == 실패)가
           부동소수 오차 없이 올바르게 동작하는가.
        4. auto/manual 격발 모드의 트리거 엣지 검출과 쿨다운(연사속도) 로직이
           합성 시간열에서 의도한 발수만큼만 격발하는가.
        5. ground truth 최신성(timeout) 로직이 오래된 실측값을 올바르게
           무효화하는가.

    검증하지 않는 것 (사용자가 WSL2 에서 확인해야 함):
        1. ROS 2 노드가 실제로 기동해 /smash_fcs/fire_event, /smash_fcs/hit_result
           토픽이 흐르는가.
        2. Gazebo PosePublisher 플러그인이 실제로 /model/drone/pose 를 그
           이름으로 발행하는가(gz-sim 버전에 따라 토픽 이름이 다를 수 있다).
        3. HUD 상의 탄환 궤적/명중 이펙트 렌더링이 실제로 보기 좋게 나오는가.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np


def _resolve_core_root() -> Path:
    """SMASH 코어 저장소 루트(aiming_engine, bridge 를 담은 곳)를 찾는다.

    engine_bootstrap.resolve_core_root() 와 같은 우선순위다: 환경변수
    SMASH_CORE_PATH -> 이 파일 기준 상대 경로 후보. 이 파일은
    <core>/ciws_turret_aerial_object_detection-main/drone_sim/verify_stage3_fire_control.py
    에 두는 것을 전제로 parents[2] 를 우선 시도한다.
    """

    env = os.environ.get("SMASH_CORE_PATH")
    candidates = []
    if env:
        candidates.append(Path(env).expanduser())
    here = Path(__file__).resolve()
    for up in range(1, 6):
        if len(here.parents) > up:
            candidates.append(here.parents[up])
    for candidate in candidates:
        if (candidate / "aiming_engine" / "__init__.py").is_file():
            return candidate
    raise RuntimeError(
        "SMASH 코어 저장소(aiming_engine 을 담은 폴더)를 찾지 못했습니다. "
        "환경변수 SMASH_CORE_PATH 를 코어 루트로 지정하세요. 시도한 경로: "
        + ", ".join(str(c) for c in candidates)
    )


CORE_ROOT = _resolve_core_root()
sys.path.insert(0, str(CORE_ROOT))
# fire_control.py 는 drone_sim 패키지 안에 있다(smash_fcs_node.py 와 같은 위치).
sys.path.insert(0, str(Path(__file__).resolve().parent / "drone_sim" / "smash_fcs"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aiming_engine.aiming_manager import AimingManager  # noqa: E402
from aiming_engine.config import load_config  # noqa: E402
from aiming_engine.coordinate_transform import CoordinateTransform  # noqa: E402
from aiming_engine.projectile_model import ProjectileModel, azimuth_elevation_to_direction  # noqa: E402
from aiming_engine.types import TrackerFrame, Vector3  # noqa: E402

from fire_control import FireController, direction_to_world_azimuth_elevation  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def default_config():
    from dataclasses import replace

    cfg_path = CORE_ROOT / "config" / "aiming_engine.yaml"
    config = load_config(cfg_path)
    # gravity-only 로 고정해 analytic parabola 오라클과 직접 비교한다.
    projectile = replace(
        config.projectile,
        forces=[f for f in config.projectile.forces if f != "drag"],
        muzzle_velocity=920.0,
    )
    return replace(config, projectile=projectile)


# ── 1. 탄도 재적분 vs 해석해 (gravity-only) ──────────────────────────────


def test_ballistic_analytic_match():
    config = default_config()
    fc = FireController(
        projectile_config=config.projectile,
        muzzle_velocity_mps=config.projectile.muzzle_velocity,
        fire_mode="auto",
        fire_rate_rpm=600.0,
        target_radius_m=0.5,
        ground_truth_timeout_s=0.5,
    )

    launch = np.array([1.0, 2.0, 0.5])
    az, el = math.radians(15.0), math.radians(10.0)
    direction = azimuth_elevation_to_direction(az, el)
    v0 = config.projectile.muzzle_velocity * direction
    g = np.array([0.0, 0.0, -9.81])

    from fire_control import FiredShot

    shot = FiredShot(
        shot_id=1,
        fire_time=0.0,
        launch_point_world=launch,
        azimuth_world=az,
        elevation_world=el,
        predicted_tof_s=1.0,
        predicted_impact_point_world=launch,
        target_radius_m=0.5,
        muzzle_velocity_mps=config.projectile.muzzle_velocity,
        trigger_mode="auto",
    )

    max_err = 0.0
    for t in [0.0, 0.05, 0.2, 0.5, 1.0, 1.5]:
        analytic = launch + v0 * t + 0.5 * g * t * t
        got = fc.projectile_position_at(shot, t)
        err = float(np.linalg.norm(got - analytic))
        max_err = max(max_err, err)

    check(
        "탄도 재적분이 gravity-only 해석해와 일치",
        max_err < 1e-9,
        f"max_err={max_err:.3e} m",
    )


# ── 2. FireController 재적분 vs AimingManager.lead_point_world 교차검증 ──


def test_cross_check_against_aiming_manager():
    config = default_config()
    manager = AimingManager(config=config)
    ct = CoordinateTransform(config.scope)

    launch_point = Vector3(0.0, 0.0, 0.0)
    extrinsics = np.eye(4)
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 320.0], [0, 0, 1.0]])

    # 표적: 35 m 전방, 12 m 상공에서 y 방향으로 3 m/s 등속 이동. 이 좌표를
    # 그대로 pixel/range 로 정확히 역투영해서 TrackerFrame 을 만든다 — 그래야
    # AimingManager 가 내부적으로 image_to_world() 로 복원하는 위치가 우리가
    # 의도한 (35, v*t, 12) 와 정확히 일치한다(단순히 화면 중앙 픽셀을 고정하면
    # 시선 방향이 항상 보어사이트를 향하게 되어 전혀 다른 가상의 궤적이 된다).
    target_speed = 3.0
    t0 = 0.0
    dt = 1.0 / 30.0
    aim_solution = None
    for i in range(40):
        t = t0 + i * dt
        true_world_pos = np.array([35.0, target_speed * t, 12.0])
        # extrinsics 가 항등행렬이므로 scope_to_world/world_to_scope 도 항등이다:
        # p_scope == p_world. scope_to_camera 는 image_to_camera+camera_to_scope 의
        # 역함수이므로, 이렇게 만든 pixel/range 는 image_to_world() 에 넣었을 때
        # true_world_pos 를 수치오차 수준까지 정확히 복원한다.
        p_cam = ct.scope_to_camera(Vector3.from_array(true_world_pos)).to_array()
        range_m = float(np.linalg.norm(p_cam))
        u = intrinsics[0, 0] * p_cam[0] / p_cam[2] + intrinsics[0, 2]
        v = intrinsics[1, 1] * p_cam[1] / p_cam[2] + intrinsics[1, 2]

        frame = TrackerFrame(
            timestamp=t,
            bbox=(u - 20.0, v - 20.0, u + 20.0, v + 20.0),
            center_px=(u, v),
            velocity_px=(0.0, 0.0),
            tracking_confidence=1.0,
            follower_state="stable",
            camera_intrinsics=intrinsics,
            camera_extrinsics=extrinsics,
            laser_range_m=range_m,
        )
        output = manager.update(frame)
        aim_solution = output.aim_solution

    assert aim_solution is not None

    fc = FireController(
        projectile_config=config.projectile,
        muzzle_velocity_mps=config.projectile.muzzle_velocity,
        fire_mode="auto",
        fire_rate_rpm=600.0,
        target_radius_m=0.5,
        ground_truth_timeout_s=0.5,
    )

    fire_time = t0 + 39 * dt

    # AimSolver 가 실제로 푼 세계 프레임 방향은 scope 프레임 각도를 extrinsics
    # 회전(여기서는 항등행렬)으로 되돌려 얻는다. smash_fcs_node._aim_direction_in_base
    # 와 동일한 방법.
    from aiming_engine.projectile_model import azimuth_elevation_to_direction as az_el_to_dir

    direction_scope = az_el_to_dir(aim_solution.azimuth, aim_solution.elevation)
    direction_world = extrinsics[:3, :3] @ direction_scope

    shot = fc.maybe_fire(
        now=fire_time,
        aim_ready=True,
        trigger_pressed=False,
        aim_solution=aim_solution,
        launch_point_world=launch_point.to_array(),
        direction_world=direction_world,
    )
    assert shot is not None

    reintegrated = fc.projectile_position_at(shot, shot.predicted_tof_s)
    reference = aim_solution.lead_point_world.to_array()
    err = float(np.linalg.norm(reintegrated - reference))

    # 허용 오차는 뉴턴 솔버의 수렴 판정 기준(config.solver.tolerance_m, 잔차
    # norm 이 이 값 밑으로 내려가면 수렴으로 본다)에 맞춘다. lead_point_world 는
    # "해가 수렴했다고 판정된" 시점의 근사해이므로, 그 잔차 자체가 이 크기까지는
    # 남아 있을 수 있다 — 0에 수렴할 이유가 없다. 대신 이 오차가 solver
    # tolerance 를 넘지 않는지, 즉 재적분 결과가 솔버의 수렴 기준과 모순되지
    # 않는지를 확인한다.
    tolerance_m = config.solver.tolerance_m
    check(
        "FireController 재적분 명중점이 AimSolution.lead_point_world와 solver tolerance 이내로 일치",
        err <= tolerance_m,
        f"err={err:.3e} m (solver tolerance {tolerance_m:.3e} m), tof={aim_solution.time_of_flight * 1000:.2f} ms",
    )

    # 표적이 실제로 등속 가정을 정확히 지켰다면(이 테스트의 전제), 그 시각의
    # 실제 위치는 target_state 예측과 일치해야 하고, 그 지점을 ground truth로
    # 주면 HIT 이 나와야 한다.
    impact_time = shot.fire_time + shot.predicted_tof_s
    true_pos_at_impact = np.array([35.0, target_speed * impact_time, 12.0])
    results = fc.update(
        now=impact_time + 1e-6,
        ground_truth_position_world=true_pos_at_impact,
        ground_truth_timestamp=impact_time,
    )
    check(
        "등속 가정이 실제로 성립하는 표적에 대해 HIT 판정",
        len(results) == 1 and results[0].verdict == "HIT",
        f"verdict={results[0].verdict if results else None}, "
        f"miss_distance={results[0].miss_distance_m if results else None}",
    )


# ── 3. Hit/Miss 판정 경계 ────────────────────────────────────────────────


def test_hit_miss_boundary():
    config = default_config()
    fc = FireController(
        projectile_config=config.projectile,
        muzzle_velocity_mps=config.projectile.muzzle_velocity,
        fire_mode="auto",
        fire_rate_rpm=600.0,
        target_radius_m=1.0,
        ground_truth_timeout_s=1.0,
    )
    from fire_control import FiredShot

    launch = np.zeros(3)
    az, el = 0.0, 0.0
    shot_at = fc.projectile_position_at(
        FiredShot(2, 0.0, launch, az, el, 0.5, launch, 1.0, 920.0, "auto"), 0.5
    )

    shot_hit = FiredShot(3, 0.0, launch, az, el, 0.5, launch, 1.0, 920.0, "auto")
    fc._pending.append(shot_hit)  # noqa: SLF001 — 테스트 전용 내부 접근
    exactly_on_boundary = shot_at + np.array([1.0, 0.0, 0.0])  # 정확히 반경만큼 이탈
    results = fc.update(now=0.5 + 1e-9, ground_truth_position_world=exactly_on_boundary, ground_truth_timestamp=0.5)
    check("반경과 정확히 같은 이탈 거리는 HIT (<=)", results[0].verdict == "HIT", f"{results[0].miss_distance_m}")

    shot_miss = FiredShot(4, 0.0, launch, az, el, 0.5, launch, 1.0, 920.0, "auto")
    fc._pending.append(shot_miss)  # noqa: SLF001
    just_beyond = shot_at + np.array([1.0 + 1e-6, 0.0, 0.0])
    results = fc.update(now=0.5 + 1e-9, ground_truth_position_world=just_beyond, ground_truth_timestamp=0.5)
    check("반경을 살짝 초과한 이탈 거리는 MISS", results[0].verdict == "MISS", f"{results[0].miss_distance_m}")


# ── 4. 격발 모드(auto/manual)와 쿨다운 ───────────────────────────────────


def test_fire_modes_and_cooldown():
    config = default_config()

    class DummySolution:
        azimuth = 0.0
        elevation = 0.0
        time_of_flight = 0.05
        lead_point_world = Vector3(10.0, 0.0, 0.0)

    direction = np.array([1.0, 0.0, 0.0])

    # auto 모드: READY 가 계속 참이면 fire_rate_rpm 쿨다운 간격으로만 격발한다.
    fc_auto = FireController(
        projectile_config=config.projectile,
        muzzle_velocity_mps=920.0,
        fire_mode="auto",
        fire_rate_rpm=600.0,  # 쿨다운 0.1 s
        target_radius_m=0.5,
        ground_truth_timeout_s=0.5,
    )
    fired_times = []
    for i in range(50):  # 0.0 ~ 0.98s, dt=0.02s
        t = i * 0.02
        shot = fc_auto.maybe_fire(t, True, False, DummySolution(), np.zeros(3), direction)
        if shot is not None:
            fired_times.append(t)
    intervals_ok = all((b - a) >= 0.1 - 1e-9 for a, b in zip(fired_times, fired_times[1:]))
    check(
        "auto 모드 쿨다운 간격 준수",
        len(fired_times) >= 8 and intervals_ok,
        f"fired {len(fired_times)}회, intervals_ok={intervals_ok}",
    )

    # manual 모드: 트리거를 누르고 있어도 상승 엣지에서만 한 발.
    fc_manual = FireController(
        projectile_config=config.projectile,
        muzzle_velocity_mps=920.0,
        fire_mode="manual",
        fire_rate_rpm=6000.0,  # 쿨다운 0.01s, 사실상 무제한
        target_radius_m=0.5,
        ground_truth_timeout_s=0.5,
    )
    trigger_sequence = [False, True, True, True, False, True, False, True, True]
    manual_fires = 0
    for i, pressed in enumerate(trigger_sequence):
        shot = fc_manual.maybe_fire(i * 0.02, True, pressed, DummySolution(), np.zeros(3), direction)
        if shot is not None:
            manual_fires += 1
    # 상승 엣지는 인덱스 1(False->True), 5(False->True), 7(False->True) 총 3회.
    check("manual 모드는 트리거 상승 엣지에서만 격발", manual_fires == 3, f"manual_fires={manual_fires}")

    # aim_ready 가 거짓이면 트리거를 눌러도 격발하지 않는다(조준해 없이 사격 금지).
    fc_manual2 = FireController(
        projectile_config=config.projectile,
        muzzle_velocity_mps=920.0,
        fire_mode="manual",
        fire_rate_rpm=6000.0,
        target_radius_m=0.5,
        ground_truth_timeout_s=0.5,
    )
    shot = fc_manual2.maybe_fire(0.0, False, True, DummySolution(), np.zeros(3), direction)
    check("aim_ready=False 이면 트리거를 눌러도 격발하지 않음", shot is None)

    # aim_solution 이 없으면(조준각 없음) 어떤 모드든 격발하지 않는다.
    shot = fc_auto.maybe_fire(100.0, True, False, None, np.zeros(3), None)
    check("aim_solution 이 없으면 격발하지 않음", shot is None)


# ── 5. Ground truth 최신성(timeout) ──────────────────────────────────────


def test_ground_truth_staleness():
    config = default_config()
    fc = FireController(
        projectile_config=config.projectile,
        muzzle_velocity_mps=920.0,
        fire_mode="auto",
        fire_rate_rpm=600.0,
        target_radius_m=0.5,
        ground_truth_timeout_s=0.2,
    )
    from fire_control import FiredShot

    launch = np.zeros(3)
    shot = FiredShot(5, 0.0, launch, 0.0, 0.0, 0.5, launch, 0.5, 920.0, "auto")
    fc._pending.append(shot)  # noqa: SLF001

    impact_point = fc.projectile_position_at(shot, 0.5)

    # ground truth 시각이 판정 시각보다 0.3초 전(허용 timeout 0.2초 초과)이면 무효.
    results = fc.update(now=0.5, ground_truth_position_world=impact_point, ground_truth_timestamp=0.2)
    check(
        "오래된(timeout 초과) ground truth는 UNVERIFIED 처리",
        len(results) == 1 and results[0].verdict == "UNVERIFIED" and not results[0].ground_truth_available,
        f"verdict={results[0].verdict}",
    )

    shot2 = FiredShot(6, 1.0, launch, 0.0, 0.0, 0.5, launch, 0.5, 920.0, "auto")
    fc._pending.append(shot2)  # noqa: SLF001
    impact_point2 = fc.projectile_position_at(shot2, 0.5)
    results2 = fc.update(now=1.5, ground_truth_position_world=impact_point2, ground_truth_timestamp=1.45)
    check(
        "timeout 이내의 최신 ground truth는 정상 판정에 반영",
        len(results2) == 1 and results2[0].ground_truth_available and results2[0].verdict == "HIT",
        f"verdict={results2[0].verdict if results2 else None}",
    )


def main() -> int:
    test_ballistic_analytic_match()
    test_cross_check_against_aiming_manager()
    test_hit_miss_boundary()
    test_fire_modes_and_cooldown()
    test_ground_truth_staleness()

    print()
    if FAILURES:
        print(f"{len(FAILURES)}건 실패: {FAILURES}")
        return 1
    print("전체 통과.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
