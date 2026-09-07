#!/usr/bin/env python3
"""08_stage1_fcs_offline_validation.py — SMASH FCS 1단계 오프라인 정량 검증.

ROS 2 와 Gazebo 없이, 1단계에서 새로 만든 수학 경로만 떼어 내어 검증한다.
검증 대상은 다음 네 가지다.

    A. 포탑 순기구학 / 역기구학 정합성 및 포구 시차의 크기
    B. Image -> Camera -> Scope(포구) -> World 좌표변환 왕복 정확도
    C. 거리 추정기(레이저 게이팅, 바운딩박스 역추정) 정확도와 오차 전파
    D. 종단 조준 파이프라인: 합성 등속 표적에 대한 리드 조준 각오차

검증 방법론은 level1_level2_validation_report.md 의 오라클 방식을 따른다.
즉 정답(GT)을 해석적으로 알고 있는 합성 시나리오를 만들고, 파이프라인의 출력을
그 GT 와 비교한다. 실제 Gazebo 렌더링 이미지나 YOLO 추론은 포함하지 않는다.
따라서 본 스크립트는 "수학 경로가 옳은가"만 답하며, "Gazebo 3D 렌더에서 탐지가
되는가"(2단계 과제)는 답하지 않는다.

실행:
    python scripts/08_stage1_fcs_offline_validation.py
결과:
    results/intermediate_results/stage1_fcs_offline_validation.json  (전체 수치)
    results/intermediate_results/stage1_range_sensitivity.csv        (거리오차 민감도, utf-8-sig)
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CIWS_ROOT = REPO_ROOT / "ciws_turret_aerial_object_detection-main"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CIWS_ROOT / "drone_sim"))

from aiming_engine.aiming_manager import AimingManager  # noqa: E402
from aiming_engine.config import load_config  # noqa: E402
from aiming_engine.coordinate_transform import CoordinateTransform  # noqa: E402
from aiming_engine.projectile_model import azimuth_elevation_to_direction  # noqa: E402
from aiming_engine.types import TrackerFrame, Vector3  # noqa: E402
from drone_sim.smash_fcs import turret_kinematics as tk  # noqa: E402
from drone_sim.smash_fcs.pointcloud_utils import ranges_from_xyz_buffer  # noqa: E402
from drone_sim.smash_fcs.range_estimator import (  # noqa: E402
    BoundingBoxRange,
    ConeLaserRange,
    HybridRangeEstimator,
)

# ── 카메라 모델 (ciws_turret.urdf.xacro 의 turret_camera 센서와 동일) ──────────
IMAGE_W = 640
IMAGE_H = 640
HORIZONTAL_FOV_RAD = 0.4
FX = (IMAGE_W / 2.0) / math.tan(HORIZONTAL_FOV_RAD / 2.0)
FY = (IMAGE_H / 2.0) / math.tan(HORIZONTAL_FOV_RAD / 2.0)  # 정사각 이미지라 종횡비 1
CX = IMAGE_W / 2.0
CY = IMAGE_H / 2.0
INTRINSICS = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], dtype=np.float64)

TARGET_SIZE_M = 0.47  # 표적 특성 크기 가정 (smash_fcs.yaml 과 동일)
FPS = 30.0
# 동축 광각 레이저 원뿔 반각 (ciws_turret.urdf.xacro 의 gpu_lidar 와 동일)
LASER_CONE_HALF_ANGLE_RAD = 0.04

SCOPE_OFFSET_PINHOLE = tk.scope_to_muzzle_offset_in_pinhole_frame()

# ── 탄종: 5.56 x 45mm M855 (62 gr) ───────────────────────────────────────────
# 포구초속은 20인치 총열 기준값에 가깝다. 14.5인치 M4 계열은 약 880 m/s.
MUZZLE_VELOCITY_MPS = 920.0
PROJECTILE_MASS_KG = 0.00402
PROJECTILE_DIAMETER_M = 0.00570  # 강선 홈 기준 실제 탄두 직경
DRAG_MODEL = "G7"  # M855 는 보트테일 스피처이므로 평저탄용 G1 보다 G7 이 형상에 가깝다


def project_to_pixel(point_base: np.ndarray, camera_pose: np.ndarray) -> tuple[float, float, float]:
    """base 프레임 점을 이미지 픽셀로 투영. (u, v, 카메라로부터의 경사거리) 반환.

    camera_link_optical 은 FLU(X 전, Y 좌, Z 상) 이므로 표준 핀홀 축으로 라벨을
    바꾼다: x_right = -y_flu, y_down = -z_flu, z_forward = x_flu.
    """

    rel = np.asarray(camera_pose)[:3, :3].T @ (np.asarray(point_base) - np.asarray(camera_pose)[:3, 3])
    slant = float(np.linalg.norm(rel))
    z = float(rel[0])
    if z <= 1e-9:
        return float("nan"), float("nan"), slant
    u = FX * (-rel[1] / z) + CX
    v = FY * (-rel[2] / z) + CY
    return float(u), float(v), slant


def bbox_from_target(u: float, v: float, slant_m: float, size_m: float = TARGET_SIZE_M):
    """표적을 픽셀 크기 s = f*S/R 의 정사각 바운딩박스로 근사한다."""

    s = FX * size_m / max(slant_m, 1e-6)
    return (u - s / 2.0, v - s / 2.0, u + s / 2.0, v + s / 2.0)


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(math.acos(max(-1.0, min(1.0, cos))))


# ══════════════════════════════════════════════════════════════════════════
# A. 포탑 기구학
# ══════════════════════════════════════════════════════════════════════════
def test_kinematics() -> dict:
    rng = np.random.default_rng(20260904)
    results: dict = {}

    # A1. boresight_direction 이 gun_rotation @ (1,0,0) 과 일치하는가
    fk_err = []
    for _ in range(2000):
        pan = float(rng.uniform(-math.pi, math.pi))
        tilt = float(rng.uniform(tk.TILT_MIN_RAD, tk.TILT_MAX_RAD))
        d1 = tk.boresight_direction(pan, tilt)
        d2 = tk.gun_rotation(pan, tilt) @ np.array([1.0, 0.0, 0.0])
        fk_err.append(float(np.linalg.norm(d1 - d2)))
    results["A1_boresight_vs_rotation_max_err"] = max(fk_err)

    # A2. 역기구학 왕복: 구한 (pan,tilt) 로 포구에서 표적을 정확히 겨누는가
    ik_err_rad = []
    naive_err_rad = []
    ranges = []
    for _ in range(2000):
        azimuth = float(rng.uniform(-math.pi, math.pi))
        elevation = float(rng.uniform(0.05, 1.3))
        distance = float(rng.uniform(15.0, 300.0))
        target = distance * np.array(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ]
        ) + np.array([0.0, 0.0, 1.0])

        pan, tilt = tk.solve_pan_tilt_for_point(target, iterations=8)
        muzzle = tk.point_in_base(np.asarray(tk.GUN_TO_MUZZLE_M), pan, tilt)
        ik_err_rad.append(angle_between(target - muzzle, tk.boresight_direction(pan, tilt)))

        # 포구 오프셋을 무시한 순진한 지향각과의 차이 = 시차의 크기
        pan_n, tilt_n = tk.direction_to_pan_tilt(target)
        naive_err_rad.append(
            angle_between(target - muzzle, tk.boresight_direction(pan_n, tilt_n))
        )
        ranges.append(distance)

    results["A2_ik_residual_max_rad"] = max(ik_err_rad)
    results["A2_ik_residual_max_urad"] = max(ik_err_rad) * 1e6

    # A3. 시차를 두 층으로 분리해 보고한다.
    #   A3a 제품 고유: 조준경 광축과 총열 중심의 높이차가 만드는 시차.
    #       실제 소총에 존재하는 물리량이며 사거리에 반비례한다.
    #   A3b 시뮬레이션 고유: Gazebo 포탑의 회전축 지렛대가 추가로 만드는 시차.
    #       실제 제품에는 없다. 사수는 총 전체를 들고 움직이지 고정 피벗이 없다.
    #       서보를 올바로 구동하려면 시뮬레이션에서는 반영해야 한다.
    sight_height = abs(tk.SCOPE_TO_MUZZLE_M[2])
    results["A3a_product_sight_over_bore_m"] = sight_height
    results["A3a_product_parallax_mrad"] = {
        f"{r:g}m": round(sight_height / r * 1000.0, 4) for r in (10.0, 20.0, 35.0, 50.0, 100.0)
    }
    results["A3a_product_parallax_at_35m_deg"] = round(
        math.degrees(sight_height / 35.0), 4
    )
    results["A3b_sim_turret_total_parallax_mean_mrad"] = float(np.mean(naive_err_rad) * 1000.0)
    idx35 = int(np.argmin(np.abs(np.array(ranges) - 35.0)))
    results["A3b_sim_turret_total_parallax_at_35m_mrad"] = float(naive_err_rad[idx35] * 1000.0)
    results["A3b_설명"] = (
        "A3b 는 포탑 회전축 지렛대까지 포함한 시뮬레이션상의 총 시차다. "
        "제품 정확도에 물리적으로 의미가 있는 값은 A3a 이며, A3b 는 Gazebo 서보 "
        "구동을 올바로 하기 위해 코드가 처리해야 하는 양이다."
    )

    # A4. Pan 언랩: +-pi 경계에서 최단 회전을 선택하는가
    results["A4_unwrap_pi_boundary"] = {
        "target_deg": 179.0,
        "reference_deg": -179.0,
        "unwrapped_deg": math.degrees(
            tk.unwrap_pan(math.radians(179.0), math.radians(-179.0))
        ),
        "step_deg": math.degrees(
            tk.unwrap_pan(math.radians(179.0), math.radians(-179.0)) - math.radians(-179.0)
        ),
    }
    return results


# ══════════════════════════════════════════════════════════════════════════
# B. 좌표변환 왕복
# ══════════════════════════════════════════════════════════════════════════
def test_coordinate_roundtrip() -> dict:
    """픽셀 + 경사거리 -> World 복원이 원래 표적 위치와 일치하는지 확인한다.

    이 검증이 통과해야 smash_fcs_node 의 핵심 가정
    (ScopeConfig.mount_offset_m = 카메라->포구 오프셋(핀홀축),
     camera_extrinsics = 포구 자세) 이 옳다고 말할 수 있다.
    """

    base_config = load_config(REPO_ROOT / "config" / "aiming_engine.yaml")
    scope = replace(base_config.scope, mount_offset_m=SCOPE_OFFSET_PINHOLE)
    transform = CoordinateTransform(scope)

    rng = np.random.default_rng(11)
    errors = []
    launch_point_errors = []
    for _ in range(2000):
        pan = float(rng.uniform(-math.pi, math.pi))
        tilt = float(rng.uniform(0.05, tk.TILT_MAX_RAD))
        camera_pose = tk.camera_optical_pose(pan, tilt)
        scope_pose = tk.muzzle_pose_from_camera_pose(camera_pose)

        # 카메라 화각 안쪽 임의 방향의 표적
        distance = float(rng.uniform(10.0, 400.0))
        du = float(rng.uniform(-0.18, 0.18))
        dv = float(rng.uniform(-0.18, 0.18))
        direction_flu = np.array([1.0, -math.tan(du), -math.tan(dv)])
        direction_flu /= np.linalg.norm(direction_flu)
        target = camera_pose[:3, 3] + camera_pose[:3, :3] @ (direction_flu * distance)

        u, v, slant = project_to_pixel(target, camera_pose)
        frame = TrackerFrame(
            timestamp=0.0,
            bbox=(u - 5, v - 5, u + 5, v + 5),
            center_px=(u, v),
            velocity_px=(0.0, 0.0),
            tracking_confidence=0.9,
            follower_state="stable",
            camera_intrinsics=INTRINSICS,
            camera_extrinsics=scope_pose,
            laser_range_m=slant,
        )
        recovered = transform.image_to_world(frame).to_array()
        errors.append(float(np.linalg.norm(recovered - target)))

        # 발사점이 실제 포구 위치와 일치하는가
        launch = transform.scope_to_world(Vector3(0.0, 0.0, 0.0), scope_pose).to_array()
        muzzle_true = tk.point_in_base(np.asarray(tk.GUN_TO_MUZZLE_M), pan, tilt)
        launch_point_errors.append(float(np.linalg.norm(launch - muzzle_true)))

    return {
        "B1_image_to_world_max_err_m": max(errors),
        "B1_image_to_world_mean_err_m": float(np.mean(errors)),
        "B2_launch_point_vs_muzzle_max_err_m": max(launch_point_errors),
    }


# ══════════════════════════════════════════════════════════════════════════
# C. 거리 추정기
# ══════════════════════════════════════════════════════════════════════════
def test_range_estimator() -> tuple[dict, list[dict]]:
    results: dict = {}

    bbox_estimator = BoundingBoxRange(characteristic_size_m=TARGET_SIZE_M)
    laser = ConeLaserRange(staleness_timeout_s=0.2, noise_sigma_m=0.02)
    hybrid = HybridRangeEstimator(laser=laser, bbox=bbox_estimator)

    # C1. 바운딩박스 역추정: 정수 픽셀 반올림만 있는 이상적 조건
    rows = []
    for true_range in [10.0, 20.0, 35.0, 50.0, 80.0, 120.0, 200.0, 350.0]:
        bbox = bbox_from_target(CX, CY, true_range)
        bbox_int = tuple(float(round(v)) for v in bbox)
        est = bbox_estimator.estimate(bbox_int, FX, FY)
        size_px = bbox_int[2] - bbox_int[0]
        rows.append(
            {
                "true_range_m": true_range,
                "bbox_size_px": round(size_px, 3),
                "estimated_range_m": round(est.distance_m, 4) if est.valid else None,
                "abs_error_m": round(abs(est.distance_m - true_range), 4) if est.valid else None,
                "rel_error_pct": round(abs(est.distance_m - true_range) / true_range * 100.0, 4)
                if est.valid
                else None,
                "reported_sigma_m": round(est.sigma_m, 4) if est.valid else None,
                "valid": est.valid,
            }
        )
    results["C1_bbox_range_table"] = rows

    # C2. 최근접 반사 선택과 교차검증 게이트
    #     원뿔 안에 표적(35 m)과 배경 지면(180 m)이 함께 들어온 경우
    hybrid.reset()
    laser.submit_ranges([float("inf"), 180.0, 35.0, 35.02, float("nan")], timestamp=0.0)
    on_target_bbox = bbox_from_target(CX, CY, 35.0)
    nearest = hybrid.estimate(0.0, on_target_bbox, (CX, CY), FX, FY)

    #     원뿔이 엉뚱한 물체(5 m 앞 조류)를 물어 바운딩박스 추정과 크게 어긋나는 경우
    hybrid.reset()
    laser.submit_ranges([5.0], timestamp=0.0)
    rejected = hybrid.estimate(0.0, on_target_bbox, (CX, CY), FX, FY)

    #     레이저가 오래되어 무효인 경우
    hybrid.reset()
    laser.submit_ranges([35.0], timestamp=0.0)
    stale = hybrid.estimate(5.0, on_target_bbox, (CX, CY), FX, FY)

    results["C2_gating"] = {
        "nearest_return_source": nearest.source,
        "nearest_return_range_m": round(nearest.distance_m, 4),
        "outlier_laser_source": rejected.source,
        "outlier_laser_range_m": round(rejected.distance_m, 4),
        "outlier_rejected_count": hybrid.laser_rejections,
        "stale_laser_falls_back_to": stale.source,
    }

    # C2b. PointCloud2 버퍼 파싱 단위검증
    #      x, y, z (float32) + intensity 로 구성된 점 4개를 손으로 만들어 검증한다.
    pts = [(35.0, 0.0, 0.0, 1.0), (0.0, 3.0, 4.0, 1.0),
           (float("inf"), 0.0, 0.0, 1.0), (float("nan"), 1.0, 1.0, 1.0)]
    buf = b"".join(np.array(p, dtype="<f4").tobytes() for p in pts)
    parsed = ranges_from_xyz_buffer(buf, point_step=16, x_offset=0, y_offset=4, z_offset=8)
    results["C2b_pointcloud_parse"] = {
        "input_points": 4,
        "valid_points": int(parsed.size),
        "ranges": [round(float(v), 6) for v in sorted(parsed)],
        "expected_ranges": [5.0, 35.0],
    }

    # C3. 거리 오차가 조준각 오차로 얼마나 전파되는가
    #     정지 표적이면 시선 방향 자체는 거리와 무관하므로, 오차는 오직 중력 낙하
    #     보정(비행시간 t = R/v0 에 비례)을 통해서만 들어온다.
    #     이동 표적이면 리드각(= v_target * t / R)에도 직접 들어간다.
    muzzle_velocity = MUZZLE_VELOCITY_MPS
    g = 9.81
    sensitivity = []
    for true_range in [20.0, 35.0, 60.0, 100.0]:
        for rel_err in [0.05, 0.10, 0.20, 0.30]:
            wrong = true_range * (1.0 + rel_err)
            drop_true = g * (true_range / muzzle_velocity) / (2.0 * muzzle_velocity)
            drop_wrong = g * (wrong / muzzle_velocity) / (2.0 * muzzle_velocity)
            # 5 m/s 횡방향 표적의 리드각 = v*t/R = v/v0 (거리와 무관) 이므로
            # 거리 오차는 리드각 자체에는 1차적으로 들어가지 않는다.
            # 다만 리드 후 표적 위치를 잘못 잡으므로 각오차 = v*(t_wrong - t_true)/R 이다.
            v_lat = 5.0
            lead_err = v_lat * (wrong - true_range) / muzzle_velocity / true_range
            sensitivity.append(
                {
                    "true_range_m": true_range,
                    "range_rel_error_pct": rel_err * 100.0,
                    "aim_error_mrad_static_target": round(abs(drop_wrong - drop_true) * 1000.0, 6),
                    "aim_error_mrad_lateral_5mps": round(abs(lead_err) * 1000.0, 6),
                }
            )
    results["C3_range_error_sensitivity"] = sensitivity
    return results, rows


# ══════════════════════════════════════════════════════════════════════════
# D. 종단 조준 파이프라인 (합성 등속 표적)
# ══════════════════════════════════════════════════════════════════════════
def run_engagement(
    target_start: np.ndarray,
    target_velocity: np.ndarray,
    duration_s: float,
    range_mode: str,
    enable_drag: bool,
    muzzle_velocity: float = MUZZLE_VELOCITY_MPS,
    feedforward_s: float = 0.0,
) -> dict:
    """폐루프 교전 1회. 포탑이 실제로 움직이며 조준이 수렴하는지까지 본다.

    range_mode:
        "exact"  : 표적까지의 참 경사거리를 그대로 준다 (수학 상한 성능)
        "hybrid" : 보어사이트 레이저 게이팅 + 바운딩박스 폴백 (실제 노드와 동일)
    """

    base_config = load_config(REPO_ROOT / "config" / "aiming_engine.yaml")
    forces = ["gravity", "drag"] if enable_drag else ["gravity"]
    projectile = replace(
        base_config.projectile,
        muzzle_velocity=muzzle_velocity,
        forces=forces,
        mass_kg=PROJECTILE_MASS_KG,
        diameter_m=PROJECTILE_DIAMETER_M,
        drag_model=DRAG_MODEL,
    )
    config = replace(
        base_config,
        projectile=projectile,
        scope=replace(base_config.scope, mount_offset_m=SCOPE_OFFSET_PINHOLE),
    )
    manager = AimingManager(config=config)

    bbox_estimator = BoundingBoxRange(characteristic_size_m=TARGET_SIZE_M)
    laser = ConeLaserRange(staleness_timeout_s=0.2, noise_sigma_m=0.02)
    hybrid = HybridRangeEstimator(laser=laser, bbox=bbox_estimator)

    noise_rng = np.random.default_rng(4242)
    feedforward = tk.GoalFeedForward(
        lead_time_s=feedforward_s,
        max_rate_rad_s=max(tk.PAN_MAX_RATE_RAD_S, tk.TILT_MAX_RATE_RAD_S),
    )
    dt = 1.0 / FPS
    n_steps = int(round(duration_s / dt))

    # 포탑 초기 지향: 표적을 대충 향하되 정확히는 아니게 (수렴 과정을 보기 위함)
    pan_cmd, tilt_cmd = tk.direction_to_pan_tilt(target_start)
    pan_cmd += math.radians(1.0)
    tilt_cmd = tk.clamp(tilt_cmd + math.radians(1.0), tk.TILT_MIN_RAD, tk.TILT_MAX_RAD)
    pan_goal, tilt_goal = pan_cmd, tilt_cmd

    log = []
    for i in range(n_steps):
        t = i * dt
        target = target_start + target_velocity * t

        camera_pose = tk.camera_optical_pose(pan_cmd, tilt_cmd)
        scope_pose = tk.muzzle_pose_from_camera_pose(camera_pose)

        u, v, slant = project_to_pixel(target, camera_pose)
        in_fov = (0.0 <= u < IMAGE_W) and (0.0 <= v < IMAGE_H) and math.isfinite(u)
        if not in_fov:
            log.append({"t": t, "in_fov": False})
            continue

        # 추적기 출력 근사: 정수 픽셀 바운딩박스
        bbox = tuple(float(round(x)) for x in bbox_from_target(u, v, slant))
        center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

        if range_mode == "exact":
            range_m = slant
            range_source = "exact"
        else:
            # 광각 원뿔 레이저 모사: 표적이 카메라 보어사이트 기준 원뿔 반각 안에
            # 들어와 있을 때만 표적 반사를 준다. 원뿔 밖이면 반사 없음(하늘).
            # 실제 gpu_lidar 의 가우시안 노이즈(stddev 0.02 m)도 반영한다.
            offset_rad = angle_between(
                target - camera_pose[:3, 3], camera_pose[:3, :3] @ np.array([1.0, 0.0, 0.0])
            )
            if offset_rad <= LASER_CONE_HALF_ANGLE_RAD:
                laser.submit_ranges([slant + float(noise_rng.normal(0.0, 0.02))], timestamp=t)
            est = hybrid.estimate(t, bbox, (CX, CY), FX, FY)
            if not est.valid:
                log.append({"t": t, "in_fov": True, "range_valid": False})
                continue
            range_m = est.distance_m
            range_source = est.source

        frame = TrackerFrame(
            timestamp=t,
            bbox=bbox,
            center_px=center,
            velocity_px=(0.0, 0.0),
            tracking_confidence=0.9,
            follower_state="stable",
            camera_intrinsics=INTRINSICS,
            camera_extrinsics=scope_pose,
            laser_range_m=range_m,
        )
        output = manager.update(frame)
        solution = output.aim_solution

        direction_scope = azimuth_elevation_to_direction(solution.azimuth, solution.elevation)
        aim_dir_base = scope_pose[:3, :3] @ direction_scope
        aim_dir_base = aim_dir_base / np.linalg.norm(aim_dir_base)

        pan_raw, tilt_raw = tk.direction_to_pan_tilt(aim_dir_base)
        pan_goal, tilt_goal = feedforward.apply(tk.unwrap_pan(pan_raw, pan_cmd), tilt_raw, t)
        tilt_goal = tk.clamp(tilt_goal, tk.TILT_MIN_RAD, tk.TILT_MAX_RAD)

        # 현재 명령 지향과 요구 지향의 각차 = 조준 잔차
        pointing_error_rad = angle_between(tk.boresight_direction(pan_cmd, tilt_cmd), aim_dir_base)
        # 리드각 크기: 표적 직접 조준과 탄도 보정 조준의 각차
        muzzle = scope_pose[:3, 3]
        lead_magnitude_rad = angle_between(target - muzzle, aim_dir_base)

        log.append(
            {
                "t": t,
                "in_fov": True,
                "range_valid": True,
                "range_source": range_source,
                "range_m": range_m,
                "true_range_m": slant,
                "state": output.state.value,
                "aim_ready": bool(output.aim_ready),
                "tof_s": solution.time_of_flight,
                "converged": bool(solution.solver_converged),
                "iterations": int(solution.solver_iterations),
                "residual_m": solution.residual_norm_m,
                "pointing_error_deg": math.degrees(pointing_error_rad),
                "lead_magnitude_mrad": lead_magnitude_rad * 1000.0,
                "aim_error_mrad": float(output.debug.get("aim_error_mrad", float("nan"))),
                "target_speed_est": float(output.debug.get("target_speed", float("nan"))),
                # 표적의 시선각속도. 지향 지연 오차가 이 값에 비례하는지 보기 위함.
                "target_angular_rate_deg_s": math.degrees(
                    float(np.linalg.norm(np.cross(target - muzzle, target_velocity)))
                    / max(float(np.dot(target - muzzle, target - muzzle)), 1e-9)
                ),
            }
        )

        # 포탑 서보 (속도 제한)
        pan_cmd = tk.rate_limit(pan_goal, pan_cmd, tk.PAN_MAX_RATE_RAD_S, dt)
        tilt_cmd = tk.clamp(
            tk.rate_limit(tilt_goal, tilt_cmd, tk.TILT_MAX_RATE_RAD_S, dt),
            tk.TILT_MIN_RAD,
            tk.TILT_MAX_RAD,
        )

    solved = [r for r in log if r.get("range_valid")]
    if not solved:
        return {"frames": len(log), "solved_frames": 0}

    # 정상상태 = 마지막 1초
    tail = solved[-int(FPS) :]
    settle_frame = None
    for idx, row in enumerate(solved):
        if row["pointing_error_deg"] < 0.5 and all(
            r["pointing_error_deg"] < 0.5 for r in solved[idx : idx + 5]
        ):
            settle_frame = idx
            break

    speed_true = float(np.linalg.norm(target_velocity))
    return {
        "frames": len(log),
        "solved_frames": len(solved),
        "range_sources": sorted({r["range_source"] for r in solved}),
        "laser_share_all_frames": float(np.mean([r["range_source"] == "laser" for r in solved])),
        "laser_share_tail": float(np.mean([r["range_source"] == "laser" for r in tail])),
        "converged_ratio": float(np.mean([r["converged"] for r in solved])),
        "mean_iterations": float(np.mean([r["iterations"] for r in solved])),
        "settle_frames_to_0p5deg": settle_frame,
        "settle_time_s": None if settle_frame is None else round(settle_frame / FPS, 4),
        "steady_pointing_error_deg_mean": float(np.mean([r["pointing_error_deg"] for r in tail])),
        "steady_pointing_error_deg_max": float(np.max([r["pointing_error_deg"] for r in tail])),
        "steady_aim_error_mrad_mean": float(np.mean([r["aim_error_mrad"] for r in tail])),
        "steady_lead_magnitude_mrad_mean": float(np.mean([r["lead_magnitude_mrad"] for r in tail])),
        "steady_tof_ms_mean": float(np.mean([r["tof_s"] for r in tail]) * 1000.0),
        "steady_range_abs_error_m_mean": float(
            np.mean([abs(r["range_m"] - r["true_range_m"]) for r in tail])
        ),
        "target_speed_true_mps": speed_true,
        "steady_target_speed_est_mps_mean": float(np.mean([r["target_speed_est"] for r in tail])),
        "ready_ratio_tail": float(np.mean([r["aim_ready"] for r in tail])),
        "steady_target_angular_rate_deg_s": float(
            np.mean([r["target_angular_rate_deg_s"] for r in tail])
        ),
        # 지향 지연 오차를 표적 각속도로 나눈 값. 서보 명령이 조준해보다 몇 초
        # 뒤처지는지를 뜻하며, 프레임 주기(1/30 s = 0.0333 s)와 비교하면 지연의
        # 원인이 프레임 지연인지 속도 한계인지 구분할 수 있다.
        "implied_command_lag_s": (
            None
            if float(np.mean([r["target_angular_rate_deg_s"] for r in tail])) < 1e-6
            else float(
                np.mean([r["pointing_error_deg"] for r in tail])
                / np.mean([r["target_angular_rate_deg_s"] for r in tail])
            )
        ),
    }


def test_engagements() -> dict:
    # 1단계 검증 조건: 사거리 약 35 m, 앙각 약 20도
    hover = np.array([33.0, 0.0, 12.5])
    scenarios = {
        "D1_정지호버링_거리exact_중력만": dict(
            target_start=hover, target_velocity=np.zeros(3), duration_s=6.0,
            range_mode="exact", enable_drag=False),
        "D2_정지호버링_거리hybrid_중력만": dict(
            target_start=hover, target_velocity=np.zeros(3), duration_s=6.0,
            range_mode="hybrid", enable_drag=False),
        "D3_정지호버링_거리hybrid_공기저항": dict(
            target_start=hover, target_velocity=np.zeros(3), duration_s=6.0,
            range_mode="hybrid", enable_drag=True),
        "D4_등속직선_2mps_거리exact": dict(
            target_start=hover, target_velocity=np.array([0.0, 2.0, 0.0]), duration_s=6.0,
            range_mode="exact", enable_drag=False),
        "D5_등속직선_2mps_거리hybrid": dict(
            target_start=hover, target_velocity=np.array([0.0, 2.0, 0.0]), duration_s=6.0,
            range_mode="hybrid", enable_drag=False),
        "D6_등속직선_5mps_거리hybrid": dict(
            target_start=hover, target_velocity=np.array([0.0, 5.0, 0.0]), duration_s=6.0,
            range_mode="hybrid", enable_drag=False),
        # 로드맵 4단계(FPV 15~20 m/s)의 조준 지연 한계를 미리 확인하기 위한 시나리오.
        # 1단계 범위 밖이지만, 지향 지연이 각속도에 비례하는지 확인하고 4단계에서
        # 어떤 보정이 필요한지 근거를 남기기 위해 함께 측정한다.
        "D7_등속직선_15mps_거리hybrid_4단계선행확인": dict(
            target_start=np.array([33.0, -25.0, 12.5]),
            target_velocity=np.array([0.0, 15.0, 0.0]), duration_s=4.0,
            range_mode="hybrid", enable_drag=False),
        # 속도 피드포워드(1프레임 외삽)를 켰을 때의 개선 효과. 잡음 없는 합성
        # 궤적 기준이므로 상한 성능이며, 실제 추적 잡음에서는 재평가가 필요하다.
        "D8_등속직선_5mps_피드포워드ON": dict(
            target_start=hover, target_velocity=np.array([0.0, 5.0, 0.0]), duration_s=6.0,
            range_mode="hybrid", enable_drag=False, feedforward_s=1.0 / FPS),
        "D9_등속직선_15mps_피드포워드ON": dict(
            target_start=np.array([33.0, -25.0, 12.5]),
            target_velocity=np.array([0.0, 15.0, 0.0]), duration_s=4.0,
            range_mode="hybrid", enable_drag=False, feedforward_s=1.0 / FPS),
    }
    return {name: run_engagement(**kwargs) for name, kwargs in scenarios.items()}


# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    report = {
        "설명": "SMASH FCS 1단계 오프라인 정량 검증 결과",
        "카메라모델": {
            "image_w": IMAGE_W, "image_h": IMAGE_H,
            "horizontal_fov_rad": HORIZONTAL_FOV_RAD,
            "fx": FX, "fy": FY, "cx": CX, "cy": CY,
            "pixel_ifov_mrad": HORIZONTAL_FOV_RAD / IMAGE_W * 1000.0,
        },
        "무기제원": {
            "탄종": "5.56 x 45mm M855 (62 gr)",
            "muzzle_velocity_mps": MUZZLE_VELOCITY_MPS,
            "projectile_mass_kg": PROJECTILE_MASS_KG,
            "projectile_diameter_m": PROJECTILE_DIAMETER_M,
            "drag_model": DRAG_MODEL,
            "단위질량당_단면적_m2_per_kg": round(
                math.pi * (PROJECTILE_DIAMETER_M / 2.0) ** 2 / PROJECTILE_MASS_KG, 6
            ),
            "sight_height_over_bore_m": tk.SIGHT_HEIGHT_OVER_BORE_M,
            "muzzle_forward_of_scope_m": tk.MUZZLE_FORWARD_OF_SCOPE_M,
        },
        "scope_mount_offset_pinhole_m": list(SCOPE_OFFSET_PINHOLE),
        "A_기구학": test_kinematics(),
        "B_좌표변환": test_coordinate_roundtrip(),
    }
    range_results, range_rows = test_range_estimator()
    report["C_거리추정"] = range_results
    report["D_종단조준"] = test_engagements()

    out_dir = REPO_ROOT / "results" / "intermediate_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "stage1_fcs_offline_validation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    csv_path = out_dir / "stage1_range_sensitivity.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(range_rows[0].keys()))
        writer.writeheader()
        writer.writerows(range_rows)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n저장 완료: {json_path}")
    print(f"저장 완료: {csv_path}")


if __name__ == "__main__":
    main()
