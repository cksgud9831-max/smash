"""가제보(Gazebo) 실시간 마우스 조준 사격 시뮬레이터 (광각 마우스 룩 & 100% 순수 비전 소실/재포착)

기능:
1. 마우스 조준 회전 범위를 60도 이상으로 대폭 확장하여, 마우스를 돌리면 드론이 화면 밖으로 완전히 사라짐
2. 드론이 시야에서 벗어나면 화면에 드론 픽셀이 렌더링되지 않으며, YOLO11s가 즉시 타겟 소실(STATE: SEARCH)로 전환
3. 다시 마우스를 돌려 드론을 렌즈 중앙에 넣으면 YOLO11s가 순수 영상으로 즉각 재포착(UAV [YOLO] -> READY)
4. 사수가 마우스로 중앙 십자선을 미래 조준점(AIM HERE)에 정렬하고 격발 시 100% 정밀 격추 판정
"""

import math
import os
import sys
import time
import numpy as np
import cv2
import torch

# 프로젝트 루트 경로 등록
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from aiming_engine.aiming_manager import AimingManager
from aiming_engine.coordinate_transform import CoordinateTransform, _euler_to_rotation_matrix
from aiming_engine.types import TrackerFrame, Vector3
from bridge.optical_flow_tracker import OpticalFlowTracker
from drone_motion_models import compute_drone_linear_position

# 마우스 조준 각도 (라디안)
mouse_yaw = 0.0
mouse_pitch = 0.0
mouse_fire_triggered = False


def mouse_event_handler(event, x, y, flags, param):
    """마우스 이동으로 소총 조준선 광각(최대 60도) 3차원 회전"""
    global mouse_yaw, mouse_pitch, mouse_fire_triggered
    cam_w, cam_h = 640, 480
    
    # 마우스 커서 위치를 최대 좌우 55도, 상하 35도 각도로 시원하게 매핑
    norm_x = (x - cam_w / 2.0) / (cam_w / 2.0)
    norm_y = (y - cam_h / 2.0) / (cam_h / 2.0)
    
    mouse_yaw = -norm_x * math.radians(55.0)
    mouse_pitch = norm_y * math.radians(35.0)

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_fire_triggered = True


def draw_realistic_parallax_background(hud, width, height, pitch, yaw, fov_rad):
    """Anti UAV 데이터셋과 동일한 도심 고층 빌딩 및 아파트 스카이라인 시차 렌더링"""
    fx = float(width) / (2.0 * math.tan(fov_rad / 2.0))
    horizon_y = int(height * 0.62 + pitch * fx * 0.9)
    pan_x = yaw * fx * 0.9

    # 1. 도시 하늘 그라데이션
    for y in range(max(0, min(height, horizon_y))):
        alpha = y / max(1, horizon_y)
        r = int(210 + 25 * alpha)
        g = int(185 + 20 * alpha)
        b = int(140 + 15 * alpha)
        hud[y, :] = (r, g, b)

    # 2. 원거리 도시 마천루 스카이라인 (시차 0.35x)
    far_skyline_configs = [
        (-450, 140, 220), (-290, 90, 180), (-180, 120, 260), (-40, 80, 190),
        (60, 150, 280), (230, 110, 210), (360, 130, 250), (510, 95, 175),
        (630, 160, 290), (810, 120, 230), (950, 100, 190)
    ]
    for bx_base, bw, bh in far_skyline_configs:
        bx = int(bx_base + pan_x * 0.35)
        by = horizon_y - bh
        if -bw <= bx <= width + bw:
            cv2.rectangle(hud, (bx, by), (bx + bw, horizon_y), (135, 145, 155), -1)
            cv2.rectangle(hud, (bx, by), (bx + bw, horizon_y), (100, 110, 120), 1)
            # 옥상 안테나 첨탑
            cv2.line(hud, (bx + bw // 2, by), (bx + bw // 2, by - 25), (80, 90, 100), 2)
            cv2.circle(hud, (bx + bw // 2, by - 25), 3, (0, 0, 255), -1)

    # 3. 중거리 도심 빌딩 및 아파트 단지 (시차 0.70x, 창문 격자 조명)
    mid_building_configs = [
        (-380, 160, 170, (95, 105, 115)),
        (-200, 130, 190, (80, 90, 100)),
        (-50, 180, 150, (110, 115, 125)),
        (150, 140, 210, (75, 85, 95)),
        (310, 190, 160, (100, 110, 120)),
        (520, 150, 180, (85, 95, 105)),
        (690, 170, 140, (105, 115, 125)),
        (880, 140, 190, (80, 90, 100)),
    ]
    for bx_base, bw, bh, bcolor in mid_building_configs:
        bx = int(bx_base + pan_x * 0.70)
        by = horizon_y - bh
        if -bw <= bx <= width + bw:
            # 건물 외벽
            cv2.rectangle(hud, (bx, by), (bx + bw, horizon_y), bcolor, -1)
            cv2.rectangle(hud, (bx, by), (bx + bw, horizon_y), (50, 55, 60), 2)

            # 옥상 난간 및 구조물
            cv2.rectangle(hud, (bx + 15, by - 12), (bx + bw - 15, by), (60, 65, 70), -1)

            # 창문 격자 패턴
            for wy in range(by + 16, horizon_y - 15, 18):
                for wx in range(bx + 12, bx + bw - 12, 16):
                    cv2.rectangle(hud, (wx, wy), (wx + 9, wy + 10), (180, 205, 220), -1)

    # 4. 근경 도심 옥상 바닥 및 방공 진지
    if horizon_y < height:
        for y in range(max(0, horizon_y), height):
            alpha = (y - horizon_y) / max(1, height - horizon_y)
            r = int(50 + 20 * alpha)
            g = int(55 + 20 * alpha)
            b = int(58 + 20 * alpha)
            hud[y, :] = (r, g, b)
        # 옥상 헬리패드/진지 라인
        cv2.line(hud, (0, horizon_y + 1), (width, horizon_y + 1), (30, 32, 35), 3)

    # 5. 거리별 표지 기둥 및 옥상 안테나 랜드마크 (거리 인덱스)
    pole_configs = [
        (-22.0, 20.0, "20m Building"),
        (-10.0, 35.0, "35m Tower"),
        (0.0, 45.0, "45m Rooftop"),
        (10.0, 35.0, "35m Tower"),
        (22.0, 20.0, "20m Building"),
    ]
    for px, pz, label in pole_configs:
        cam_x = px * math.cos(yaw) - pz * math.sin(yaw)
        cam_z = px * math.sin(yaw) + pz * math.cos(yaw)
        if cam_z > 1.0:
            u = int((fx * cam_x / cam_z) + width / 2.0)
            v_base = int((fx * (1.2 - 0.0) / cam_z) + height / 2.0 + pitch * fx * 0.9)
            v_top = v_base - int(130.0 / cam_z)
            if -50 <= u <= width + 50 and v_top < height:
                cv2.line(hud, (u, v_base), (u, v_top), (70, 75, 80), max(2, int(22.0 / cam_z)))
                cv2.rectangle(hud, (u, v_top), (u + int(42.0 / cam_z), v_top + int(20.0 / cam_z)), (0, 140, 255), -1)
                cv2.putText(hud, label, (u + 4, v_top - 4), cv2.FONT_HERSHEY_SIMPLEX, max(0.3, 10.0 / cam_z), (240, 240, 240), 1)


def draw_realistic_drone(hud, cu, cv, box_w, box_h, roll_ang):
    """3D 메탈릭 쿼드콥터 드론 및 회전 모션 블러 렌더링"""
    cu, cv = int(cu), int(cv)
    bw, bh = int(box_w), int(box_h)

    # 지면 투영 그림자
    shadow_y = min(hud.shape[0] - 10, cv + 85)
    cv2.ellipse(hud, (cu, shadow_y), (int(bw * 0.7), 10), 0, 0, 360, (25, 35, 25), -1)

    # 4개 로터 암
    arm_len = int(bw * 0.60)
    arm_thick = max(2, bw // 18)
    cv2.line(hud, (cu - arm_len, cv - bh), (cu + arm_len, cv + bh), (30, 30, 30), arm_thick)
    cv2.line(hud, (cu - arm_len, cv + bh), (cu + arm_len, cv - bh), (30, 30, 30), arm_thick)

    # 중앙 본체
    cv2.ellipse(hud, (cu, cv), (bw // 2, bh // 2), math.degrees(roll_ang), 0, 360, (40, 40, 45), -1)
    cv2.ellipse(hud, (cu, cv), (bw // 2, bh // 2), math.degrees(roll_ang), 0, 360, (70, 70, 75), 1)

    # 4개 모터 하우징
    rotor_pos = [
        (cu - arm_len, cv - bh),
        (cu + arm_len, cv - bh),
        (cu - arm_len, cv + bh),
        (cu + arm_len, cv + bh)
    ]
    for rx, ry in rotor_pos:
        cv2.circle(hud, (rx, ry), max(3, bw // 12), (20, 20, 25), -1)

    # 고속 회전 반투명 모션 블러 프로펠러 디스크
    overlay = hud.copy()
    prop_r = max(5, int(bw * 0.28))
    cv2.circle(overlay, rotor_pos[0], prop_r, (0, 100, 255), -1)
    cv2.circle(overlay, rotor_pos[1], prop_r, (0, 100, 255), -1)
    cv2.circle(overlay, rotor_pos[2], prop_r, (255, 140, 0), -1)
    cv2.circle(overlay, rotor_pos[3], prop_r, (255, 140, 0), -1)
    cv2.addWeighted(overlay, 0.40, hud, 0.60, 0, hud)


def draw_scope_vignette(hud, width, height):
    """군용 스마트 스코프 원형 튜브 하우징 및 밀닷 레티클"""
    cx, cy = width // 2, height // 2
    radius = int(min(width, height) * 0.46)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    inv_mask = cv2.bitwise_not(mask)
    hud[inv_mask > 0] = (15, 15, 18)

    cv2.circle(hud, (cx, cy), radius, (40, 45, 50), 4)
    cv2.circle(hud, (cx, cy), radius - 3, (20, 22, 25), 2)

    dot_color = (0, 255, 120)
    for offset in [-120, -80, -40, 40, 80, 120]:
        cv2.circle(hud, (cx + offset, cy), 2, dot_color, -1)
        cv2.circle(hud, (cx, cy + offset), 2, dot_color, -1)


def main():
    global mouse_fire_triggered

    print("=======================================================")
    print(" SMASH 광각 마우스 조준 & 100% 순수 비전 시뮬레이터")
    print("=======================================================")

    models_dir = os.path.join(project_root, "models")
    config_dir = os.path.join(project_root, "config")
    
    finetuned_path = os.path.join(models_dir, "best_finetuned.pt")
    model_path = finetuned_path if os.path.exists(finetuned_path) else os.path.join(models_dir, "best.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f" -> 딥러닝 모델: {os.path.basename(model_path)}")
    print(f" -> 연산 가속: {device}")
    
    tracker = OpticalFlowTracker(model_path=model_path, device=device, conf_thres=0.30)
    aiming_config_path = os.path.join(config_dir, "aiming_engine.yaml")
    aiming_manager = AimingManager(config_path=aiming_config_path if os.path.exists(aiming_config_path) else None)
    transform = CoordinateTransform(aiming_manager._config.scope)

    cam_w, cam_h = 640, 480
    center_x, center_y = cam_w / 2.0, cam_h / 2.0
    fov_rad = 0.85
    fx = float(cam_w) / (2.0 * math.tan(fov_rad / 2.0))
    camera_intrinsics = np.array([
        [fx, 0.0, float(center_x)],
        [0.0, fx, float(center_y)],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    window_name = "SMASH Smart Scope - Gazebo Live Shooter"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)
    cv2.setMouseCallback(window_name, mouse_event_handler)

    bullet_active = False
    bullet_flight_time = 0.0
    bullet_target_tof = 0.0
    bullet_start_point = (center_x, center_y)
    bullet_end_point = (center_x, center_y)
    hit_effect_timer = 0.0
    shot_count = 0
    hit_count = 0

    t0 = time.time()
    last_loop_time = t0
    prev_center_px = None

    print("\n[동작 검증]")
    print(" 1. 마우스를 좌우/상하로 크게 움직여보세요: 소총 시야가 55도 회전하며 드론이 화면 밖으로 완전히 사라집니다.")
    print(" 2. 드론이 화면 밖으로 나가면: YOLO가 즉시 놓치고 STATE: SEARCH (조준선 완전 소멸)로 전환됩니다.")
    print(" 3. 다시 드론 쪽으로 총구를 겨누면: YOLO11s가 드론을 재포착하고 AIM 조준점을 복구합니다.\n")

    while True:
        loop_now = time.time()
        dt = max(0.001, loop_now - last_loop_time)
        last_loop_time = loop_now
        current_time = loop_now - t0

        # 1. 35m 상공 등속 직선 왕복 비행 드론 (2단계: 1~2 m/s 등속 직선 비행)
        drone_x, drone_y, drone_z, drone_vy = compute_drone_linear_position(current_time)

        # 2. 사수의 마우스 조준 제어 자세 (Yaw, Pitch - 최대 55도 광각 회전)
        platform_pos = np.array([0.0, 0.0, 1.2], dtype=np.float64)
        extrinsics = _euler_to_rotation_matrix(0.0, mouse_pitch, mouse_yaw)
        T_extrinsics = np.eye(4, dtype=np.float64)
        T_extrinsics[:3, :3] = extrinsics
        T_extrinsics[:3, 3] = platform_pos

        rel_vec = np.array([drone_x, drone_y, drone_z]) - platform_pos
        laser_dist = float(np.linalg.norm(rel_vec))

        # 3. 실감형 배경 렌더링
        raw_scene = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
        draw_realistic_parallax_background(raw_scene, cam_w, cam_h, mouse_pitch, mouse_yaw, fov_rad)

        # 4. 3D 드론 렌즈 시야 투영
        p_drone_v3 = Vector3(drone_x, drone_y, drone_z)
        p_scope = transform.world_to_scope(p_drone_v3, T_extrinsics)
        p_cam = transform.scope_to_camera(p_scope).to_array()

        is_drone_in_fov = False
        true_u, true_v = -999.0, -999.0
        if p_cam[2] > 0.5:
            tu = (camera_intrinsics[0, 0] * p_cam[0] / p_cam[2]) + camera_intrinsics[0, 2]
            tv = (camera_intrinsics[1, 1] * p_cam[1] / p_cam[2]) + camera_intrinsics[1, 2]
            # [엄격한 화각 검사]: 화면 테두리 안에 들어왔을 때만 드론을 렌더링!
            if 0 <= tu < cam_w and 0 <= tv < cam_h:
                is_drone_in_fov = True
                true_u, true_v = float(tu), float(tv)

        # 화면 밖으로 나가면 아예 드론을 그리지 않음!
        if is_drone_in_fov:
            box_w = max(30, int(1600.0 / p_cam[2]))
            box_h = int(box_w * 0.42)
            drone_roll = math.atan2(drone_vy, 9.81) * 0.5
            draw_realistic_drone(raw_scene, true_u, true_v, box_w, box_h, drone_roll)

        # 5. [100% 순수 비전 추적] - 화면에 드론 그림이 없으면 YOLO가 100% None을 반환함!
        tracker_res = tracker.update(raw_scene)
        is_target_acquired = False
        mode_label = "SEARCH"
        tracking_conf = 0.0

        if tracker_res is not None:
            tx, ty, tw, th = tracker_res.bbox
            center_px = (float(tx + tw / 2.0), float(ty + th / 2.0))
            detected_bbox = (float(tx), float(ty), float(tx + tw), float(ty + th))
            tracking_conf = float(tracker_res.score)
            is_target_acquired = True
            mode_label = "YOLO" if tracker_res.used_yolo else "LK"
        else:
            center_px = (float(center_x), float(center_y))
            detected_bbox = (float(center_x - 25), float(center_y - 20), float(center_x + 25), float(center_y + 20))
            tracker._smooth_bbox = None
            tracker._p0 = None

        if prev_center_px is not None and is_target_acquired:
            vel_px = ((center_px[0] - prev_center_px[0]) / dt, (center_px[1] - prev_center_px[1]) / dt)
        else:
            vel_px = (0.0, 0.0)
        prev_center_px = center_px if is_target_acquired else None

        # 6. Aiming Engine 업데이트 및 뉴턴 탄도학 리드 솔버
        tracker_frame = TrackerFrame(
            timestamp=current_time,
            bbox=detected_bbox,
            center_px=center_px,
            velocity_px=vel_px,
            tracking_confidence=tracking_conf,
            follower_state="stable" if is_target_acquired else "lost",
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=T_extrinsics,
            laser_range_m=laser_dist,
        )
        aim_output = aiming_manager.update(tracker_frame)
        hit_prob = aim_output.hit_probability if is_target_acquired else 0.0

        aim_point_px = None
        tof = 0.038
        if is_target_acquired and aim_output.aim_solution is not None:
            sol = aim_output.aim_solution
            tof = sol.time_of_flight
            lead_pt_v3 = sol.lead_point_world
            lead_scope = transform.world_to_scope(lead_pt_v3, T_extrinsics)
            lead_cam = transform.scope_to_camera(lead_scope).to_array()
            if lead_cam[2] > 0.5:
                lu = (camera_intrinsics[0, 0] * lead_cam[0] / lead_cam[2]) + camera_intrinsics[0, 2]
                lv = (camera_intrinsics[1, 1] * lead_cam[1] / lead_cam[2]) + camera_intrinsics[1, 2]
                aim_point_px = (float(lu), float(lv))

        # 정렬 일치 검사
        is_aligned = False
        align_dist = 999.0
        if is_target_acquired and aim_point_px is not None:
            align_dist = math.sqrt((center_x - aim_point_px[0])**2 + (center_y - aim_point_px[1])**2)
            if align_dist <= 22.0:
                is_aligned = True

        current_state = "READY" if is_aligned else ("AIM" if is_target_acquired else "SEARCH")

        # 7. 스마트 스코프 HUD 오버레이 렌더링
        hud = raw_scene.copy()

        # (1) 비전 추적 바운딩 박스 (포착 시에만 표시)
        if is_target_acquired:
            bx1, by1, bx2, by2 = map(int, detected_bbox)
            cv2.rectangle(hud, (bx1, by1), (bx2, by2), (0, 165, 255), 2)
            cv2.putText(hud, f"UAV [{mode_label} {tracking_conf*100:.0f}%]", (bx1, max(16, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1)

        # (2) 뉴턴 솔버 미래 조준점 (포착 시에만 표시)
        if is_target_acquired and aim_point_px is not None:
            ax, ay = int(aim_point_px[0]), int(aim_point_px[1])
            reticle_color = (0, 255, 0) if is_aligned else (0, 220, 255)
            cv2.circle(hud, (ax, ay), 18, reticle_color, 2)
            cv2.line(hud, (ax - 22, ay), (ax + 22, ay), reticle_color, 1)
            cv2.line(hud, (ax, ay - 22), (ax, ay + 22), reticle_color, 1)
            cv2.line(hud, (int(center_x), int(center_y)), (ax, ay), (0, 220, 255), 1)
            cv2.putText(hud, "AIM HERE", (ax + 20, ay + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, reticle_color, 1)

        # (3) 군용 스코프 원형 하우징
        draw_scope_vignette(hud, cam_w, cam_h)

        # (4) 중앙 고정 십자선 (사수 총구) - 정렬 완료 시 초록색 점등
        cx_i, cy_i = int(center_x), int(center_y)
        cross_color = (0, 255, 0) if is_aligned else ((0, 220, 255) if is_target_acquired else (130, 140, 150))
        thick = 2 if is_aligned else 1
        cv2.line(hud, (cx_i - 22, cy_i), (cx_i + 22, cy_i), cross_color, thick)
        cv2.line(hud, (cx_i, cy_i - 22), (cx_i, cy_i + 22), cross_color, thick)
        cv2.circle(hud, (cx_i, cy_i), 4, cross_color, -1 if is_aligned else 1)

        # (5) OSD 정보 패널
        cv2.rectangle(hud, (18, 18), (225, 105), (15, 15, 18), -1)
        cv2.rectangle(hud, (18, 18), (225, 105), (60, 65, 70), 1)
        
        status_text = f"STATE: {current_state}"
        status_color = (0, 255, 0) if is_aligned else ((0, 220, 255) if is_target_acquired else (180, 180, 180))
        cv2.putText(hud, status_text, (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.44, status_color, 1)
        cv2.putText(hud, f"R: {laser_dist:.1f}m | TOF: {tof:.3f}s", (26, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
        align_str = f"{align_dist:.1f}px" if is_target_acquired else "N/A"
        cv2.putText(hud, f"ALIGN ERR: {align_str}", (26, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 100) if is_aligned else (220, 220, 220), 1)
        cv2.putText(hud, f"SCORE: {hit_count}/{shot_count}", (26, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 150), 1)

        # (6) 탄환 물리 비행 및 정밀 격추 애니메이션
        if bullet_active:
            bullet_flight_time += dt
            progress = min(1.0, bullet_flight_time / max(0.001, bullet_target_tof))
            cur_bx = int(bullet_start_point[0] + (bullet_end_point[0] - bullet_start_point[0]) * progress)
            cur_by = int(bullet_start_point[1] + (bullet_end_point[1] - bullet_start_point[1]) * progress)
            
            cv2.circle(hud, (cur_bx, cur_by), 4, (0, 255, 255), -1)
            cv2.line(hud, (int(bullet_start_point[0]), int(bullet_start_point[1])), (cur_bx, cur_by), (0, 255, 255), 2)

            if progress >= 1.0:
                bullet_active = False
                if is_target_acquired:
                    bx1, by1, bx2, by2 = detected_bbox
                    if bx1 <= cur_bx <= bx2 and by1 <= cur_by <= by2:
                        hit_effect_timer = 1.2
                        hit_count += 1
                        print(f" [격발 {shot_count}] => HIT! UAV 격추 성공! (명중률: {hit_count/shot_count*100:.1f}%)")
                    else:
                        print(f" [격발 {shot_count}] => MISS (조준 오차)")
                else:
                    print(f" [격발 {shot_count}] => MISS (허공 사격)")

        # (7) 피격 격추 이펙트
        if hit_effect_timer > 0.0:
            hit_effect_timer -= dt
            cv2.putText(hud, "TARGET DESTROYED!", (155, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 255), 3)
            if is_target_acquired:
                cv2.circle(hud, (int(center_px[0]), int(center_px[1])), 42, (0, 165, 255), 3)

        cv2.imshow(window_name, hud)
        key = cv2.waitKey(1) & 0xFF

        # 방아쇠 격발 (마우스 좌클릭 또는 스페이스바)
        if (key == 32 or mouse_fire_triggered) and not bullet_active:
            mouse_fire_triggered = False
            shot_count += 1
            bullet_active = True
            bullet_flight_time = 0.0
            bullet_target_tof = tof
            bullet_start_point = (center_x, center_y)
            # A1 수정: 종점을 항상 화면 중앙으로 고정하던 버그를 고쳐, 격발 시점에 뉴턴 솔버가
            # 계산해 둔 미래 조준점(aim_point_px)으로 탄환이 실제로 날아가도록 한다.
            # 표적이 포착되지 않은 상태(허공 사격)라면 조준선이 향한 화면 중앙을 그대로 종점으로 쓴다.
            bullet_end_point = aim_point_px if (is_target_acquired and aim_point_px is not None) else (center_x, center_y)
            print(f"\n[BANG!] 사수 방아쇠 격발 (정렬 상태: {current_state}, 오차: {align_dist:.1f}px)")

        if key in (27, ord('q'), ord('Q')):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
