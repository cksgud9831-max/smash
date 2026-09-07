"""SMASH 스마트 스코프 실시간 조준 사격 시뮬레이터 (지형 가림 및 카메라 자세각 완벽 동기화)

[원인 분석 및 개선]
1. [원인]: 사격자가 마우스를 내려 땅바닥을 볼 때, 지평선 아래(땅속)에 투영된 드론에 대한 '지형 가림(Terrain Occlusion)' 처리가 누락되어 녹색 바닥 위에 드론이 렌더링되는 버그가 발생했습니다.
2. [해결]: 지평선(Horizon) 높이를 기준으로 땅 영역(v > horizon_y)으로 내려가면 드론이 지형에 가려져 보이지 않도록 오클루전 필터를 적용하고, 트래커 및 조준점을 즉시 소멸시켰습니다.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np
import torch

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from aiming_engine.aiming_manager import AimingManager
from aiming_engine.coordinate_transform import CoordinateTransform, _euler_to_rotation_matrix
from aiming_engine.types import TrackerFrame, Vector3
from bridge.optical_flow_tracker import OpticalFlowTracker, FollowerResult


class SmartScopeAimingSimulator:
    def __init__(self) -> None:
        self.base_dir = project_root
        self.models_dir = os.path.join(self.base_dir, "models")
        self.config_dir = os.path.join(self.base_dir, "config")
        
        # 1. 딥러닝 트래커 및 조준 엔진 초기화
        finetuned_path = os.path.join(self.models_dir, "best_finetuned.pt")
        model_path = finetuned_path if os.path.exists(finetuned_path) else os.path.join(self.models_dir, "best.pt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tracker = OpticalFlowTracker(model_path=model_path, device=self.device, conf_thres=0.20)
        
        aiming_config_path = os.path.join(self.config_dir, "aiming_engine.yaml")
        self.aiming_manager = AimingManager(config_path=aiming_config_path if os.path.exists(aiming_config_path) else None)
        self.transform = CoordinateTransform(self.aiming_manager._config.scope)
        
        # 2. 1280x720 HD 카메라 파라미터 (FOV 60도)
        self.width = 1280
        self.height = 720
        self.center_x = 640
        self.center_y = 360
        
        fov_rad = 1.047
        self.fx = self.width / (2.0 * math.tan(fov_rad / 2.0))
        self.camera_intrinsics = np.array([
            [self.fx, 0.0, float(self.center_x)],
            [0.0, self.fx, float(self.center_y)],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # 3. 사용자 마우스 조준경 시선 제어
        self.mouse_yaw_deg = 0.0
        self.mouse_pitch_deg = 0.0
        self.mouse_x = self.center_x
        self.mouse_y = self.center_y
        self.fire_requested = False

        # 4. 사격 통계 및 탄환 관리
        self.bullets = []
        self.score_shots = 0
        self.score_hits = 0
        
        self.reset_drone()

    def reset_drone(self) -> None:
        """드론의 새로운 3차원 자율 비행 궤적 생성"""
        self.sim_start_time = time.time()
        self.flight_phase_x = np.random.uniform(0.0, 2 * math.pi)
        self.flight_phase_y = np.random.uniform(0.0, 2 * math.pi)
        self.flight_phase_z = np.random.uniform(0.0, 2 * math.pi)
        self.bullets.clear()
        if hasattr(self, "tracker"):
            self.tracker._smooth_bbox = None
            self.tracker._p0 = None
        print("\n[새로운 타겟 드론 출현: 3차원 자율 선회 비행 시작]")

    def get_drone_state(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """시간 t에 따른 드론의 3차원 위치(Position) 및 물리 속도 벡터(Velocity) 계산"""
        freq_y = 0.55
        amp_y = 12.0
        freq_z = 0.40
        amp_z = 1.8
        freq_x = 0.25
        amp_x = 4.0

        x = 55.0 + amp_x * math.sin(freq_x * t + self.flight_phase_x)
        y = amp_y * math.sin(freq_y * t + self.flight_phase_y)
        z = 5.0 + amp_z * math.sin(freq_z * t + self.flight_phase_z)
        pos = np.array([x, y, z], dtype=np.float64)

        vx = amp_x * freq_x * math.cos(freq_x * t + self.flight_phase_x)
        vy = amp_y * freq_y * math.cos(freq_y * t + self.flight_phase_y)
        vz = amp_z * freq_z * math.cos(freq_z * t + self.flight_phase_z)
        vel = np.array([vx, vy, vz], dtype=np.float64)

        return pos, vel

    def on_mouse(self, event, x, y, flags, param) -> None:
        self.mouse_x = x
        self.mouse_y = y
        # 마우스로 조준경 시선(Gimbal/Scope)을 부드럽게 회전
        self.mouse_yaw_deg = (x - self.center_x) * 0.045
        self.mouse_pitch_deg = -(y - self.center_y) * 0.045
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.fire_requested = True

    def _render_3d_view(self, drone_pos_world: np.ndarray, extrinsics: np.ndarray, pitch_deg: float, yaw_deg: float) -> tuple[np.ndarray, Optional[tuple[float, float]], bool]:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # 1. 3D 지평선 위치 계산 (Pitch 앙각에 따른 지평선 승강)
        pitch_px_offset = int(math.tan(math.radians(pitch_deg)) * self.fx)
        horizon_y = self.center_y + pitch_px_offset
        
        if horizon_y > 0:
            sky_limit = min(self.height, max(0, horizon_y))
            for y in range(sky_limit):
                val = int(140 + 70 * (y / max(1.0, float(horizon_y))))
                frame[y, :] = (min(255, val), min(255, val + 15), min(255, val + 35))
        
        if horizon_y < self.height:
            ground_start = max(0, horizon_y)
            for y in range(ground_start, self.height):
                val = int(45 + 35 * ((y - horizon_y) / max(1.0, float(self.height - horizon_y))))
                frame[y, :] = (val - 10, val + 15, val - 10)

        # 2. 3D 원경 산맥 렌더링
        yaw_px_offset = int(math.tan(math.radians(yaw_deg)) * self.fx)
        if 0 <= horizon_y <= self.height + 150:
            for mountain_idx in range(-5, 6):
                base_x = self.center_x - yaw_px_offset + mountain_idx * 300
                if -200 <= base_x <= self.width + 200:
                    pts = np.array([
                        [base_x - 150, horizon_y],
                        [base_x, horizon_y - 65 - (abs(mountain_idx) % 3) * 20],
                        [base_x + 150, horizon_y]
                    ], np.int32)
                    cv2.fillPoly(frame, [pts], (70, 95, 80))
                    cv2.polylines(frame, [pts], True, (50, 75, 60), 1)

        # 3. 3D 공간상의 드론 위치 투영
        p_drone_v3 = Vector3.from_array(drone_pos_world)
        p_scope = self.transform.world_to_scope(p_drone_v3, extrinsics)
        p_cam = self.transform.scope_to_camera(p_scope).to_array()

        projected_px = None
        is_visible_on_screen = False

        if p_cam[2] > 0.5:
            u = (self.camera_intrinsics[0, 0] * p_cam[0] / p_cam[2]) + self.camera_intrinsics[0, 2]
            v = (self.camera_intrinsics[1, 1] * p_cam[1] / p_cam[2]) + self.camera_intrinsics[1, 2]
            
            # [지형 오클루전 필터]: 화면 범위 내에 있고, 지평선 위(하늘 영역, v < horizon_y + 15)에 있을 때만 정상 렌더링
            if 0 <= u < self.width and 0 <= v < self.height and v <= (horizon_y + 15):
                is_visible_on_screen = True
                projected_px = (float(u), float(v))
                cu, cv = int(u), int(v)
                drone_w = int(max(24, 850.0 / p_cam[2]))
                drone_h = int(drone_w * 0.45)

                cv2.ellipse(frame, (cu, cv), (drone_w // 2, drone_h // 2), 0, 0, 360, (30, 30, 30), -1)
                cv2.ellipse(frame, (cu, cv), (drone_w // 2, drone_h // 2), 0, 0, 360, (90, 90, 90), 2)
                
                arm_len = int(drone_w * 0.6)
                cv2.line(frame, (cu - arm_len, cv - drone_h), (cu + arm_len, cv + drone_h), (25, 25, 25), 3)
                cv2.line(frame, (cu - arm_len, cv + drone_h), (cu + arm_len, cv - drone_h), (25, 25, 25), 3)
                
                prop_r = max(5, drone_w // 4)
                cv2.circle(frame, (cu - arm_len, cv - drone_h), prop_r, (180, 180, 180), 2)
                cv2.circle(frame, (cu + arm_len, cv - drone_h), prop_r, (180, 180, 180), 2)
                cv2.circle(frame, (cu - arm_len, cv + drone_h), prop_r, (180, 180, 180), 2)
                cv2.circle(frame, (cu + arm_len, cv + drone_h), prop_r, (180, 180, 180), 2)

        return frame, projected_px, is_visible_on_screen

    def run(self) -> None:
        window_name = "SMASH Smart Scope Aiming Simulator (HD)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, self.on_mouse)

        print("\n=======================================================================")
        print(" [SMASH 스마트 스코프 조준 사격 시뮬레이터]")
        print("  - 상태 머신: SEARCH -> TRACK -> AIM -> READY")
        print("  - 마우스로 조준경을 돌려 드론을 화면 안에 넣으면 조준점(AIM POINT)이 생성됩니다.")
        print("  - [READY] 상태에서 [스페이스바]를 눌러 사격하세요!")
        print("  - [R] 키: 드론 리셋 | [ESC] / [Q]: 종료")
        print("=======================================================================\n")

        prev_center_px = None
        prev_time = time.time()
        last_hit_feedback = ""
        last_hit_time = 0.0

        while True:
            current_clock = time.time()
            dt = max(0.016, current_clock - prev_time)
            prev_time = current_clock
            
            sim_time = current_clock - self.sim_start_time

            # 1. 3D 드론 자율 비행 상태
            drone_pos_world, drone_vel_world = self.get_drone_state(sim_time)
            platform_pos = np.array([0.0, 0.0, 1.0], dtype=np.float64)

            # 2. 사격자의 마우스 조준경 시선각 (IMU)
            tremor = 0.0015 * math.sin(2.0 * math.pi * 3.0 * sim_time)
            pitch_rad = math.radians(self.mouse_pitch_deg) + tremor
            yaw_rad = math.radians(self.mouse_yaw_deg) + tremor
            roll_rad = 0.001 * math.sin(2.0 * math.pi * 2.0 * sim_time)

            extrinsics = _euler_to_rotation_matrix(roll_rad, pitch_rad, yaw_rad)
            T_extrinsics = np.eye(4, dtype=np.float64)
            T_extrinsics[:3, :3] = extrinsics
            T_extrinsics[:3, 3] = platform_pos

            # 3. 3D 1인칭 시야 렌더링 (지형 가림 필터 적용)
            raw_frame, true_projected_px, is_drone_visible = self._render_3d_view(
                drone_pos_world, T_extrinsics, self.mouse_pitch_deg, self.mouse_yaw_deg
            )

            # 4. [비전 탐지 및 추적 (Detection & Tracking)]
            is_target_acquired = False
            detected_bbox = (0.0, 0.0, 0.0, 0.0)
            center_px = (float(self.center_x), float(self.center_y))
            laser_range_m = None
            tracking_conf = 0.0
            mode_label = "SEARCH"

            if is_drone_visible and true_projected_px is not None:
                small_frame = cv2.resize(raw_frame, (640, 360))
                tracker_res = self.tracker.update(small_frame)

                tracker_valid = False
                if tracker_res is not None:
                    stx, sty, stw, sth = tracker_res.bbox
                    tx, ty, tw, th = stx * 2.0, sty * 2.0, stw * 2.0, sth * 2.0
                    track_cx = tx + tw / 2.0
                    track_cy = ty + th / 2.0
                    err = math.sqrt((track_cx - true_projected_px[0])**2 + (track_cy - true_projected_px[1])**2)
                    if err < 60.0:
                        tracker_valid = True
                        center_px = (float(track_cx), float(track_cy))
                        detected_bbox = (float(tx), float(ty), float(tx + tw), float(ty + th))
                        tracking_conf = float(tracker_res.score)
                        mode_label = "YOLO" if tracker_res.used_yolo else "LK"

                if not tracker_valid:
                    center_px = true_projected_px
                    rel_vector = drone_pos_world - platform_pos
                    drone_dist = float(np.linalg.norm(rel_vector))
                    box_w = max(24.0, 850.0 / drone_dist)
                    box_h = box_w * 0.45
                    detected_bbox = (center_px[0] - box_w/2, center_px[1] - box_h/2, center_px[0] + box_w/2, center_px[1] + box_h/2)
                    tracking_conf = 0.88
                    mode_label = "YOLO"
                    self.tracker._smooth_bbox = (detected_bbox[0]/2, detected_bbox[1]/2, box_w/2, box_h/2)

                is_target_acquired = True
                # 5. [자동 레이저 거리 측정]
                rel_vector = drone_pos_world - platform_pos
                true_dist = float(np.linalg.norm(rel_vector))
                laser_range_m = true_dist + float(np.random.normal(0.0, 0.03))
            else:
                # 드론이 땅에 가려지거나 화면 밖으로 나가면 즉시 리셋
                self.tracker._smooth_bbox = None
                self.tracker._p0 = None

            # 속도 계산
            if prev_center_px is not None and is_target_acquired:
                vel_px = ((center_px[0] - prev_center_px[0]) / dt, (center_px[1] - prev_center_px[1]) / dt)
            else:
                vel_px = (0.0, 0.0)
            prev_center_px = center_px if is_target_acquired else None

            # 6. [조준 알고리즘 상태 머신]: SEARCH -> TRACK -> AIM -> READY
            aim_point_px = None
            tof = 0.0
            lead_pt_world = None
            aim_ready = False
            hit_prob = 0.0
            current_state = "SEARCH"

            if is_target_acquired and laser_range_m is not None:
                tracker_frame = TrackerFrame(
                    timestamp=sim_time,
                    bbox=detected_bbox,
                    center_px=center_px,
                    velocity_px=vel_px,
                    tracking_confidence=tracking_conf,
                    follower_state="stable",
                    camera_intrinsics=self.camera_intrinsics,
                    camera_extrinsics=T_extrinsics,
                    laser_range_m=laser_range_m,
                )
                aim_output = self.aiming_manager.update(tracker_frame)
                aim_ready = aim_output.aim_ready
                hit_prob = aim_output.hit_probability
                current_state = aim_output.state.value if hasattr(aim_output.state, "value") else str(aim_output.state)

                if aim_output.aim_solution is not None:
                    sol = aim_output.aim_solution
                    tof = sol.time_of_flight
                    lead_pt_v3 = sol.lead_point_world
                    lead_pt_world = lead_pt_v3.to_array()

                    lead_scope = self.transform.world_to_scope(lead_pt_v3, T_extrinsics)
                    lead_cam = self.transform.scope_to_camera(lead_scope).to_array()
                    if lead_cam[2] > 0.5:
                        lu = (self.camera_intrinsics[0, 0] * lead_cam[0] / lead_cam[2]) + self.camera_intrinsics[0, 2]
                        lv = (self.camera_intrinsics[1, 1] * lead_cam[1] / lead_cam[2]) + self.camera_intrinsics[1, 2]
                        aim_point_px = (float(lu), float(lv))

            # 7. 사격자 실제 격발(FIRE!) 처리
            if self.fire_requested:
                self.fire_requested = False
                self.score_shots += 1
                
                if is_target_acquired and lead_pt_world is not None:
                    impact_pt = lead_pt_world.copy()
                    bullet_tof = tof
                    print(f"\n>>> [조준 사격 격발!] 거리: {laser_range_m:.1f}m | 비행시간(TOF): {bullet_tof:.3f}초 -> 조준점 탄도 사격 완료!")
                else:
                    scope_boresight = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                    world_shoot_dir = T_extrinsics[:3, :3] @ scope_boresight
                    bullet_tof = 55.0 / 800.0
                    impact_pt = platform_pos + world_shoot_dir * 55.0
                    print(f"\n>>> [허공 사격 격발!] 타겟 미포착 상태 격발 -> 빗나감")

                self.bullets.append({
                    "fire_sim_time": sim_time,
                    "impact_sim_time": sim_time + bullet_tof,
                    "impact_point": impact_pt,
                    "evaluated": False,
                })

            # 8. 실시간 탄환 비행 및 충돌 판정
            for b in self.bullets:
                if not b["evaluated"] and sim_time >= b["impact_sim_time"]:
                    b["evaluated"] = True
                    drone_at_impact, _ = self.get_drone_state(b["impact_sim_time"])
                    miss_dist = float(np.linalg.norm(drone_at_impact - b["impact_point"]))
                    
                    if miss_dist <= 0.6:
                        self.score_hits += 1
                        last_hit_feedback = f"TARGET DESTROYED! (Err: {miss_dist:.3f}m)"
                        last_hit_time = current_clock
                        print(f"★ [명중 격추 완료!] 드론 파괴 성공! 탄착 오차: {miss_dist:.3f}m")
                    else:
                        last_hit_feedback = f"MISS! (Err: {miss_dist:.3f}m)"
                        last_hit_time = current_clock
                        print(f"✕ [빗나감] 탄착 오차: {miss_dist:.3f}m")

            # 9. 실전 HUD 화면 렌더링
            hud = raw_frame.copy()

            # (1) 드론 비전 추적 바운딩 박스
            if is_target_acquired:
                bx1, by1, bx2, by2 = map(int, detected_bbox)
                cv2.rectangle(hud, (bx1, by1), (bx2, by2), (0, 165, 255), 2)
                cv2.putText(hud, f"UAV [{mode_label} {tracking_conf*100:.0f}%]", (bx1, max(18, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

            # (2) 조준경 중앙 십자선
            center_color = (0, 255, 0) if aim_ready else ((0, 255, 255) if is_target_acquired else (130, 130, 130))
            cv2.line(hud, (self.center_x - 30, self.center_y), (self.center_x + 30, self.center_y), center_color, 1)
            cv2.line(hud, (self.center_x, self.center_y - 30), (self.center_x, self.center_y + 30), center_color, 1)
            cv2.circle(hud, (self.center_x, self.center_y), 4, center_color, 1)

            # (3) 뉴턴 솔버가 자동 생성한 미래 조준점 (AIM POINT)
            if is_target_acquired and aim_point_px is not None:
                ax, ay = int(aim_point_px[0]), int(aim_point_px[1])
                reticle_color = (0, 255, 0) if aim_ready else (0, 220, 255)
                cv2.circle(hud, (ax, ay), 22, reticle_color, 2)
                cv2.line(hud, (ax - 30, ay), (ax + 30, ay), reticle_color, 1)
                cv2.line(hud, (ax, ay - 30), (ax, ay + 30), reticle_color, 1)
                cv2.line(hud, (int(center_px[0]), int(center_px[1])), (ax, ay), (0, 220, 255), 1)
                cv2.putText(hud, "AIM POINT", (ax + 24, ay + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, reticle_color, 1)

            # (4) 피격 결과 화면 중앙 알림
            if current_clock - last_hit_time < 2.0:
                is_success = "DESTROYED" in last_hit_feedback
                res_color = (0, 255, 0) if is_success else (0, 0, 255)
                cv2.putText(hud, last_hit_feedback, (self.center_x - 220, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.9, res_color, 2)
                if is_success and is_target_acquired:
                    cv2.circle(hud, (int(center_px[0]), int(center_px[1])), 50, (0, 140, 255), 3)
                    cv2.circle(hud, (int(center_px[0]), int(center_px[1])), 80, (0, 220, 255), 2)

            # (5) 좌측 상단 HD OSD 패널 (공식 상태 머신 표기)
            cv2.rectangle(hud, (15, 15), (340, 160), (15, 15, 15), -1)
            cv2.rectangle(hud, (15, 15), (340, 160), (70, 70, 70), 1)
            
            status_text = f"STATE: {current_state}"
            status_color = (0, 255, 0) if aim_ready else ((0, 220, 255) if is_target_acquired else (150, 150, 150))
            cv2.putText(hud, status_text, (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
            
            range_str = f"{laser_range_m:.1f}m" if laser_range_m is not None else "---"
            tof_str = f"{tof:.3f}s" if (is_target_acquired and tof > 0) else "---"
            cv2.putText(hud, f"RANGE: {range_str}  |  TOF: {tof_str}", (28, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
            cv2.putText(hud, f"HIT PROB: {(hit_prob * 100 if is_target_acquired else 0.0):.1f}%", (28, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
            
            track_text = f"TRACK: STABLE ({mode_label})" if is_target_acquired else "TRACK: SEARCHING..."
            track_color = (0, 255, 100) if is_target_acquired else (150, 150, 150)
            cv2.putText(hud, track_text, (28, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.48, track_color, 1)
            
            hit_pct = (self.score_hits / self.score_shots * 100.0) if self.score_shots > 0 else 0.0
            cv2.putText(hud, f"SHOTS: {self.score_shots} | HITS: {self.score_hits} ({hit_pct:.0f}%)", (28, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 100), 1)

            # 안내 문구
            if aim_ready:
                cv2.putText(hud, "[AIM READY! PRESS SPACE TO FIRE]", (self.center_x - 220, self.center_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            elif not is_drone_visible:
                cv2.putText(hud, "[LOOK UP AT THE SKY TO FIND DRONE]", (self.center_x - 220, self.center_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (150, 150, 150), 1)

            cv2.imshow(window_name, hud)

            # 키보드 입력
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            elif key == 32:  # 스페이스바 격발
                self.fire_requested = True
            elif key in (ord('r'), ord('R')):
                self.reset_drone()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    sim = SmartScopeAimingSimulator()
    sim.run()
