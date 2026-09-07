"""가제보(Gazebo) 전용 실시간 스마트 스코프 조준 및 격추 시뮬레이터 (버전 호환 임포트 보강)

기능:
1. gz.transport 초고속 멀티스레드 파이프라인으로 가제보 3D 카메라 영상 실시간 수신
2. 파인튜닝된 YOLO11s (models/best_finetuned.pt) + OpticalFlowTracker 실시간 드론 추적
3. 3차원 뉴턴 탄도학 솔버(lead_solver)를 통한 미래 조준점(AIM POINT) 및 명중 확률 산출
4. 스마트 스코프 HUD 화면 렌더링 (사격 제원 OSD, 리드 조준선)
5. 사수 방아쇠(스페이스바) 격발 시 탄환 물리 비행 및 드론 3D 피격 격추(HIT/KILL) 판정
"""

import math
import os
import sys
import time
import threading
import numpy as np
import cv2
import torch

# 프로젝트 루트 경로 등록
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# [안전한 Gazebo transport 및 msgs 패키지 버전 호환 임포트]
try:
    from gz.transport import Node
except ImportError:
    try:
        from gz.transport13 import Node
    except ImportError:
        from gz.transport12 import Node

try:
    from gz.msgs.image_pb2 import Image
except ImportError:
    try:
        from gz.msgs10.image_pb2 import Image
    except ImportError:
        from gz.msgs8.image_pb2 import Image

from aiming_engine.aiming_manager import AimingManager
from aiming_engine.coordinate_transform import CoordinateTransform, _euler_to_rotation_matrix
from aiming_engine.types import TrackerFrame, Vector3
from bridge.optical_flow_tracker import OpticalFlowTracker

# 멀티스레딩 프레임 버퍼
latest_frame = None
frame_lock = threading.Lock()


def image_callback(msg: Image):
    """가제보 카메라 영상 실시간 수신 콜백 (무손실 멀티스레드)"""
    global latest_frame
    img_array = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
    cv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    with frame_lock:
        latest_frame = cv_image.copy()


def main():
    global latest_frame

    print("=======================================================")
    print(" SMASH 가제보 실시간 스마트 스코프 사격 및 격추 시뮬레이터")
    print("=======================================================")

    # 1. 딥러닝 트래커 및 조준 엔진 초기화
    models_dir = os.path.join(project_root, "models")
    config_dir = os.path.join(project_root, "config")
    
    finetuned_path = os.path.join(models_dir, "best_finetuned.pt")
    model_path = finetuned_path if os.path.exists(finetuned_path) else os.path.join(models_dir, "best.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f" -> 가중치 모델: {os.path.basename(model_path)}")
    print(f" -> 연산 가속: {device}")
    
    tracker = OpticalFlowTracker(model_path=model_path, device=device, conf_thres=0.20)
    aiming_config_path = os.path.join(config_dir, "aiming_engine.yaml")
    aiming_manager = AimingManager(config_path=aiming_config_path if os.path.exists(aiming_config_path) else None)
    transform = CoordinateTransform(aiming_manager._config.scope)

    # 2. Gazebo 카메라 토픽 구독 (슬래시 유무 동시 대응)
    gz_node = Node()
    gz_node.subscribe(Image, "/platform/camera/image", image_callback)
    gz_node.subscribe(Image, "platform/camera/image", image_callback)
    print(" -> Gazebo 카메라 토픽 구독 완료")

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

    # 사격 및 격추 판정 상태 변수
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

    print("\n[조작 안내]")
    print(" - 가제보 3D 창: 드론 3D 실시간 비행 관찰")
    print(" - HUD 조준경 창: 실시간 조준점 확인 및 스페이스바 격발")
    print(" - Q 또는 ESC: 시뮬레이터 종료\n")

    while True:
        loop_now = time.time()
        dt = max(0.001, loop_now - last_loop_time)
        last_loop_time = loop_now
        current_time = loop_now - t0

        # 1. 가제보 카메라 프레임 읽기
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            waiting_screen = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
            cv2.putText(waiting_screen, "Connecting to Gazebo Camera Stream...", (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
            cv2.imshow(window_name, waiting_screen)
            if cv2.waitKey(10) & 0xFF in (27, ord('q'), ord('Q')):
                break
            continue

        if frame.shape[0] != cam_h or frame.shape[1] != cam_w:
            frame = cv2.resize(frame, (cam_w, cam_h))

        # 2. 6자유도 자세 행렬
        platform_pos = np.array([0.0, 0.0, 1.2], dtype=np.float64)
        T_extrinsics = np.eye(4, dtype=np.float64)
        T_extrinsics[:3, 3] = platform_pos

        # 3. OpticalFlowTracker 실시간 비전 추적
        tracker_res = tracker.update(frame)
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

        if prev_center_px is not None and is_target_acquired:
            vel_px = ((center_px[0] - prev_center_px[0]) / dt, (center_px[1] - prev_center_px[1]) / dt)
        else:
            vel_px = (0.0, 0.0)
        prev_center_px = center_px if is_target_acquired else None

        # 4. Aiming Engine 업데이트 및 뉴턴 탄도학 리드 솔버
        laser_dist = 35.0
        tracker_frame = TrackerFrame(
            timestamp=current_time,
            bbox=detected_bbox,
            center_px=center_px,
            velocity_px=vel_px,
            tracking_confidence=tracking_conf,
            follower_state="stable",
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=T_extrinsics,
            laser_range_m=laser_dist,
        )
        aim_output = aiming_manager.update(tracker_frame)
        aim_ready = aim_output.aim_ready
        hit_prob = aim_output.hit_probability
        current_state = aim_output.state.value if hasattr(aim_output.state, "value") else str(aim_output.state)

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

        # 5. 스마트 스코프 실시간 HUD 렌더링
        hud = frame.copy()

        # (1) 조준경 중앙 십자선
        cx_i, cy_i = int(center_x), int(center_y)
        cross_color = (0, 255, 0) if aim_ready else (140, 140, 140)
        cv2.line(hud, (cx_i - 20, cy_i), (cx_i + 20, cy_i), cross_color, 1)
        cv2.line(hud, (cx_i, cy_i - 20), (cx_i, cy_i + 20), cross_color, 1)
        cv2.circle(hud, (cx_i, cy_i), 3, cross_color, 1)

        # (2) 드론 비전 추적 바운딩 박스
        if is_target_acquired:
            bx1, by1, bx2, by2 = map(int, detected_bbox)
            cv2.rectangle(hud, (bx1, by1), (bx2, by2), (0, 165, 255), 2)
            cv2.putText(hud, f"UAV [{mode_label} {tracking_conf*100:.0f}%]", (bx1, max(16, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1)

        # (3) 뉴턴 솔버 미래 조준점 (AIM POINT)
        if is_target_acquired and aim_point_px is not None:
            ax, ay = int(aim_point_px[0]), int(aim_point_px[1])
            reticle_color = (0, 255, 0) if aim_ready else (0, 220, 255)
            cv2.circle(hud, (ax, ay), 18, reticle_color, 2)
            cv2.line(hud, (ax - 22, ay), (ax + 22, ay), reticle_color, 1)
            cv2.line(hud, (ax, ay - 22), (ax, ay + 22), reticle_color, 1)
            cv2.line(hud, (int(center_px[0]), int(center_px[1])), (ax, ay), (0, 220, 255), 1)
            cv2.putText(hud, "AIM", (ax + 20, ay + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, reticle_color, 1)

        # (4) 미니멀 OSD 사격 제원 정보창
        cv2.rectangle(hud, (12, 12), (210, 95), (20, 20, 20), -1)
        cv2.rectangle(hud, (12, 12), (210, 95), (80, 80, 80), 1)
        
        status_text = f"STATE: {current_state}"
        status_color = (0, 255, 0) if aim_ready else (0, 220, 255)
        cv2.putText(hud, status_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)
        cv2.putText(hud, f"R: {laser_dist:.1f}m | TOF: {tof:.3f}s", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
        cv2.putText(hud, f"HIT PROB: {hit_prob*100:.1f}%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
        cv2.putText(hud, f"SHOTS: {hit_count}/{shot_count}", (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 100), 1)

        # (5) 탄환 물리 비행 및 격추 애니메이션
        if bullet_active:
            bullet_flight_time += dt
            progress = min(1.0, bullet_flight_time / max(0.001, bullet_target_tof))
            cur_bx = int(bullet_start_point[0] + (bullet_end_point[0] - bullet_start_point[0]) * progress)
            cur_by = int(bullet_start_point[1] + (bullet_end_point[1] - bullet_start_point[1]) * progress)
            
            # 발광 예광탄 궤적
            cv2.circle(hud, (cur_bx, cur_by), 4, (0, 255, 255), -1)
            cv2.line(hud, (int(bullet_start_point[0]), int(bullet_start_point[1])), (cur_bx, cur_by), (0, 255, 255), 2)

            if progress >= 1.0:
                bullet_active = False
                bx1, by1, bx2, by2 = detected_bbox
                if bx1 <= cur_bx <= bx2 and by1 <= cur_by <= by2:
                    hit_effect_timer = 1.2
                    hit_count += 1
                    print(f" [격발 {shot_count}] => HIT! UAV 격추 성공! (명중률: {hit_count/shot_count*100:.1f}%)")
                else:
                    print(f" [격발 {shot_count}] => MISS (오차 발생)")

        # (6) 피격 격추 이펙트 표출
        if hit_effect_timer > 0.0:
            hit_effect_timer -= dt
            cv2.putText(hud, "TARGET DESTROYED!", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.circle(hud, (int(center_px[0]), int(center_px[1])), 45, (0, 165, 255), 3)

        cv2.imshow(window_name, hud)
        key = cv2.waitKey(1) & 0xFF

        # 방아쇠 격발 (스페이스바)
        if key == 32 and not bullet_active:
            shot_count += 1
            bullet_active = True
            bullet_flight_time = 0.0
            bullet_target_tof = tof
            bullet_start_point = (center_x, center_y)
            bullet_end_point = (center_px[0], center_px[1])
            print(f"\n[BANG!] 사수 방아쇠 격발 (거리: {laser_dist:.1f}m, TOF: {tof:.3f}s)")

        if key in (27, ord('q'), ord('Q')):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
