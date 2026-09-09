#!/usr/bin/env python3
"""smash_fcs_node.py — SMASH 지능형 사격 통제(FCS) 통합 노드 [로드맵 1단계]

■ 이 노드가 어떤 무기를 전제로 하는가 (가장 먼저 읽을 것)
    SMASH 의 실제 제품은 5.56mm 개인화기에 장착하는 스마트 조준경이다. 서보로
    총을 돌리는 무기가 아니라 사수에게 조준점을 알려 주는 자문 장비이며,
    aiming_engine 전체가 그 전제로 설계되어 있다.

    Gazebo 의 CIWS 포탑은 참고 오픈소스에서 빌려 온 3D 비행 환경이다. 여기서
    Pan/Tilt 서보는 **사수가 총을 겨누는 동작의 시뮬레이션 대체물**이며 제품
    기능이 아니다. 사람 대신 서보가 조준을 따라가 주어야 폐루프 수렴 성능
    (정착시간, 정상상태 조준오차)을 정량 측정할 수 있어서 유지한다.

    따라서 탄도와 기하는 포탑이 아니라 소총 제원을 따른다.
      1. 탄종 5.56 x 45mm M855 (62 gr), 포구초속 920 m/s, 질량 4.02 g
      2. 조준경 광축과 총열 중심의 높이차 66 mm (AR 계열 플랫탑 기준)
      3. 조준경 광학중심에서 총구까지 전방 0.45 m
    Gazebo 포탑 메시의 포신 끝 좌표는 쓰지 않는다.


기존 오픈소스의 단순 YOLO 탐지기(drone_detector) + 단순 비례제어 추종기
(turret_tracker) 경로를 대체한다. 이 노드 하나가 다음을 모두 수행한다.

    /turret_camera/image_raw
        -> cv_bridge -> OpticalFlowTracker (YOLO11s + Lucas-Kanade)
        -> 거리 추정 (동축 광각 레이저 우선 / 바운딩박스 기하 역추정 폴백)
        -> TF 기반 포구(scope) 외부파라미터 산출
        -> aiming_engine.AimingManager (좌표변환 -> 표적상태 -> 뉴턴 탄도 리드 솔버)
        -> 포탑 Pan/Tilt 관절 명령 -> /turret_controller/commands
        -> 기본 HUD 오버레이 -> /turret_camera/image_annotated

■ 1단계 범위와 경계 (투명성)
    1. 본 노드는 조준까지만 담당한다. aiming_engine 자체는 설계상 사격 명령을
       만들지 않는다(aim_state_machine.py 참고). READY 는 "조준 해가 신뢰할
       만하다"는 의미이지 "발사하라"가 아니며, 이 원칙은 아래 3단계 격발 제어를
       추가한 뒤에도 그대로 유지된다 — "지금 쏜다"는 결정은 aiming_engine 밖의
       별도 모듈(smash_fcs.fire_control.FireController)이 내리고, aiming_engine
       코드는 이 노드가 처음 만들어진 이후 한 줄도 바뀌지 않았다.
    2. HUD 는 상태 확인에 필요한 최소한만 그린다. 군용 스마트 스코프 HUD 렌더링은
       4단계 과제다.
    3. 표적 선택은 OpticalFlowTracker 의 단일표적 추종 정책을 그대로 따른다.
       다중표적 우선순위 결정 로직은 포함하지 않는다.

■ 3단계 범위와 경계 (가상 탄환 격발 및 Hit/Kill 판정, 투명성)
    enable_fire_control 파라미터로 켠다(기본값 false — 1/2단계 검증 결과에
    회귀가 없도록 기본 동작은 이전과 동일하게 유지한다). 자세한 설계 근거와
    한계는 smash_fcs/fire_control.py 모듈 docstring 을 반드시 함께 읽을 것.
    핵심만 요약하면 다음과 같다.

    1. Hit/Kill 판정은 FCS 자신의 표적 추정치가 아니라 Gazebo 가 world 프레임
       으로 발행하는 실측 위치(ground truth, ground_truth_pose_topic)를 기준
       으로 한다. 이는 표적이 등속 가정을 위반해 예측과 다르게 움직이는
       상황(로드맵 4단계 회피 기동)을 포착하기 위한 의도적 설계다.
    2. Gazebo 실측 위치(world 프레임)를 aiming_engine 이 쓰는 base_footprint
       프레임으로 옮기려면 포탑 스폰 변환을 알아야 한다. turret_world_
       translation_m / turret_world_yaw_rad 파라미터가 이 값이며, 반드시
       view.launch.py 의 스폰 인자(-z 0.05, x=y=0, 회전 없음)와 일치해야
       한다. 스폰 인자를 바꾸면 이 파라미터도 함께 바꿔야 하고, 이 동기화는
       코드로 강제되지 않는다.
    3. ground_truth_pose_topic 의 실제 이름과 메시지 발행 여부는 Gazebo
       PosePublisher 플러그인(drone_sim/models/drone/model.sdf)이 실제로
       기동해야만 확인할 수 있다. 이 세션은 gz 런타임을 실행할 수 없으므로
       WSL2 에서 `gz topic -l` 로 실제 토픽 이름을 사용자가 확인해야 한다.
    4. Hit == Kill 로 단순화했다(치명도 모델 없음). 자세한 한계는
       fire_control.py 6절과 동일하다.

■ 좌표계와 발사점 처리 (수학적으로 중요)
    aiming_engine 은 scope 프레임 원점을 발사점으로 삼는다(aiming_manager.py 의
    launch_point_world 계산). 소총에서 조준경 광축과 총열 중심은 66 mm 어긋나
    있으므로, 발사점을 조준경이 아니라 총구로 잡아야 한다. 이 수직 어긋남이
    만드는 시차는 사거리 R 에서 대략 0.066 / R [rad] 이며, 35 m 에서 약
    1.9 mrad(0.108도), 10 m 에서 약 6.6 mrad(0.38도)다. 4단계 READY 판정
    임계값 0.5도보다는 작지만 0 이 아니므로 탄도 해에 반영한다.

    참고로 이 시스템에는 소총의 "영점(zero)" 개념이 필요 없다. 영점은 고정된
    한 거리에서만 맞도록 조준선과 탄도를 미리 맞춰 두는 방식인데, 여기서는 매
    프레임 실제 측정 거리로 탄도를 직접 풀기 때문이다.

    그래서 다음과 같이 처리한다.
        1. ScopeConfig.mount_offset_m 을 "카메라 핀홀 프레임에서 본 포구 위치"로 설정
           (turret_kinematics.camera_to_muzzle_offset_in_pinhole_frame)
        2. TrackerFrame.camera_extrinsics 로 카메라 자세가 아니라 포구 자세를 전달
           (turret_kinematics.muzzle_pose_from_camera_pose)
    이렇게 하면 픽셀 역투영은 카메라 원점 기준으로, 탄도 해는 포구 기준으로
    각각 올바르게 계산된다. aiming_engine 코드 수정 없이 성립한다.

■ 포탑 지향각 산출 (사수 조준의 시뮬레이션 대체)
    AimSolver 는 scope 프레임 방위/앙각을 반환한다. 같은 외부파라미터로 다시 월드로
    되돌리면 뉴턴 솔버가 실제로 푼 월드 프레임 방향(중력/공기저항 보정이 포함된
    방향)을 정확히 복원할 수 있다. 이 방향의 방위각과 앙각이 곧 turret_pan_joint /
    turret_tilt_joint 명령값이다(turret_kinematics 모듈 상단 유도 참고).
    표적 자체를 직접 겨누는 것이 아니라 미래 탄착점을 선행 지향하게 된다.

    실제 제품에서는 이 방향이 HUD 조준점으로 표시되고 사수가 총을 그쪽으로
    움직인다. 시뮬레이션에서는 서보가 그 역할을 대신한다.
"""

from __future__ import annotations

import json
import math
from typing import Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Bool, Float64MultiArray, String

import tf2_ros

from drone_sim.smash_fcs import turret_kinematics as tk
from drone_sim.smash_fcs.engine_bootstrap import build_aim_config, ensure_core_on_path
from drone_sim.smash_fcs.pointcloud_utils import field_offsets, ranges_from_xyz_buffer
from drone_sim.smash_fcs.range_estimator import (
    SOURCE_BBOX,
    SOURCE_LASER,
    BoundingBoxRange,
    ConeLaserRange,
    HybridRangeEstimator,
)

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.VOLATILE,
)


def _stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class SmashFcsNode(Node):
    def __init__(self) -> None:
        super().__init__("smash_fcs")

        # ── 파라미터 선언 ───────────────────────────────────────────────
        self.declare_parameter("core_path", "")
        self.declare_parameter("aiming_config_path", "")
        self.declare_parameter("model_path", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("conf_thres", 0.25)

        self.declare_parameter("image_topic", "/turret_camera/image_raw")
        self.declare_parameter("camera_info_topic", "/turret_camera/camera_info")
        self.declare_parameter("laser_topic", "/turret_camera/boresight_range/points")
        self.declare_parameter("command_topic", "/turret_controller/commands")
        self.declare_parameter("annotated_topic", "/turret_camera/image_annotated")
        self.declare_parameter("state_topic", "/smash_fcs/state")

        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("camera_optical_frame", "camera_link_optical")
        self.declare_parameter("tf_timeout_s", 0.05)

        # 5.56 x 45mm M855 (62 gr) 기준. 총열 길이에 따라 포구초속이 달라진다.
        self.declare_parameter("muzzle_velocity_mps", 920.0)
        self.declare_parameter("enable_drag", True)
        self.declare_parameter("projectile_mass_kg", 0.00402)
        # 5.56 탄의 실제 탄두 직경(강선 홈 기준)은 5.70 mm 다.
        self.declare_parameter("projectile_diameter_m", 0.00570)
        # M855 는 보트테일 스피처이므로 평저탄용 G1 보다 G7 계열이 형상에 가깝다.
        self.declare_parameter("drag_model", "G7")

        # 조준경 광학중심 기준 총구 위치, FLU (X 전방, Y 좌측, Z 상방).
        # 세 번째 성분의 절댓값이 곧 조준경 광축과 총열 중심의 높이차다.
        self.declare_parameter("scope_to_muzzle_m", list(tk.SCOPE_TO_MUZZLE_M))
        self.declare_parameter("gun_to_camera_optical_m", list(tk.GUN_TO_CAMERA_OPTICAL_M))

        self.declare_parameter("target_characteristic_size_m", 0.47)
        self.declare_parameter("target_size_uncertainty_ratio", 0.25)
        self.declare_parameter("bbox_pixel_sigma", 2.0)
        self.declare_parameter("range_min_m", 3.0)
        self.declare_parameter("range_max_m", 800.0)
        self.declare_parameter("range_smoothing_alpha", 1.0)
        self.declare_parameter("laser_noise_sigma_m", 0.02)
        self.declare_parameter("laser_staleness_timeout_s", 0.2)
        self.declare_parameter("laser_cross_check_sigma", 3.0)
        self.declare_parameter("max_range_rate_mps", 60.0)

        self.declare_parameter("command_rate_hz", 30.0)
        self.declare_parameter("pan_max_rate_rad_s", tk.PAN_MAX_RATE_RAD_S)
        self.declare_parameter("tilt_max_rate_rad_s", tk.TILT_MAX_RATE_RAD_S)
        self.declare_parameter("tilt_min_rad", tk.TILT_MIN_RAD)
        self.declare_parameter("tilt_max_rad", tk.TILT_MAX_RAD)
        self.declare_parameter("search_pan_rad", 0.0)
        self.declare_parameter("search_tilt_rad", 0.0)
        self.declare_parameter("command_feedforward_s", 0.0)

        self.declare_parameter("border_margin_px", 2)
        self.declare_parameter("border_frames_before_reset", 3)
        self.declare_parameter("publish_annotated", True)

        # ── 3단계: 가상 탄환 격발 및 Hit/Kill 판정 ─────────────────────
        # 기본값 false. 1/2단계 검증에는 아무 영향이 없다(설계 근거는 위
        # 모듈 docstring 및 smash_fcs/fire_control.py 참고).
        self.declare_parameter("enable_fire_control", False)
        # "auto": READY 인 동안 fire_rate_rpm 쿨다운으로 자동 격발.
        # "manual": trigger_topic 상승 엣지에서만 격발(반자동 방아쇠 근사).
        self.declare_parameter("fire_mode", "manual")
        self.declare_parameter("fire_rate_rpm", 700.0)
        # 0 이하이면 target_characteristic_size_m / 2 로 자동 유도한다.
        self.declare_parameter("target_hit_radius_m", -1.0)
        self.declare_parameter("trigger_topic", "/smash_fcs/trigger")
        self.declare_parameter("fire_event_topic", "/smash_fcs/fire_event")
        self.declare_parameter("hit_result_topic", "/smash_fcs/hit_result")
        self.declare_parameter("engagement_stats_topic", "/smash_fcs/engagement_stats")
        self.declare_parameter("fire_control_rate_hz", 60.0)
        self.declare_parameter("impact_effect_hold_s", 0.35)
        # Gazebo PosePublisher 가 발행하는(가정) 표적 실측 자세 토픽. 실제 이름은
        # gz-sim 버전에 따라 다를 수 있으므로 WSL2 에서 `gz topic -l` 로 확인할 것
        # (drone_sim/models/drone/model.sdf 의 PosePublisher 플러그인 참고).
        self.declare_parameter("ground_truth_pose_topic", "/model/drone/pose")
        self.declare_parameter("ground_truth_timeout_s", 0.5)
        # world -> base_footprint 스폰 변환. view.launch.py 의
        # `ros_gz_sim create -topic robot_description -name ciws_turret -z 0.05`
        # 와 반드시 일치해야 한다(x=y=0, 회전 없음이 현재 기본값).
        self.declare_parameter("turret_world_translation_m", [0.0, 0.0, 0.05])
        self.declare_parameter("turret_world_yaw_rad", 0.0)

        # ── 파라미터 반영 ───────────────────────────────────────────────
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._camera_frame = str(self.get_parameter("camera_optical_frame").value)
        self._tf_timeout_s = float(self.get_parameter("tf_timeout_s").value)
        self._tilt_min = float(self.get_parameter("tilt_min_rad").value)
        self._tilt_max = float(self.get_parameter("tilt_max_rad").value)
        self._pan_max_rate = float(self.get_parameter("pan_max_rate_rad_s").value)
        self._tilt_max_rate = float(self.get_parameter("tilt_max_rate_rad_s").value)
        self._border_margin_px = int(self.get_parameter("border_margin_px").value)
        self._border_frames_limit = int(self.get_parameter("border_frames_before_reset").value)
        self._publish_annotated = bool(self.get_parameter("publish_annotated").value)

        self._scope_to_muzzle = tuple(float(v) for v in self.get_parameter("scope_to_muzzle_m").value)
        self._gun_to_camera = tuple(float(v) for v in self.get_parameter("gun_to_camera_optical_m").value)
        # 포탑 링크 좌표로 옮긴 파생값. Gazebo 서보 구동 배선에만 쓰인다.
        self._gun_to_muzzle = tk.gun_to_muzzle_from_scope_offset(
            scope_to_muzzle_m=self._scope_to_muzzle, gun_to_camera_m=self._gun_to_camera
        )

        # ── 조준 코어 부트스트랩 ────────────────────────────────────────
        core_path = str(self.get_parameter("core_path").value) or None
        self._core_root = ensure_core_on_path(core_path)
        self.get_logger().info(f"SMASH 조준 코어 경로: {self._core_root}")

        scope_offset = tk.scope_to_muzzle_offset_in_pinhole_frame(self._scope_to_muzzle)
        aiming_config_path = str(self.get_parameter("aiming_config_path").value) or None
        self._aim_config = build_aim_config(
            core_root=self._core_root,
            config_path=aiming_config_path,
            muzzle_velocity_mps=float(self.get_parameter("muzzle_velocity_mps").value),
            scope_mount_offset_m=scope_offset,
            enable_drag=bool(self.get_parameter("enable_drag").value),
            projectile_mass_kg=float(self.get_parameter("projectile_mass_kg").value),
            projectile_diameter_m=float(self.get_parameter("projectile_diameter_m").value),
            drag_model=str(self.get_parameter("drag_model").value),
        )
        self.get_logger().info(
            "탄도 설정: 포구초속 {:.1f} m/s, 질량 {:.5f} kg, 직경 {:.5f} m, 힘 모델 {} ({})".format(
                self._aim_config.projectile.muzzle_velocity,
                self._aim_config.projectile.mass_kg,
                self._aim_config.projectile.diameter_m,
                self._aim_config.projectile.forces,
                self._aim_config.projectile.drag_model,
            )
        )
        self.get_logger().info(
            "조준경 기하: 광축 대비 총열 하방 {:.1f} mm, 총구 전방 {:.3f} m. "
            "35 m 시차 약 {:.2f} mrad".format(
                abs(self._scope_to_muzzle[2]) * 1000.0,
                self._scope_to_muzzle[0],
                abs(self._scope_to_muzzle[2]) / 35.0 * 1000.0,
            )
        )
        if "drag" in self._aim_config.projectile.forces:
            self.get_logger().warn(
                "공기저항이 켜져 있습니다. aiming_engine/forces.py 의 G1/G7 항력계수 표는 "
                "해당 파일 주석에 명시된 대로 아직 1차 출처 검증이 끝나지 않은 근사치입니다. "
                "정량 보고서에는 이 한계를 반드시 병기하십시오."
            )

        from aiming_engine.aiming_manager import AimingManager  # noqa: E402
        from aiming_engine.types import AimState  # noqa: E402
        from bridge.confidence import compute_follower_state, compute_tracking_confidence  # noqa: E402
        from bridge.optical_flow_tracker import OpticalFlowTracker  # noqa: E402

        self._AimingManager = AimingManager
        self._AimState = AimState
        self._compute_follower_state = compute_follower_state
        self._compute_tracking_confidence = compute_tracking_confidence
        self._aiming_manager = AimingManager(config=self._aim_config)

        model_path = str(self.get_parameter("model_path").value)
        if not model_path or model_path.strip() == "":
            candidate = os.path.join(
                self._core_path,
                "detector+tracker", "ga_results", "yolo11s_ga_final" + chr(45) + "3", "weights", "best.pt"
            )
            if os.path.exists(candidate):
                model_path = candidate
                self.get_logger().info(f"model_path 미지정으로 기본 GA 가중치 자동 채택: {model_path}")
            else:
                raise RuntimeError(
                    "model_path 파라미터가 비어 있고 기본 가중치를 찾을 수 없습니다: " + candidate
                )
        self.get_logger().info(f"추적기 가중치 로딩: {model_path}")
        self._tracker = OpticalFlowTracker(
            model_path=model_path,
            device=str(self.get_parameter("device").value),
            conf_thres=float(self.get_parameter("conf_thres").value),
        )

        # ── 거리 추정기 ─────────────────────────────────────────────────
        self._range_estimator = HybridRangeEstimator(
            laser=ConeLaserRange(
                staleness_timeout_s=float(self.get_parameter("laser_staleness_timeout_s").value),
                noise_sigma_m=float(self.get_parameter("laser_noise_sigma_m").value),
                range_min_m=float(self.get_parameter("range_min_m").value),
                range_max_m=float(self.get_parameter("range_max_m").value),
            ),
            bbox=BoundingBoxRange(
                characteristic_size_m=float(self.get_parameter("target_characteristic_size_m").value),
                size_uncertainty_ratio=float(self.get_parameter("target_size_uncertainty_ratio").value),
                bbox_pixel_sigma=float(self.get_parameter("bbox_pixel_sigma").value),
                min_range_m=float(self.get_parameter("range_min_m").value),
                max_range_m=float(self.get_parameter("range_max_m").value),
            ),
            smoothing_alpha=float(self.get_parameter("range_smoothing_alpha").value),
            cross_check_sigma=float(self.get_parameter("laser_cross_check_sigma").value),
            max_range_rate_mps=float(self.get_parameter("max_range_rate_mps").value),
        )

        # ── 3단계: 격발 제어 (기본 비활성) ──────────────────────────────
        self._enable_fire_control = bool(self.get_parameter("enable_fire_control").value)
        self._fire_control = None
        self._turret_world_translation = np.array(
            [float(v) for v in self.get_parameter("turret_world_translation_m").value], dtype=np.float64
        )
        self._turret_world_yaw = float(self.get_parameter("turret_world_yaw_rad").value)
        if self._enable_fire_control:
            from drone_sim.smash_fcs.fire_control import FireController  # noqa: E402

            target_hit_radius_m = float(self.get_parameter("target_hit_radius_m").value)
            if target_hit_radius_m <= 0.0:
                target_hit_radius_m = float(self.get_parameter("target_characteristic_size_m").value) / 2.0
            fire_mode = str(self.get_parameter("fire_mode").value)
            self._fire_control = FireController(
                projectile_config=self._aim_config.projectile,
                muzzle_velocity_mps=self._aim_config.projectile.muzzle_velocity,
                fire_mode=fire_mode,
                fire_rate_rpm=float(self.get_parameter("fire_rate_rpm").value),
                target_radius_m=target_hit_radius_m,
                ground_truth_timeout_s=float(self.get_parameter("ground_truth_timeout_s").value),
                impact_effect_hold_s=float(self.get_parameter("impact_effect_hold_s").value),
            )
            self.get_logger().info(
                "격발 제어 활성화: mode={} target_hit_radius={:.3f}m ground_truth_topic={}".format(
                    fire_mode, target_hit_radius_m, str(self.get_parameter("ground_truth_pose_topic").value)
                )
            )
            self.get_logger().warn(
                "Hit/Kill 판정은 Gazebo 실측(ground truth) 위치에 의존합니다. "
                "drone_sim/models/drone/model.sdf 의 PosePublisher 플러그인과 "
                "smash_scene.launch.py 의 ground-truth 브리지가 실제로 떠 있는지, "
                "그리고 ground_truth_pose_topic 이름이 실제 gz 토픽과 일치하는지 "
                "`gz topic -l` 로 확인하십시오. 확인 전에는 판정이 전부 UNVERIFIED 로 "
                "남습니다."
            )

        # ── 런타임 상태 ─────────────────────────────────────────────────
        self._bridge_cv = CvBridge()
        self._intrinsics: Optional[np.ndarray] = None
        self._pan_cmd = float(self.get_parameter("search_pan_rad").value)
        self._tilt_cmd = tk.clamp(
            float(self.get_parameter("search_tilt_rad").value), self._tilt_min, self._tilt_max
        )
        self._user_pan = self._pan_cmd
        self._user_tilt = self._tilt_cmd
        self._pan_goal = self._user_pan
        self._tilt_goal = self._user_tilt
        self._search_pan = self._user_pan
        self._search_tilt = self._user_tilt
        self._manual_pan_offset = 0.0
        self._manual_tilt_offset = 0.0
        self._aim_aligned = False
        self._align_err_px = 999.0
        self._align_tolerance_px = 35.0
        self._feedforward = tk.GoalFeedForward(
            lead_time_s=float(self.get_parameter("command_feedforward_s").value),
            max_rate_rad_s=max(self._pan_max_rate, self._tilt_max_rate),
        )
        self._last_command_time: Optional[float] = None
        self._border_frames = 0
        self._state_label = "SEARCH"
        self._last_output = None
        self._last_bbox: Optional[tuple[float, float, float, float]] = None
        self._last_range = None
        self._last_aim_dir_base: Optional[np.ndarray] = None
        self._last_camera_pose: Optional[np.ndarray] = None
        self._last_scope_pose: Optional[np.ndarray] = None
        self._last_frame_timestamp: Optional[float] = None
        self._frames_seen = 0
        self._frames_solved = 0

        # 3단계 격발 제어 런타임 상태
        self._trigger_pressed = False
        self._gt_position_base: Optional[np.ndarray] = None
        self._gt_timestamp: Optional[float] = None

        # ── ROS 인터페이스 ──────────────────────────────────────────────
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._cmd_pub = self.create_publisher(
            Float64MultiArray, str(self.get_parameter("command_topic").value), 10
        )
        self._state_pub = self.create_publisher(String, str(self.get_parameter("state_topic").value), 10)
        self._annotated_pub = self.create_publisher(
            Image, str(self.get_parameter("annotated_topic").value), 10
        )

        self.create_subscription(
            CameraInfo, str(self.get_parameter("camera_info_topic").value), self._on_camera_info, 10
        )
        self.create_subscription(
            PointCloud2, str(self.get_parameter("laser_topic").value), self._on_laser, SENSOR_QOS
        )
        self.create_subscription(
            Image, str(self.get_parameter("image_topic").value), self._on_image, SENSOR_QOS
        )
        self.create_subscription(
            Float64MultiArray, "/smash_fcs/manual_slew", self._on_manual_slew, 10
        )

        rate = max(1.0, float(self.get_parameter("command_rate_hz").value))
        self.create_timer(1.0 / rate, self._publish_command)

        if self._enable_fire_control:
            self.create_subscription(
                PoseStamped,
                str(self.get_parameter("ground_truth_pose_topic").value),
                self._on_ground_truth_pose,
                SENSOR_QOS,
            )
            self.create_subscription(
                Bool, str(self.get_parameter("trigger_topic").value), self._on_trigger, 10
            )
            self._fire_event_pub = self.create_publisher(
                String, str(self.get_parameter("fire_event_topic").value), 10
            )
            self._hit_result_pub = self.create_publisher(
                String, str(self.get_parameter("hit_result_topic").value), 10
            )
            self._stats_pub = self.create_publisher(
                String, str(self.get_parameter("engagement_stats_topic").value), 10
            )
            fc_rate = max(1.0, float(self.get_parameter("fire_control_rate_hz").value))
            self.create_timer(1.0 / fc_rate, self._fire_control_tick)

        self.get_logger().info("SMASH FCS 노드 기동 완료. 상태: SEARCH")

    # ── 콜백 ────────────────────────────────────────────────────────────

    def _on_camera_info(self, msg: CameraInfo) -> None:
        k = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        if k[0, 0] <= 0.0 or k[1, 1] <= 0.0:
            return
        if self._intrinsics is None:
            self.get_logger().info(
                "카메라 내부파라미터 수신: fx={:.2f} fy={:.2f} cx={:.2f} cy={:.2f} ({}x{})".format(
                    k[0, 0], k[1, 1], k[0, 2], k[1, 2], msg.width, msg.height
                )
            )
        self._intrinsics = k

    def _on_laser(self, msg: PointCloud2) -> None:
        """광각 레이저 점군에서 각 빔의 경사거리를 뽑아 추정기에 넣는다.

        점들은 camera_link_optical 원점 기준이므로 노름이 곧 경사거리다.
        최근접 유효 반사를 고르는 일은 ConeLaserRange 가 담당한다.
        """

        try:
            x_off, y_off, z_off, datatype = field_offsets(msg.fields)
            distances = ranges_from_xyz_buffer(
                data=bytes(msg.data),
                point_step=int(msg.point_step),
                x_offset=x_off,
                y_offset=y_off,
                z_offset=z_off,
                datatype=datatype,
                is_bigendian=bool(msg.is_bigendian),
            )
        except ValueError as exc:
            self.get_logger().warn(f"레이저 점군 파싱 실패: {exc}", throttle_duration_sec=5.0)
            return

        if distances.size == 0:
            return
        self._range_estimator.laser.submit_ranges(
            distances.tolist(), _stamp_to_seconds(msg.header.stamp)
        )

    def _on_image(self, msg: Image) -> None:
        if self._intrinsics is None:
            self.get_logger().warn(
                "camera_info 를 아직 받지 못해 프레임을 건너뜁니다.", throttle_duration_sec=5.0
            )
            return

        timestamp = _stamp_to_seconds(msg.header.stamp)
        frame_bgr = self._bridge_cv.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._frames_seen += 1

        result = self._tracker.update(frame_bgr)
        if result is None:
            self._enter_search("표적 미포착")
            self._render(msg, frame_bgr)
            return

        x, y, w, h = result.bbox
        bbox = (float(x), float(y), float(x + w), float(y + h))
        center_px = (float(x + w / 2.0), float(y + h / 2.0))
        self._last_bbox = bbox

        # 화면 이탈 판정: 로드맵 2단계 요구사항("이탈 시 즉시 SEARCH 초기화")의
        # 실행 지점. OpticalFlowTracker 는 한 번 초기화되면 표적을 잃어도 계속
        # 마지막 bbox 를 반환하므로, 여기서 명시적으로 끊어 주지 않으면 화면 밖으로
        # 나간 표적을 계속 추종하는 것처럼 보인다.
        if self._bbox_on_border(bbox, frame_bgr.shape):
            self._border_frames += 1
            if self._border_frames >= self._border_frames_limit:
                self._enter_search("표적 화면 이탈")
                self._render(msg, frame_bgr)
                return
        else:
            self._border_frames = 0

        range_estimate = self._range_estimator.estimate(
            now=timestamp,
            bbox_xyxy=bbox,
            principal_point=(float(self._intrinsics[0, 2]), float(self._intrinsics[1, 2])),
            fx=float(self._intrinsics[0, 0]),
            fy=float(self._intrinsics[1, 1]),
        )
        self._last_range = range_estimate
        if not range_estimate.valid:
            self._state_label = "TRACK(NO RANGE)"
            self._render(msg, frame_bgr)
            return

        camera_pose = self._lookup_camera_pose(msg.header.stamp)
        if camera_pose is None:
            self._state_label = "TRACK(NO TF)"
            self._render(msg, frame_bgr)
            return
        self._last_camera_pose = camera_pose

        scope_pose = tk.muzzle_pose_from_camera_pose(
            camera_pose, gun_to_muzzle_m=self._gun_to_muzzle, gun_to_camera_m=self._gun_to_camera
        )
        self._last_scope_pose = scope_pose

        from aiming_engine.types import TrackerFrame  # noqa: E402

        tracker_frame = TrackerFrame(
            timestamp=timestamp,
            bbox=bbox,
            center_px=center_px,
            # OpticalFlowTracker 는 bbox 속도를 자체 추정하지 않는다. aiming_engine 의
            # TargetState 가 월드 프레임 위치 이력에서 직접 속도를 추정하며 이 필드를
            # 읽지 않으므로 자리표시자를 넣는다(bridge/frame_builder.py 와 동일한 규약).
            velocity_px=(0.0, 0.0),
            tracking_confidence=self._compute_tracking_confidence(result.score, result.is_recovery_event),
            follower_state=self._compute_follower_state(result.is_recovery_event),
            camera_intrinsics=self._intrinsics,
            camera_extrinsics=scope_pose,
            laser_range_m=float(range_estimate.distance_m),
        )

        output = self._aiming_manager.update(tracker_frame)
        self._last_output = output
        self._last_frame_timestamp = timestamp
        self._frames_solved += 1

        aim_dir_base = self._aim_direction_in_base(output.aim_solution, scope_pose)
        self._last_aim_dir_base = aim_dir_base

        # 사수 수동 조준: 포탑은 드론을 자동 추종하지 않고, 오직 사수가 조작한 각도를 유지한다.
        self._pan_goal = self._user_pan
        self._tilt_goal = self._user_tilt

        # 사수 조준선(보어사이트 십자선)과 탄착 리드각 레티클 간 정렬 오차 판정
        cx = int(round(self._intrinsics[0, 2])) if self._intrinsics is not None else frame_bgr.shape[1] // 2
        cy = int(round(self._intrinsics[1, 2])) if self._intrinsics is not None else frame_bgr.shape[0] // 2
        lead_pixel = self._project_direction_to_pixel(aim_dir_base) if aim_dir_base is not None else None

        if lead_pixel is not None:
            self._align_err_px = math.hypot(lead_pixel[0] - cx, lead_pixel[1] - cy)
            self._aim_aligned = (self._align_err_px <= self._align_tolerance_px)
        else:
            self._align_err_px = 999.0
            self._aim_aligned = False

        if output.aim_ready and self._aim_aligned:
            self._state_label = "READY"
        elif output.aim_solution is not None and output.aim_solution.solver_converged:
            self._state_label = f"ALIGN({self._align_err_px:.0f}px)"
        else:
            self._state_label = output.state.value

        self._render(msg, frame_bgr)

    # ── 내부 로직 ───────────────────────────────────────────────────────

    def _bbox_on_border(self, bbox: tuple[float, float, float, float], shape) -> bool:
        height, width = shape[0], shape[1]
        m = self._border_margin_px
        x1, y1, x2, y2 = bbox
        return x1 <= m or y1 <= m or x2 >= (width - 1 - m) or y2 >= (height - 1 - m)

    def _enter_search(self, reason: str) -> None:
        """추적기와 조준 상태를 완전히 초기화하고 SEARCH 로 되돌린다.

        AimingManager 를 새로 만드는 이유: TargetState 가 지난 표적의 위치 이력을
        들고 있으면 다음 표적을 재포착했을 때 두 궤적이 섞여 속도 추정이 오염된다.
        모델 로딩이 없는 가벼운 객체라 매 초기화마다 재생성해도 비용이 무시할 만하다.
        """

        if self._state_label != "SEARCH":
            self.get_logger().info(f"상태 초기화 -> SEARCH ({reason})")
        self._tracker.reset()
        self._aiming_manager = self._AimingManager(config=self._aim_config)
        self._range_estimator.reset()
        self._feedforward.reset()
        self._border_frames = 0
        self._state_label = "SEARCH"
        self._last_output = None
        self._last_bbox = None
        self._last_range = None
        self._last_aim_dir_base = None
        # 사수가 조준하고 있는 현재 지향각을 그대로 유지
        self._pan_goal = self._user_pan
        self._tilt_goal = self._user_tilt
        self._aim_aligned = False
        self._align_err_px = 999.0
        if self._fire_control is not None:
            # 트리거 엣지 검출만 초기화한다. 이미 발사되어 비행 중인 탄환은
            # 취소하지 않는다 — 표적 추적 상실과 무관하게 물리적으로 계속
            # 날아가는 것이 맞고, 판정은 예정대로 진행된다(fire_control.py
            # FireController.reset_trigger_edge 참고).
            self._fire_control.reset_trigger_edge()

    def _lookup_camera_pose(self, stamp) -> Optional[np.ndarray]:
        """base_frame <- camera_optical_frame 4x4 동차변환을 TF 에서 가져온다."""

        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_frame,
                self._camera_frame,
                stamp,
                timeout=rclpy.duration.Duration(seconds=self._tf_timeout_s),
            )
        except tf2_ros.ExtrapolationException:
            try:
                tf = self._tf_buffer.lookup_transform(
                    self._base_frame,
                    self._camera_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=self._tf_timeout_s),
                )
            except Exception as exc:
                self.get_logger().warn(f"TF 조회 실패: {exc}", throttle_duration_sec=2.0)
                return None
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
        ) as exc:
            self.get_logger().warn(f"TF 조회 실패: {exc}", throttle_duration_sec=2.0)
            return None

        t = tf.transform.translation
        q = tf.transform.rotation
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = tk.quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
        pose[:3, 3] = np.array([t.x, t.y, t.z], dtype=np.float64)
        return pose

    def _aim_direction_in_base(self, aim_solution, scope_pose: np.ndarray) -> Optional[np.ndarray]:
        """AimSolution 의 scope 프레임 방위/앙각을 base 프레임 방향벡터로 되돌린다.

        AimSolver 는 월드 프레임에서 [t, 방위각, 앙각] 을 풀고, 같은 외부파라미터로
        scope 프레임에 투영해 반환한다(aim_solver.py). 동일한 회전행렬로 되돌리면
        솔버가 실제로 푼 월드(=base) 방향을 정확히 복원한다. 이 방향에는 중력과
        (활성화 시) 공기저항 보정이 이미 포함되어 있다.
        """

        if aim_solution is None:
            return None
        if not np.all(np.isfinite([aim_solution.azimuth, aim_solution.elevation])):
            return None

        from aiming_engine.projectile_model import azimuth_elevation_to_direction  # noqa: E402

        direction_scope = azimuth_elevation_to_direction(aim_solution.azimuth, aim_solution.elevation)
        direction_base = np.asarray(scope_pose, dtype=np.float64)[:3, :3] @ direction_scope
        norm = float(np.linalg.norm(direction_base))
        if norm < 1e-9:
            return None
        return direction_base / norm

    def _publish_command(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 1.0 / 30.0 if self._last_command_time is None else max(1e-3, now - self._last_command_time)
        self._last_command_time = now

        self._pan_cmd = tk.rate_limit(self._pan_goal, self._pan_cmd, self._pan_max_rate, dt)
        self._tilt_cmd = tk.clamp(
            tk.rate_limit(self._tilt_goal, self._tilt_cmd, self._tilt_max_rate, dt),
            self._tilt_min,
            self._tilt_max,
        )

        msg = Float64MultiArray()
        msg.data = [float(self._pan_cmd), float(self._tilt_cmd)]
        self._cmd_pub.publish(msg)

        state = String()
        state.data = self._state_label
        self._state_pub.publish(state)

    # ── 3단계: 격발 제어 ───────────────────────────────────────────────

    def _world_to_base(self, world_pos: np.ndarray) -> np.ndarray:
        """Gazebo world 프레임 위치를 base_footprint 프레임으로 옮긴다.

        world = Rz(yaw) @ base + translation 이라는 가정(포탑은 스폰 이후 움직이지
        않는 고정 강체이므로 이 변환은 시간에 무관한 상수) 아래, 역변환
        base = Rz(yaw)^T @ (world - translation) 을 적용한다. Rz(yaw)^T 는
        회전행렬의 직교성 덕에 Rz(-yaw) 와 같다. translation/yaw 은
        turret_world_translation_m / turret_world_yaw_rad 파라미터이며, 이 값이
        실제 스폰 인자와 다르면 이 변환 자체가 틀리다(모듈 상단 docstring 3단계
        절 및 fire_control.py 한계 3번 참고).
        """

        rel = np.asarray(world_pos, dtype=np.float64) - self._turret_world_translation
        yaw = self._turret_world_yaw
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1], rel[2]], dtype=np.float64)

    def _on_manual_slew(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 2:
            d_pan = float(msg.data[0])
            d_tilt = float(msg.data[1])
            if abs(d_pan - 999.0) < 1e-3:
                self._user_pan = 0.0
                self._user_tilt = 0.0
                self.get_logger().info("[MANUAL SLEW] 사수 조준 수평 정렬 초기화 (PAN 0.0deg, TILT 0.0deg)")
            elif abs(d_pan - 888.0) < 1e-3:
                self._user_pan = 0.0
                self._user_tilt = 0.38
                self.get_logger().info("[MANUAL SLEW] 대공 경계 자세 프리셋 가동 -> PAN 0.0deg, TILT 21.8deg (드론 정면 직시)")
            else:
                self._user_pan += d_pan
                self._user_tilt = float(np.clip(self._user_tilt + d_tilt, self._tilt_min, self._tilt_max))
                self.get_logger().info(
                    f"[MANUAL SLEW] 사수 수동 조작 -> 지향각 PAN: {math.degrees(self._user_pan):+.1f}deg, TILT: {math.degrees(self._user_tilt):+.1f}deg"
                )
            self._pan_goal = self._user_pan
            self._tilt_goal = self._user_tilt

    def _on_ground_truth_pose(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        world_pos = np.array([p.x, p.y, p.z], dtype=np.float64)
        self._gt_position_base = self._world_to_base(world_pos)
        self._gt_timestamp = _stamp_to_seconds(msg.header.stamp)

    def _on_trigger(self, msg: Bool) -> None:
        self._trigger_pressed = bool(msg.data)

    def _fire_control_tick(self) -> None:
        """command 타이머와 별개의 고정 주기 타이머. 새 영상 프레임이 없어도
        (예: 표적을 놓친 동안) 이미 비행 중인 탄환의 판정 시각은 실시간으로
        도래하므로, 판정만큼은 영상 콜백에 묶지 않고 독립적으로 돈다."""

        if self._fire_control is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        if (
            self._last_scope_pose is not None
            and self._last_output is not None
            and self._last_frame_timestamp is not None
        ):
            actual_ready = bool(self._last_output.aim_ready and self._aim_aligned)
            actual_bore_dir_base = np.asarray(self._last_scope_pose, dtype=np.float64)[:3, 0]
            norm = float(np.linalg.norm(actual_bore_dir_base))
            bore_dir = actual_bore_dir_base / norm if norm > 1e-6 else self._last_aim_dir_base

            shot = self._fire_control.maybe_fire(
                now=self._last_frame_timestamp,
                aim_ready=actual_ready,
                trigger_pressed=self._trigger_pressed,
                aim_solution=self._last_output.aim_solution,
                launch_point_world=self._last_scope_pose[:3, 3],
                direction_world=bore_dir,
            )
            if shot is not None:
                self._publish_fire_event(shot)
                self._publish_engagement_stats()

        results = self._fire_control.update(
            now=now,
            ground_truth_position_world=self._gt_position_base,
            ground_truth_timestamp=self._gt_timestamp,
        )
        for result in results:
            self._publish_hit_result(result)
        if results:
            self._publish_engagement_stats()

    def _publish_fire_event(self, shot) -> None:
        payload = {
            "event": "fire",
            "shot_id": shot.shot_id,
            "trigger_mode": shot.trigger_mode,
            "fire_time_s": shot.fire_time,
            "muzzle_world_m": shot.launch_point_world.tolist(),
            "launch_azimuth_deg": math.degrees(shot.azimuth_world),
            "launch_elevation_deg": math.degrees(shot.elevation_world),
            "muzzle_velocity_mps": shot.muzzle_velocity_mps,
            "predicted_tof_s": shot.predicted_tof_s,
            "predicted_impact_world_m": shot.predicted_impact_point_world.tolist(),
            "target_radius_m": shot.target_radius_m,
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self._fire_event_pub.publish(msg)
        self.get_logger().info(
            "격발 #{} ({}): TOF {:.1f}ms".format(shot.shot_id, shot.trigger_mode, shot.predicted_tof_s * 1000.0)
        )

    def _publish_hit_result(self, result) -> None:
        payload = {
            "event": "hit_result",
            "shot_id": result.shot_id,
            "verdict": result.verdict,
            "impact_time_s": result.impact_time,
            "projectile_point_world_m": result.projectile_point_world.tolist(),
            "ground_truth_point_world_m": (
                None if result.ground_truth_point_world is None else result.ground_truth_point_world.tolist()
            ),
            "miss_distance_m": result.miss_distance_m,
            "ground_truth_available": result.ground_truth_available,
            "ground_truth_age_s": result.ground_truth_age_s,
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self._hit_result_pub.publish(msg)
        detail = (
            f"miss_distance={result.miss_distance_m:.3f}m"
            if result.miss_distance_m is not None
            else "ground truth 없음(UNVERIFIED)"
        )
        self.get_logger().info(f"판정 #{result.shot_id}: {result.verdict} ({detail})")

    def _publish_engagement_stats(self) -> None:
        stats = self._fire_control.stats()
        msg = String()
        msg.data = json.dumps(stats, ensure_ascii=False)
        self._stats_pub.publish(msg)

    # ── HUD ─────────────────────────────────────────────────────────────

    def _project_direction_to_pixel(self, direction_base: np.ndarray) -> Optional[tuple[int, int]]:
        """base 프레임 방향벡터를 이미지 픽셀로 투영한다.

        camera_link_optical 은 FLU(X 전, Y 좌, Z 상) 이므로 표준 핀홀 축으로 라벨을
        바꾼 뒤 투영한다: x_right = -y_flu, y_down = -z_flu, z_forward = x_flu.
        """

        if self._last_camera_pose is None or self._intrinsics is None:
            return None
        d_flu = np.asarray(self._last_camera_pose, dtype=np.float64)[:3, :3].T @ direction_base
        z = float(d_flu[0])
        if z <= 1e-6:
            return None
        x_right = float(-d_flu[1])
        y_down = float(-d_flu[2])
        u = self._intrinsics[0, 0] * (x_right / z) + self._intrinsics[0, 2]
        v = self._intrinsics[1, 1] * (y_down / z) + self._intrinsics[1, 2]
        return int(round(u)), int(round(v))

    def _project_world_point_to_pixel(self, point_base: np.ndarray) -> Optional[tuple[int, int]]:
        """base 프레임의 절대 좌표점을 이미지 픽셀로 투영한다.

        현재 카메라 위치에서 그 점을 향하는 방향벡터를 만들어
        _project_direction_to_pixel 에 그대로 위임한다(3단계 탄환 궤적/명중점
        HUD 표시용). 카메라가 계속 회전하므로 매 프레임 최신 카메라 자세로
        다시 계산해야 한다 — 한 프레임 전 자세로 그리면 포탑이 움직이는 동안
        탄착점 마커가 실제 화면 위치에서 어긋난다.
        """

        if self._last_camera_pose is None or self._intrinsics is None:
            return None
        camera_pos = np.asarray(self._last_camera_pose, dtype=np.float64)[:3, 3]
        direction = np.asarray(point_base, dtype=np.float64) - camera_pos
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return None
        return self._project_direction_to_pixel(direction / norm)

    def _render(self, msg: Image, frame_bgr: np.ndarray) -> None:
        """1단계용 최소 HUD. 정식 스마트 스코프 HUD 는 4단계 과제다.

        텍스트는 ASCII 만 사용한다. OpenCV 의 Hershey 폰트는 한글 글리프가 없어
        한글을 그리면 전부 사각형으로 깨지기 때문이다.
        """

        if not self._publish_annotated:
            return

        import cv2  # 지연 임포트

        canvas = frame_bgr.copy()
        height, width = canvas.shape[:2]
        cx = int(round(self._intrinsics[0, 2])) if self._intrinsics is not None else width // 2
        cy = int(round(self._intrinsics[1, 2])) if self._intrinsics is not None else height // 2


        ready = bool(self._last_output is not None and self._last_output.aim_ready and self._aim_aligned)
        cross_color = (0, 255, 0) if ready else (0, 255, 255)
        cross_thick = 2 if ready else 1

        # 보어사이트 십자선 (총구 지향 중심)
        cv2.line(canvas, (cx - 16, cy), (cx - 4, cy), cross_color, cross_thick)
        cv2.line(canvas, (cx + 4, cy), (cx + 16, cy), cross_color, cross_thick)
        cv2.line(canvas, (cx, cy - 16), (cx, cy - 4), cross_color, cross_thick)
        cv2.line(canvas, (cx, cy + 4), (cx, cy + 16), cross_color, cross_thick)

        box_color = (0, 255, 0) if ready else (0, 165, 255)

        if self._last_bbox is not None:
            x1, y1, x2, y2 = (int(round(v)) for v in self._last_bbox)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)

        # 탄도 보정이 포함된 지향점 (미래 탄착 리드각 레티클)
        if self._last_aim_dir_base is not None:
            pixel = self._project_direction_to_pixel(self._last_aim_dir_base)
            if pixel is not None:
                reticle_color = (0, 255, 0) if ready else (0, 165, 255)
                # 리드각 조준원 및 십자선
                cv2.circle(canvas, pixel, 8, reticle_color, 2)
                cv2.line(canvas, (pixel[0] - 12, pixel[1]), (pixel[0] + 12, pixel[1]), reticle_color, 1)
                cv2.line(canvas, (pixel[0], pixel[1] - 12), (pixel[0], pixel[1] + 12), reticle_color, 1)
                # 정렬 유도선 (보어사이트 십자선 -> 리드각 레티클)
                if not ready:
                    cv2.line(canvas, (cx, cy), pixel, (0, 200, 255), 1, cv2.LINE_AA)

        lines = [f"STATE {self._state_label}"]
        lines.append(
            "AIM SLEW PAN {:+6.1f}deg TILT {:+5.1f}deg".format(
                math.degrees(self._user_pan), math.degrees(self._user_tilt)
            )
        )
        if self._last_range is not None and self._last_range.valid:
            src = {SOURCE_LASER: "LSR", SOURCE_BBOX: "BOX"}.get(self._last_range.source, "---")
            lines.append(
                f"RNG {self._last_range.distance_m:6.2f}m +-{self._last_range.sigma_m:4.2f} [{src}]"
            )
        else:
            lines.append("RNG   --.--m [---]")

        out = self._last_output
        if out is not None and out.aim_solution is not None:
            sol = out.aim_solution
            lines.append(f"TOF {sol.time_of_flight * 1000.0:6.1f}ms  ITER {sol.solver_iterations}")
            lines.append(
                "LEAD AZ {:+.3f}deg EL {:+.3f}deg".format(
                    math.degrees(sol.ballistic_offset[0]), math.degrees(sol.ballistic_offset[1])
                )
            )
            lines.append(
                "ALIGN ERR {:4.1f}px  PHIT {:4.2f}".format(
                    self._align_err_px, out.hit_probability
                )
            )
            lines.append(
                "TGT SPD {:5.2f}m/s  CONV {}".format(
                    float(out.debug.get("target_speed", float("nan"))),
                    "Y" if sol.solver_converged else "N",
                )
            )
        lines.append(
            "CMD PAN {:+7.2f}deg TILT {:+6.2f}deg".format(
                math.degrees(self._pan_cmd), math.degrees(self._tilt_cmd)
            )
        )

        # ── 3단계: 탄환 궤적 및 Hit/Kill 이펙트 ─────────────────────────
        if self._fire_control is not None:
            now = self.get_clock().now().nanoseconds * 1e-9
            for shot in self._fire_control.pending_shots:
                pos = self._fire_control.projectile_position_at(shot, now - shot.fire_time)
                pixel = self._project_world_point_to_pixel(pos)
                if pixel is not None:
                    cv2.circle(canvas, pixel, 3, (0, 255, 255), -1)

            for result in self._fire_control.recent_results:
                pixel = self._project_world_point_to_pixel(result.projectile_point_world)
                if pixel is None:
                    continue
                if result.verdict == "HIT":
                    cv2.circle(canvas, pixel, 18, (0, 255, 0), 3)
                    cv2.putText(
                        canvas, "KILL", (pixel[0] - 26, pixel[1] - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA,
                    )
                    cv2.putText(
                        canvas, "KILL", (pixel[0] - 26, pixel[1] - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
                    )
                elif result.verdict == "MISS":
                    cv2.line(canvas, (pixel[0] - 12, pixel[1] - 12), (pixel[0] + 12, pixel[1] + 12), (0, 0, 255), 2)
                    cv2.line(canvas, (pixel[0] - 12, pixel[1] + 12), (pixel[0] + 12, pixel[1] - 12), (0, 0, 255), 2)
                else:  # UNVERIFIED
                    cv2.circle(canvas, pixel, 14, (0, 165, 255), 2)

            stats = self._fire_control.stats()
            hit_rate = stats["hit_rate"]
            hit_rate_str = f"{hit_rate * 100.0:5.1f}%" if hit_rate is not None else " --.-%"
            lines.append(
                "SHOTS {:3d} HIT {:3d} MISS {:3d} UNV {:3d} ACC {}".format(
                    stats["shots_fired"], stats["hits"], stats["misses"], stats["unverified"], hit_rate_str
                )
            )

        for i, text in enumerate(lines):
            origin = (8, 20 + 18 * i)
            cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        if ready:
            cv2.putText(
                canvas, "[ FIRE READY ]", (width - 160, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA
            )
            cv2.putText(
                canvas, "[ FIRE READY ]", (width - 160, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA
            )
        elif self._last_output is not None and self._last_aim_dir_base is not None:
            cv2.putText(
                canvas, "ALIGNING", (width - 110, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                canvas, "ALIGNING", (width - 110, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1, cv2.LINE_AA
            )
        else:
            # 시야 밖 표적 방향 안내 (Off-Boresight Air Cue)
            cue_txt = "^ AIR CUE: PRESS [UP] OR [T] (TGT EL +22deg) ^"
            cv2.putText(
                canvas, cue_txt, (width // 2 - 190, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                canvas, cue_txt, (width // 2 - 190, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
            )

        canvas = np.ascontiguousarray(canvas, dtype=np.uint8)
        out_msg = Image()
        out_msg.header = msg.header
        out_msg.height = canvas.shape[0]
        out_msg.width = canvas.shape[1]
        out_msg.encoding = "bgr8"
        out_msg.is_bigendian = 0
        out_msg.step = canvas.shape[1] * 3
        out_msg.data = canvas.tobytes()
        self._annotated_pub.publish(out_msg)


def main() -> None:
    rclpy.init()
    node = SmashFcsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f"종료. 수신 프레임 {node._frames_seen}, 조준해 산출 프레임 {node._frames_solved}"
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
