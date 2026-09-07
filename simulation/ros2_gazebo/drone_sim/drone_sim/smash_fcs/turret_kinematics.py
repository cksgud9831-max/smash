"""2축(Pan/Tilt) 포탑 순기구학 및 역기구학, 그리고 소총 조준경 기하.

■ 이 프로젝트에서 포탑이 무엇을 뜻하는가 (중요)
    SMASH 의 실제 제품은 5.56mm 개인화기에 장착하는 스마트 조준경이다. 서보로
    총을 돌리는 무기가 아니라, 사수에게 조준점을 알려 주는 자문 장비다.
    aiming_engine 전체가 그 전제로 설계되어 있다(aim_state_machine.py 의
    "there is no firing state here", aiming_manager.py 의 "advisory only").

    Gazebo 의 CIWS 포탑은 참고한 오픈소스 환경에서 빌려 온 3D 비행 환경이며,
    여기서 Pan/Tilt 서보는 **사수가 총을 겨누는 동작의 시뮬레이션 대체물**이다.
    사람 대신 서보가 조준을 따라가 주어야 폐루프 수렴 성능(정착시간, 정상상태
    조준오차)을 정량 측정할 수 있기 때문에 유지한다. 제품 기능이 아니다.

    따라서 아래 기하는 두 층으로 나뉜다.
      1. 제품 고유 기하: 조준경 광학중심과 총구의 상대 위치. 실제 소총 제원.
      2. 시뮬레이션 고유 기하: 포탑 회전축과 링크 오프셋. Gazebo 모델 제원.
    조준 정확도에 물리적으로 의미가 있는 것은 1번이고, 2번은 서보를 올바로
    구동하기 위한 시뮬레이션 배선이다.

■ 좌표계 규약
    1. base_frame (base_footprint): 포탑 고정 관성 기준계. 오른손 좌표계, Z축이 위.
       aiming_engine 이 요구하는 World 프레임 역할을 그대로 수행한다.
       (Gazebo world 원점과는 포탑 스폰 오프셋만큼 차이가 나지만, 표적 위치도
        같은 프레임에서 산출되므로 조준 계산의 정합성에는 영향이 없다.
        중력 방향이 -Z 인 점만 만족하면 된다.)
    2. gun_link (틸트하는 포신 조립체): X 전방(보어사이트), Y 좌측, Z 상방 (FLU).
    3. camera_link_optical: gun_link 와 회전이 동일한 FLU 프레임. 원점은 카메라 광학 중심.
       주의: 이름과 달리 표준 핀홀 광학 규약(X 우, Y 하, Z 전방)이 아니라 FLU 이다.
       URDF 의 camera_optical_joint 가 rpy="0 0 0" 이기 때문이다.
    4. scope 프레임 (aiming_engine 용어): 원점이 포구(muzzle)인 FLU 프레임.
       회전은 gun_link 와 동일하다.

■ URDF 원본 수치 (ciws_turret/urdf/ciws_turret.urdf.xacro 에서 그대로 옮김)
    base_joint          : base_footprint -> base_link, 원점 일치
    turret_pan_joint    : base_link -> turret_link, origin (0, 0, 0.0885), axis (0, 0, 1), continuous
    turret_tilt_joint   : turret_link -> gun_link,
                          origin (piv_x - 0.03785, 0, piv_z + 0.05) = (-0.06185, 0, 0.5065),
                          axis (0, -1, 0), limit 0 ~ 85도
    camera_joint        : gun_link -> camera_link, origin (-0.07, 0.05, 0.15)
    camera_optical_joint: camera_link -> camera_link_optical, origin (0.229, 0.16, 0.10)
       => 카메라 광학 중심의 gun_link 좌표 = (-0.07+0.229, 0.05+0.16, 0.15+0.10)
                                          = (0.159, 0.21, 0.25)

■ 총구 위치 (제품 고유 기하)
    SCOPE_TO_MUZZLE_M 은 조준경 광학중심에서 본 총구 위치를 FLU 로 적은 값이다.
    Gazebo 포탑 메시의 포신 끝이 아니라 실제 소총 제원에서 나온다.

      1. 상하 성분 = 조준경 광축과 총열 중심의 높이차 (sight height over bore).
         AR 계열 플랫탑 상부레일에 일반적인 마운트로 광학장비를 얹은 구성을 기준으로
         0.066 m 로 두었다. 사용자 확인값이다. 로우 마운트 구성은 0.038 m 수준이다.
         이 값이 시차와 근거리 조준 오차를 직접 결정하는 유일한 수직 항이다.
      2. 전후 성분 = 조준경 광학중심에서 총구까지의 거리. 0.45 m 로 두었다.
         이 값은 시선에 평행하므로 시차에 거의 기여하지 않는다. 다만 발사점
         위치를 미세하게 바꾸므로 실측하면 더 정확해진다.
      3. 좌우 성분 = 0. 조준경이 총열 바로 위에 정렬되어 있다고 본다.

    시차의 크기는 수직 성분만이 지배한다. 사거리 R 에서 대략 0.066 / R [rad] 이며,
    35 m 에서 약 1.9 mrad(0.108도), 10 m 에서 약 6.6 mrad(0.38도)다. 4단계 READY
    임계값 0.5도에 비하면 작지만 0 은 아니므로 탄도 해에 반영한다.

    GUN_TO_MUZZLE_M 은 위 값을 Gazebo 포탑 링크 좌표로 옮긴 파생값일 뿐이다.
    포탑 메시의 포신 끝과는 무관하며, 무관한 것이 맞다.

■ 틸트 부호
    URDF axis 가 (0, -1, 0) 이므로 관절값 theta 는 -Y 축 둘레 회전, 즉 Ry(-theta) 이다.
    Ry(-theta) 를 (1,0,0) 에 적용하면 (cos theta, 0, sin theta) 가 되어 양의 관절값이
    포신을 위로 든다. 따라서 tilt 관절값 = 보어사이트 앙각(elevation) 과 부호가 같다.
"""

from __future__ import annotations

import math

import numpy as np

# URDF 미러 상수. xacro 를 수정하면 이 값도 함께 갱신해야 한다.
BASE_TO_PAN_M: tuple[float, float, float] = (0.0, 0.0, 0.0885)
PAN_TO_TILT_M: tuple[float, float, float] = (-0.06185, 0.0, 0.5065)
GUN_TO_CAMERA_OPTICAL_M: tuple[float, float, float] = (0.159, 0.21, 0.25)

# ── 제품 고유 기하: 소총 조준경과 총구 ───────────────────────────────────
# 조준경 광축과 총열 중심의 높이차. AR 계열 플랫탑 상부레일 기준(사용자 확인값).
SIGHT_HEIGHT_OVER_BORE_M: float = 0.066
# 조준경 광학중심에서 총구까지의 전방 거리. 시선에 평행하므로 시차 기여는 미미하다.
MUZZLE_FORWARD_OF_SCOPE_M: float = 0.45
# 조준경 광학중심 기준 총구 위치, FLU (X 전방, Y 좌측, Z 상방).
SCOPE_TO_MUZZLE_M: tuple[float, float, float] = (
    MUZZLE_FORWARD_OF_SCOPE_M,
    0.0,
    -SIGHT_HEIGHT_OVER_BORE_M,
)

# 위 제품 기하를 Gazebo 포탑 링크 좌표로 옮긴 파생값. 포탑 메시의 포신 끝이
# 아니라 조준경 기준으로 정의된다는 점이 핵심이다.
GUN_TO_MUZZLE_M: tuple[float, float, float] = (
    GUN_TO_CAMERA_OPTICAL_M[0] + SCOPE_TO_MUZZLE_M[0],
    GUN_TO_CAMERA_OPTICAL_M[1] + SCOPE_TO_MUZZLE_M[1],
    GUN_TO_CAMERA_OPTICAL_M[2] + SCOPE_TO_MUZZLE_M[2],
)


def gun_to_muzzle_from_scope_offset(
    scope_to_muzzle_m: tuple[float, float, float] = SCOPE_TO_MUZZLE_M,
    gun_to_camera_m: tuple[float, float, float] = GUN_TO_CAMERA_OPTICAL_M,
) -> tuple[float, float, float]:
    """조준경 기준 총구 오프셋을 Gazebo 포탑 링크 좌표로 옮긴다."""

    return (
        gun_to_camera_m[0] + scope_to_muzzle_m[0],
        gun_to_camera_m[1] + scope_to_muzzle_m[1],
        gun_to_camera_m[2] + scope_to_muzzle_m[2],
    )


def scope_to_muzzle_offset_in_pinhole_frame(
    scope_to_muzzle_m: tuple[float, float, float] = SCOPE_TO_MUZZLE_M,
) -> tuple[float, float, float]:
    """aiming_engine 의 ScopeConfig.mount_offset_m 에 넣을 값을 계산한다.

    ScopeConfig.mount_offset_m 의 정의는 "카메라 프레임에서 본 scope 원점 위치"이고,
    aiming_engine/coordinate_transform.py 는 이 값을 표준 핀홀 규약(X 우, Y 하,
    Z 전방)의 camera 프레임에서 뺀다. 반면 이 모듈의 오프셋은 FLU 기준이므로 축을
    다시 라벨링한다.

        right   = -left
        down    = -up
        forward =  forward

    기본 소총 기하(0.45, 0, -0.066)에서는 (0.0, 0.066, 0.45) 가 된다. 두 번째 성분
    0.066 이 곧 조준경 광축 아래로 내려간 총열 높이이며, 이것이 시차를 만드는
    유일한 수직 항이다.
    """

    fwd, left, up = scope_to_muzzle_m
    return (float(-left), float(-up), float(fwd))

TILT_MIN_RAD: float = 0.0
TILT_MAX_RAD: float = 85.0 * math.pi / 180.0

# URDF limit 항목의 velocity 값 (rad/s).
PAN_MAX_RATE_RAD_S: float = 3.0
TILT_MAX_RATE_RAD_S: float = 2.5


def rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def gun_rotation(pan: float, tilt: float) -> np.ndarray:
    """base 프레임에서 본 gun_link 의 회전행렬. R = Rz(pan) @ Ry(-tilt)."""

    return rotation_z(pan) @ rotation_y(-tilt)


def boresight_direction(pan: float, tilt: float) -> np.ndarray:
    """base 프레임 보어사이트 단위벡터. gun_rotation @ (1, 0, 0) 과 동일."""

    ct = math.cos(tilt)
    return np.array([ct * math.cos(pan), ct * math.sin(pan), math.sin(tilt)], dtype=np.float64)


def point_in_base(p_gun: np.ndarray, pan: float, tilt: float) -> np.ndarray:
    """gun_link 좌표의 점을 base 프레임으로 변환."""

    r_pan = rotation_z(pan)
    r_tilt = rotation_y(-tilt)
    p_turret = np.asarray(PAN_TO_TILT_M, dtype=np.float64) + r_tilt @ np.asarray(p_gun, dtype=np.float64)
    return r_pan @ p_turret + np.asarray(BASE_TO_PAN_M, dtype=np.float64)


def _pose_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def camera_optical_pose(pan: float, tilt: float) -> np.ndarray:
    """base <- camera_link_optical 4x4 동차변환 (FLU 회전)."""

    return _pose_matrix(
        gun_rotation(pan, tilt),
        point_in_base(np.asarray(GUN_TO_CAMERA_OPTICAL_M, dtype=np.float64), pan, tilt),
    )


def muzzle_pose(
    pan: float,
    tilt: float,
    gun_to_muzzle_m: tuple[float, float, float] = GUN_TO_MUZZLE_M,
) -> np.ndarray:
    """base <- scope(포구 원점, FLU) 4x4 동차변환."""

    return _pose_matrix(
        gun_rotation(pan, tilt),
        point_in_base(np.asarray(gun_to_muzzle_m, dtype=np.float64), pan, tilt),
    )


def muzzle_pose_from_camera_pose(
    camera_pose: np.ndarray,
    gun_to_muzzle_m: tuple[float, float, float] = GUN_TO_MUZZLE_M,
    gun_to_camera_m: tuple[float, float, float] = GUN_TO_CAMERA_OPTICAL_M,
) -> np.ndarray:
    """TF 로 얻은 카메라 광학 프레임 자세로부터 포구 프레임 자세를 유도한다.

    카메라와 포신은 gun_link 에 강체 고정이므로 회전은 동일하고, 병진만
    (포구 - 카메라) 오프셋을 gun_link 회전으로 돌려 더하면 된다.
    관절각을 몰라도 되므로 TF 만으로 동작한다.
    """

    rotation = np.asarray(camera_pose, dtype=np.float64)[:3, :3]
    offset_gun = np.asarray(gun_to_muzzle_m, dtype=np.float64) - np.asarray(gun_to_camera_m, dtype=np.float64)
    translation = np.asarray(camera_pose, dtype=np.float64)[:3, 3] + rotation @ offset_gun
    return _pose_matrix(rotation, translation)


def camera_to_muzzle_offset_in_pinhole_frame(
    gun_to_muzzle_m: tuple[float, float, float] = GUN_TO_MUZZLE_M,
    gun_to_camera_m: tuple[float, float, float] = GUN_TO_CAMERA_OPTICAL_M,
) -> tuple[float, float, float]:
    """포탑 링크 좌표 두 개에서 ScopeConfig.mount_offset_m 을 유도하는 변형.

    조준경 기준 오프셋을 이미 알고 있다면 scope_to_muzzle_offset_in_pinhole_frame
    쪽이 더 직접적이다. 이 함수는 포탑 링크 좌표만 주어졌을 때 쓴다.
    """

    fwd, left, up = (
        np.asarray(gun_to_muzzle_m, dtype=np.float64) - np.asarray(gun_to_camera_m, dtype=np.float64)
    )
    return (float(-left), float(-up), float(fwd))


def direction_to_pan_tilt(direction: np.ndarray) -> tuple[float, float]:
    """base 프레임 방향벡터를 (pan, tilt) 관절각으로 환산."""

    d = np.asarray(direction, dtype=np.float64)
    horizontal = float(math.hypot(d[0], d[1]))
    pan = float(math.atan2(d[1], d[0]))
    tilt = float(math.atan2(d[2], horizontal))
    return pan, tilt


def solve_pan_tilt_for_point(
    aim_point_base: np.ndarray,
    pan_seed: float = 0.0,
    tilt_seed: float = 0.0,
    gun_to_muzzle_m: tuple[float, float, float] = GUN_TO_MUZZLE_M,
    iterations: int = 6,
) -> tuple[float, float]:
    """포구에서 aim_point_base 를 정확히 겨누는 (pan, tilt) 역기구학.

    포구 위치 자체가 (pan, tilt) 의 함수이므로 닫힌 해가 없다. 다만 포구 오프셋
    (약 0.68 m) 이 교전 거리(수십 m) 대비 매우 작아 고정점 반복이 빠르게 수렴한다.
    반환값은 관절 한계로 클램프하지 않는다. 클램프는 호출자 책임이다.
    """

    target = np.asarray(aim_point_base, dtype=np.float64)
    pan, tilt = float(pan_seed), float(tilt_seed)
    for _ in range(max(1, iterations)):
        muzzle = point_in_base(np.asarray(gun_to_muzzle_m, dtype=np.float64), pan, tilt)
        direction = target - muzzle
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            break
        pan, tilt = direction_to_pan_tilt(direction)
    return pan, tilt


def wrap_to_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def unwrap_pan(target_pan: float, reference_pan: float) -> float:
    """turret_pan_joint 는 continuous 이므로 명령값이 +-pi 경계에서 튀지 않도록
    직전 명령 기준으로 최단 회전이 되게 펼친다."""

    return float(reference_pan + wrap_to_pi(target_pan - reference_pan))


class GoalFeedForward:
    """지향 목표각의 1차 외삽(속도 피드포워드).

    왜 필요한가: 조준해는 프레임 k 의 영상으로 계산되지만 관절 명령은 그 이후에
    적용된다. 표적이 시선각속도 w 로 움직이면 명령은 항상 w * (지연시간) 만큼
    뒤처진다. 오프라인 검증(D 시나리오)에서 이 지연은 표적 속도와 무관하게
    0.0327 ~ 0.0328 초로 측정되었고, 이는 프레임 주기 1/30 = 0.0333 초와
    사실상 일치한다. 즉 정상상태 지향오차는 솔버 오차도 서보 속도한계도 아니고
    순수한 한 프레임 지연이다.

        2 m/s  -> 각속도 3.03 deg/s  -> 지향오차 0.099 deg
        5 m/s  -> 각속도 5.15 deg/s  -> 지향오차 0.169 deg
        15 m/s -> 각속도 15.59 deg/s -> 지향오차 0.510 deg  (4단계 READY 임계 0.5도 초과)

    보정 방법: 직전 두 목표각의 차분으로 각속도를 추정해 lead_time 만큼 외삽한다.

    주의 (기본값이 0.0 인 이유): 이 보정은 목표각을 수치 미분하므로 추적 잡음을
    증폭한다. 합성 궤적(잡음 없음)에서는 순이득이지만, 실제 YOLO/광학흐름
    바운딩박스의 잔여 잡음에서는 손해일 수 있다. 1단계 표적(정지 및 저속)은
    보정 없이도 여유가 크므로 기본값을 0.0(비활성)으로 두었다. 3~4단계에서
    실제 추적 잡음으로 재평가한 뒤 켤 것을 권장한다.
    """

    def __init__(self, lead_time_s: float = 0.0, max_rate_rad_s: float = 6.0):
        self._lead_time_s = float(lead_time_s)
        self._max_rate_rad_s = float(max_rate_rad_s)
        self._prev: tuple[float, float] | None = None
        self._prev_time: float | None = None

    def reset(self) -> None:
        self._prev = None
        self._prev_time = None

    def apply(self, pan_goal: float, tilt_goal: float, now: float) -> tuple[float, float]:
        if self._lead_time_s <= 0.0:
            self._prev = (pan_goal, tilt_goal)
            self._prev_time = now
            return pan_goal, tilt_goal

        result = (pan_goal, tilt_goal)
        if self._prev is not None and self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 1e-6:
                pan_rate = wrap_to_pi(pan_goal - self._prev[0]) / dt
                tilt_rate = (tilt_goal - self._prev[1]) / dt
                # 잡음이 만든 비현실적인 각속도는 관절 속도한계로 잘라 낸다.
                pan_rate = clamp(pan_rate, -self._max_rate_rad_s, self._max_rate_rad_s)
                tilt_rate = clamp(tilt_rate, -self._max_rate_rad_s, self._max_rate_rad_s)
                result = (
                    pan_goal + pan_rate * self._lead_time_s,
                    tilt_goal + tilt_rate * self._lead_time_s,
                )
        self._prev = (pan_goal, tilt_goal)
        self._prev_time = now
        return result


def clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def rate_limit(target: float, current: float, max_rate: float, dt: float) -> float:
    """관절 속도 한계를 넘지 않도록 명령 변화량을 제한한다."""

    max_step = abs(max_rate) * max(dt, 0.0)
    delta = target - current
    if delta > max_step:
        return float(current + max_step)
    if delta < -max_step:
        return float(current - max_step)
    return float(target)


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """TF 쿼터니언(x, y, z, w) -> 3x3 회전행렬. 정규화를 먼저 수행한다."""

    q = np.array([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("영벡터 쿼터니언은 회전으로 변환할 수 없습니다")
    x, y, z, w = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
