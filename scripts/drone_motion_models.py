"""SMASH 로드맵 단계별 드론 기동 모델 모음.

06_gazebo_interactive_shooter.py(인터랙티브 시뮬레이터)와 별도의 정밀도 검증 스크립트가
같은 궤적 정의를 공유하도록 이 모듈로 분리했다. 대드론 사격 통제 시뮬레이터의 로드맵
2단계(저속 등속도 직선 비행) 검증을 위한 것이며, 3단계(선회 기동) 이후 새 모델을 추가할
때도 이 파일에 함수만 더 만들면 된다.

현재 구현:
    compute_drone_linear_position: 2단계용 등속 직선 왕복 비행 모델 (1~2 m/s)
"""

import math

# 2단계 목표: 저속 등속도 직선 비행 (1~2 m/s, 기초 리드각 정밀도 검증)
DRONE_RANGE_X = 35.0
DRONE_ALT_Z = 3.5
DRONE_SPEED_MPS = 1.5
DRONE_PATH_HALF_LENGTH = 14.0


def compute_drone_linear_position(t, speed_mps=DRONE_SPEED_MPS, half_length=DRONE_PATH_HALF_LENGTH,
                                   range_x=DRONE_RANGE_X, alt_z=DRONE_ALT_Z):
    """등속 직선 왕복 비행 위치/속도 계산.

    x(사거리)와 z(고도)는 고정하고 y축만 왕복 구간에서 순수 등속(구간별 상수 속도)으로
    이동시킨다. 방향이 바뀌는 두 끝점을 제외하면 언제나 정확히 일정한 속도 벡터를
    유지하므로, 사인파 진동과 달리 실제 등속 직선 비행 리드각 검증에 쓸 수 있다.

    반환값: (x, y, z, vy) — vy는 현재 순간의 y축 속도(m/s), 부호로 방향을 나타낸다.
    """
    leg_duration = (2.0 * half_length) / speed_mps
    phase = math.fmod(t, 2.0 * leg_duration)
    if phase < leg_duration:
        y = -half_length + speed_mps * phase
        vy = speed_mps
    else:
        y = half_length - speed_mps * (phase - leg_duration)
        vy = -speed_mps
    return range_x, y, alt_z, vy
