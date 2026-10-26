"""operating_envelope.py — 화각별 포착 가능 거리(운용 범위).

탐지기가 표적을 최초로 잡으려면 표적이 화면에서 일정 크기 이상이어야 한다. 한계는 거리 자체가
아니라 화면 속 표적 폭(px)이며, 그래서 화각(초점거리)에 비례한다.

실측 근거(2026-10-14, 광각 0.4 rad, fx 1578 px, 정지 호버링 표적, 경사거리 기준):
  A. 격추 후 재출현(포 큐 선회로 표적이 화면 중심, 구름 배경):
      약 21 m 이하     3~6 초에 포착(10.8·16·21.5 m 표본 14건)
      약 27.7 m        3.6~15.7 초(표본 4건)
      약 28.9 m        9~99 초(불안정, 표본 4건)
      약 32 m 이상     240 초 안에 포착 0건, 35 m 는 150 초 동안 못 잡음
  B. 최초 시작(정지 호버, 포는 탐색 자세):
      12·18·22·25·28 m 모두 7~24 초에 잠금(25 m 방위 6도 포함, 이동 표적 3 km/h 도 25 m 에서 10·23 초)
      그러나 중심 (24.7, 0, 9.4) = 26.4 m 의 이동 표적 시작은 240 초 이상 3회 잠금 실패
      (화면에서 약 22 px 로 보였지만 밝은 구름 위라 대비가 낮았다)
  C. 화각 0.2 rad 에서는 경사거리 35 m 를 16 초 만에 포착
  E. 같은 위치 (20.5, 0, 7.7)=21.9 m, 이동 표적 시작을 화각만 바꿔 반복:
      0.4 rad(약 24 px) 7회 모두 100 초 안에 잠금 실패, 0.3 rad(약 32 px) 3회 5~12 초,
      0.2 rad(약 48 px) 3회 5 초. 25 m 도 0.3 rad(약 28 px) 12 초, 0.2 rad 5 초.
표적 폭 약 0.336 m 로 환산한 화면 폭: 19 m(0.4 rad) 약 28 px, 22 m 약 24 px, 25 m 약 21 px, 35 m 약 15 px.

결론: 한계는 거리 하나로 정해지지 않고 표적 화면 폭과 배경 대비에 함께 달린다. 24 px 전후에서는
위치를 0.1 m 만 옮겨도 잠금이 되기도 안 되기도 해(D) 확률적이다. 시험한 모든 조건에서 일관되게
잡힌 것은 화면 폭 약 28 px 이상이므로 MIN_TARGET_PX 를 28 로 둔다. 표본이 작고 고도 20도·구름
배경 한 조건이라 이 값은 추정이며, 맑은 하늘이나 이동이 큰 표적에서는 달라질 수 있다.
ROS 의존성이 없는 순수 파이썬이다.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

# 시뮬레이션 드론 모델의 화면상 유효 폭(m). 35.3 m 에서 15 px 실측값(fx 1578)에서 역산한 값이다.
TARGET_WIDTH_M = 0.336
# 최초 포착을 안정적으로 기대하는 화면상 최소 표적 폭(px)
MIN_TARGET_PX = 28.0
# 터릿 카메라 기준점(world, m). smash_fcs.yaml 의 turret_world_translation_m 과 같아야 한다.
DEFAULT_ORIGIN = (0.0, 0.0, 0.05)

# 이동 방식별 중심에서 벗어나는 폭: (x 방향, y 방향, z 방향) 이 쓰는 인자
_MOTIONS_LATERAL = ("straight", "cross", "maneuver")


def focal_px(hfov_rad: float, width_px: float = 640.0) -> float:
    """수평 화각과 해상도에서 초점거리(px)."""

    return (float(width_px) / 2.0) / math.tan(float(hfov_rad) / 2.0)


def max_range_m(hfov_rad: float, width_px: float = 640.0,
                min_target_px: float = MIN_TARGET_PX, target_width_m: float = TARGET_WIDTH_M) -> float:
    """표적이 min_target_px 이상으로 보이는 최대 경사거리(m)."""

    return focal_px(hfov_rad, width_px) * float(target_width_m) / float(min_target_px)


def slant_range_m(position: Sequence[float], origin: Sequence[float] = DEFAULT_ORIGIN) -> float:
    return math.dist([float(v) for v in position], [float(v) for v in origin])


def worst_case_range_m(
    center: Sequence[float],
    motion: str = "none",
    half_length_m: float = 0.0,
    half_height_m: float = 0.0,
    origin: Sequence[float] = DEFAULT_ORIGIN,
) -> float:
    """이동 표적이 중심에서 가장 멀어졌을 때의 경사거리(m)."""

    x, y, z = (float(v) for v in center)
    ox, oy, oz = (float(v) for v in origin)
    dx = abs(x - ox)
    dy = abs(y - oy)
    dz = abs(z - oz)
    if motion == "approach":
        dx += abs(half_length_m)
    elif motion in _MOTIONS_LATERAL:
        dy += abs(half_length_m)
        if motion == "maneuver":
            dz += abs(half_height_m)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def clamp_to_envelope(
    center: Sequence[float],
    limit_m: float,
    motion: str = "none",
    half_length_m: float = 0.0,
    half_height_m: float = 0.0,
    origin: Sequence[float] = DEFAULT_ORIGIN,
) -> tuple[tuple[float, float, float], bool]:
    """표적 중심을 원점 방향으로 당겨 이동 폭까지 포함한 최악 거리가 limit_m 이하가 되게 한다.

    반환: (조정된 중심, 조정 여부). 이미 범위 안이면 그대로 돌려준다. 이동 폭만으로 이미 범위를
    넘으면 중심을 원점 쪽 끝까지(비율 0.2) 당기며, 그 경우 호출자가 경고해야 한다.
    """

    c = tuple(float(v) for v in center)
    if limit_m <= 0.0 or worst_case_range_m(c, motion, half_length_m, half_height_m, origin) <= limit_m:
        return c, False
    o = tuple(float(v) for v in origin)

    def at(scale: float) -> tuple[float, float, float]:
        return tuple(o[i] + (c[i] - o[i]) * scale for i in range(3))  # type: ignore[return-value]

    lo, hi = 0.2, 1.0
    if worst_case_range_m(at(lo), motion, half_length_m, half_height_m, origin) > limit_m:
        return at(lo), True
    for _ in range(40):  # 단조 증가이므로 이분법
        mid = (lo + hi) / 2.0
        if worst_case_range_m(at(mid), motion, half_length_m, half_height_m, origin) <= limit_m:
            lo = mid
        else:
            hi = mid
    return at(lo), True


def clamp_flat_positions(
    flat: Sequence[float], limit_m: float, origin: Sequence[float] = DEFAULT_ORIGIN
) -> tuple[list[float], list[int]]:
    """[x,y,z, x,y,z, ...] 목록(재출현 위치)을 범위 안으로 당긴다. 반환: (목록, 조정된 항목 번호)."""

    out: list[float] = []
    changed: list[int] = []
    values = [float(v) for v in flat]
    for i in range(0, len(values) - 2, 3):
        pos, moved = clamp_to_envelope(values[i:i + 3], limit_m, origin=origin)
        out.extend(pos)
        if moved:
            changed.append(i // 3)
    return out, changed


def parse_flat_positions(text: str) -> Optional[list[float]]:
    """'[1.0, 2.0, 3.0]' 형태의 launch 인자 문자열을 숫자 목록으로 바꾼다. 실패하면 None."""

    try:
        values = [float(v) for v in text.strip().strip("[]").split(",") if v.strip()]
    except ValueError:
        return None
    return values
