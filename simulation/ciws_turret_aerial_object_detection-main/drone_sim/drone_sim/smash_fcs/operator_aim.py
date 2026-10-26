"""ROS-independent pointing math for the operator emulation (operator_follow.py)."""

from __future__ import annotations

import math

import numpy as np

# 이 거리(m)보다 크게 한 번에 움직이면 속도가 아니라 순간 이동(재출현·끝점 되돌아옴)으로 본다.
TELEPORT_M = 2.0


def world_to_base(world, translation, yaw) -> np.ndarray:
    """smash_fcs_node._world_to_base 와 같은 변환. base = Rz(yaw)^T (world - translation)."""

    rel = np.asarray(world, dtype=np.float64) - np.asarray(translation, dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1], rel[2]], dtype=np.float64)


def shift_left(base_point, meters: float) -> np.ndarray:
    """base 프레임 점을 시선에 수직인 수평 왼쪽(+y 방향 회전)으로 meters 만큼 옮긴다.

    turret_command 의 cue 는 turret_kinematics.solve_pan_tilt_for_point 로 풀고, 그 기본 포구 위치(GUN_TO_MUZZLE_M)는
    카메라 좌우 0.21 m 를 가정한다. 실제 URDF·TF 의 카메라는 좌우 0 이라 그대로 쓰면 카메라 축이 표적에서 0.21 m
    오른쪽으로 빗나간다. 겨냥점을 같은 만큼(0.21 m) 왼쪽으로 옮겨 상쇄한다.
    (2026-10-18: 처음에는 화면 오차를 bbox 왼쪽 위 모서리 기준으로 재서 0.47 m 로 잘못 정했다. 0.21 m 로 고치자
     빗나간 좌우 편향이 +0.22 m 에서 0 으로 사라지고 명중이 24 % 에서 92 % 가 되었다.)
    """

    p = np.asarray(base_point, dtype=np.float64).copy()
    r = math.hypot(p[0], p[1])
    if r < 1e-9:
        return p
    p[0], p[1] = p[0] - p[1] / r * meters, p[1] + p[0] / r * meters
    return p


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def lead_offset(lead_dir, bbox_dir):
    """FCS 리드 방향이 FCS 가 본 표적(bbox 중심) 방향에서 벗어난 각도 (d_pan, d_tilt) [rad].

    둘 다 같은 프레임(base)의 방향 벡터라 카메라 자세와 무관하다. 조준선을 리드점에 두려면 표적을 향한
    방향에 이 각도를 더하면 된다. 입력이 올바르지 않으면 None.
    """

    try:
        lx, ly, lz = (float(v) for v in lead_dir)
        bx, by, bz = (float(v) for v in bbox_dir)
    except (TypeError, ValueError):
        return None
    vals = (lx, ly, lz, bx, by, bz)
    if not all(math.isfinite(v) for v in vals):
        return None
    if math.hypot(lx, ly) < 1e-9 or math.hypot(bx, by) < 1e-9:
        return None
    d_pan = _wrap(math.atan2(ly, lx) - math.atan2(by, bx))
    d_tilt = math.atan2(lz, math.hypot(lx, ly)) - math.atan2(bz, math.hypot(bx, by))
    return d_pan, d_tilt


def rotate_aim(point_base, d_pan: float, d_tilt: float) -> np.ndarray:
    """base 원점에서 본 점의 방위·고각에 (d_pan, d_tilt) 를 더해 같은 거리로 다시 놓는다."""

    p = np.asarray(point_base, dtype=np.float64)
    r = float(np.linalg.norm(p))
    h = math.hypot(p[0], p[1])
    if r < 1e-9 or h < 1e-9:
        return p.copy()
    pan = math.atan2(p[1], p[0]) + d_pan
    tilt = math.atan2(p[2], h) + d_tilt
    return np.array([r * math.cos(tilt) * math.cos(pan), r * math.cos(tilt) * math.sin(pan), r * math.sin(tilt)])


class LeadPredictor:
    """표적 실측 위치로 속도를 EMA 로 구해 lead_s 만큼 앞선 위치를 낸다."""

    def __init__(self, smoothing: float = 0.6, teleport_m: float = TELEPORT_M) -> None:
        self._smoothing = min(0.95, max(0.0, float(smoothing)))
        self._teleport = float(teleport_m)
        self.reset()

    def reset(self) -> None:
        self.pos: np.ndarray | None = None
        self.time: float | None = None
        self.vel = np.zeros(3)

    def update(self, pos, now: float) -> None:
        pos = np.asarray(pos, dtype=np.float64)
        if self.pos is not None and self.time is not None and now > self.time:
            step = pos - self.pos
            if float(np.linalg.norm(step)) > self._teleport:
                self.vel = np.zeros(3)
            else:
                self.vel = self._smoothing * self.vel + (1.0 - self._smoothing) * step / (now - self.time)
        self.pos, self.time = pos, now

    def aim_point(self, lead_s: float):
        if self.pos is None:
            return None
        return self.pos + self.vel * max(0.0, float(lead_s))
