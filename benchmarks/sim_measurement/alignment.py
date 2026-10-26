"""auto_aim_probe.py 의 정렬 계산 (ROS 없이 시험할 수 있게 분리)."""


def axis_offset_from_probe(e0: float, ep: float, step: float, k: float) -> float:
    """탐침 이동 전후 오차 크기로 한 축의 보정량(rad)을 구한다.

    오차 벡터를 (a, b) px, 탐침 이동 step rad 이 이 축 오차를 k*step px 만큼 바꾼다고 두면
    e0^2 = a^2 + b^2, ep^2 = (a + k*step)^2 + b^2 이므로 a = (ep^2 - e0^2 - k^2 step^2) / (2 k step).
    오차를 0 으로 만드는 이동량은 -a/k 이다.
    """
    return -((ep ** 2 - e0 ** 2) - k * k * step * step) / (2 * step * k * k)


def probe_is_plausible(e0: float, ep: float, step_px: float, margin_px: float = 10.0) -> bool:
    """탐침 이동 뒤 오차 변화가 물리적으로 가능한지.

    조준점을 step_px 만큼 옮기면 오차 크기는 삼각부등식에 따라 많아야 step_px 만큼 변한다.
    그보다 크게 튀면 추적기가 순간적으로 엉뚱한 것을 물었던 것이다. 2026-09-22 실측: 3 px 이던 오차가
    8 px 탐침 뒤 145 px 로 튀었고, 이를 그대로 믿어 -410 mrad 보정을 내려 정렬이 망가졌다.
    """
    return abs(ep - e0) <= step_px + margin_px


def clamp_correction(d: float, e0: float, k: float, margin_px: float = 10.0) -> float:
    """한 축 보정량(rad)을 "지금 오차 전체를 한 번에 없애는 크기" 이하로 제한한다."""
    limit = (e0 + margin_px) / k
    return max(-limit, min(limit, d))
