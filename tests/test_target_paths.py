"""drone_sim/target_paths.py (이동 표적 궤적) 시험. ROS 불필요."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation",
                                "ciws_turret_aerial_object_detection-main", "drone_sim", "drone_sim"))

from target_paths import Figure8Path, cross_offset  # noqa: E402

KMH30 = 30.0 / 3.6


def _speeds(path, dt=0.002, n=5000):
    pts = [path.offset(i * dt) for i in range(n)]
    return [math.dist(a, b) / dt for a, b in zip(pts, pts[1:])]


def test_figure8_speed_is_constant_30kmh():
    path = Figure8Path(KMH30, 5.0, 2.5)
    v = _speeds(path)
    assert min(v) == pytest.approx(KMH30, rel=0.01)
    assert max(v) == pytest.approx(KMH30, rel=0.01)


def test_figure8_starts_at_center_and_stays_in_bounds():
    path = Figure8Path(KMH30, 5.0, 2.5)
    assert path.offset(0.0) == pytest.approx((0.0, 0.0), abs=1e-9)
    pts = [path.offset(i * 0.01) for i in range(int(path.period / 0.01) + 1)]
    assert max(abs(x) for x, _ in pts) == pytest.approx(5.0, abs=0.01)
    assert max(abs(y) for _, y in pts) == pytest.approx(2.5, abs=0.01)
    # 한 주기에 중심을 두 번(출발점 포함 세 번) 지난다
    assert path.offset(path.period / 2) == pytest.approx((0.0, 0.0), abs=0.02)
    assert path.offset(path.period) == pytest.approx((0.0, 0.0), abs=0.02)


def test_figure8_peak_accel_matches_formula():
    path = Figure8Path(KMH30, 5.0, 2.5)
    dt = 0.001
    t = path.period / 4  # 좌우 끝점
    p0, p1, p2 = (path.offset(t + k * dt) for k in (-1, 0, 1))
    acc = math.hypot(*((a - 2 * b + c) / dt ** 2 for a, b, c in zip(p0, p1, p2)))
    assert acc == pytest.approx(path.max_lateral_accel(), rel=0.02)


def test_zero_speed_is_stationary():
    assert Figure8Path(0.0, 5.0, 2.5).offset(3.0) == (0.0, 0.0)
    assert cross_offset(3.0, 0.0, 6.0) == (0.0, 0.0)


def test_cross_offset_reverses_at_ends():
    assert cross_offset(0.5, 5.0, 3.0) == pytest.approx((2.5, 1.0))
    assert cross_offset(0.7, 5.0, 3.0) == pytest.approx((2.5, -1.0))
    assert cross_offset(1.2, 5.0, 3.0)[0] == pytest.approx(0.0)


def test_straight_is_constant_speed_and_repeats():
    from target_paths import straight_offset
    assert straight_offset(0.0, KMH30, 15.0) == pytest.approx(0.0)
    assert straight_offset(1.0, KMH30, 15.0) == pytest.approx(KMH30)
    assert straight_offset(15.0 / KMH30 + 0.1, KMH30, 15.0) == pytest.approx(-15.0 + 0.1 * KMH30)
    period = 30.0 / KMH30
    assert straight_offset(period + 0.5, KMH30, 15.0) == pytest.approx(straight_offset(0.5, KMH30, 15.0))
    assert straight_offset(3.0, 0.0, 15.0) == 0.0
