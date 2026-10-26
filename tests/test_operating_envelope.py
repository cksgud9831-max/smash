"""화각별 포착 가능 거리(operating_envelope) 단위 테스트. ROS 없이 돈다."""

import math
import os
import sys

import pytest

_SIM_PKG = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, "simulation",
    "ciws_turret_aerial_object_detection-main", "drone_sim",
))
sys.path.insert(0, _SIM_PKG)

from drone_sim.smash_fcs import operating_envelope as env  # noqa: E402


def test_focal_length_matches_simulator_camera():
    # smash_fcs 노드가 받은 camera_info: 0.4 rad -> fx 1578.61, 0.2 rad -> fx 3189.33
    assert env.focal_px(0.4) == pytest.approx(1578.6, abs=0.5)
    assert env.focal_px(0.2) == pytest.approx(3189.3, abs=0.5)


def test_max_range_scales_with_zoom():
    wide, narrow = env.max_range_m(0.4), env.max_range_m(0.2)
    assert wide == pytest.approx(18.9, abs=0.3)
    assert narrow / wide == pytest.approx(env.focal_px(0.2) / env.focal_px(0.4))
    # 협각에서 실측으로 잡힌 35 m 는 범위 안이어야 한다
    assert narrow > 35.3
    # 0.3 rad 는 실측으로 21.9 m·25 m 가 잠겼다
    assert env.max_range_m(0.3) > 25.0


def test_measured_wide_angle_results_fall_on_expected_side_of_limit():
    limit = env.max_range_m(0.4)
    assert env.slant_range_m((15.0, -2.5, 5.6)) < limit     # 4~5 s 에 포착
    assert env.slant_range_m((20.5, 0.0, 7.7)) > limit      # 21.9 m: 이동 표적 시작이 7/7 잠금 실패
    assert env.slant_range_m((23.5, 0.0, 8.8)) > limit      # 25 m 부근: 같은 조건에서 성공·실패가 갈림
    assert env.slant_range_m((33.0, 0.0, 12.5)) > limit     # 포착 실패


def test_inside_envelope_is_left_untouched():
    pos, moved = env.clamp_to_envelope((10.0, 0.0, 4.2), env.max_range_m(0.4))
    assert pos == (10.0, 0.0, 4.2) and not moved


def test_outside_target_is_pulled_in_along_the_line_of_sight():
    limit = env.max_range_m(0.4)
    pos, moved = env.clamp_to_envelope((33.0, 0.0, 12.5), limit)
    assert moved
    assert env.slant_range_m(pos) == pytest.approx(limit, abs=0.05)
    # 방향(앙각·방위)은 유지된다
    assert pos[2] / pos[0] == pytest.approx(12.45 / 33.0, rel=0.02)
    assert pos[1] == 0.0


def test_motion_excursion_is_counted():
    limit = env.max_range_m(0.4)
    # 정면 접근 ±8 m: 중심이 범위 안이어도 가장 먼 끝이 넘으면 중심을 당긴다
    pos, moved = env.clamp_to_envelope((15.0, 0.0, 5.5), limit, motion="approach", half_length_m=8.0)
    assert moved
    assert env.worst_case_range_m(pos, "approach", 8.0) <= limit + 0.05
    # 정지 표적은 같은 중심이어도 그대로
    _, moved_static = env.clamp_to_envelope((15.0, 0.0, 5.5), limit)
    assert not moved_static


def test_excursion_larger_than_limit_pins_to_minimum_scale():
    pos, moved = env.clamp_to_envelope((20.0, 0.0, 7.0), 10.0, motion="straight", half_length_m=40.0)
    assert moved
    assert math.dist(pos, env.DEFAULT_ORIGIN) == pytest.approx(
        0.2 * math.dist((20.0, 0.0, 7.0), env.DEFAULT_ORIGIN), rel=0.02)


def test_flat_positions_report_which_entries_changed():
    limit = env.max_range_m(0.4)
    flat, changed = env.clamp_flat_positions([15.0, 3.0, 5.4, 33.0, 0.0, 12.5, 10.0, 0.0, 4.2], limit)
    assert changed == [1]
    assert flat[:3] == [15.0, 3.0, 5.4] and flat[6:] == [10.0, 0.0, 4.2]
    assert env.slant_range_m(flat[3:6]) <= limit + 0.05


def test_default_respawn_positions_are_all_inside_wide_angle_envelope():
    # launch 기본값. 바뀌면 이 테스트가 범위 이탈을 잡는다.
    import re
    src = open(os.path.join(_SIM_PKG, "launch", "smash_scene.launch.py"), encoding="utf-8").read()
    text = re.search(r"'respawn_positions',\s*default_value='(\[[^']*\])'", src).group(1)
    flat = env.parse_flat_positions(text)
    assert flat and len(flat) % 3 == 0
    _, changed = env.clamp_flat_positions(flat, env.max_range_m(0.4))
    assert changed == []


@pytest.mark.parametrize("text,expected", [("[1.0, 2.0, 3.0]", [1.0, 2.0, 3.0]), ("[0.0]", [0.0]),
                                           ("bad", None), ("", [])])
def test_parse_flat_positions(text, expected):
    assert env.parse_flat_positions(text) == expected
