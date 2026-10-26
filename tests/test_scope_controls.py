import os
import sys

import numpy as np

from scripts.scope_caption import letterbox
from scripts.scope_controls import (
    SPEED_STEPS, SpawnControls, control_layout, format_spawn, hit_test, window_to_view,
)
from scripts.scope_monitor import StreamHealth, compose_view
from scripts.scope_panel import PanelRenderer, StatusHistory

_SIM_PKG = os.path.join(
    os.path.dirname(__file__), os.pardir, "simulation", "ciws_turret_aerial_object_detection-main", "drone_sim"
)
sys.path.insert(0, os.path.abspath(_SIM_PKG))

from drone_sim.smash_fcs import spawn_command  # noqa: E402


def test_windows_and_wsl_sides_agree_on_the_command_format():
    for motion in spawn_command.SPAWN_MOTIONS:
        for kmh in (1, 2.5, 10, 30, 150):
            assert spawn_command.parse_spawn(format_spawn(motion, kmh)) == (motion, float(kmh))


def test_parse_rejects_bad_commands():
    parse = spawn_command.parse_spawn
    assert parse("spawn:hover:10") is None            # 정지는 생성 대상이 아니다
    assert parse("spawn:cross:0") is None             # 1 km/h 미만
    assert parse("spawn:cross:151") is None
    assert parse("spawn:cross:abc") is None
    assert parse("spawn:cross") is None
    assert parse("spawn:cross:10;rm") is None
    assert parse("toggle_debug") is None


def test_format_clamps_speed():
    assert format_spawn("cross", 0.2) == "spawn:cross:1"
    assert format_spawn("approach", 999) == "spawn:approach:150"


def test_layout_buttons_do_not_overlap_and_fit_panel():
    layout = control_layout(682)
    rects = [(n, r) for n, r in layout.items() if n != "top"]
    for i, (_, a) in enumerate(rects):
        assert 0 <= a[0] < a[2] <= 290 and layout["top"] <= a[1] < a[3] <= 682 - 54
        for _, b in rects[i + 1:]:
            if a is layout["speed:value"] or b is layout["speed:value"]:
                continue
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def test_hit_test_finds_each_button_and_ignores_gaps():
    layout = control_layout(682)
    for name, rect in layout.items():
        if name in ("top", "speed:value"):
            continue
        x0, y0, x1, y1 = rect
        assert hit_test(layout, (x0 + x1) / 2, (y0 + y1) / 2) == name
    assert hit_test(layout, 2, 2) is None
    x0, y0, x1, y1 = layout["speed:value"]
    assert hit_test(layout, (x0 + x1) / 2, (y0 + y1) / 2) is None  # 속도 표시칸은 버튼이 아니다


def test_window_to_view_inverts_letterbox_and_rejects_black_bars():
    view = np.zeros((682, 930, 3), np.uint8)
    for win_w, win_h in ((930, 682), (1040, 740), (1920, 1080), (800, 900)):
        lb = letterbox(view, win_w, win_h)
        assert lb.shape[:2] == (win_h, win_w)
        scale = min(win_w / 930, win_h / 682)
        nw, nh = round(930 * scale), round(682 * scale)
        ox, oy = (win_w - nw) // 2, (win_h - nh) // 2
        vx, vy = window_to_view(ox + nw / 2, oy + nh / 2, win_w, win_h, 930, 682)
        assert abs(vx - 465) < 2 and abs(vy - 341) < 2
        if ox > 2:
            assert window_to_view(ox - 2, oy + 10, win_w, win_h, 930, 682) is None
        if oy > 2:
            assert window_to_view(ox + 10, oy - 2, win_w, win_h, 930, 682) is None


def test_speed_steps_walk_the_ladder_and_stop_at_the_ends():
    c = SpawnControls()
    c.kmh = 10.0
    c.step_speed(+1)
    assert c.kmh == 15.0
    c.step_speed(-1), c.step_speed(-1)
    assert c.kmh == 9.0
    c.kmh = 12.5  # 표에 없는 값에서도 이웃 칸으로
    c.step_speed(+1)
    assert c.kmh == 15.0
    c.kmh = 12.5
    c.step_speed(-1)
    assert c.kmh == 10.0
    c.kmh = 150.0
    c.step_speed(+1)
    assert c.kmh == SPEED_STEPS[-1]
    c.kmh = 1.0
    c.step_speed(-1)
    assert c.kmh == SPEED_STEPS[0]


def test_press_selects_and_only_spawn_returns_a_command():
    c = SpawnControls()
    assert c.press("motion:maneuver") is None and c.motion == "maneuver"
    assert c.press("speed:+") is None
    assert c.press("spawn") == f"spawn:maneuver:{c.kmh:g}"


def test_sync_takes_the_running_scenario_once():
    c = SpawnControls()
    c.sync({"motion": "none", "kmh": 30})        # 호버링은 생성 방식이 아니라 무시
    assert (c.motion, c.kmh) == ("cross", 10.0)
    c.sync({"motion": "approach", "kmh": 15.0})
    assert (c.motion, c.kmh) == ("approach", 15.0)
    c.select_motion("straight")
    c.sync({"motion": "cross", "kmh": 5.0})      # 이미 맞췄으니 사용자의 선택을 덮지 않는다
    assert c.motion == "straight"


def test_panel_draws_controls_and_old_three_tuple_still_works():
    renderer = PanelRenderer()
    if not renderer.available:
        return
    h = StatusHistory()
    status = {"fcs_state": "READY", "reliability": "STABLE", "motion": "cross", "kmh": 10.0,
              "state": "DOWN", "event": None, "event_age_s": None, "spawn_enabled": True}
    source = np.full((640, 640, 3), 90, np.uint8)
    health = StreamHealth()
    health.received(1.0)
    controls = SpawnControls()
    with_controls = compose_view(source, health, 1.0, audience=(renderer, h, status, controls))
    layout = control_layout(with_controls.shape[0])
    x0, y0, x1, y1 = layout["spawn"]
    spawn_area = with_controls[y0:y1, 640 + x0:640 + x1]
    assert spawn_area.std() > 5 or spawn_area.mean() > 60  # 생성 버튼이 그려졌다
    assert np.array_equal(with_controls[:640, :640], source)
    renderer2 = PanelRenderer()
    legacy = compose_view(source, health, 1.0, audience=(renderer2, h, status))
    assert legacy.shape == with_controls.shape
