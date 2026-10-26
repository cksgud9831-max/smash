"""시뮬레이터 시연 HUD(drone_sim/smash_fcs/hud.py) 단위 테스트.

ROS 없이 돈다. hud.py 는 cv2/numpy 만 쓰므로 시뮬레이터 패키지 경로를 직접 붙여 임포트한다.
"""

import os
import sys

import numpy as np
import pytest

_SIM_PKG = os.path.join(
    os.path.dirname(__file__), os.pardir, "simulation", "ciws_turret_aerial_object_detection-main", "drone_sim"
)
sys.path.insert(0, os.path.abspath(_SIM_PKG))

from drone_sim.smash_fcs.hud import HudFrame, HudRenderer, ImpactMark, impact_mark  # noqa: E402


# ── 탄착 좌표 변환 ─────────────────────────────────────────────────────


def test_impact_mark_right_and_up_signs():
    # base_footprint: X 전방, Y 좌측, Z 위. 사수는 +X 를 본다.
    launch = np.array([0.0, 0.0, 0.0])
    target = np.array([10.0, 0.0, 0.0])
    right_hit = impact_mark(np.array([10.0, -0.1, 0.0]), target, launch, hit=True)  # -Y = 오른쪽
    up_hit = impact_mark(np.array([10.0, 0.0, 0.2]), target, launch, hit=False)

    assert right_hit.right_m == pytest.approx(0.1)
    assert right_hit.up_m == pytest.approx(0.0, abs=1e-12)
    assert up_hit.up_m == pytest.approx(0.2)
    assert up_hit.right_m == pytest.approx(0.0, abs=1e-12)
    assert right_hit.hit and not up_hit.hit


def test_impact_mark_ignores_along_line_of_sight_component():
    launch = np.array([0.0, 0.0, 0.0])
    target = np.array([30.0, 0.0, 10.0])
    los = target / np.linalg.norm(target)
    projectile = target + 0.5 * los  # 시선 방향으로만 0.5 m 어긋남
    mark = impact_mark(projectile, target, launch, hit=True)
    assert mark.right_m == pytest.approx(0.0, abs=1e-9)
    assert mark.up_m == pytest.approx(0.0, abs=1e-9)


def test_impact_mark_preserves_perpendicular_distance():
    rng = np.random.default_rng(0)
    launch = np.array([0.0, 0.0, 0.5])
    target = np.array([33.0, 0.0, 12.5])
    los = (target - launch) / np.linalg.norm(target - launch)
    for _ in range(20):
        d = rng.normal(size=3) * 0.3
        d -= (d @ los) * los  # 시선에 수직인 성분만
        mark = impact_mark(target + d, target, launch, hit=True)
        assert np.hypot(mark.right_m, mark.up_m) == pytest.approx(np.linalg.norm(d), rel=1e-9)


def test_impact_mark_degenerate_inputs_return_none():
    p = np.array([1.0, 2.0, 3.0])
    assert impact_mark(p, p, p, hit=True) is None  # 발사점 == 표적
    # 시선이 연직이면 좌우 축을 정의할 수 없다
    assert impact_mark(np.zeros(3), np.array([0.0, 0.0, 10.0]), np.zeros(3), hit=True) is None


# ── 렌더러 ─────────────────────────────────────────────────────────────


def _frame(**kw):
    base = dict(now=1.0, principal_point=(320, 320), state="SEARCH")
    base.update(kw)
    return HudFrame(**base)


@pytest.mark.parametrize("state", ["SEARCH", "TRACK", "ALIGN", "READY"])
def test_render_every_state_keeps_shape_and_does_not_touch_input(state):
    img = np.full((640, 640, 3), 180, np.uint8)
    before = img.copy()
    out = HudRenderer().render(
        img,
        _frame(
            state=state,
            state_detail="12px" if state == "ALIGN" else "",
            bbox_xyxy=(300, 310, 340, 325),
            lock_age_s=0.1,
            lead_pixel=(330, 318),
            range_m=35.3,
            range_src="LASER",
            tof_ms=40.0,
            hit_probability=0.9,
            fire_enabled=True,
            shots=3,
            hits=2,
            misses=1,
            impacts=[ImpactMark(0.05, -0.02, True), ImpactMark(0.4, 0.3, False)],
            last_hit_age_s=0.1,
            kill_banner_age_s=0.5,
        ),
    )
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(img, before)  # 원본 영상은 그대로
    assert not np.array_equal(out, before)  # 무언가 그려졌다


def test_render_other_resolution_and_debug_mode():
    img = np.full((320, 320, 3), 200, np.uint8)
    out = HudRenderer().render(
        img, _frame(principal_point=(160, 160), debug=True, debug_lines=["STATE SEARCH", "RNG --"], target_down=True)
    )
    assert out.shape == (320, 320, 3)


def test_vignette_darkens_edges_but_not_center():
    img = np.full((640, 640, 3), 200, np.uint8)
    out = HudRenderer().render(img, _frame())
    # 중심 근처(레티클 선을 피한 대각 위치)는 원본 밝기 유지, 모서리는 어두워진다
    assert out[250, 250].mean() == pytest.approx(200, abs=2)
    assert out[5, 630].mean() < 80


def test_impact_panel_only_when_fire_control_enabled():
    img = np.full((640, 640, 3), 200, np.uint8)
    r = HudRenderer()
    off = r.render(img, _frame(fire_enabled=False))
    on = r.render(img, _frame(fire_enabled=True))
    panel = (slice(60, 180), slice(510, 620))  # 우상단 과녁 패널 영역
    assert not np.array_equal(off[panel], on[panel])


# ── 일반 관객용 요소 ───────────────────────────────────────────────────

from drone_sim.smash_fcs.hud import Engagement  # noqa: E402


def _frame(**kw):
    base = dict(now=1.0, principal_point=(320, 320), state="TRACK", fire_enabled=True)
    base.update(kw)
    return HudFrame(**base)


def test_friendly_detail_hides_pixel_numbers_and_jargon_unless_debug():
    d = HudRenderer._friendly_detail
    assert d(_frame(state="ALIGN", state_detail="14px")) == ""
    assert d(_frame(state="ALIGN", state_detail="READY BLOCK: MOTION FIT")) == "SETTLING SPEED"
    assert d(_frame(state="ALIGN", state_detail="READY BLOCK: REL HOLD")) == "LOW CONFIDENCE"
    assert d(_frame(state="TRACK", state_detail="NO RANGE")) == "RANGING"
    assert d(_frame(state="ALIGN", state_detail="14px", debug=True)) == "14px"


def test_new_overlays_render_without_error_and_change_the_image():
    bg = np.full((640, 640, 3), 100, dtype=np.uint8)
    plain = HudRenderer().render(bg, _frame())
    rich = HudRenderer().render(bg, _frame(
        state="ALIGN", bbox_xyxy=(380, 250, 420, 280), lead_pixel=(350, 300), align_tol_px=20.0,
        reliability="STABLE", quality=0.8, target_kmh=3.0,
        trail=[(300 + i, 280) for i in range(10)],
        engagement=Engagement(1, 5.0, 4, 2), engagement_age_s=1.0,
    ))
    assert rich.shape == plain.shape
    assert not np.array_equal(plain, rich)


def test_engagement_card_only_while_target_down_and_fresh():
    bg = np.full((640, 640, 3), 100, dtype=np.uint8)
    eng = Engagement(1, 5.0, 4, 2)
    shown = HudRenderer().render(bg, _frame(engagement=eng, engagement_age_s=1.0, target_down=True))
    expired = HudRenderer().render(bg, _frame(engagement=eng, engagement_age_s=9.0, target_down=True))
    respawned = HudRenderer().render(bg, _frame(engagement=eng, engagement_age_s=1.0, target_down=False))
    none = HudRenderer().render(bg, _frame(target_down=True))
    assert not np.array_equal(shown, none)
    assert np.array_equal(expired, none)
    assert np.array_equal(respawned, HudRenderer().render(bg, _frame()))


def test_quality_gauge_hidden_while_searching():
    bg = np.full((640, 640, 3), 100, dtype=np.uint8)
    searching = HudRenderer().render(bg, _frame(state="SEARCH", reliability="LOST"))
    baseline = HudRenderer().render(bg, _frame(state="SEARCH"))
    assert np.array_equal(searching, baseline)


def test_audience_view_hides_estimated_speed_and_rounds_distance_unless_debug():
    """외부 관객용 기본 화면: 흔들리는 추정 속도는 숨기고 거리는 정수 미터. 상세 모드(H)에서는 원래 값."""

    r = HudRenderer
    tracking = dict(state="ALIGN", target_kmh=23.8, range_m=34.03, bbox_xyxy=(380, 250, 420, 280))
    assert r._speed_label(_frame(**tracking)) is None
    assert r._speed_label(_frame(debug=True, **tracking)) == "23.8 km/h"
    assert r._speed_label(_frame(debug=True, **{**tracking, "state": "SEARCH"})) is None
    assert r._speed_label(_frame(debug=True, **{**tracking, "target_kmh": 0.3})) == "HOVERING"
    assert r._range_label(_frame(**tracking)) == "RNG 34 m"
    assert r._range_label(_frame(debug=True, **tracking)) == "RNG  34.0 m"


def test_render_draws_speed_text_only_in_debug(monkeypatch):
    drawn = []
    original = HudRenderer._text

    def spy(img, text, *a, **k):
        drawn.append(text)
        return original(img, text, *a, **k)

    monkeypatch.setattr(HudRenderer, "_text", staticmethod(spy))
    bg = np.full((640, 640, 3), 100, dtype=np.uint8)
    kw = dict(state="ALIGN", bbox_xyxy=(380, 250, 420, 280), target_kmh=12.3, range_m=35.2, range_src="OPTICAL")
    HudRenderer().render(bg, _frame(**kw))
    assert not any("km/h" in t for t in drawn) and "RNG 35 m" in drawn and "OPTICAL" not in drawn
    drawn.clear()
    HudRenderer().render(bg, _frame(debug=True, debug_lines=["x"], **kw))
    assert any("12.3 km/h" in t for t in drawn) and any(t.startswith("RNG  35.2") for t in drawn) and "OPTICAL" in drawn
