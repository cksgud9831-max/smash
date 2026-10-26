import numpy as np

from scripts.scope_monitor import StreamHealth, compose_view
from scripts.scope_panel import PanelRenderer, StatusHistory, phase_of, range_text


def _st(state, rel="STABLE", event=None, age=None):
    return {"fcs_state": state, "reliability": rel, "event": event, "event_age_s": age,
            "motion": "straight", "kmh": 10.0, "state": "MOVE", "range_m": 35.0}


def _feed(history, state, seconds, t, rel="STABLE", step=0.25):
    for _ in range(int(seconds / step)):
        history.update(_st(state, rel), t)
        t += step
    return t


def test_phase_follows_caption_tone():
    assert phase_of(_st("SEARCH", "LOST")) == "idle"
    assert phase_of(_st("ALIGN(30px)")) == "align"
    assert phase_of(_st("READY")) == "ready"
    assert phase_of({}) == "idle"


def test_events_are_logged_in_order_and_flicker_is_ignored():
    h = StatusHistory()
    t = _feed(h, "SEARCH", 2, 0.0, "LOST")
    t = _feed(h, "TRACK", 1, t)
    t = _feed(h, "ALIGN(40px)", 1, t)
    t = _feed(h, "READY", 1, t)
    t = _feed(h, "ALIGN(40px)", 0.25, t)  # 한 표본짜리 깜빡임
    t = _feed(h, "READY", 1, t)
    texts = [e[2] for e in h.events]
    assert texts == ["표적 포착", "조준선 정렬", "발사 준비 완료"]
    assert h.ready_count == 1


def test_kill_counted_once_per_event():
    h = StatusHistory()
    t = 0.0
    for age in (0.0, 0.25, 0.5, 0.75):
        h.update(_st("READY", event="destroyed", age=age), t)
        t += 0.25
    assert h.kills == 1
    h.update(_st("SEARCH", "LOST", event="destroyed", age=0.0), t)  # 새 격추: 나이가 다시 0
    assert h.kills == 2


def test_kills_follow_hud_count_when_viewer_starts_late():
    # 뷰어를 늦게 켜면 이벤트로는 0 이지만 HUD 는 이미 11 이다. hud_kills 가 오면 그 값을 쓴다.
    h = StatusHistory()
    h.update({**_st("READY"), "hud_kills": 11}, 0.0)
    assert h.kills == 11
    h.update({**_st("READY"), "hud_kills": 12}, 0.25)
    assert h.kills == 12
    h.update({**_st("READY"), "hud_kills": 0}, 0.5)  # HUD 통계 초기화(reset_stats)를 따라간다
    assert h.kills == 0


def test_range_text_matches_hud_and_never_shows_scenario_value_as_measured():
    assert range_text({**_st("READY"), "hud_range_m": 31.4}) == "31 m"  # HUD 는 정수 m
    assert range_text({**_st("READY"), "hud_range_m": None}) == "--"  # 측정 없음(HUD 에도 RNG 없음)
    assert range_text(_st("READY")) == "설정 35 m"  # 옛 노드: 시나리오 값임을 밝힌다
    assert range_text({}) == "--"


def test_timeline_covers_window_and_merges_runs():
    h = StatusHistory()
    t = _feed(h, "SEARCH", 10, 0.0, "LOST")
    t = _feed(h, "READY", 10, t)
    segs = h.timeline(t, window_s=30.0)
    assert [s[2] for s in segs] == ["idle", "ready"]
    assert abs(segs[0][0] - (30 - 20) / 30) < 0.02 and abs(segs[-1][1] - 1.0) < 0.02


def test_audience_panel_renders_and_keeps_source_pixels():
    renderer = PanelRenderer()
    if not renderer.available:  # 글꼴이 없는 환경
        return
    h = StatusHistory()
    _feed(h, "READY", 2, 0.0)
    source = np.full((640, 640, 3), 90, np.uint8)
    health = StreamHealth()
    health.received(1.0)
    view = compose_view(source, health, 1.0, audience=(renderer, h, _st("READY")))
    assert view.shape == (682, 930, 3)
    assert np.array_equal(view[:640, :640], source)
    assert view[:, 640:].std() > 5  # 패널이 실제로 그려졌다
