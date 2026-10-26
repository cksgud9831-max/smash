import numpy as np
import pytest

from scripts.scope_caption import CaptionRenderer, caption_for


def _cap(state, rel="STABLE", event=None, age=None):
    return caption_for({"fcs_state": state, "reliability": rel, "event": event, "event_age_s": age})


@pytest.mark.parametrize("state,rel,expected_tone", [
    ("SEARCH", "LOST", "idle"),
    ("TRACK(NO RANGE)", "HOLD", "track"),
    ("TRACK(NO TF)", "STABLE", "track"),
    ("TRACK(REL HOLD)", "HOLD", "warn"),
    ("TRACK(REL LOST)", "LOST", "warn"),
    ("TRACK", "STABLE", "track"),
    ("ALIGN(12px)", "STABLE", "align"),
    ("READY BLOCK: MOTION FIT", "STABLE", "align"),
    ("READY BLOCK: REL HOLD", "HOLD", "warn"),
    ("READY", "STABLE", "ready"),
])
def test_every_fcs_state_has_korean_caption(state, rel, expected_tone):
    text, tone = _cap(state, rel)
    assert tone == expected_tone
    assert any("가" <= ch <= "힣" for ch in text)


def test_lost_reliability_says_target_lost():
    text, _ = _cap("TRACK(REL LOST)", "LOST")
    assert "놓침" in text


def test_kill_event_overrides_state_until_respawn():
    assert _cap("READY", event="destroyed", age=1.0)[1] == "kill"
    # 격추 뒤 추적이 풀리는 것은 정상 동작이라 "놓침" 경고를 띄우지 않는다
    assert _cap("TRACK(REL LOST)", "LOST", event="destroyed", age=9.0) == (
        "표적 격추 완료 — 다음 표적 대기 중", "idle")


def test_respawn_caption_only_while_searching():
    assert "새 표적" in _cap("SEARCH", event="respawn", age=1.0)[0]
    assert "새 표적" not in _cap("READY", event="respawn", age=1.0)[0]


def test_no_status_means_no_caption():
    assert caption_for({}) is None
    assert _cap("UNKNOWN_STATE") is None


def test_renderer_only_touches_bottom_strip():
    r = CaptionRenderer()
    if not r.available:
        pytest.skip("한글 글꼴 없음")
    frame = np.full((640, 640, 3), 120, dtype=np.uint8)
    r.draw(frame, ("발사 준비 완료 — 자동 발사", "ready"))
    assert (frame[:560] == 120).all()
    assert not (frame[-30:] == 120).all()


def test_renderer_ignores_none_caption():
    frame = np.full((100, 100, 3), 7, dtype=np.uint8)
    CaptionRenderer().draw(frame, None)
    assert (frame == 7).all()


def test_letterbox_keeps_aspect_and_pads():
    from scripts.scope_caption import letterbox

    frame = np.full((100, 100, 3), 200, dtype=np.uint8)
    out = letterbox(frame, 320, 180)
    assert out.shape == (180, 320, 3)
    assert (out[:, :10] == 0).all() and (out[:, -10:] == 0).all()  # 좌우 검은 띠
    assert (out[90, 160] == 200).all()
    assert letterbox(frame, 100, 100) is frame
    assert letterbox(frame, 0, 0) is frame


def test_ready_caption_tells_the_user_to_pull_the_trigger_in_manual_mode():
    auto = caption_for({"fcs_state": "READY", "reliability": "STABLE", "trigger_mode": "auto"})
    manual = caption_for({"fcs_state": "READY", "reliability": "STABLE", "trigger_mode": "manual"})
    assert auto == ("발사 준비 완료 — 자동 발사", "ready")
    assert manual[1] == "ready" and "클릭" in manual[0] and "스페이스" in manual[0]
    # 모드 정보가 없던 예전 /status 도 자동으로 취급한다
    assert caption_for({"fcs_state": "READY", "reliability": "STABLE"}) == auto
