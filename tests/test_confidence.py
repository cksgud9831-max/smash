import pytest

from bridge.confidence import compute_follower_state, compute_tracking_confidence


def test_follower_state_mapping():
    assert compute_follower_state(is_recovery_event=False) == "stable"
    assert compute_follower_state(is_recovery_event=True) == "unstable"


def test_tracking_confidence_passes_through_score_when_not_recovering():
    conf = compute_tracking_confidence(last_detection_score=0.8, is_recovery_event=False)
    assert conf == pytest.approx(0.8)


def test_tracking_confidence_zero_during_recovery_event():
    conf = compute_tracking_confidence(last_detection_score=0.9, is_recovery_event=True)
    assert conf == pytest.approx(0.0)


def test_tracking_confidence_clips_to_valid_range():
    assert compute_tracking_confidence(last_detection_score=1.5, is_recovery_event=False) == pytest.approx(1.0)
    assert compute_tracking_confidence(last_detection_score=-0.5, is_recovery_event=False) == pytest.approx(0.0)
