"""Maps OpticalFlowTracker's native signals onto the tracking_confidence /
follower_state fields aiming_engine.TrackerFrame expects. OpticalFlowTracker
has no native concept of either -- both are derived heuristics, split out
here so they're independently unit-testable.
"""

from __future__ import annotations

import numpy as np

# Must match aiming_engine.aiming_manager._STABLE_FOLLOWER_STATE.
STABLE = "stable"
UNSTABLE = "unstable"


def compute_follower_state(is_recovery_event: bool) -> str:
    """is_recovery_event = a non-periodic YOLO redetect just fired this
    frame (the follower actually lost the target and had to recover, not a
    routine periodic correction) -- OpticalFlowTracker's analogue of
    SmartTracker's old crisis_mode signal."""

    return UNSTABLE if is_recovery_event else STABLE


def compute_tracking_confidence(last_detection_score: float, is_recovery_event: bool) -> float:
    """last_detection_score is the most recent YOLO confidence, persisted
    across pure-optical-flow frames (the follower doesn't re-run YOLO every
    frame, so there's nothing fresher to report most frames). Zeroed out on
    a recovery-event frame, mirroring the old crisis-mode gate."""

    if is_recovery_event:
        return 0.0
    return float(np.clip(last_detection_score, 0.0, 1.0))
