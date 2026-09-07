"""Turns (camera frame, sensor readings) into aiming_engine.TrackerFrame.

Owns OpticalFlowTracker plus the RangeSensor/PoseSource/LaserAligner it
needs to fill in the fields the tracker alone can't provide
(tracking_confidence, follower_state, camera_extrinsics, laser_range_m).
One TrackerFrameBuilder per engagement; call build() once per camera frame.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from aiming_engine.types import TrackerFrame

from .config import BridgeConfig
from .confidence import compute_follower_state, compute_tracking_confidence
from .laser_alignment import LaserAligner
from .optical_flow_tracker import OpticalFlowTracker
from .pose_source import PoseSource
from .range_sensor import RangeSensor


class TrackerFrameBuilder:
    def __init__(
        self,
        config: BridgeConfig,
        range_sensor: RangeSensor,
        pose_source: PoseSource,
        tracker: Optional[OpticalFlowTracker] = None,
    ):
        """`tracker` is a test/DI hook: pass an already-constructed
        tracker-like object (only needs an `update(frame_bgr) ->
        Optional[FollowerResult]` method, see optical_flow_tracker.py) to
        bypass loading a real ultralytics model. Normal callers just omit
        it and let this build the real OpticalFlowTracker from config."""

        self._config = config
        self._tracker = tracker if tracker is not None else OpticalFlowTracker(
            model_path=config.tracker.model_path,
            device=config.tracker.device,
            conf_thres=config.tracker.conf_thres,
        )
        self._range_sensor = range_sensor
        self._pose_source = pose_source
        self._laser_aligner = LaserAligner(np.array(config.laser.mount_offset_m, dtype=np.float64))
        self._intrinsics = config.camera.intrinsics_matrix()

    def build(self, frame_bgr: np.ndarray, timestamp: float) -> Optional[TrackerFrame]:
        """Returns None (no output this frame) if there's nothing to track,
        or no valid range sample, or no valid pose sample -- callers should
        just skip calling AimingManager.update() for that frame rather than
        treat this as an error; there's no fire command downstream to gate
        on it anyway."""

        result = self._tracker.update(frame_bgr)
        if result is None:
            return None

        x, y, w, h = result.bbox
        bbox = (float(x), float(y), float(x + w), float(y + h))
        center_px = (float(x + w / 2.0), float(y + h / 2.0))

        tracking_confidence = compute_tracking_confidence(result.score, result.is_recovery_event)
        follower_state = compute_follower_state(result.is_recovery_event)

        range_sample = self._range_sensor.read(timestamp)
        if range_sample is None or not range_sample.valid:
            return None

        try:
            laser_range_m = self._laser_aligner.correct(center_px, range_sample.distance_m, self._intrinsics)
        except ValueError:
            return None

        pose_sample = self._pose_source.read(timestamp)
        if not pose_sample.valid:
            return None

        return TrackerFrame(
            timestamp=timestamp,
            bbox=bbox,
            center_px=center_px,
            # OpticalFlowTracker doesn't estimate bbox velocity the way
            # SmartTracker's Kalman filter did; aiming_engine derives its own
            # world-frame velocity from position history anyway
            # (target_state.py) and never reads this field (see
            # known_issues.md) so a placeholder here changes nothing.
            velocity_px=(0.0, 0.0),
            tracking_confidence=tracking_confidence,
            follower_state=follower_state,
            camera_intrinsics=self._intrinsics,
            camera_extrinsics=pose_sample.extrinsic,
            laser_range_m=laser_range_m,
        )
