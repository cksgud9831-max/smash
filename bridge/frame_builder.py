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

        # Zero-order hold caches for sensor dropout robustness (~100ms tolerance)
        self._last_valid_range: Optional[float] = None
        self._last_valid_range_ts: float = -1.0
        self._last_valid_extrinsic: Optional[np.ndarray] = None
        self._last_valid_pose_ts: float = -1.0
        self._sensor_hold_timeout_s: float = 0.1  # hold valid sample for up to 3 frames @ 30fps

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

        # 1. Range sample retrieval with Zero-Order Hold (dropout tolerance)
        range_sample = self._range_sensor.read(timestamp)
        effective_range_m: Optional[float] = None
        if range_sample is not None and range_sample.valid:
            self._last_valid_range = range_sample.distance_m
            self._last_valid_range_ts = timestamp
            effective_range_m = range_sample.distance_m
        elif self._last_valid_range is not None and (timestamp - self._last_valid_range_ts) <= self._sensor_hold_timeout_s:
            effective_range_m = self._last_valid_range

        if effective_range_m is None:
            return None

        try:
            laser_range_m = self._laser_aligner.correct(center_px, effective_range_m, self._intrinsics)
        except ValueError:
            return None

        # 2. Pose sample retrieval with Zero-Order Hold (dropout tolerance)
        pose_sample = self._pose_source.read(timestamp)
        effective_extrinsic: Optional[np.ndarray] = None
        if pose_sample is not None and pose_sample.valid:
            self._last_valid_extrinsic = pose_sample.extrinsic
            self._last_valid_pose_ts = timestamp
            effective_extrinsic = pose_sample.extrinsic
        elif self._last_valid_extrinsic is not None and (timestamp - self._last_valid_pose_ts) <= self._sensor_hold_timeout_s:
            effective_extrinsic = self._last_valid_extrinsic

        if effective_extrinsic is None:
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
            camera_extrinsics=effective_extrinsic,
            laser_range_m=laser_range_m,
        )

