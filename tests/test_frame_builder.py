import numpy as np

from aiming_engine.aiming_manager import AimingManager
from aiming_engine.types import TrackerFrame
from bridge.config import (
    BridgeConfig,
    CameraConfig,
    DetectorConfig,
    LaserConfig,
    PoseConfig,
    TrackerConfig,
)
from bridge.frame_builder import TrackerFrameBuilder
from bridge.optical_flow_tracker import FollowerResult
from bridge.pose_source import MockPoseSource, PoseSample, PoseSource
from bridge.range_sensor import MockRangeSensor


def make_config() -> BridgeConfig:
    return BridgeConfig(
        camera=CameraConfig(fx=500.0, fy=500.0, cx=320.0, cy=240.0),
        laser=LaserConfig(mount_offset_m=(0.0, 0.0, 0.0), sensor="mock", mock_fixed_distance_m=250.0),
        pose=PoseConfig(source="mock", mock_extrinsic="identity"),
        tracker=TrackerConfig(model_path="unused.pt", device="cpu", conf_thres=0.25),
        detector=DetectorConfig(
            backend="onnx",
            onnx_weights_path="unused.onnx",
            imgsz=(416, 416),
            conf_thres=0.25,
            iou_thres=0.45,
            trt_engine_path=None,
        ),
    )


class _FakeTracker:
    """Stand-in for OpticalFlowTracker: returns a scripted sequence of
    FollowerResult (or None) instead of running ultralytics + optical flow
    on real frames."""

    def __init__(self, results: list):
        self._results = list(results)
        self._calls = 0
        self.close_calls = 0

    def update(self, frame_bgr):
        result = self._results[min(self._calls, len(self._results) - 1)]
        self._calls += 1
        return result

    def close(self):
        self.close_calls += 1


def make_result(cx=320.0, cy=240.0, w=40.0, h=40.0, score=0.9, is_recovery_event=False) -> FollowerResult:
    return FollowerResult(bbox=(cx - w / 2.0, cy - h / 2.0, w, h), score=score, is_recovery_event=is_recovery_event)


def make_builder(tracker) -> TrackerFrameBuilder:
    config = make_config()
    return TrackerFrameBuilder(config, MockRangeSensor(fixed_distance_m=250.0), MockPoseSource(np.eye(4)), tracker=tracker)


def make_frame_bgr() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_no_track_returns_none():
    builder = make_builder(_FakeTracker([None]))
    assert builder.build(make_frame_bgr(), timestamp=0.0) is None


def test_tracker_config_defaults_to_legacy_backend():
    tracker_config = TrackerConfig(model_path="unused.pt", device="cpu", conf_thres=0.25)
    assert tracker_config.backend == "legacy"


def test_injected_tracker_takes_priority_over_final_v20_backend():
    config = make_config()
    config = BridgeConfig(
        camera=config.camera,
        laser=config.laser,
        pose=config.pose,
        tracker=TrackerConfig("unused.engine", "0", 0.25, backend="final_v20"),
        detector=config.detector,
    )
    fake = _FakeTracker([None])
    builder = TrackerFrameBuilder(
        config, MockRangeSensor(fixed_distance_m=250.0), MockPoseSource(np.eye(4)), tracker=fake
    )
    assert builder._tracker is fake


def test_legacy_backend_constructs_optical_flow_tracker(monkeypatch):
    captured = {}
    fake = _FakeTracker([None])

    def constructor(**kwargs):
        captured.update(kwargs)
        return fake

    monkeypatch.setattr("bridge.frame_builder.OpticalFlowTracker", constructor)
    config = make_config()
    builder = TrackerFrameBuilder(
        config, MockRangeSensor(fixed_distance_m=250.0), MockPoseSource(np.eye(4))
    )
    assert builder._tracker is fake
    assert captured == {"model_path": "unused.pt", "device": "cpu", "conf_thres": 0.25}


def test_final_v20_backend_constructs_final_tracker(monkeypatch):
    captured = {}
    fake = _FakeTracker([None])

    def constructor(**kwargs):
        captured.update(kwargs)
        return fake

    monkeypatch.setattr("bridge.frame_builder.FinalTracker", constructor)
    config = make_config()
    config = BridgeConfig(
        camera=config.camera,
        laser=config.laser,
        pose=config.pose,
        tracker=TrackerConfig("model.engine", "0", 0.99, backend="final_v20"),
        detector=config.detector,
    )
    builder = TrackerFrameBuilder(
        config, MockRangeSensor(fixed_distance_m=250.0), MockPoseSource(np.eye(4))
    )
    assert builder._tracker is fake
    assert captured == {"engine_path": "model.engine", "device": "0"}


def test_unknown_backend_raises_value_error():
    config = make_config()
    config = BridgeConfig(
        camera=config.camera,
        laser=config.laser,
        pose=config.pose,
        tracker=TrackerConfig("unused", "cpu", 0.25, backend="something_invalid"),
        detector=config.detector,
    )
    try:
        TrackerFrameBuilder(
            config, MockRangeSensor(fixed_distance_m=250.0), MockPoseSource(np.eye(4))
        )
    except ValueError as exc:
        assert str(exc) == "Unsupported tracker backend: something_invalid"
    else:
        raise AssertionError("unknown tracker backend was accepted")


def test_builder_close_forwards_once_and_is_idempotent():
    fake = _FakeTracker([None])
    builder = make_builder(fake)
    builder.close()
    builder.close()
    assert fake.close_calls == 1


def test_track_produces_valid_trackerframe():
    builder = make_builder(_FakeTracker([make_result()]))
    result = builder.build(make_frame_bgr(), timestamp=0.0)

    assert isinstance(result, TrackerFrame)
    assert result.laser_range_m is not None and result.laser_range_m > 0.0
    assert result.follower_state in ("stable", "unstable")
    assert 0.0 <= result.tracking_confidence <= 1.0
    assert result.camera_intrinsics.shape == (3, 3)
    assert result.camera_extrinsics.shape == (4, 4)


def test_recovery_event_maps_to_unstable_and_zero_confidence():
    builder = make_builder(_FakeTracker([make_result(is_recovery_event=True, score=0.9)]))
    result = builder.build(make_frame_bgr(), timestamp=0.0)

    assert result.follower_state == "unstable"
    assert result.tracking_confidence == 0.0


class _AlwaysInvalidPoseSource(PoseSource):
    """PoseSource stand-in that always reports an invalid sample, e.g. an
    IMU that's stale/disconnected -- build() must skip the frame rather
    than hand aiming_engine a stale extrinsic."""

    def read(self, timestamp: float) -> PoseSample:
        return PoseSample(extrinsic=np.eye(4), timestamp=timestamp, valid=False)


def test_invalid_pose_sample_skips_the_frame():
    config = make_config()
    builder = TrackerFrameBuilder(
        config, MockRangeSensor(fixed_distance_m=250.0), _AlwaysInvalidPoseSource(), tracker=_FakeTracker([make_result()])
    )
    assert builder.build(make_frame_bgr(), timestamp=0.0) is None


def test_aiming_manager_accepts_builder_output_without_raising():
    builder = make_builder(_FakeTracker([make_result()]))
    result = builder.build(make_frame_bgr(), timestamp=0.0)
    assert result is not None

    manager = AimingManager()  # default config/aiming_engine.yaml
    output = manager.update(result)
    assert 0.0 <= output.hit_probability <= 1.0
