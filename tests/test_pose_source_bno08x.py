import time

import numpy as np
from scipy.spatial.transform import Rotation

from bridge.pose_source import Bno08xPoseSource, quaternion_to_platform_rotation


def _wait_until(predicate, timeout_s=1.0, interval_s=0.01):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


# ── quaternion_to_platform_rotation (pure function) ─────────────────────────


def test_identity_quaternion_with_identity_mount_gives_identity_rotation():
    rotation = quaternion_to_platform_rotation(
        quat=(0.0, 0.0, 0.0, 1.0), mount_rotation=np.eye(3), imu_to_platform_axes=np.eye(3)
    )
    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-9)


def test_quaternion_matches_scipy_reference_rotation():
    # 90 degree rotation about Z, expressed as a quaternion.
    quat_xyzw = Rotation.from_euler("Z", 90, degrees=True).as_quat()  # (x, y, z, w)
    rotation = quaternion_to_platform_rotation(
        quat=tuple(quat_xyzw), mount_rotation=np.eye(3), imu_to_platform_axes=np.eye(3)
    )
    expected = Rotation.from_euler("Z", 90, degrees=True).as_matrix()
    np.testing.assert_allclose(rotation, expected, atol=1e-9)


def test_mount_rotation_is_applied_on_top_of_imu_reading():
    # IMU itself reports identity; a 90-degree mount calibration offset
    # about Z should show up directly in the output rotation.
    mount_rotation = Rotation.from_euler("Z", 90, degrees=True).as_matrix()
    rotation = quaternion_to_platform_rotation(
        quat=(0.0, 0.0, 0.0, 1.0), mount_rotation=mount_rotation, imu_to_platform_axes=np.eye(3)
    )
    np.testing.assert_allclose(rotation, mount_rotation, atol=1e-9)


# ── Bno08xPoseSource (background thread + injected fake IMU) ───────────────


class FakeImu:
    """Minimal stand-in for adafruit_bno08x's BNO08X_I2C: reports a fixed
    quaternion via the property the driver actually reads from."""

    def __init__(self, quat=(0.0, 0.0, 0.0, 1.0)):
        self._quat = quat

    @property
    def game_quaternion(self):
        return self._quat

    @property
    def quaternion(self):
        return self._quat


class FailingImu:
    @property
    def game_quaternion(self):
        raise RuntimeError("simulated I2C read failure")

    @property
    def quaternion(self):
        raise RuntimeError("simulated I2C read failure")


def test_pose_source_reports_valid_sample_with_configured_fixed_position():
    source = Bno08xPoseSource(imu_obj=FakeImu(), fixed_position_m=(1.0, 2.0, 3.0), read_timeout_s=1.0)
    try:
        assert _wait_until(lambda: source.read(timestamp=0.0).valid)
        sample = source.read(timestamp=0.0)
        np.testing.assert_allclose(sample.extrinsic[:3, :3], np.eye(3), atol=1e-9)
        np.testing.assert_allclose(sample.extrinsic[:3, 3], [1.0, 2.0, 3.0])
    finally:
        source.close()


def test_pose_source_invalid_before_first_sample_and_when_imu_always_fails():
    source = Bno08xPoseSource(imu_obj=FailingImu(), read_timeout_s=0.5)
    try:
        time.sleep(0.05)
        assert source.read(timestamp=0.0).valid is False
    finally:
        source.close()


def test_pose_source_marks_stale_sample_invalid_after_read_timeout():
    source = Bno08xPoseSource(imu_obj=FakeImu(), read_timeout_s=0.05)
    try:
        assert _wait_until(lambda: source.read(timestamp=0.0).valid)

        source._stop_event.set()  # stop the background thread without tearing down other state
        time.sleep(0.15)  # exceed read_timeout_s with no further updates
        assert source.read(timestamp=0.0).valid is False
    finally:
        source.close()


def test_rejects_unknown_report_type():
    import pytest

    with pytest.raises(ValueError):
        Bno08xPoseSource(imu_obj=FakeImu(), report_type="not_a_real_report_type")
