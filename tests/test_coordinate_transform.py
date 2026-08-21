import numpy as np
import pytest

from aiming_engine.config import ScopeConfig
from aiming_engine.coordinate_transform import CoordinateTransform
from aiming_engine.types import TrackerFrame, Vector3


def make_transform(offset=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0)) -> CoordinateTransform:
    return CoordinateTransform(ScopeConfig(mount_offset_m=offset, mount_rotation_rad=rotation))


def test_image_to_camera_norm_equals_range():
    ct = make_transform()
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    p = ct.image_to_camera((320.0, 240.0), range_m=10.0, intrinsics=K)
    assert p.norm() == pytest.approx(10.0, abs=1e-9)
    # principal point -> ray is straight down +Z
    assert p.x == pytest.approx(0.0, abs=1e-9)
    assert p.y == pytest.approx(0.0, abs=1e-9)
    assert p.z == pytest.approx(10.0, abs=1e-9)


def test_camera_scope_round_trip_with_offset_and_rotation():
    ct = make_transform(offset=(0.1, -0.2, 0.05), rotation=(0.05, -0.1, 0.2))
    p_cam = Vector3(1.0, 2.0, 3.0)
    p_scope = ct.camera_to_scope(p_cam)
    p_back = ct.scope_to_camera(p_scope)
    assert p_back.to_array() == pytest.approx(p_cam.to_array(), abs=1e-9)


def test_scope_world_round_trip_with_nontrivial_extrinsic():
    ct = make_transform()
    # 90 degree yaw + translation extrinsic
    theta = np.pi / 2
    r = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    t = np.array([5.0, -3.0, 1.5])
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = r
    extrinsic[:3, 3] = t

    p_scope = Vector3(2.0, 0.0, 0.0)
    p_world = ct.scope_to_world(p_scope, extrinsic)
    p_back = ct.world_to_scope(p_world, extrinsic)
    assert p_back.to_array() == pytest.approx(p_scope.to_array(), abs=1e-9)
    # sanity: 90deg yaw rotates +X(scope) -> +Y(world) direction, then translate
    assert p_world.to_array() == pytest.approx(np.array([5.0, -1.0, 1.5]), abs=1e-9)


def test_direction_transform_ignores_translation():
    ct = make_transform()
    theta = np.pi / 2
    r = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = r
    extrinsic[:3, 3] = [100.0, -50.0, 20.0]  # large translation must not affect direction

    d_scope = np.array([1.0, 0.0, 0.0])
    d_world = ct.scope_to_world_direction(d_scope, extrinsic)
    assert d_world == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)

    d_back = ct.world_to_scope_direction(d_world, extrinsic)
    assert d_back == pytest.approx(d_scope, abs=1e-9)


def test_camera_to_scope_axis_convention():
    # Camera is pinhole (X right, Y down, Z forward); scope is FLU
    # (X forward/boresight, Y left, Z up). camera_to_scope must remap
    # axes, not just apply the (nominally near-identity) mount calibration.
    ct = make_transform()
    assert ct.camera_to_scope(Vector3(0.0, 0.0, 1.0)).to_array() == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
    assert ct.camera_to_scope(Vector3(1.0, 0.0, 0.0)).to_array() == pytest.approx([0.0, -1.0, 0.0], abs=1e-9)
    assert ct.camera_to_scope(Vector3(0.0, 1.0, 0.0)).to_array() == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)


def test_image_to_world_full_chain_identity_setup():
    ct = make_transform()
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    frame = TrackerFrame(
        timestamp=0.0,
        bbox=(300.0, 220.0, 340.0, 260.0),
        center_px=(320.0, 240.0),
        velocity_px=(0.0, 0.0),
        tracking_confidence=0.9,
        follower_state="stable",
        camera_intrinsics=K,
        camera_extrinsics=np.eye(4),
        laser_range_m=50.0,
    )
    p_world = ct.image_to_world(frame)
    # principal point = straight down the boresight -> scope/world +X (forward), with
    # identity extrinsic and zero mount offset/rotation
    assert p_world.to_array() == pytest.approx(np.array([50.0, 0.0, 0.0]), abs=1e-9)


def test_image_to_world_requires_laser_range():
    ct = make_transform()
    K = np.eye(3)
    frame = TrackerFrame(
        timestamp=0.0,
        bbox=(0.0, 0.0, 1.0, 1.0),
        center_px=(0.0, 0.0),
        velocity_px=(0.0, 0.0),
        tracking_confidence=0.9,
        follower_state="stable",
        camera_intrinsics=K,
        camera_extrinsics=np.eye(4),
        laser_range_m=None,
    )
    with pytest.raises(ValueError):
        ct.image_to_world(frame)
