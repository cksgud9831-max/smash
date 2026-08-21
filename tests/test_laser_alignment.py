import numpy as np
import pytest

from bridge.laser_alignment import LaserAligner

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
PRINCIPAL_POINT = (320.0, 240.0)
OFF_AXIS_PIXEL = (320.0 + 160.0, 240.0)


def test_zero_offset_at_principal_point_passes_through_unchanged():
    aligner = LaserAligner(np.zeros(3))
    corrected = aligner.correct(PRINCIPAL_POINT, raw_range_m=250.0, intrinsics=K)
    # At the principal point the camera ray is exactly (0,0,1), so with no
    # mount offset the projection is the identity.
    assert corrected == pytest.approx(250.0, abs=1e-6)


def test_zero_offset_off_axis_scales_by_ray_z_component():
    aligner = LaserAligner(np.zeros(3))
    u, v = OFF_AXIS_PIXEL
    ray = np.linalg.inv(K) @ np.array([u, v, 1.0])
    d_hat = ray / np.linalg.norm(ray)

    corrected = aligner.correct(OFF_AXIS_PIXEL, raw_range_m=250.0, intrinsics=K)
    assert corrected == pytest.approx(250.0 * d_hat[2], rel=1e-6)
    assert corrected < 250.0  # off-axis ray is not parallel to the laser boresight


def test_nonzero_offset_matches_projection_formula():
    offset = np.array([0.05, -0.02, 0.01])
    aligner = LaserAligner(offset)
    u, v = OFF_AXIS_PIXEL
    ray = np.linalg.inv(K) @ np.array([u, v, 1.0])
    d_hat = ray / np.linalg.norm(ray)

    raw_range = 100.0
    expected = float(np.dot(offset + raw_range * np.array([0.0, 0.0, 1.0]), d_hat))

    corrected = aligner.correct(OFF_AXIS_PIXEL, raw_range_m=raw_range, intrinsics=K)
    assert corrected == pytest.approx(expected, rel=1e-9)


def test_nonphysical_correction_raises():
    # Offset places the "true" point far behind the camera relative to the
    # (small) raw range, so the projection onto the pixel ray goes negative.
    offset = np.array([0.0, 0.0, -300.0])
    aligner = LaserAligner(offset)
    with pytest.raises(ValueError):
        aligner.correct(PRINCIPAL_POINT, raw_range_m=250.0, intrinsics=K)
