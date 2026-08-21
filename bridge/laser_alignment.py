"""Corrects for the physical offset between the laser rangefinder's
boresight and the camera's optical center.

The laser is rigidly mounted with a small known translational offset `o`
(camera frame) from the camera's optical center, its own boresight assumed
parallel to the camera's optical (Z) axis. It measures raw slant range
`R_laser` along that boresight, i.e. along camera-frame direction (0,0,1).
The true 3D target point in camera frame is therefore:

    P = o + R_laser * (0, 0, 1)

`aiming_engine.coordinate_transform.CoordinateTransform.image_to_camera`
computes `p_cam = s * d_hat`, where `d_hat` is the normalized camera pixel
ray through the tracked bbox center -- that contract is fixed (aiming_engine
is not modified here), so the best single scalar `s` to hand it is the
projection of P onto d_hat, which minimizes ||s*d_hat - P||:

    laser_range_m := P . d_hat = o . d_hat + R_laser * d_hat.z

This removes the boresight/pixel-ray misalignment term entirely. It does
NOT remove the lateral component of `o` (perpendicular to the camera axis)
-- that's real parallax with no single-scalar fix, roughly constant in
meters / shrinking as offset/range in bearing. It is still strictly better
than passing R_laser straight through, which carries the full offset error
uncorrected.
"""

from __future__ import annotations

import numpy as np


class LaserAligner:
    def __init__(self, laser_offset_cam_m: np.ndarray):
        self._offset = np.asarray(laser_offset_cam_m, dtype=np.float64)

    def correct(
        self,
        pixel: tuple[float, float],
        raw_range_m: float,
        intrinsics: np.ndarray,
    ) -> float:
        u, v = pixel
        ray_h = np.array([u, v, 1.0], dtype=np.float64)
        direction_cam = np.linalg.inv(intrinsics) @ ray_h
        direction_cam = direction_cam / np.linalg.norm(direction_cam)

        target_point = self._offset + raw_range_m * np.array([0.0, 0.0, 1.0])
        corrected_range = float(np.dot(target_point, direction_cam))

        if not np.isfinite(corrected_range) or corrected_range <= 0.0:
            raise ValueError(
                f"laser range correction produced a non-physical range "
                f"({corrected_range}) for pixel={pixel}, raw_range_m={raw_range_m}"
            )
        return corrected_range
