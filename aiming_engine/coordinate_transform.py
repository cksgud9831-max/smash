"""Image -> Camera -> Scope -> World coordinate transforms.

Conventions:
    - World frame: right-handed, Z-up, inertial. All target-motion physics
      (TargetState, LeadPrediction, HitEquation, ProjectileModel) operates
      here, never in a frame that moves with the platform.
    - Camera frame: standard pinhole convention (X right, Y down, Z forward
      along the optical axis).
    - Scope frame: forward-left-up (X forward/boresight, Y left, Z up) —
      the convention azimuth/elevation are measured in (azimuth=0,
      elevation=0 means "straight down the boresight"). This is NOT the
      same axis labeling as the camera frame, so `camera_to_scope` is the
      fixed pinhole->FLU convention rotation composed with the small
      residual mount misalignment from `ScopeConfig` (nominally identity).
    - `laser_range_m` is treated as slant range along the pixel's viewing
      ray (as reported by a laser rangefinder), not a z-depth.
    - `camera_extrinsics` (bridge PoseSource output) is the 4x4 homogeneous pose
      that locates the rigidly-mounted camera/scope platform in World
      frame; it is applied at the Scope -> World step, matching the
      Image -> Camera -> Scope -> World order fixed by the project spec.

Units are meters and radians throughout; degrees only ever appear at
external I/O boundaries (none exist in this module).
"""

from __future__ import annotations

import numpy as np

from .config import ScopeConfig
from .types import TrackerFrame, Vector3


def _euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Aerospace ZYX convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


# Fixed pinhole-camera-axes (X right, Y down, Z forward) -> scope FLU-axes
# (X forward, Y left, Z up) convention rotation. Not a calibration value —
# this is a constant axis relabeling, always composed with the (nominally
# near-identity) mount_rotation_rad calibration from ScopeConfig.
_CAMERA_TO_SCOPE_AXES = np.array(
    [
        [0.0, 0.0, 1.0],  # scope X (forward) = camera Z (forward)
        [-1.0, 0.0, 0.0],  # scope Y (left) = -camera X (right)
        [0.0, -1.0, 0.0],  # scope Z (up) = -camera Y (down)
    ]
)


class CoordinateTransform:
    """Stateless transforms between Image, Camera, Scope, and World frames."""

    def __init__(self, scope_config: ScopeConfig):
        self._mount_offset = np.array(scope_config.mount_offset_m, dtype=np.float64)
        mount_calibration = _euler_to_rotation_matrix(*scope_config.mount_rotation_rad)
        # p_scope = _camera_to_scope_rotation @ (p_cam - offset)
        self._camera_to_scope_rotation = mount_calibration @ _CAMERA_TO_SCOPE_AXES
        self._scope_to_camera_rotation = self._camera_to_scope_rotation.T  # orthogonal -> transpose is inverse

    def image_to_camera(self, pixel: tuple[float, float], range_m: float, intrinsics: np.ndarray) -> Vector3:
        """Back-project a pixel + slant range into a Camera-frame point."""

        u, v = pixel
        ray_h = np.array([u, v, 1.0], dtype=np.float64)
        direction_cam = np.linalg.inv(intrinsics) @ ray_h
        direction_cam = direction_cam / np.linalg.norm(direction_cam)
        return Vector3.from_array(direction_cam * range_m)

    def camera_to_scope(self, p_cam: Vector3) -> Vector3:
        p = p_cam.to_array() - self._mount_offset
        return Vector3.from_array(self._camera_to_scope_rotation @ p)

    def scope_to_camera(self, p_scope: Vector3) -> Vector3:
        p = self._scope_to_camera_rotation @ p_scope.to_array()
        return Vector3.from_array(p + self._mount_offset)

    def scope_to_world(self, p_scope: Vector3, extrinsic: np.ndarray) -> Vector3:
        r = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        return Vector3.from_array(r @ p_scope.to_array() + t)

    def world_to_scope(self, p_world: Vector3, extrinsic: np.ndarray) -> Vector3:
        r = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        return Vector3.from_array(r.T @ (p_world.to_array() - t))

    def world_to_scope_direction(self, direction_world: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
        """Rotate-only variant of `world_to_scope`, for directions/angles (no translation)."""

        r = extrinsic[:3, :3]
        return r.T @ direction_world

    def scope_to_world_direction(self, direction_scope: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
        r = extrinsic[:3, :3]
        return r @ direction_scope

    def image_to_world(self, frame: TrackerFrame) -> Vector3:
        """Convenience: full forward chain for one TrackerFrame."""

        if frame.laser_range_m is None:
            raise ValueError("image_to_world requires a valid laser_range_m")
        p_cam = self.image_to_camera(frame.center_px, frame.laser_range_m, frame.camera_intrinsics)
        p_scope = self.camera_to_scope(p_cam)
        return self.scope_to_world(p_scope, frame.camera_extrinsics)
