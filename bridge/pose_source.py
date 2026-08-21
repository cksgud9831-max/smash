"""Scope-platform world-pose abstraction.

`PoseSource` is a swappable strategy, same shape as range_sensor.py's
RangeSensor.

`Bno08xPoseSource` is the real-hardware implementation, targeting a
BNO085/BNO086 9-DoF IMU (BNO08x family) rigidly mounted to the scope body.
This is a *handheld/shoulder-mounted scope*, not a motorized gimbal: the
IMU's own orientation reading *is* the platform orientation, continuously
updated every frame as the operator aims -- there is no fixed-platform
shortcut and no separate gimbal encoder to fuse in.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .config import BridgeConfig


@dataclass
class PoseSample:
    extrinsic: np.ndarray  # 4x4, World<-Platform (camera/scope platform pose in World frame)
    timestamp: float
    valid: bool = True  # default True keeps MockPoseSource / existing callers unchanged


class PoseSource(ABC):
    @abstractmethod
    def read(self, timestamp: float) -> PoseSample:
        ...

    def close(self) -> None:
        """Release any hardware resources (I2C bus, background thread).
        No-op by default; real hardware implementations override this.
        Callers should always call this on shutdown regardless of which
        backend is configured."""


class MockPoseSource(PoseSource):
    """Returns a fixed configured extrinsic every frame. Stand-in for
    tests/demos that don't need real hardware."""

    def __init__(self, extrinsic: np.ndarray):
        self._extrinsic = extrinsic

    def read(self, timestamp: float) -> PoseSample:
        return PoseSample(extrinsic=self._extrinsic, timestamp=timestamp)


# ── BNO08x (BNO085 / BNO086) ─────────────────────────────────────────────────
#
# Chip does its own sensor fusion (CEVA SH-2 firmware on an onboard
# Cortex-M0+) and reports orientation directly as a quaternion -- no AHRS
# filter needs to be implemented on our side.
#
# Two report types are available:
#   - "rotation_vector": accel+gyro+magnetometer fusion, absolute heading,
#     no drift -- but the magnetometer is vulnerable to distortion from
#     nearby ferrous mass / motors, and this IMU is bolted directly to a
#     metal firearm body, which is exactly that situation.
#   - "game_rotation_vector": accel+gyro fusion only (no magnetometer),
#     immune to magnetic distortion, but has no absolute heading reference
#     and will drift slowly over time.
# Default is "game_rotation_vector" for that reason (config: pose.bno08x.report_type).
#
# `_IMU_TO_PLATFORM_AXES` is a placeholder identity matrix. The real value
# depends on which way the IMU board is physically glued/screwed onto the
# scope body and can only be determined empirically once hardware is in
# hand (e.g. rotate the assembly to a few known orientations -- level,
# nose-up, rolled 90 degrees -- and see which quaternion axis moves to
# match). `mount_rotation_rad` in PoseConfig is the fine calibration
# offset layered on top of that fixed remap, mirroring
# aiming_engine.config.ScopeConfig.mount_rotation_rad.


def _euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Aerospace ZYX convention, matching aiming_engine.coordinate_transform's
    convention for scope mount calibration -- kept as a local copy rather
    than importing aiming_engine internals, since bridge only depends on
    aiming_engine's public `types` module by design."""

    return Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()


def quaternion_to_platform_rotation(
    quat: Sequence[float],
    mount_rotation: np.ndarray,
    imu_to_platform_axes: np.ndarray,
) -> np.ndarray:
    """quat = (i, j, k, real) as reported by adafruit_bno08x's
    `.quaternion` / `.game_quaternion` properties -> 3x3 rotation matrix
    in Platform frame (the same frame aiming_engine expects
    camera_extrinsics[:3, :3] to be in)."""

    i, j, k, real = quat
    r_imu = Rotation.from_quat([i, j, k, real]).as_matrix()
    return mount_rotation @ imu_to_platform_axes @ r_imu


class Bno08xSensorLike(Protocol):
    """Minimal surface of adafruit_bno08x's BNO08X_I2C this driver depends
    on -- lets tests inject a fake without needing the real library/hardware."""

    def enable_feature(self, feature_id: int) -> None: ...
    @property
    def quaternion(self) -> tuple[float, float, float, float]: ...
    @property
    def game_quaternion(self) -> tuple[float, float, float, float]: ...


class Bno08xPoseSource(PoseSource):
    """BNO085/BNO086 9-DoF IMU, rigidly mounted to the scope, read over I2C.

    Position is treated as fixed (`fixed_position_m`) -- valid for a
    handheld/shoulder-mounted scope where the operator's own translation
    over the few seconds of one engagement is negligible next to how much
    the aim direction itself moves. This assumption breaks down if the
    operator walks a meaningful distance while tracking a target; there is
    currently no position-tracking sensor in this pipeline to correct for
    that.

    Orientation, by contrast, is read fresh every frame via a background
    polling thread (same non-blocking cache pattern as
    Tf02ProRangeSensor) -- this is a real-time requirement, not an
    optional optimization, because the scope's aim direction changes
    continuously as the operator aims.
    """

    _IMU_TO_PLATFORM_AXES = np.eye(3)  # TODO: replace once IMU mount orientation is measured (see module docstring)

    def __init__(
        self,
        i2c_bus: int = 1,
        i2c_address: int = 0x4A,
        report_type: str = "game_rotation_vector",
        mount_rotation_rad: tuple[float, float, float] = (0.0, 0.0, 0.0),
        fixed_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        read_timeout_s: float = 0.2,
        imu_obj: Optional[Bno08xSensorLike] = None,
    ):
        """`imu_obj` is a test/DI hook: pass an already-initialized
        BNO08x-like object (see `Bno08xSensorLike`) to bypass opening a
        real I2C bus. Normal callers just pass `i2c_bus`/`i2c_address`."""

        if report_type not in ("game_rotation_vector", "rotation_vector"):
            raise ValueError(f"Unknown pose.bno08x.report_type: {report_type!r}")
        self._report_type = report_type
        self._mount_rotation = _euler_to_rotation_matrix(*mount_rotation_rad)
        self._position = np.array(fixed_position_m, dtype=np.float64)
        self._read_timeout_s = read_timeout_s

        if imu_obj is not None:
            self._imu: Bno08xSensorLike = imu_obj
        else:
            # Lazy import: board/busio/adafruit_bno08x require Adafruit Blinka
            # and real I2C hardware, neither present off-device.
            import board
            import busio
            from adafruit_bno08x import BNO_REPORT_GAME_ROTATION_VECTOR, BNO_REPORT_ROTATION_VECTOR
            from adafruit_bno08x.i2c import BNO08X_I2C

            i2c = busio.I2C(board.SCL, board.SDA)
            self._imu = BNO08X_I2C(i2c, address=i2c_address)
            feature = (
                BNO_REPORT_GAME_ROTATION_VECTOR
                if report_type == "game_rotation_vector"
                else BNO_REPORT_ROTATION_VECTOR
            )
            self._imu.enable_feature(feature)

        self._lock = threading.Lock()
        self._latest: Optional[PoseSample] = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _read_quaternion(self) -> tuple[float, float, float, float]:
        if self._report_type == "game_rotation_vector":
            return self._imu.game_quaternion
        return self._imu.quaternion

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                quat = self._read_quaternion()
            except Exception:
                # Transient I2C hiccup -- leave the cached sample in place;
                # read() will mark it invalid once it goes stale (read_timeout_s).
                time.sleep(0.01)
                continue

            rotation = quaternion_to_platform_rotation(quat, self._mount_rotation, self._IMU_TO_PLATFORM_AXES)
            extrinsic = np.eye(4, dtype=np.float64)
            extrinsic[:3, :3] = rotation
            extrinsic[:3, 3] = self._position

            with self._lock:
                self._latest = PoseSample(extrinsic=extrinsic, timestamp=time.time(), valid=True)

    def read(self, timestamp: float) -> PoseSample:
        with self._lock:
            sample = self._latest
        if sample is None:
            return PoseSample(extrinsic=np.eye(4, dtype=np.float64), timestamp=timestamp, valid=False)
        if time.time() - sample.timestamp > self._read_timeout_s:
            return PoseSample(extrinsic=sample.extrinsic, timestamp=sample.timestamp, valid=False)
        return sample

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)


def build_pose_source(config: BridgeConfig) -> PoseSource:
    if config.pose.source == "mock":
        if config.pose.mock_extrinsic == "identity":
            extrinsic = np.eye(4, dtype=np.float64)
        else:
            raise ValueError(f"Unknown pose.mock.extrinsic: {config.pose.mock_extrinsic!r}")
        return MockPoseSource(extrinsic)
    if config.pose.source == "bno08x":
        return Bno08xPoseSource(
            i2c_bus=config.pose.bno08x_i2c_bus,
            i2c_address=config.pose.bno08x_i2c_address,
            report_type=config.pose.bno08x_report_type,
            mount_rotation_rad=config.pose.bno08x_mount_rotation_rad,
            fixed_position_m=config.pose.bno08x_fixed_position_m,
            read_timeout_s=config.pose.bno08x_read_timeout_s,
        )
    raise ValueError(f"Unknown pose.source: {config.pose.source!r}")
