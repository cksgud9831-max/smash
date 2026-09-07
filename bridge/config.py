"""YAML configuration loading for the bridge layer.

Mirrors aiming_engine/config.py's pattern (frozen dataclasses per section,
single load_* entry point, nothing else reads YAML directly) but is a
separate file/concern from aiming_engine's config: this one is hardware
calibration (camera, laser, pose source, tracker tuning), not aim-assist
math parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "bridge.yaml"


@dataclass(frozen=True)
class CameraConfig:
    fx: float
    fy: float
    cx: float
    cy: float

    def intrinsics_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class LaserConfig:
    mount_offset_m: tuple[float, float, float]
    sensor: str  # mock | serial
    mock_fixed_distance_m: float
    # `serial` backend (Benewake TF02-Pro over UART) -- unused unless sensor == "serial".
    # min_signal_strength/read_timeout_s are placeholders pending real-hardware validation
    # (see bridge/range_sensor.py:Tf02ProRangeSensor docstring).
    serial_port: Optional[str] = None
    serial_baud_rate: int = 115200
    serial_min_signal_strength: int = 0
    serial_read_timeout_s: float = 0.3


@dataclass(frozen=True)
class PoseConfig:
    source: str  # mock | bno08x
    mock_extrinsic: str  # "identity" for now; extend when a real pose source lands
    # `bno08x` backend (BNO085/BNO086 9-DoF IMU over I2C) -- unused unless source == "bno08x".
    # game_rotation_vector chosen as the default report type because the IMU is rigidly
    # mounted to a metal firearm body, which is prone to distorting a magnetometer-based
    # heading (see bridge/pose_source.py:Bno08xPoseSource docstring).
    bno08x_i2c_bus: int = 1
    bno08x_i2c_address: int = 0x4A
    bno08x_report_type: str = "game_rotation_vector"  # game_rotation_vector | rotation_vector
    bno08x_mount_rotation_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bno08x_fixed_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bno08x_read_timeout_s: float = 0.2


@dataclass(frozen=True)
class TrackerConfig:
    """bridge.optical_flow_tracker.OpticalFlowTracker's deployment-specific
    knobs. Its algorithm/GA-tuned constants (ROI_MARGIN, CENTER_SHIFT_THRES,
    etc.) are NOT here -- they're validated, hardcoded constants in that
    module, not meant to be re-tuned per deployment."""

    model_path: str
    device: str  # "0" (GPU index, as a string) or "cpu"
    conf_thres: float


@dataclass(frozen=True)
class DetectorConfig:
    """bridge.detector.Detector (ONNX/TensorRT). Currently unused by the
    default pipeline -- bridge.optical_flow_tracker.OpticalFlowTracker does
    its own internal YOLO calls via ultralytics instead (see its module
    docstring for why). Kept around/still loaded in case a future backend
    wants a standalone Detector again; onnx_weights_path currently points
    at a file that no longer exists in this checkout (bridge/weights/ was
    removed alongside bridge/smart_tracking/), so constructing an
    OnnxDetector from this config will fail until that's restored."""

    backend: str
    onnx_weights_path: str
    imgsz: tuple[int, int]
    conf_thres: float
    iou_thres: float
    trt_engine_path: Optional[str]


@dataclass(frozen=True)
class BridgeConfig:
    camera: CameraConfig
    laser: LaserConfig
    pose: PoseConfig
    tracker: TrackerConfig
    detector: DetectorConfig


def _vec3(d: dict[str, Any]) -> tuple[float, float, float]:
    return (float(d["x"]), float(d["y"]), float(d["z"]))


def _rot3(d: dict[str, Any]) -> tuple[float, float, float]:
    return (float(d["roll"]), float(d["pitch"]), float(d["yaw"]))


def load_bridge_config(path: str | Path | None = None) -> BridgeConfig:
    """Load and validate the bridge YAML config into typed dataclasses."""

    resolved = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(resolved, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return BridgeConfig(
        camera=CameraConfig(
            fx=float(raw["camera"]["intrinsics"]["fx"]),
            fy=float(raw["camera"]["intrinsics"]["fy"]),
            cx=float(raw["camera"]["intrinsics"]["cx"]),
            cy=float(raw["camera"]["intrinsics"]["cy"]),
        ),
        laser=LaserConfig(
            mount_offset_m=_vec3(raw["laser"]["mount_offset_m"]),
            sensor=str(raw["laser"]["sensor"]),
            mock_fixed_distance_m=float(raw["laser"]["mock"]["fixed_distance_m"]),
            serial_port=(raw["laser"].get("serial") or {}).get("port"),
            serial_baud_rate=int((raw["laser"].get("serial") or {}).get("baud_rate", 115200)),
            serial_min_signal_strength=int((raw["laser"].get("serial") or {}).get("min_signal_strength", 0)),
            serial_read_timeout_s=float((raw["laser"].get("serial") or {}).get("read_timeout_s", 0.3)),
        ),
        pose=PoseConfig(
            source=str(raw["pose"]["source"]),
            mock_extrinsic=str(raw["pose"]["mock"]["extrinsic"]),
            bno08x_i2c_bus=int((raw["pose"].get("bno08x") or {}).get("i2c_bus", 1)),
            bno08x_i2c_address=int((raw["pose"].get("bno08x") or {}).get("i2c_address", 0x4A)),
            bno08x_report_type=str((raw["pose"].get("bno08x") or {}).get("report_type", "game_rotation_vector")),
            bno08x_mount_rotation_rad=_rot3(
                (raw["pose"].get("bno08x") or {}).get("mount_rotation_rad", {"roll": 0.0, "pitch": 0.0, "yaw": 0.0})
            ),
            bno08x_fixed_position_m=_vec3(
                (raw["pose"].get("bno08x") or {}).get("fixed_position_m", {"x": 0.0, "y": 0.0, "z": 0.0})
            ),
            bno08x_read_timeout_s=float((raw["pose"].get("bno08x") or {}).get("read_timeout_s", 0.2)),
        ),
        tracker=TrackerConfig(
            model_path=str(raw["tracker"]["model_path"]),
            device=str(raw["tracker"]["device"]),
            conf_thres=float(raw["tracker"]["conf_thres"]),
        ),
        detector=DetectorConfig(
            backend=str(raw["detector"]["backend"]),
            onnx_weights_path=str(raw["detector"]["onnx"]["weights_path"]),
            imgsz=(int(raw["detector"]["onnx"]["imgsz"][0]), int(raw["detector"]["onnx"]["imgsz"][1])),
            conf_thres=float(raw["detector"]["onnx"]["conf_thres"]),
            iou_thres=float(raw["detector"]["onnx"]["iou_thres"]),
            trt_engine_path=raw["detector"]["trt"]["engine_path"],
        ),
    )
