"""Laser rangefinder abstraction.

`RangeSensor` is a swappable strategy (same shape as target_state.py's
MotionEstimator/build_estimator pattern) so a real serial/SDK driver can
replace MockRangeSensor later without frame_builder.py changing.

`Tf02ProRangeSensor` is the real-hardware implementation, targeting the
Benewake TF02-Pro UART LiDAR rangefinder chosen for this project.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol

from .config import BridgeConfig


@dataclass
class RangeSample:
    distance_m: float
    timestamp: float
    valid: bool


class RangeSensor(ABC):
    @abstractmethod
    def read(self, timestamp: float) -> Optional[RangeSample]:
        """Return the latest range sample, or None if no reading is available."""
        ...

    def close(self) -> None:
        """Release any hardware resources (serial port, background thread).
        No-op by default; real hardware implementations override this.
        Callers should always call this on shutdown regardless of which
        backend is configured."""


class MockRangeSensor(RangeSensor):
    """Returns a constant configured distance. Stand-in for tests/demos that
    don't need real hardware."""

    def __init__(self, fixed_distance_m: float):
        self._fixed_distance_m = fixed_distance_m

    def read(self, timestamp: float) -> Optional[RangeSample]:
        return RangeSample(distance_m=self._fixed_distance_m, timestamp=timestamp, valid=True)


# ── Benewake TF02-Pro ────────────────────────────────────────────────────────
#
# Standard Benewake TF-series 9-byte UART binary frame (shared across
# TFmini/TF02/TF02-Pro/TF-Luna; default 115200 baud, 8N1):
#
#   byte 0-1: header, always 0x59 0x59
#   byte 2-3: distance, low/high byte, centimeters
#   byte 4-5: signal strength ("AMP"), low/high byte
#   byte 6:   reliability/signal-quality byte -- exact semantics vary by
#             firmware revision; treated here only as "higher is better"
#             via the strength field, NOT parsed for meaning. Reserved for
#             a future refinement once real hardware is in hand.
#   byte 7:   reserved
#   byte 8:   checksum = low 8 bits of sum(byte0..byte7)
#
# NOTE: the exact invalid-reading sentinel (some TF-series firmware reports
# a fixed max-range distance, e.g. "no target") and the reliability byte's
# precise meaning should be confirmed against the physical unit / official
# TF02-Pro manual before field use -- this implementation instead falls
# back to a configurable minimum signal-strength gate (`min_signal_strength`)
# for the invalid/low-confidence case, which is the one detail confirmed
# consistent across the TF-series datasheets.

TF02_FRAME_LENGTH = 9
TF02_HEADER_BYTE = 0x59


class SerialLike(Protocol):
    """Minimal surface of `serial.Serial` this driver depends on -- lets
    tests inject a fake without needing pyserial or a real port."""

    def read(self, size: int = 1) -> bytes: ...
    def close(self) -> None: ...


def parse_tf02_frame(frame: bytes, min_signal_strength: int) -> Optional[RangeSample]:
    """Parse one 9-byte TF02-Pro frame. Returns None if the frame fails the
    header/checksum sanity check (caller should treat that as "drop this
    frame, keep the previous cached sample" rather than as a hard error --
    a single corrupted UART byte is expected noise, not a fault)."""

    if len(frame) != TF02_FRAME_LENGTH:
        return None
    if frame[0] != TF02_HEADER_BYTE or frame[1] != TF02_HEADER_BYTE:
        return None
    if (sum(frame[0:8]) & 0xFF) != frame[8]:
        return None

    distance_cm = frame[2] | (frame[3] << 8)
    strength = frame[4] | (frame[5] << 8)
    valid = distance_cm > 0 and strength >= min_signal_strength

    return RangeSample(distance_m=distance_cm / 100.0, timestamp=time.time(), valid=valid)


class Tf02ProRangeSensor(RangeSensor):
    """Benewake TF02-Pro laser rangefinder over UART.

    A background thread continuously reads bytes off the serial port,
    resyncs on the 0x59 0x59 header, parses complete frames, and caches the
    latest sample. `read()` itself never touches the serial port -- it just
    returns the cached sample (marked invalid if it's gone stale) -- so it's
    safe to call every frame from the main ~30fps loop without risking a
    blocking UART read stalling the pipeline.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baud_rate: int = 115200,
        min_signal_strength: int = 0,
        read_timeout_s: float = 0.3,
        serial_obj: Optional[SerialLike] = None,
    ):
        """`serial_obj` is a test/DI hook: pass an already-open serial-like
        object (see `SerialLike`) to bypass opening a real port. Normal
        callers just pass `port`."""

        self._min_signal_strength = min_signal_strength
        self._read_timeout_s = read_timeout_s

        if serial_obj is not None:
            self._serial: SerialLike = serial_obj
        else:
            if not port:
                raise ValueError("Tf02ProRangeSensor requires a serial `port` (or an injected serial_obj)")
            import serial  # lazy import: pyserial only needed when actually talking to hardware

            self._serial = serial.Serial(port, baudrate=baud_rate, timeout=0.1)

        self._lock = threading.Lock()
        self._latest: Optional[RangeSample] = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self) -> None:
        buffer = bytearray()
        while not self._stop_event.is_set():
            chunk = self._serial.read(TF02_FRAME_LENGTH)
            if not chunk:
                continue
            buffer.extend(chunk)

            while len(buffer) >= TF02_FRAME_LENGTH:
                if buffer[0] != TF02_HEADER_BYTE or buffer[1] != TF02_HEADER_BYTE:
                    # Not aligned to a frame boundary -- drop one byte and resync.
                    del buffer[0:1]
                    continue
                frame = bytes(buffer[0:TF02_FRAME_LENGTH])
                del buffer[0:TF02_FRAME_LENGTH]

                sample = parse_tf02_frame(frame, self._min_signal_strength)
                if sample is not None:
                    with self._lock:
                        self._latest = sample

    def read(self, timestamp: float) -> Optional[RangeSample]:
        with self._lock:
            sample = self._latest
        if sample is None:
            return None
        if time.time() - sample.timestamp > self._read_timeout_s:
            return RangeSample(distance_m=sample.distance_m, timestamp=sample.timestamp, valid=False)
        return sample

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self._serial.close()


def build_range_sensor(config: BridgeConfig) -> RangeSensor:
    if config.laser.sensor == "mock":
        return MockRangeSensor(config.laser.mock_fixed_distance_m)
    if config.laser.sensor == "serial":
        if not config.laser.serial_port:
            raise ValueError("laser.sensor is 'serial' but laser.serial.port is not set")
        return Tf02ProRangeSensor(
            port=config.laser.serial_port,
            baud_rate=config.laser.serial_baud_rate,
            min_signal_strength=config.laser.serial_min_signal_strength,
            read_timeout_s=config.laser.serial_read_timeout_s,
        )
    raise ValueError(f"Unknown laser.sensor: {config.laser.sensor!r}")
