import time

from bridge.range_sensor import (
    TF02_HEADER_BYTE,
    Tf02ProRangeSensor,
    parse_tf02_frame,
)


def make_frame(distance_cm: int, strength: int, reliability: int = 0x03) -> bytes:
    body = bytes(
        [
            TF02_HEADER_BYTE,
            TF02_HEADER_BYTE,
            distance_cm & 0xFF,
            (distance_cm >> 8) & 0xFF,
            strength & 0xFF,
            (strength >> 8) & 0xFF,
            reliability,
            0x00,
        ]
    )
    checksum = sum(body) & 0xFF
    return body + bytes([checksum])


# ── parse_tf02_frame (pure function) ────────────────────────────────────────


def test_parse_valid_frame_above_signal_threshold():
    frame = make_frame(distance_cm=2500, strength=800)
    sample = parse_tf02_frame(frame, min_signal_strength=100)
    assert sample is not None
    assert sample.distance_m == 25.0
    assert sample.valid is True


def test_parse_frame_below_signal_threshold_is_invalid_not_dropped():
    frame = make_frame(distance_cm=2500, strength=50)
    sample = parse_tf02_frame(frame, min_signal_strength=100)
    assert sample is not None  # frame itself parsed fine
    assert sample.valid is False  # just below the confidence gate


def test_parse_frame_zero_distance_is_invalid():
    frame = make_frame(distance_cm=0, strength=800)
    sample = parse_tf02_frame(frame, min_signal_strength=100)
    assert sample is not None
    assert sample.valid is False


def test_parse_frame_rejects_bad_checksum():
    frame = bytearray(make_frame(distance_cm=2500, strength=800))
    frame[-1] ^= 0xFF  # corrupt checksum byte
    assert parse_tf02_frame(bytes(frame), min_signal_strength=100) is None


def test_parse_frame_rejects_bad_header():
    frame = bytearray(make_frame(distance_cm=2500, strength=800))
    frame[0] = 0x00
    assert parse_tf02_frame(bytes(frame), min_signal_strength=100) is None


def test_parse_frame_rejects_wrong_length():
    assert parse_tf02_frame(b"\x59\x59\x01\x02", min_signal_strength=0) is None


# ── Tf02ProRangeSensor (background thread + injected fake serial port) ─────


class FakeSerial:
    """Minimal stand-in for pyserial's Serial: yields queued chunks, then
    empty reads (as a real non-blocking/timeout'd port would when idle)."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = list(chunks)
        self.closed = False

    def read(self, size: int = 1) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        time.sleep(0.01)
        return b""

    def close(self) -> None:
        self.closed = True


def _wait_until(predicate, timeout_s=1.0, interval_s=0.01):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


def test_sensor_resyncs_past_garbage_bytes_and_reports_parsed_sample():
    garbage = b"\x00\x01\x02"
    frame = make_frame(distance_cm=1234, strength=900)
    sensor = Tf02ProRangeSensor(serial_obj=FakeSerial([garbage, frame]), min_signal_strength=100, read_timeout_s=5.0)
    try:
        assert _wait_until(lambda: sensor.read(timestamp=0.0) is not None)
        sample = sensor.read(timestamp=0.0)
        assert sample.valid is True
        assert sample.distance_m == 12.34
    finally:
        sensor.close()


def test_sensor_reports_none_before_first_frame_arrives():
    sensor = Tf02ProRangeSensor(serial_obj=FakeSerial([]), read_timeout_s=5.0)
    try:
        assert sensor.read(timestamp=0.0) is None
    finally:
        sensor.close()


def test_sensor_marks_stale_sample_invalid_after_read_timeout():
    frame = make_frame(distance_cm=1000, strength=900)
    sensor = Tf02ProRangeSensor(serial_obj=FakeSerial([frame]), min_signal_strength=100, read_timeout_s=0.05)
    try:
        assert _wait_until(lambda: sensor.read(timestamp=0.0) is not None)
        assert sensor.read(timestamp=0.0).valid is True

        time.sleep(0.15)  # exceed read_timeout_s with no further frames arriving
        stale = sensor.read(timestamp=0.0)
        assert stale is not None
        assert stale.valid is False
    finally:
        sensor.close()
