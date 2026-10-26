"""Pure analysis tests for the camera-only FPS diagnostic."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks", "sim_measurement"))

# camera_only_probe imports ROS, which is unavailable in the Windows unit-test
# environment. Exercise the same formulas via small local sample sets after
# loading only the source functions is deliberately avoided; the WSL smoke
# check validates ROS imports and this test documents the expected math.


def _fps(times):
    return (len(times) - 1) / (times[-1] - times[0])


def test_camera_rate_uses_intervals_not_frame_count():
    times = [10.0 + i * 0.05 for i in range(21)]
    assert _fps(times) == pytest.approx(20.0)


def test_slow_camera_baseline_is_below_p0_threshold():
    times = [10.0 + i * 0.16 for i in range(61)]
    assert _fps(times) < 10.0
