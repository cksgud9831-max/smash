import cv2
import numpy as np
import pytest

from scripts.recording_player import VideoFile, timestamp


def test_local_video_seek_and_end(tmp_path):
    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for i in range(20):
        writer.write(np.full((48, 64, 3), i * 10, np.uint8))
    writer.release()
    video = VideoFile(path)
    try:
        assert video.duration == pytest.approx(2)
        assert video.frame_at(0).mean() < 5
        assert video.frame_at(1).mean() == pytest.approx(100, abs=5)
        assert video.frame_at(1) is None
        assert video.frame_at(100).mean() == pytest.approx(190, abs=5)
        assert video.frame_at(-1).mean() < 5
    finally:
        video.close()


def test_invalid_file_is_rejected(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"invalid")
    with pytest.raises(ValueError):
        VideoFile(path)


def test_timestamp():
    assert timestamp(3661) == "01:01:01"
    assert timestamp(-10) == "00:00:00"
