"""다중 스케일 탐지(bridge/optical_flow_tracker.py) 단위 시험.

실제 YOLO 없이, 넘겨받은 imgsz 를 기록하는 가짜 모델로 입력 크기 선택만 검사한다.
"""

import numpy as np
import pytest

from bridge import optical_flow_tracker as oft
from bridge.optical_flow_tracker import (
    ACQUIRE_IMGSZ_CYCLE,
    MAX_MODEL_TARGET_PX,
    OpticalFlowTracker,
    imgsz_for_target,
    yolo_detect,
)


class _Boxes:
    def __init__(self):
        self.conf = None

    def __len__(self):
        return 0


class _Result:
    boxes = _Boxes()


class FakeModel:
    """아무것도 탐지하지 않고, 호출마다 (source 크기, imgsz) 를 기록한다."""

    def __init__(self):
        self.calls = []

    def predict(self, source, conf, device, verbose, imgsz=None):
        self.calls.append((source.shape[:2], imgsz))
        return [_Result()]


def _tracker(model, acquire_imgsz=960, multi_scale=True):
    t = OpticalFlowTracker.__new__(OpticalFlowTracker)  # ultralytics 로딩 없이 생성
    t._model = model
    t._device = "cpu"
    t._conf_thres = 0.25
    t._acquire_imgsz = acquire_imgsz
    t._multi_scale = bool(multi_scale) and acquire_imgsz is not None
    t._acquire_cycle_index = 0
    t.reset()
    return t


# ── imgsz_for_target ───────────────────────────────────────────────────


def test_small_target_keeps_cap():
    # 10 px 표적을 59 px crop 에서: 640 으로 키워도 108 px 안쪽이 아니면 줄인다.
    # ROI_MARGIN(약 5.9)배 crop 을 640 으로 키우면 표적은 약 108 px 이 되어 상한(100)을
    # 살짝 넘으므로 576 이 된다. 더 작은 crop 비율에서는 cap 그대로다.
    assert imgsz_for_target(147, 25, 640) == 576
    assert imgsz_for_target(300, 25, 640) == 640
    # 전체 프레임 640 에서 작은 표적은 탐색 입력 960 그대로 (이전 동작)
    assert imgsz_for_target(640, 25, 960) == 960


@pytest.mark.parametrize("target_px", [100, 135, 181, 250, 400])
def test_large_target_is_scaled_under_limit(target_px):
    size = imgsz_for_target(640, target_px, 960)
    assert size % 32 == 0
    assert target_px * size / 640 <= MAX_MODEL_TARGET_PX or size == oft.MIN_IMGSZ


def test_imgsz_never_below_minimum_and_handles_degenerate():
    assert imgsz_for_target(640, 5000, 960) == oft.MIN_IMGSZ
    assert imgsz_for_target(640, 0, 960) == 960
    assert imgsz_for_target(0, 50, 640) == 640


# ── 첫 포착: 프레임마다 입력 크기 순환 ─────────────────────────────────


def test_acquisition_cycles_input_sizes():
    m = FakeModel()
    t = _tracker(m)
    frame = np.zeros((640, 640, 3), np.uint8)
    for _ in range(6):
        assert t.update(frame) is None
    assert [c[1] for c in m.calls] == list(ACQUIRE_IMGSZ_CYCLE) * 2


def test_cycle_survives_reset_every_frame():
    # smash_fcs_node 는 SEARCH 동안 매 프레임 reset() 을 부른다. 그래도 순환해야 한다.
    m = FakeModel()
    t = _tracker(m)
    frame = np.zeros((640, 640, 3), np.uint8)
    for _ in range(6):
        t.update(frame)
        t.reset()
    assert [c[1] for c in m.calls] == list(ACQUIRE_IMGSZ_CYCLE) * 2


def test_multi_scale_off_keeps_single_input_size():
    m = FakeModel()
    t = _tracker(m, multi_scale=False)
    frame = np.zeros((640, 640, 3), np.uint8)
    for _ in range(3):
        t.update(frame)
    assert [c[1] for c in m.calls] == [960, 960, 960]


def test_cycle_respects_smaller_acquire_imgsz():
    m = FakeModel()
    t = _tracker(m, acquire_imgsz=640)
    frame = np.zeros((640, 640, 3), np.uint8)
    for _ in range(4):
        t.update(frame)
    assert [c[1] for c in m.calls] == [640, 320, 640, 320]


# ── 추적 중 재탐지(ROI) ────────────────────────────────────────────────


def test_roi_input_keeps_target_under_limit():
    frame = np.zeros((640, 640, 3), np.uint8)

    m = FakeModel()
    yolo_detect(frame, m, "cpu", 0.25, roi_bbox=(300, 300, 25, 12), prev_bbox=(300, 300, 25, 12),
                full_frame_imgsz=960, limit_target_size=True)
    roi_call, full_call = m.calls
    assert 25 * roi_call[1] / max(roi_call[0]) <= MAX_MODEL_TARGET_PX  # 작은 표적도 상한 안
    assert roi_call[1] >= 512
    assert full_call[1] == 960         # 전체 프레임 재포착도 960 그대로

    m = FakeModel()
    yolo_detect(frame, m, "cpu", 0.25, roi_bbox=(200, 250, 250, 110), prev_bbox=(200, 250, 250, 110),
                full_frame_imgsz=960, limit_target_size=True)
    (roi_shape, roi_imgsz), (_, full_imgsz) = m.calls
    assert 250 * roi_imgsz / max(roi_shape) <= MAX_MODEL_TARGET_PX or roi_imgsz == oft.MIN_IMGSZ
    assert 250 * full_imgsz / 640 <= MAX_MODEL_TARGET_PX


def test_roi_default_behaviour_without_limit_flag():
    frame = np.zeros((640, 640, 3), np.uint8)
    m = FakeModel()
    yolo_detect(frame, m, "cpu", 0.25, roi_bbox=(200, 250, 250, 110), prev_bbox=(200, 250, 250, 110),
                full_frame_imgsz=960)
    assert m.calls[0][1] is None  # 이전처럼 Ultralytics 기본값
    assert m.calls[1][1] == 960
