"""The production optical-flow tracker must never wait for YOLO in update()."""

from concurrent.futures import Future

import cv2
import numpy as np

from bridge.optical_flow_tracker import OpticalFlowTracker


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def item(self):
        return self.value.item()

    def __getitem__(self, index):
        return Tensor(self.value[index])


class Confidence(Tensor):
    def argmax(self):
        return Tensor(np.argmax(self.value))


class Boxes:
    def __init__(self, bbox=None, score=0.9):
        self.xyxy = Tensor([] if bbox is None else [bbox])
        self.conf = Confidence([] if bbox is None else [score])

    def __len__(self):
        return len(self.conf.value)


class Model:
    def __init__(self, detections):
        self.detections = list(detections)
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        bbox, score = self.detections.pop(0)
        return [type("Result", (), {"boxes": Boxes(bbox, score)})()]


class ControlledExecutor:
    def __init__(self, **kwargs):
        self.submissions = []
        self.shutdown_calls = []

    def submit(self, fn, *args):
        future = Future()
        self.submissions.append((future, fn, args))
        return future

    def shutdown(self, wait=True):
        self.shutdown_calls.append(wait)


def frame(shift=0):
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(image, (20 + shift, 20), (59 + shift, 59), (255, 255, 255), 1)
    for y in range(25, 56, 5):
        for x in range(25, 56, 5):
            image[y, x + shift] = 255
    return image


def tracker(detections, *, warmup=False, max_age=1.0):
    model = Model(detections)
    executor = ControlledExecutor()
    instance = OpticalFlowTracker.__new__(OpticalFlowTracker)
    # Build through the real initializer except for the heavy YOLO constructor.
    instance._model = model
    instance._device = "cpu"
    instance._acquire_imgsz = 960
    instance._conf_thres = 0.25
    instance._multi_scale = False
    instance._acquire_cycle_index = 0
    instance._executor = executor
    instance._future = None
    instance._request = None
    instance._generation = 0
    instance._closed = False
    instance._async_result_max_age_s = max_age
    instance._async_submitted = 0
    instance._async_applied = 0
    instance._async_stale = 0
    instance._async_failed = 0
    instance._search_hits = 0
    instance._recovery_misses = 0
    instance._smooth_bbox = None
    instance._base_w = instance._base_h = 0.0
    instance._old_gray = instance._p0 = instance._lk_params = None
    instance._frame_index = 0
    instance._low_feature_count = 0
    instance._prev_area = 1.0
    instance._prev_center = (0.0, 0.0)
    instance._last_score = 0.0
    if warmup:
        instance._submit_detection(np.zeros((640, 640, 3), np.uint8), reason="warmup", roi_bbox=None,
                                   full_frame_imgsz=960)
    return instance, executor, model


def complete(executor, index=0):
    future, fn, args = executor.submissions[index]
    future.set_result(fn(*args))


def test_search_yolo_is_submitted_without_running_on_update_thread():
    instance, executor, model = tracker([([20, 20, 60, 60], 0.9)])
    assert instance.update(frame()) is None
    assert len(executor.submissions) == 1
    assert model.calls == []


def test_completed_search_result_is_applied_on_a_later_frame():
    instance, executor, _ = tracker([
        ([20, 20, 60, 60], 0.9),
        ([23, 20, 63, 60], 0.92),
    ])
    assert instance.update(frame()) is None
    complete(executor)
    assert instance.update(frame(3)) is None
    complete(executor, 1)
    result = instance.update(frame(3))
    assert result is not None
    assert result.bbox[0] >= 20
    assert result.used_yolo
    assert instance._async_applied == 1


def test_reset_invalidates_inflight_result():
    instance, executor, _ = tracker([([20, 20, 60, 60], 0.9), (None, 0.0)])
    instance.update(frame())
    instance.reset()
    complete(executor)
    assert instance.update(frame()) is None
    assert instance._smooth_bbox is None
    assert instance._async_stale == 1


def test_old_completed_track_correction_is_discarded():
    instance, executor, _ = tracker(
        [([20, 20, 60, 60], 0.9), ([20, 20, 60, 60], 0.91),
         ([22, 20, 62, 60], 0.95)], max_age=0.01
    )
    instance.update(frame())
    complete(executor)
    assert instance.update(frame()) is None
    complete(executor, 1)
    assert instance.update(frame()) is not None
    instance._frame_index = 29
    assert not instance.update(frame(1)).used_yolo
    assert instance._future is not None
    instance._request["submitted_at"] -= 1.0
    complete(executor, 2)
    assert instance.update(frame(2)) is not None
    assert instance._smooth_bbox is not None
    assert instance._async_stale == 1


def test_track_keeps_returning_optical_flow_while_yolo_is_pending():
    instance, executor, _ = tracker([
        ([20, 20, 60, 60], 0.9), ([20, 20, 60, 60], 0.91),
        ([22, 20, 62, 60], 0.95),
    ])
    instance.update(frame())
    complete(executor)
    assert instance.update(frame()) is None
    complete(executor, 1)
    assert instance.update(frame()) is not None
    instance._frame_index = 29
    submitted = instance.update(frame(1))
    assert not submitted.used_yolo
    assert instance._future is not None
    pending = instance.update(frame(2))
    assert pending is not None
    assert not pending.used_yolo
    assert len(executor.submissions) == 3


def test_track_result_compensation_cannot_follow_large_background_drift():
    request = {"submit_bbox": (20, 20, 40, 40)}
    detected = (22, 20, 40, 40)
    # LK box drifted 100 px while the detector worked.  The delayed detector
    # result may be advanced only by the small physical latency allowance.
    corrected = OpticalFlowTracker._compensate_track_bbox(
        request, detected, (120, 20, 40, 40), (200, 240, 3)
    )
    assert corrected[0] == 28
    assert corrected[1] == 20


def test_warmup_occupies_worker_before_live_search():
    instance, executor, model = tracker([(None, 0.0), ([20, 20, 60, 60], 0.9)], warmup=True)
    assert instance.update(frame()) is None
    assert len(executor.submissions) == 1
    complete(executor)
    assert instance.update(frame()) is None
    assert len(executor.submissions) == 2
    assert len(model.calls) == 1


def test_close_releases_worker_once():
    instance, executor, _ = tracker([])
    instance.close()
    instance.close()
    assert executor.shutdown_calls == [True]


def test_feature_override_changes_only_requested_feature_params():
    from bridge.optical_flow_tracker import make_lk_params

    base_features, base_lk = make_lk_params(44, 17)
    features, lk = make_lk_params(44, 17, {"qualityLevel": 0.0005, "blockSize": 3})
    assert base_features["qualityLevel"] == 0.01 and base_features["blockSize"] == 7
    assert features["qualityLevel"] == 0.0005 and features["blockSize"] == 3
    assert features["maxCorners"] == base_features["maxCorners"]
    assert features["minDistance"] == base_features["minDistance"]
    assert lk == base_lk
    assert make_lk_params(44, 17, None) == (base_features, base_lk)
    assert make_lk_params(44, 17, {}) == (base_features, base_lk)
