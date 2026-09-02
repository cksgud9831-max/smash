from concurrent.futures import Future

import cv2
import numpy as np

from bridge.final_tracker import FinalTracker
from bridge.optical_flow_tracker import FollowerResult


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


def frame():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (59, 59), (255, 255, 255), 1)
    for y in range(25, 56, 5):
        for x in range(25, 56, 5):
            image[y, x] = 255
    return image


def tracker(detections):
    executor = ControlledExecutor()
    instance = FinalTracker("unused.engine", model=Model(detections), executor_factory=lambda **_: executor)
    return instance, executor


def acquire(instance):
    assert instance.update(frame()) is None
    for _ in range(4):
        assert instance.update(frame()) is None
    return instance.update(frame())


def test_first_search_detection_does_not_enter_track():
    instance, _ = tracker([([20, 20, 60, 60], 0.8)])
    assert instance.update(frame()) is None
    assert instance._state == "SEARCH"


def test_two_consecutive_search_detections_enter_track():
    instance, _ = tracker([([20, 20, 60, 60], 0.8), ([22, 20, 62, 60], 0.9)])
    result = acquire(instance)
    assert instance._state == "TRACK"
    assert result.bbox == (22, 20, 40, 40)
    assert result.score == 0.9


def test_search_runs_only_every_five_frames():
    instance, _ = tracker([([20, 20, 60, 60], 0.8), ([20, 20, 60, 60], 0.9)])
    model = instance._model
    instance.update(frame())
    for _ in range(4):
        instance.update(frame())
    assert len(model.calls) == 1
    instance.update(frame())
    assert len(model.calls) == 2


def test_failed_search_attempt_resets_confirmation_hits():
    instance, _ = tracker(
        [
            ([20, 20, 60, 60], 0.8),
            (None, 0.0),
            ([20, 20, 60, 60], 0.85),
            ([20, 20, 60, 60], 0.9),
        ]
    )
    assert instance.update(frame()) is None
    for _ in range(4):
        assert instance.update(frame()) is None
    assert instance.update(frame()) is None
    assert instance._search_hits == 0
    for _ in range(4):
        assert instance.update(frame()) is None
    assert instance.update(frame()) is None
    assert instance._state == "SEARCH"
    for _ in range(4):
        assert instance.update(frame()) is None
    assert isinstance(instance.update(frame()), FollowerResult)


def test_pure_optical_flow_does_not_call_or_submit_yolo():
    instance, executor = tracker([([20, 20, 60, 60], 0.8), ([20, 20, 60, 60], 0.9)])
    acquire(instance)
    result = instance.update(frame())
    assert not result.used_yolo
    assert executor.submissions == []


def test_periodic_twentieth_track_frame_submits_async_yolo():
    instance, executor = tracker([([20, 20, 60, 60], 0.8), ([20, 20, 60, 60], 0.9)])
    acquire(instance)
    for _ in range(19):
        assert not instance.update(frame()).used_yolo
    result = instance.update(frame())
    assert result.used_yolo and not result.is_recovery_event
    assert len(executor.submissions) == 1


def test_pending_yolo_prevents_duplicate_submission():
    instance, executor = tracker([([20, 20, 60, 60], 0.8), ([20, 20, 60, 60], 0.9)])
    acquire(instance)
    instance._track_frame_index = 19
    instance.update(frame())
    instance._low_feature_count = 3
    instance.update(frame())
    assert len(executor.submissions) == 1


def test_async_result_is_motion_compensated_and_reinitializes_features():
    instance, executor = tracker([([20, 20, 60, 60], 0.8), ([20, 20, 60, 60], 0.9)])
    acquire(instance)
    instance._track_frame_index = 19
    instance.update(frame())
    future, _, _ = executor.submissions[0]
    instance._smooth_bbox = (30, 20, 40, 40)  # moved +10 after submission
    future.set_result(((22, 20, 40, 40), 0.95))
    result = instance.update(frame())
    assert result.bbox == (32, 20, 40, 40)
    assert result.score == 0.95
    assert not result.used_yolo and not result.is_recovery_event
    assert instance._p0 is not None


def test_result_is_existing_bridge_type_and_xywh():
    instance, _ = tracker([([20, 20, 60, 60], 0.8), ([20, 20, 60, 60], 0.9)])
    result = acquire(instance)
    assert isinstance(result, FollowerResult)
    assert result.bbox == (20, 20, 40, 40)


def test_close_shuts_executor_down_once():
    instance, executor = tracker([])
    instance.close()
    instance.close()
    assert executor.shutdown_calls == [True]
