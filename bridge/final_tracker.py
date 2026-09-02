"""Stateful Bridge adapter for the validated v20 YOLO/LK follower.

Only the real-time SEARCH and TRACK paths are retained from
``detector+tracker/final/src/yolo_follower_v20_final.py``.  Benchmarking,
replay pacing, rendering, and LOST/reacquisition policy deliberately remain
outside this adapter.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Optional

import cv2
import numpy as np

from bridge.optical_flow_tracker import FollowerResult

CONF_THRES = 0.25
IMGSZ = (384, 640)
INITIAL_SEARCH_INTERVAL = 5
SEARCH_CONFIRM_HITS = 2

FEATURE_MAX_CORNERS = 300
MIN_FLOW_FEATURES = 6
LOW_FEATURE_LIMIT = 3
REFRESH_FEATURE_LIMIT = 20
BBOX_SIZE_ALPHA = 0.90
ROI_MARGIN = 5.896049312441447
PERIODIC_DETECT_INTERVAL = 20
CENTER_SHIFT_THRES = 153
AREA_CHANGE_THRES = 0.21069533031912427
MOTION_WEIGHT = 0.3791670489989446
CONF_WEIGHT = 0.3485399290530977
SCALE_WEIGHT = 0.27229302194795774
MAX_CENTER_DIST = 141
YOLO_BBOX_BLEND_ALPHA = 0.35
YOLO_BBOX_Y_BIAS = 0.0
ROI_MIN_AREA_RATIO = 0.55
ROI_MIN_CONF_FOR_ACCEPT = 0.60
PERIODIC_YOLO_BLEND_ALPHA = 1.00
FEATURE_KEEP_MARGIN = 2.0
FEATURE_ROI_SCALE = 1.0
FLOW_RESIDUAL_THRES = 999.0
FLOW_KEEP_RESIDUAL = 30.0
FEATURE_CENTER_OFFSET_THRES = 999.0
UPWARD_FLOW_SCALE = 0.65
UPWARD_DAMPING_MAX_STEP = 35.0

_RECOVERY_REASONS = {"low_features", "center_jump", "area_jump", "feature_offset"}


def _clip_bbox(bbox, frame_shape):
    x, y, w, h = map(int, bbox)
    height, width = frame_shape[:2]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(3, min(w, width - x))
    h = max(3, min(h, height - y))
    return x, y, w, h


def _expand_bbox(bbox, frame_shape, margin=ROI_MARGIN):
    x, y, w, h = bbox
    height, width = frame_shape[:2]
    cx, cy = x + w / 2, y + h / 2
    x1, y1 = int(cx - w * margin / 2), int(cy - h * margin / 2)
    x2, y2 = int(cx + w * margin / 2), int(cy + h * margin / 2)
    x1, y1 = max(0, min(x1, width - 1)), max(0, min(y1, height - 1))
    x2, y2 = max(0, min(x2, width - 1)), max(0, min(y2, height - 1))
    return x1, y1, x2 - x1, y2 - y1


def _apply_yolo_bbox_bias(bbox, frame_shape):
    x, y, w, h = bbox
    return _clip_bbox((x, y + int(h * YOLO_BBOX_Y_BIAS), w, h), frame_shape)


def _roi_scale_valid(previous, detected):
    if previous is None or detected is None:
        return True
    previous_area = max(previous[2] * previous[3], 1)
    detected_area = max(detected[2] * detected[3], 1)
    return detected_area / previous_area >= ROI_MIN_AREA_RATIO


def _motion_consistent_bbox(boxes, previous, frame_shape, offset=(0, 0)):
    if boxes is None or len(boxes) == 0:
        return None, 0.0
    ox, oy = offset
    px, py, pw, ph = previous
    pcx, pcy = px + pw / 2, py + ph / 2
    previous_area = max(pw * ph, 1)
    best_score, best_bbox, best_conf = -1e9, None, 0.0
    for index in range(len(boxes)):
        confidence = float(boxes.conf[index].cpu().item())
        x1, y1, x2, y2 = boxes.xyxy[index].cpu().numpy()
        x1, y1, x2, y2 = map(int, (x1 + ox, y1 + oy, x2 + ox, y2 + oy))
        width, height = x2 - x1, y2 - y1
        if width <= 2 or height <= 2:
            continue
        center_distance = np.linalg.norm((x1 + width / 2 - pcx, y1 + height / 2 - pcy))
        motion_score = max(0.0, 1.0 - center_distance / max(MAX_CENTER_DIST, 1))
        scale_ratio = max(max(width * height, 1) / previous_area, 1e-6)
        scale_score = np.exp(-abs(np.log(scale_ratio)))
        score = CONF_WEIGHT * confidence + MOTION_WEIGHT * motion_score + SCALE_WEIGHT * scale_score
        if score > best_score:
            best_score = score
            best_bbox = _apply_yolo_bbox_bias((x1, y1, width, height), frame_shape)
            best_conf = confidence
    return best_bbox, best_conf


def _yolo_detect(frame, model, device, roi_bbox=None, prev_bbox=None, conf=CONF_THRES):
    """v20 ROI detection, including its motion selection and full fallback."""
    if roi_bbox is not None:
        rx, ry, rw, rh = _expand_bbox(roi_bbox, frame.shape)
        crop = frame[ry : ry + rh, rx : rx + rw]
        if crop.size:
            boxes = model.predict(
                source=crop, conf=conf, imgsz=IMGSZ, device=device, verbose=False
            )[0].boxes
            if boxes is not None and len(boxes) > 0:
                if prev_bbox is not None:
                    bbox, score = _motion_consistent_bbox(boxes, prev_bbox, frame.shape, (rx, ry))
                    if bbox is not None and score >= ROI_MIN_CONF_FOR_ACCEPT and _roi_scale_valid(prev_bbox, bbox):
                        return bbox, score
                best = boxes.conf.argmax().item()
                x1, y1, x2, y2 = map(int, boxes.xyxy[best].cpu().numpy())
                score = float(boxes.conf[best].cpu().item())
                bbox = _apply_yolo_bbox_bias((rx + x1, ry + y1, x2 - x1, y2 - y1), frame.shape)
                if score >= ROI_MIN_CONF_FOR_ACCEPT and _roi_scale_valid(prev_bbox, bbox):
                    return bbox, score

    boxes = model.predict(
        source=frame, conf=conf, imgsz=IMGSZ, device=device, verbose=False
    )[0].boxes
    if boxes is None or len(boxes) == 0:
        return None, 0.0
    if prev_bbox is not None:
        bbox, score = _motion_consistent_bbox(boxes, prev_bbox, frame.shape)
        if bbox is not None:
            return bbox, score
    best = boxes.conf.argmax().item()
    x1, y1, x2, y2 = map(int, boxes.xyxy[best].cpu().numpy())
    score = float(boxes.conf[best].cpu().item())
    return _apply_yolo_bbox_bias((x1, y1, x2 - x1, y2 - y1), frame.shape), score


def _make_lk_params(width, height):
    if width * height < 200:
        features = dict(maxCorners=FEATURE_MAX_CORNERS, qualityLevel=0.0005, minDistance=1, blockSize=3)
        lk = dict(winSize=(5, 5), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    else:
        features = dict(maxCorners=FEATURE_MAX_CORNERS, qualityLevel=0.01, minDistance=1, blockSize=7)
        lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03))
    return features, lk


def _feature_roi_bbox(bbox, frame_shape):
    x, y, w, h = _clip_bbox(bbox, frame_shape)
    if w < 12 or h < 12:
        return x, y, w, h
    cx, cy = x + w / 2, y + h / 2
    fw, fh = max(3, int(w * FEATURE_ROI_SCALE)), max(3, int(h * FEATURE_ROI_SCALE))
    return _clip_bbox((int(cx - fw / 2), int(cy - fh / 2), fw, fh), frame_shape)


def _init_follower(frame, bbox):
    x, y, w, h = _clip_bbox(bbox, frame.shape)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    feature_params, lk_params = _make_lk_params(w, h)
    fx, fy, fw, fh = _feature_roi_bbox((x, y, w, h), frame.shape)
    points = cv2.goodFeaturesToTrack(gray[fy : fy + fh, fx : fx + fw], mask=None, **feature_params)
    if points is not None:
        points[:, 0, 0] += fx
        points[:, 0, 1] += fy
    return gray, points, lk_params


def _keep_points(points, bbox, frame_shape):
    if points is None or len(points) == 0:
        return points
    x, y, w, h = _expand_bbox(bbox, frame_shape, FEATURE_KEEP_MARGIN)
    flat = points.reshape(-1, 2)
    keep = (flat[:, 0] >= x) & (flat[:, 0] <= x + w) & (flat[:, 1] >= y) & (flat[:, 1] <= y + h)
    return flat[keep].reshape(-1, 1, 2)


def _feature_center_offset(points, bbox):
    if points is None or len(points) == 0:
        return 0.0
    x, y, w, h = bbox
    feature_center = np.median(points.reshape(-1, 2), axis=0)
    return float(np.linalg.norm(feature_center - (x + w / 2, y + h / 2)) / max(float(np.hypot(w, h)), 1.0))


def _apply_redetection(current, detected, frame_shape, reason):
    alpha = PERIODIC_YOLO_BLEND_ALPHA if reason == "periodic" else 1.0
    blended = tuple(int((1 - alpha) * a + alpha * b) for a, b in zip(current, detected))
    return _clip_bbox(blended, frame_shape)


class FinalTracker:
    """Stateful v20 follower implementing the Bridge ``update`` contract.

    ``model`` and ``executor_factory`` are injection points for engine-free,
    deterministic unit tests. Production construction loads ``YOLO(engine_path)``.

    ``used_yolo`` is true only on an update that submits a new asynchronous
    request. ``is_recovery_event`` is true on that same update only when the
    request reason is non-periodic. Consuming/applying the eventual result does
    not emit either event a second time.
    """

    def __init__(
        self,
        engine_path: str,
        *,
        device: int | str = 0,
        model=None,
        executor_factory: Callable[..., ThreadPoolExecutor] = ThreadPoolExecutor,
    ):
        if model is None:
            from ultralytics import YOLO

            model = YOLO(engine_path, task="detect")
        self._model = model
        self._device = device
        self._executor = executor_factory(max_workers=1, thread_name_prefix="yolo-gpu")
        self._future: Optional[Future] = None
        self._request = None
        self._closed = False
        self._state = "SEARCH"
        self._search_frame_index = 0
        self._search_hits = 0
        self._track_frame_index = 0
        self._smooth_bbox = None
        self._base_w = self._base_h = 0.0
        self._old_gray = self._p0 = self._lk_params = None
        self._low_feature_count = 0
        self._prev_area = 1.0
        self._prev_center = (0.0, 0.0)
        self._last_score = 0.0

    def update(self, frame_bgr: np.ndarray) -> Optional[FollowerResult]:
        if self._closed:
            raise RuntimeError("FinalTracker is closed")
        if self._state == "SEARCH":
            return self._search(frame_bgr)
        return self._track(frame_bgr)

    def _search(self, frame):
        result = None
        if self._search_frame_index == 0 or self._search_frame_index % INITIAL_SEARCH_INTERVAL == 0:
            bbox, score = _yolo_detect(frame, self._model, self._device)
            if bbox is None:
                self._search_hits = 0
            else:
                self._search_hits += 1
                if self._search_hits >= SEARCH_CONFIRM_HITS:
                    self._enter_track(frame, bbox, score)
                    result = FollowerResult(self._smooth_bbox, score, False, True)
        self._search_frame_index += 1
        return result

    def _enter_track(self, frame, bbox, score):
        self._state = "TRACK"
        self._smooth_bbox = _clip_bbox(bbox, frame.shape)
        _, _, self._base_w, self._base_h = self._smooth_bbox
        self._old_gray, self._p0, self._lk_params = _init_follower(frame, self._smooth_bbox)
        x, y, w, h = self._smooth_bbox
        self._prev_area = max(w * h, 1)
        self._prev_center = (x + w / 2, y + h / 2)
        self._last_score = score

    def _track(self, frame):
        self._track_frame_index += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        feature_offset = self._run_flow(frame, gray)
        used_yolo = False
        recovery_event = False

        # Only done() futures are consumed: update never waits for GPU inference.
        applied_this_frame = False
        if self._future is not None and self._future.done():
            request, future = self._request, self._future
            self._future = self._request = None
            try:
                detected, score = future.result()
            except Exception:
                detected, score = None, 0.0
            if detected is not None:
                sx, sy, sw, sh = request["submit_bbox"]
                cx, cy, cw, ch = self._smooth_bbox
                shift_x = (cx + cw / 2) - (sx + sw / 2)
                shift_y = (cy + ch / 2) - (sy + sh / 2)
                dx, dy, dw, dh = detected
                compensated = _clip_bbox((int(round(dx + shift_x)), int(round(dy + shift_y)), int(dw), int(dh)), frame.shape)
                self._smooth_bbox = _apply_redetection(self._smooth_bbox, compensated, frame.shape, request["reason"])
                _, _, self._base_w, self._base_h = self._smooth_bbox
                self._old_gray, self._p0, self._lk_params = _init_follower(frame, self._smooth_bbox)
                self._low_feature_count = 0
                self._last_score = score
                applied_this_frame = True

        x, y, w, h = self._smooth_bbox
        area = max(w * h, 1)
        center = (x + w // 2, y + h // 2)
        area_change = abs(area - self._prev_area) / max(self._prev_area, 1)
        center_shift = float(np.linalg.norm(np.array(center) - np.array(self._prev_center)))
        reason = ""
        if self._low_feature_count >= LOW_FEATURE_LIMIT:
            reason = "low_features"
        elif center_shift > CENTER_SHIFT_THRES:
            reason = "center_jump"
        elif area_change > AREA_CHANGE_THRES:
            reason = "area_jump"
        elif feature_offset > FEATURE_CENTER_OFFSET_THRES:
            reason = "feature_offset"
        elif self._track_frame_index % PERIODIC_DETECT_INTERVAL == 0:
            reason = "periodic"

        if reason and not applied_this_frame and self._future is None:
            submit_bbox = tuple(self._smooth_bbox)
            self._future = self._executor.submit(
                _yolo_detect, frame.copy(), self._model, self._device, submit_bbox, submit_bbox
            )
            self._request = {"reason": reason, "submit_bbox": submit_bbox}
            used_yolo = True
            recovery_event = reason in _RECOVERY_REASONS

        x, y, w, h = self._smooth_bbox
        self._prev_area = max(w * h, 1)
        self._prev_center = (x + w / 2, y + h / 2)
        return FollowerResult(self._smooth_bbox, self._last_score, recovery_event, used_yolo)

    def _run_flow(self, frame, gray):
        feature_offset = 0.0
        if self._p0 is not None and len(self._p0) >= 3:
            p1, status, _ = cv2.calcOpticalFlowPyrLK(self._old_gray, gray, self._p0, None, **self._lk_params)
            if p1 is not None and status is not None:
                new, old = p1[status == 1], self._p0[status == 1]
                if len(new) >= 3:
                    vectors = new - old
                    median = np.median(vectors, axis=0)
                    distances = np.linalg.norm(vectors - median, axis=1)
                    residual = float(np.percentile(distances, 75))
                    keep = distances < FLOW_KEEP_RESIDUAL
                    new, old = new[keep], old[keep]
                    if residual > FLOW_RESIDUAL_THRES:
                        self._old_gray = gray.copy()
                        self._p0 = _keep_points(new.reshape(-1, 1, 2), self._smooth_bbox, frame.shape)
                        self._low_feature_count = LOW_FEATURE_LIMIT
                    elif len(new) >= MIN_FLOW_FEATURES:
                        x, y, w, h = self._smooth_bbox
                        dx, dy = np.median(new - old, axis=0)
                        if dy < 0 and abs(dy) <= UPWARD_DAMPING_MAX_STEP:
                            dy *= UPWARD_FLOW_SCALE
                        cx, cy = x + w / 2 + dx, y + h / 2 + dy
                        nw = BBOX_SIZE_ALPHA * w + (1 - BBOX_SIZE_ALPHA) * self._base_w
                        nh = BBOX_SIZE_ALPHA * h + (1 - BBOX_SIZE_ALPHA) * self._base_h
                        self._smooth_bbox = _clip_bbox((int(cx - nw / 2), int(cy - nh / 2), int(nw), int(nh)), frame.shape)
                        feature_offset = _feature_center_offset(new, self._smooth_bbox)
                        self._old_gray = gray.copy()
                        self._p0 = _keep_points(new.reshape(-1, 1, 2), self._smooth_bbox, frame.shape)
                        if self._p0 is not None and len(self._p0) < REFRESH_FEATURE_LIMIT:
                            self._old_gray, self._p0, self._lk_params = _init_follower(frame, self._smooth_bbox)
                        self._low_feature_count = 0
                        return feature_offset
        self._low_feature_count += 1
        return feature_offset

    def close(self) -> None:
        """Wait for the sole worker and release its thread (idempotent)."""
        if not self._closed:
            self._executor.shutdown(wait=True)
            self._future = self._request = None
            self._closed = True
