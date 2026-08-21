"""Single-target YOLO + Lucas-Kanade optical-flow follower.

Ported from detector+tracker/code/yolo_follower_v3_reactivate.py's
evaluate_tracker() loop: same algorithm, same GA-tuned constants (copied
verbatim below), same per-frame branching -- just restructured as a
stateful update(frame) the way bridge/frame_builder.py needs one frame at a
time, instead of a script that consumes a whole video and returns only
aggregate metrics at the end. If that script's GA search is ever rerun,
port the new constants here by hand.

This replaces bridge/smart_tracking (Kalman-filter + Hungarian-matching
multi-track approach) as this project's confirmed tracker -- validated
against test/visible.mp4 (see known_issues.md): mean IoU 0.760 vs ground
truth, 0 non-periodic recovery events over 1000 frames, only 3.4% of
frames needed a YOLO call.

Deliberately NOT routed through bridge.detector.Detector: this follower
only calls YOLO on init / every PERIODIC_DETECT_INTERVAL frames / on an
actual tracking failure, via its own ROI-cropped, motion-consistent
candidate selection -- forcing every frame through an external
Detector.detect() call first would throw away exactly that efficiency and
duplicate work. It uses ultralytics.YOLO(.pt) directly, matching the
validated reference script, rather than bridge.detector's ONNX/TensorRT
path (a deliberate choice to minimize risk by porting the validated code
as-is -- see known_issues.md for the tradeoff).

Single-target only, like SmartTracker before it: no multi-track Hungarian
matching or track-ID concept, since this follower only ever holds one
bbox.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# ── GA-tuned constants, copied verbatim from
# detector+tracker/code/yolo_follower_v3_reactivate.py ────────────────────
FEATURE_MAX_CORNERS = 300
MIN_FLOW_FEATURES = 6
LOW_FEATURE_LIMIT = 3
REFRESH_FEATURE_LIMIT = 20

BBOX_SIZE_ALPHA = 0.90
ROI_MARGIN = 5.896049312441447
PERIODIC_DETECT_INTERVAL = 30
CENTER_SHIFT_THRES = 153
AREA_CHANGE_THRES = 0.21069533031912427

MOTION_WEIGHT = 0.3791670489989446
CONF_WEIGHT = 0.3485399290530977
SCALE_WEIGHT = 0.27229302194795774
MAX_CENTER_DIST = 141
YOLO_BBOX_BLEND_ALPHA = 0.35
ROI_MIN_AREA_RATIO = 0.55
ROI_MIN_CONF_FOR_ACCEPT = 0.60
PERIODIC_YOLO_BLEND_ALPHA = 1.00
YOLO_BBOX_Y_BIAS = 0.0

FEATURE_KEEP_MARGIN = 2.0
FEATURE_ROI_SCALE = 1.0
FLOW_KEEP_RESIDUAL = 30.0
UPWARD_FLOW_SCALE = 0.65
UPWARD_DAMPING_MAX_STEP = 35.0

_NON_PERIODIC_REDETECT_REASONS = {"low_features", "center_jump", "area_jump", "feature_offset"}


# ── Helper functions, ported verbatim (only global-parameter references
# adjusted into plain module constants / explicit args) ───────────────────


def clip_bbox(bbox, frame_shape):
    x, y, w, h = map(int, bbox)
    H, W = frame_shape[:2]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(3, min(w, W - x))
    h = max(3, min(h, H - y))
    return x, y, w, h


def expand_bbox(bbox, frame_shape, margin=ROI_MARGIN):
    x, y, w, h = bbox
    H, W = frame_shape[:2]
    cx = x + w / 2
    cy = y + h / 2
    nw = w * margin
    nh = h * margin

    x1 = int(cx - nw / 2)
    y1 = int(cy - nh / 2)
    x2 = int(cx + nw / 2)
    y2 = int(cy + nh / 2)

    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(0, min(x2, W - 1))
    y2 = max(0, min(y2, H - 1))
    return x1, y1, x2 - x1, y2 - y1


def apply_yolo_bbox_bias(bbox, frame_shape):
    x, y, w, h = bbox
    return clip_bbox((x, y + int(h * YOLO_BBOX_Y_BIAS), w, h), frame_shape)


def is_roi_detection_scale_valid(prev_bbox, detected_bbox):
    if prev_bbox is None or detected_bbox is None:
        return True
    _, _, pw, ph = prev_bbox
    _, _, dw, dh = detected_bbox
    prev_area = max(pw * ph, 1)
    detected_area = max(dw * dh, 1)
    return detected_area / prev_area >= ROI_MIN_AREA_RATIO


def select_motion_consistent_bbox_from_yolo(boxes, prev_bbox, frame_shape, offset=(0, 0)):
    """Picks the YOLO candidate whose center/scale best matches the
    previous bbox, not just the highest-confidence one."""

    if boxes is None or len(boxes) == 0:
        return None, 0.0

    ox, oy = offset
    px, py, pw, ph = prev_bbox
    pcx = px + pw / 2
    pcy = py + ph / 2
    prev_area = max(pw * ph, 1)

    best_score = -1e9
    best_bbox = None
    best_conf = 0.0

    for i in range(len(boxes)):
        conf = float(boxes.conf[i].cpu().item())
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = map(int, [x1 + ox, y1 + oy, x2 + ox, y2 + oy])

        bw = x2 - x1
        bh = y2 - y1
        if bw <= 2 or bh <= 2:
            continue

        cx = x1 + bw / 2
        cy = y1 + bh / 2
        center_dist = np.linalg.norm([cx - pcx, cy - pcy])
        motion_score = max(0.0, 1.0 - center_dist / max(MAX_CENTER_DIST, 1))

        area = max(bw * bh, 1)
        scale_ratio = max(area / prev_area, 1e-6)
        scale_score = np.exp(-abs(np.log(scale_ratio)))

        score = CONF_WEIGHT * conf + MOTION_WEIGHT * motion_score + SCALE_WEIGHT * scale_score

        if score > best_score:
            best_score = score
            best_bbox = apply_yolo_bbox_bias((x1, y1, bw, bh), frame_shape)
            best_conf = conf

    return best_bbox, best_conf


def yolo_detect(frame, model, device, conf_thres, roi_bbox=None, prev_bbox=None):
    """ROI-cropped YOLO detection with motion-consistent candidate
    selection; falls back to full-frame detection if the ROI crop fails."""

    if roi_bbox is not None:
        rx, ry, rw, rh = expand_bbox(roi_bbox, frame.shape, ROI_MARGIN)
        crop = frame[ry:ry + rh, rx:rx + rw]

        if crop.size != 0:
            results = model.predict(source=crop, conf=conf_thres, device=device, verbose=False)
            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                if prev_bbox is not None:
                    bbox, score = select_motion_consistent_bbox_from_yolo(
                        boxes, prev_bbox, frame.shape, offset=(rx, ry)
                    )
                    if bbox is not None:
                        if score >= ROI_MIN_CONF_FOR_ACCEPT and is_roi_detection_scale_valid(prev_bbox, bbox):
                            return bbox, score

                best_idx = boxes.conf.argmax().item()
                xyxy = boxes.xyxy[best_idx].cpu().numpy()
                score = float(boxes.conf[best_idx].cpu().item())
                x1, y1, x2, y2 = map(int, xyxy)
                bbox = (rx + x1, ry + y1, x2 - x1, y2 - y1)
                bbox = apply_yolo_bbox_bias(bbox, frame.shape)
                if score >= ROI_MIN_CONF_FOR_ACCEPT and is_roi_detection_scale_valid(prev_bbox, bbox):
                    return bbox, score

    results = model.predict(source=frame, conf=conf_thres, device=device, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None, 0.0

    if prev_bbox is not None:
        bbox, score = select_motion_consistent_bbox_from_yolo(boxes, prev_bbox, frame.shape, offset=(0, 0))
        if bbox is not None:
            return bbox, score

    best_idx = boxes.conf.argmax().item()
    xyxy = boxes.xyxy[best_idx].cpu().numpy()
    score = float(boxes.conf[best_idx].cpu().item())
    x1, y1, x2, y2 = map(int, xyxy)
    bbox = (x1, y1, x2 - x1, y2 - y1)
    return apply_yolo_bbox_bias(bbox, frame.shape), score


def make_lk_params(w, h):
    area = w * h
    if area < 200:
        feature_params = dict(maxCorners=FEATURE_MAX_CORNERS, qualityLevel=0.0005, minDistance=1, blockSize=3)
        lk_params = dict(winSize=(5, 5), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    else:
        feature_params = dict(maxCorners=FEATURE_MAX_CORNERS, qualityLevel=0.01, minDistance=1, blockSize=7)
        lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03))
    return feature_params, lk_params


def feature_roi_bbox(bbox, frame_shape, scale=FEATURE_ROI_SCALE):
    x, y, w, h = clip_bbox(bbox, frame_shape)
    if w < 12 or h < 12:
        return x, y, w, h
    cx = x + w / 2
    cy = y + h / 2
    fw = max(3, int(w * scale))
    fh = max(3, int(h * scale))
    fx = int(cx - fw / 2)
    fy = int(cy - fh / 2)
    return clip_bbox((fx, fy, fw, fh), frame_shape)


def init_follower(frame, bbox):
    x, y, w, h = clip_bbox(bbox, frame.shape)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    feature_params, lk_params = make_lk_params(w, h)
    fx, fy, fw, fh = feature_roi_bbox((x, y, w, h), frame.shape)
    roi_gray = gray[fy:fy + fh, fx:fx + fw]
    p0 = cv2.goodFeaturesToTrack(roi_gray, mask=None, **feature_params)

    if p0 is not None:
        p0[:, 0, 0] += fx
        p0[:, 0, 1] += fy

    return gray, p0, lk_params


def keep_points_near_bbox(points, bbox, frame_shape, margin=FEATURE_KEEP_MARGIN):
    if points is None or len(points) == 0:
        return points
    rx, ry, rw, rh = expand_bbox(bbox, frame_shape, margin)
    x2 = rx + rw
    y2 = ry + rh
    flat = points.reshape(-1, 2)
    keep = (flat[:, 0] >= rx) & (flat[:, 0] <= x2) & (flat[:, 1] >= ry) & (flat[:, 1] <= y2)
    return flat[keep].reshape(-1, 1, 2)


def feature_center_offset_score(points, bbox):
    if points is None or len(points) == 0:
        return 0.0
    x, y, w, h = bbox
    flat = points.reshape(-1, 2)
    feature_cx, feature_cy = np.median(flat, axis=0)
    bbox_cx = x + w / 2
    bbox_cy = y + h / 2
    diag = max(float(np.hypot(w, h)), 1.0)
    return float(np.linalg.norm([feature_cx - bbox_cx, feature_cy - bbox_cy]) / diag)


def damp_upward_flow(dx, dy):
    if dy < 0 and abs(dy) <= UPWARD_DAMPING_MAX_STEP:
        dy *= UPWARD_FLOW_SCALE
    return dx, dy


def blend_bbox(current_bbox, detected_bbox, frame_shape, alpha=YOLO_BBOX_BLEND_ALPHA):
    cx, cy, cw, ch = current_bbox
    dx, dy, dw, dh = detected_bbox
    blended = (
        int((1 - alpha) * cx + alpha * dx),
        int((1 - alpha) * cy + alpha * dy),
        int((1 - alpha) * cw + alpha * dw),
        int((1 - alpha) * ch + alpha * dh),
    )
    return clip_bbox(blended, frame_shape)


def apply_redetection_bbox(current_bbox, detected_bbox, frame_shape, redetect_reason):
    if redetect_reason == "periodic":
        return blend_bbox(current_bbox, detected_bbox, frame_shape, alpha=PERIODIC_YOLO_BLEND_ALPHA)
    return clip_bbox(detected_bbox, frame_shape)


@dataclass
class FollowerResult:
    bbox: tuple[float, float, float, float]  # x, y, w, h
    score: float  # most recent YOLO detection confidence (persists across pure-opticalflow frames)
    is_recovery_event: bool  # a non-periodic redetect fired this frame (follower actually lost the target)
    used_yolo: bool = False  # a YOLO call (periodic or recovery) actually ran this frame -- for latency profiling


class OpticalFlowTracker:
    """Stateful single-target tracker: LK optical flow every frame, YOLO
    only on init / every PERIODIC_DETECT_INTERVAL frames / on an actual
    tracking failure. See module docstring for validation numbers."""

    def __init__(self, model_path: str, device: "int | str" = 0, conf_thres: float = 0.25):
        from ultralytics import YOLO  # heavy import kept lazy -- only paid if this tracker is actually used

        self._model = YOLO(model_path)
        self._device = device
        self._conf_thres = conf_thres

        self._smooth_bbox: Optional[tuple] = None
        self._base_w = 0.0
        self._base_h = 0.0
        self._old_gray = None
        self._p0 = None
        self._lk_params = None
        self._frame_index = 0
        self._low_feature_count = 0
        self._prev_area = 1.0
        self._prev_center = (0.0, 0.0)
        self._last_score = 0.0

    def update(self, frame_bgr: np.ndarray) -> Optional[FollowerResult]:
        if self._smooth_bbox is None:
            return self._initialize(frame_bgr)
        return self._track(frame_bgr)

    def _initialize(self, frame_bgr: np.ndarray) -> Optional[FollowerResult]:
        bbox, score = yolo_detect(frame_bgr, self._model, self._device, self._conf_thres, roi_bbox=None)
        if bbox is None:
            return None

        x, y, w, h = bbox
        self._smooth_bbox = (x, y, w, h)
        self._base_w, self._base_h = w, h
        self._old_gray, self._p0, self._lk_params = init_follower(frame_bgr, self._smooth_bbox)
        self._prev_area = max(w * h, 1)
        self._prev_center = (x + w / 2, y + h / 2)
        self._last_score = score
        return FollowerResult(bbox=self._smooth_bbox, score=score, is_recovery_event=False, used_yolo=True)

    def _track(self, frame_bgr: np.ndarray) -> FollowerResult:
        self._frame_index += 1
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        feature_center_offset = 0.0

        if self._p0 is not None and len(self._p0) >= 3:
            p1, st, _err = cv2.calcOpticalFlowPyrLK(self._old_gray, frame_gray, self._p0, None, **self._lk_params)

            if p1 is not None and st is not None:
                good_new = p1[st == 1]
                good_old = self._p0[st == 1]

                if len(good_new) >= 3:
                    flow_vectors = good_new - good_old
                    median_flow = np.median(flow_vectors, axis=0)
                    distances = np.linalg.norm(flow_vectors - median_flow, axis=1)
                    keep = distances < FLOW_KEEP_RESIDUAL
                    good_new = good_new[keep]
                    good_old = good_old[keep]
                    feature_count = len(good_new)

                    if feature_count >= MIN_FLOW_FEATURES:
                        px, py, pw, ph = self._smooth_bbox
                        prev_cx, prev_cy = px + pw / 2, py + ph / 2
                        dx, dy = np.median(good_new - good_old, axis=0)
                        dx, dy = damp_upward_flow(dx, dy)

                        smooth_cx = prev_cx + dx
                        smooth_cy = prev_cy + dy
                        smooth_w = BBOX_SIZE_ALPHA * pw + (1 - BBOX_SIZE_ALPHA) * self._base_w
                        smooth_h = BBOX_SIZE_ALPHA * ph + (1 - BBOX_SIZE_ALPHA) * self._base_h

                        x = int(smooth_cx - smooth_w / 2)
                        y = int(smooth_cy - smooth_h / 2)
                        w, h = int(smooth_w), int(smooth_h)
                        self._smooth_bbox = clip_bbox((x, y, w, h), frame_bgr.shape)
                        feature_center_offset = feature_center_offset_score(good_new, self._smooth_bbox)

                        self._old_gray = frame_gray.copy()
                        self._p0 = keep_points_near_bbox(good_new.reshape(-1, 1, 2), self._smooth_bbox, frame_bgr.shape)
                        if self._p0 is not None and len(self._p0) < REFRESH_FEATURE_LIMIT:
                            self._old_gray, self._p0, self._lk_params = init_follower(frame_bgr, self._smooth_bbox)
                        self._low_feature_count = 0
                    else:
                        self._low_feature_count += 1
                else:
                    self._low_feature_count += 1
            else:
                self._low_feature_count += 1
        else:
            self._low_feature_count += 1

        x, y, w, h = self._smooth_bbox
        current_area = max(w * h, 1)
        center = (x + w // 2, y + h // 2)
        area_change = abs(current_area - self._prev_area) / max(self._prev_area, 1)
        center_shift = float(np.linalg.norm(np.array(center) - np.array(self._prev_center)))

        redetect_reason = ""
        if self._low_feature_count >= LOW_FEATURE_LIMIT:
            redetect_reason = "low_features"
        elif center_shift > CENTER_SHIFT_THRES:
            redetect_reason = "center_jump"
        elif area_change > AREA_CHANGE_THRES:
            redetect_reason = "area_jump"
        elif self._frame_index % PERIODIC_DETECT_INTERVAL == 0:
            redetect_reason = "periodic"

        if redetect_reason:
            yolo_bbox, yolo_score = yolo_detect(
                frame_bgr, self._model, self._device, self._conf_thres,
                roi_bbox=self._smooth_bbox, prev_bbox=self._smooth_bbox,
            )
            if yolo_bbox is not None:
                self._smooth_bbox = apply_redetection_bbox(self._smooth_bbox, yolo_bbox, frame_bgr.shape, redetect_reason)
                x, y, w, h = self._smooth_bbox
                self._base_w, self._base_h = w, h
                self._old_gray, self._p0, self._lk_params = init_follower(frame_bgr, self._smooth_bbox)
                self._low_feature_count = 0
                self._last_score = yolo_score

        x, y, w, h = self._smooth_bbox
        self._prev_area = max(w * h, 1)
        self._prev_center = (x + w / 2, y + h / 2)

        is_recovery_event = redetect_reason in _NON_PERIODIC_REDETECT_REASONS
        return FollowerResult(
            bbox=self._smooth_bbox, score=self._last_score, is_recovery_event=is_recovery_event,
            used_yolo=bool(redetect_reason),
        )
