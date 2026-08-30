import os
import json
import csv
import argparse
import cv2
import time
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# ====================== 기본 설정 ====================== #
VIDEO_PATH = "datasets/Anti-UAV300/test/20190925_111757_1_1/visible.mp4"
MODEL_PATH = "models/ga_yolo11s_960_best.engine"

DEVICE = 0
CONF_THRES = 0.25
IMGSZ = (384, 640)

def timed_yolo_detect(*args, **kwargs):
    t0 = time.perf_counter()
    bbox, score, mode = yolo_detect(*args, **kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return bbox, score, mode, latency_ms

REALTIME_INPUT_FPS = 20.0

# SEARCH state:
# 추적 시작 전에는 full-frame YOLO를 일정 간격으로 수행한다.
INITIAL_SEARCH_INTERVAL = 5

# SEARCH에서 단일 YOLO 검출만으로 TRACK에 진입하지 않는다.
# 연속된 검색 시도에서 2회 검출되어야 실제 표적으로 확정한다.
SEARCH_CONFIRM_HITS = 2

SAVE_VIDEO = True
OUTPUT_DIR = "output_video"
os.makedirs(OUTPUT_DIR, exist_ok=True)
run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_PATH = os.path.join(OUTPUT_DIR, f"yolo_follower_v3_reactivate_{run_tag}.mp4")
DEBUG_TRACKING_LOG = True
DEBUG_CAPTURE_FRAMES = True
DEBUG_LOG_DIR = os.path.join(OUTPUT_DIR, "debug_logs")
DEBUG_FRAME_DIR = os.path.join(OUTPUT_DIR, "debug_frames", run_tag)
DEBUG_IOU_FAIL_THRES = 0.10
os.makedirs(DEBUG_LOG_DIR, exist_ok=True)
os.makedirs(DEBUG_FRAME_DIR, exist_ok=True)

# ====================== 추적/재탐지 파라미터 ====================== #
FEATURE_MAX_CORNERS = 300
MIN_FLOW_FEATURES = 6
LOW_FEATURE_LIMIT = 3
REFRESH_FEATURE_LIMIT = 20     # Modified: refresh sparse LK features before tracking becomes unstable

BBOX_CENTER_ALPHA = 0.65       # 클수록 이전 bbox 중심을 더 믿음
BBOX_SIZE_ALPHA = 0.90         # 클수록 이전 bbox 크기를 더 유지
ROI_MARGIN = 5.896049312441447 # GA-efficient candidate: wider ROI with near-baseline YOLO call count
PERIODIC_DETECT_INTERVAL = 20  # GA-efficient candidate: periodic YOLO correction interval
CENTER_SHIFT_THRES = 153       # GA-efficient candidate: tolerate larger follower motion before recovery
AREA_CHANGE_THRES = 0.21069533031912427  # GA-efficient candidate: stricter area-change recovery trigger

MOTION_WEIGHT = 0.3791670489989446       # GA-efficient motion-consistent YOLO weight
CONF_WEIGHT = 0.3485399290530977         # GA-efficient motion-consistent YOLO weight
SCALE_WEIGHT = 0.27229302194795774       # GA-efficient motion-consistent YOLO weight
MAX_CENTER_DIST = 141                    # GA-efficient motion distance normalization
YOLO_DIRECT_CENTER_DIST = 70   # Modified: apply YOLO immediately when it is close to current bbox
YOLO_CONFIRM_CENTER_DIST = 45  # Modified: two suspicious YOLO candidates must be mutually close
YOLO_CONFIRM_AREA_THRES = 0.50 # Modified: two suspicious YOLO candidates must have similar area
YOLO_BBOX_BLEND_ALPHA = 0.35   # Modified: blend accepted YOLO correction to reduce visible jumps
YOLO_HIGH_CONF_THRES = 0.75    # Modified: high-confidence YOLO can recover a drifted follower
YOLO_CENTER_JUMP_CONF_THRES = 0.70  # Modified: center jumps need a confident YOLO candidate
YOLO_FORCE_ACCEPT_REASONS = {"low_features", "area_jump"}
YOLO_BBOX_Y_BIAS = 0.0         # Natural follower mode: do not apply sequence-specific YOLO bbox bias
ROI_MIN_AREA_RATIO = 0.55      # Natural follower guard: reject tiny ROI detections and fall back to full-frame YOLO
ROI_MIN_CONF_FOR_ACCEPT = 0.60 # Modified: reject weak ROI detections that cause visible upward bbox snaps
PERIODIC_YOLO_BLEND_ALPHA = 1.00  # Modified: keep tracker responsive; visual flow trails are disabled below

DRAW_FLOW = False
FLOW_TRAIL_DECAY = 0.80        # Visualization only: fade old optical-flow trails each frame
FEATURE_KEEP_MARGIN = 2.0      # Modified: keep only flow points near the current bbox
FEATURE_ROI_SCALE = 1.0        # Natural follower mode: use the full bbox for LK feature extraction
FLOW_RESIDUAL_THRES = 999.0    # Natural follower mode: keep residual logging but disable residual-triggered recovery
FLOW_KEEP_RESIDUAL = 30.0      # Modified: per-point residual gate around median optical flow
FEATURE_CENTER_OFFSET_THRES = 999.0  # Natural follower mode: log offset but disable offset-triggered refine
UPWARD_FLOW_SCALE = 0.65       # Modified: damp small upward LK drift during near-stationary tracking
UPWARD_DAMPING_MAX_STEP = 35.0 # Modified: keep large real upward motion responsive

DEFAULT_TRACKER_PARAMS = {
    "ROI_MARGIN": ROI_MARGIN,
    "MIN_FLOW_FEATURES": MIN_FLOW_FEATURES,
    "LOW_FEATURE_LIMIT": LOW_FEATURE_LIMIT,
    "CENTER_SHIFT_THRES": CENTER_SHIFT_THRES,
    "AREA_CHANGE_THRES": AREA_CHANGE_THRES,
    "PERIODIC_DETECT_INTERVAL": PERIODIC_DETECT_INTERVAL,
    "MOTION_WEIGHT": MOTION_WEIGHT,
    "CONF_WEIGHT": CONF_WEIGHT,
    "SCALE_WEIGHT": SCALE_WEIGHT,
    "MAX_CENTER_DIST": MAX_CENTER_DIST,
}

GA_PARAM_BOUNDS = {
    "ROI_MARGIN": (2.0, 6.0),
    "MIN_FLOW_FEATURES": (3, 15),
    "LOW_FEATURE_LIMIT": (1, 8),
    "CENTER_SHIFT_THRES": (30, 160),
    "AREA_CHANGE_THRES": (0.20, 1.20),
    "PERIODIC_DETECT_INTERVAL": (15, 120),
    "MOTION_WEIGHT": (0.10, 0.80),
    "CONF_WEIGHT": (0.10, 0.80),
    "SCALE_WEIGHT": (0.05, 0.60),
    "MAX_CENTER_DIST": (80, 500),
}

INT_PARAMS = {"MIN_FLOW_FEATURES", "LOW_FEATURE_LIMIT", "CENTER_SHIFT_THRES", "PERIODIC_DETECT_INTERVAL", "MAX_CENTER_DIST"}

GA_ACCURACY_FIRST_REFERENCE = {
    # GA reference: high-accuracy 300-frame candidate before tightening YOLO-call pressure.
    "best_params": {
        "ROI_MARGIN": 2.9089548871391075,
        "MIN_FLOW_FEATURES": 12,
        "LOW_FEATURE_LIMIT": 5,
        "CENTER_SHIFT_THRES": 110,
        "AREA_CHANGE_THRES": 0.8095155195001188,
        "PERIODIC_DETECT_INTERVAL": 20,
        "MOTION_WEIGHT": 0.4933704311543616,
        "CONF_WEIGHT": 0.30841053699947657,
        "SCALE_WEIGHT": 0.19821903184616188,
        "MAX_CENTER_DIST": 80,
    },
    "best_metrics": {
        "frames": 300,
        "mean_iou": 0.776614991670214,
        "failure_frame_ratio": 0.0,
        "recovery_count": 0,
        "yolo_calls": 16,
        "yolo_calls_ratio": 0.05333333333333334,
        "fitness": 0.8552816583368806,
    },
}


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
    # Modified: compensate the detector's upward bbox bias without accumulating drift per frame.
    x, y, w, h = bbox
    return clip_bbox((x, y + int(h * YOLO_BBOX_Y_BIAS), w, h), frame_shape)


def select_motion_consistent_bbox_from_yolo(boxes, prev_bbox, frame_shape, offset=(0, 0)):
    """
    YOLO 후보 중 confidence만 보지 않고,
    이전 bbox와의 중심 거리와 크기 유사도까지 고려해 bbox를 선택한다.
    offset은 ROI crop 좌표를 원본 프레임 좌표로 복원하기 위한 값이다.
    """
    # Modified: choose the YOLO candidate that best matches previous motion/scale,
    # instead of always taking the highest-confidence detection.
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

        score = (
            CONF_WEIGHT * conf
            + MOTION_WEIGHT * motion_score
            + SCALE_WEIGHT * scale_score
        )

        if score > best_score:
            best_score = score
            best_bbox = apply_yolo_bbox_bias((x1, y1, bw, bh), frame_shape)
            best_conf = conf

    return best_bbox, best_conf


def yolo_detect(frame, model, roi_bbox=None, prev_bbox=None, conf=CONF_THRES):
    """
    roi_bbox가 있으면 ROI에서 먼저 YOLO를 수행한다.
    prev_bbox가 있으면 confidence + 위치 일관성 + 크기 일관성으로 후보를 선택한다.
    ROI 탐지 실패 시 full-frame 탐지로 fallback한다.
    """

    # 1) ROI 기반 탐지
    if roi_bbox is not None:
        rx, ry, rw, rh = expand_bbox(roi_bbox, frame.shape, ROI_MARGIN)
        crop = frame[ry:ry + rh, rx:rx + rw]

        if crop.size != 0:
            results = model.predict(source=crop, conf=conf, imgsz=IMGSZ, device=DEVICE, verbose=False)
            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                # Modified: use motion-consistent candidate selection during ROI re-detection.
                if prev_bbox is not None:
                    bbox, score = select_motion_consistent_bbox_from_yolo(
                        boxes,
                        prev_bbox,
                        frame.shape,
                        offset=(rx, ry),
                    )
                    if bbox is not None:
                        if score >= ROI_MIN_CONF_FOR_ACCEPT and is_roi_detection_scale_valid(prev_bbox, bbox):
                            return bbox, score, "roi"

                best_idx = boxes.conf.argmax().item()
                xyxy = boxes.xyxy[best_idx].cpu().numpy()
                score = float(boxes.conf[best_idx].cpu().item())
                x1, y1, x2, y2 = map(int, xyxy)
                bbox = (rx + x1, ry + y1, x2 - x1, y2 - y1)
                bbox = apply_yolo_bbox_bias(bbox, frame.shape)
                if score >= ROI_MIN_CONF_FOR_ACCEPT and is_roi_detection_scale_valid(prev_bbox, bbox):
                    return bbox, score, "roi"

    # 2) ROI 탐지 실패 시 full-frame 탐지
    results = model.predict(source=frame, conf=conf, imgsz=IMGSZ, device=DEVICE, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None, 0.0, "full_fail"

    # Modified: use the same motion-consistent selection for full-frame fallback.
    if prev_bbox is not None:
        bbox, score = select_motion_consistent_bbox_from_yolo(
            boxes,
            prev_bbox,
            frame.shape,
            offset=(0, 0),
        )
        if bbox is not None:
            return bbox, score, "full"

    best_idx = boxes.conf.argmax().item()
    xyxy = boxes.xyxy[best_idx].cpu().numpy()
    score = float(boxes.conf[best_idx].cpu().item())
    x1, y1, x2, y2 = map(int, xyxy)
    bbox = (x1, y1, x2 - x1, y2 - y1)
    return apply_yolo_bbox_bias(bbox, frame.shape), score, "full"

def make_lk_params(w, h):
    area = w * h
    if area < 200:
        feature_params = dict(maxCorners=FEATURE_MAX_CORNERS, qualityLevel=0.0005, minDistance=1, blockSize=3)
        lk_params = dict(
            winSize=(5, 5),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
    else:
        feature_params = dict(maxCorners=FEATURE_MAX_CORNERS, qualityLevel=0.01, minDistance=1, blockSize=7)
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03),
        )
    return feature_params, lk_params


def feature_roi_bbox(bbox, frame_shape, scale=FEATURE_ROI_SCALE):
    # Modified: focus LK feature extraction on the bbox center, not the full box/background edge.
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

    return gray, p0, feature_params, lk_params


def keep_points_near_bbox(points, bbox, frame_shape, margin=FEATURE_KEEP_MARGIN):
    # Modified: remove flow points that drift far outside the current target area.
    if points is None or len(points) == 0:
        return points

    rx, ry, rw, rh = expand_bbox(bbox, frame_shape, margin)
    x2 = rx + rw
    y2 = ry + rh
    flat = points.reshape(-1, 2)
    keep = (
        (flat[:, 0] >= rx)
        & (flat[:, 0] <= x2)
        & (flat[:, 1] >= ry)
        & (flat[:, 1] <= y2)
    )
    return flat[keep].reshape(-1, 1, 2)


def bbox_center_dist(bbox_a, bbox_b):
    # Modified: shared distance metric for YOLO jump filtering.
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b
    ac = np.array([ax + aw / 2, ay + ah / 2], dtype=np.float32)
    bc = np.array([bx + bw / 2, by + bh / 2], dtype=np.float32)
    return float(np.linalg.norm(ac - bc))


def bbox_area_change(bbox_a, bbox_b):
    # Modified: shared scale-change metric for YOLO candidate confirmation.
    _, _, aw, ah = bbox_a
    _, _, bw, bh = bbox_b
    area_a = max(aw * ah, 1)
    area_b = max(bw * bh, 1)
    return abs(area_a - area_b) / max(area_a, area_b, 1)


def is_roi_detection_scale_valid(prev_bbox, detected_bbox):
    # Natural follower guard: small ROI false positives shrink the tracker box.
    if prev_bbox is None or detected_bbox is None:
        return True
    _, _, pw, ph = prev_bbox
    _, _, dw, dh = detected_bbox
    prev_area = max(pw * ph, 1)
    detected_area = max(dw * dh, 1)
    return detected_area / prev_area >= ROI_MIN_AREA_RATIO


def blend_bbox(current_bbox, detected_bbox, frame_shape, alpha=YOLO_BBOX_BLEND_ALPHA):
    # Modified: soften accepted YOLO corrections so the visible box does not snap.
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
    # Modified: periodic YOLO is a refinement, so blend it to avoid abrupt visual bbox snaps.
    if redetect_reason == "periodic":
        return blend_bbox(current_bbox, detected_bbox, frame_shape, alpha=PERIODIC_YOLO_BLEND_ALPHA)
    return clip_bbox(detected_bbox, frame_shape)


def feature_center_offset_score(points, bbox):
    # Modified: detects when LK points are stable but concentrated away from bbox center.
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
    # Modified: small negative dy can accumulate while the UAV is nearly still, so damp only that drift.
    if dy < 0 and abs(dy) <= UPWARD_DAMPING_MAX_STEP:
        dy *= UPWARD_FLOW_SCALE
    return dx, dy


def apply_yolo_correction(current_bbox, yolo_bbox, frame_shape, redetect_reason, yolo_score):
    # Modified: periodic YOLO should refine softly; direct jumps are reserved for recovery.
    if redetect_reason in YOLO_FORCE_ACCEPT_REASONS:
        return clip_bbox(yolo_bbox, frame_shape)
    if redetect_reason == "center_jump" and yolo_score >= YOLO_HIGH_CONF_THRES:
        return clip_bbox(yolo_bbox, frame_shape)
    return blend_bbox(current_bbox, yolo_bbox, frame_shape)


def normalize_tracker_params(params):
    # GA entry point: merge a candidate with defaults and clamp it to valid ranges.
    merged = DEFAULT_TRACKER_PARAMS.copy()
    if params:
        merged.update(params)

    for key, (low, high) in GA_PARAM_BOUNDS.items():
        value = merged[key]
        value = max(low, min(high, value))
        if key in INT_PARAMS:
            value = int(round(value))
        else:
            value = float(value)
        merged[key] = value

    weight_sum = merged["MOTION_WEIGHT"] + merged["CONF_WEIGHT"] + merged["SCALE_WEIGHT"]
    if weight_sum > 1e-6:
        merged["MOTION_WEIGHT"] /= weight_sum
        merged["CONF_WEIGHT"] /= weight_sum
        merged["SCALE_WEIGHT"] /= weight_sum

    return merged


def apply_tracker_params(params):
    # GA entry point: existing follower code uses globals, so evaluation temporarily applies a candidate.
    global ROI_MARGIN, MIN_FLOW_FEATURES, LOW_FEATURE_LIMIT
    global CENTER_SHIFT_THRES, AREA_CHANGE_THRES, PERIODIC_DETECT_INTERVAL
    global MOTION_WEIGHT, CONF_WEIGHT, SCALE_WEIGHT, MAX_CENTER_DIST

    normalized = normalize_tracker_params(params)
    old_params = {
        "ROI_MARGIN": ROI_MARGIN,
        "MIN_FLOW_FEATURES": MIN_FLOW_FEATURES,
        "LOW_FEATURE_LIMIT": LOW_FEATURE_LIMIT,
        "CENTER_SHIFT_THRES": CENTER_SHIFT_THRES,
        "AREA_CHANGE_THRES": AREA_CHANGE_THRES,
        "PERIODIC_DETECT_INTERVAL": PERIODIC_DETECT_INTERVAL,
        "MOTION_WEIGHT": MOTION_WEIGHT,
        "CONF_WEIGHT": CONF_WEIGHT,
        "SCALE_WEIGHT": SCALE_WEIGHT,
        "MAX_CENTER_DIST": MAX_CENTER_DIST,
    }

    ROI_MARGIN = normalized["ROI_MARGIN"]
    MIN_FLOW_FEATURES = normalized["MIN_FLOW_FEATURES"]
    LOW_FEATURE_LIMIT = normalized["LOW_FEATURE_LIMIT"]
    CENTER_SHIFT_THRES = normalized["CENTER_SHIFT_THRES"]
    AREA_CHANGE_THRES = normalized["AREA_CHANGE_THRES"]
    PERIODIC_DETECT_INTERVAL = normalized["PERIODIC_DETECT_INTERVAL"]
    MOTION_WEIGHT = normalized["MOTION_WEIGHT"]
    CONF_WEIGHT = normalized["CONF_WEIGHT"]
    SCALE_WEIGHT = normalized["SCALE_WEIGHT"]
    MAX_CENTER_DIST = normalized["MAX_CENTER_DIST"]

    return old_params, normalized


def restore_tracker_params(old_params):
    global ROI_MARGIN, MIN_FLOW_FEATURES, LOW_FEATURE_LIMIT
    global CENTER_SHIFT_THRES, AREA_CHANGE_THRES, PERIODIC_DETECT_INTERVAL
    global MOTION_WEIGHT, CONF_WEIGHT, SCALE_WEIGHT, MAX_CENTER_DIST

    ROI_MARGIN = old_params["ROI_MARGIN"]
    MIN_FLOW_FEATURES = old_params["MIN_FLOW_FEATURES"]
    LOW_FEATURE_LIMIT = old_params["LOW_FEATURE_LIMIT"]
    CENTER_SHIFT_THRES = old_params["CENTER_SHIFT_THRES"]
    AREA_CHANGE_THRES = old_params["AREA_CHANGE_THRES"]
    PERIODIC_DETECT_INTERVAL = old_params["PERIODIC_DETECT_INTERVAL"]
    MOTION_WEIGHT = old_params["MOTION_WEIGHT"]
    CONF_WEIGHT = old_params["CONF_WEIGHT"]
    SCALE_WEIGHT = old_params["SCALE_WEIGHT"]
    MAX_CENTER_DIST = old_params["MAX_CENTER_DIST"]


def bbox_iou(bbox_a, bbox_b):
    if bbox_a is None or bbox_b is None:
        return 0.0

    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    union = max(aw * ah + bw * bh - inter, 1)
    return float(inter / union)


def default_gt_json_path(video_path=VIDEO_PATH):
    base, _ = os.path.splitext(video_path)
    candidate = base + ".json"
    return candidate if os.path.exists(candidate) else None


def load_gt_annotations(gt_json_path):
    if gt_json_path is None or not os.path.exists(gt_json_path):
        return None, None

    with open(gt_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_rect = data.get("gt_rect")
    exist = data.get("exist")
    if gt_rect is None:
        return None, None

    return gt_rect, exist


def compute_reacquisition_time(failure_flags):
    # Average length of failure runs that eventually recover.
    runs = []
    current = 0
    for failed in failure_flags:
        if failed:
            current += 1
        elif current > 0:
            runs.append(current)
            current = 0
    return float(np.mean(runs)) if runs else 0.0


def compute_fitness(metrics):
    # GA fitness: keep tracking accuracy dominant while penalizing failures and YOLO cost.
    mean_iou = metrics["mean_iou"] if metrics["mean_iou"] is not None else 0.0
    fps_score = metrics["fps_normalized"]
    return (
        1.00 * mean_iou
        - 0.80 * metrics["failure_frame_ratio"]
        - 0.40 * metrics["yolo_calls_ratio"]
        + 0.10 * fps_score
    )


def _safe_mean(values):
    numeric = [float(v) for v in values if v is not None]
    return float(np.mean(numeric)) if numeric else None


def _safe_max(values):
    numeric = [float(v) for v in values if v is not None]
    return float(np.max(numeric)) if numeric else None


def read_proc_cpu_snapshot():
    try:
        with open("/proc/stat", "r", encoding="utf-8") as f:
            parts = f.readline().strip().split()[1:]
        values = [float(part) for part in parts]
        idle = values[3] + (values[4] if len(values) > 4 else 0.0)
        return sum(values), idle
    except OSError:
        return None


def proc_cpu_percent(previous, current):
    if previous is None or current is None:
        return None
    prev_total, prev_idle = previous
    cur_total, cur_idle = current
    total_delta = cur_total - prev_total
    idle_delta = cur_idle - prev_idle
    if total_delta <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))


def read_proc_memory_mb():
    try:
        values = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                key, rest = line.split(":", 1)
                values[key] = float(rest.strip().split()[0]) / 1024.0
        total = values.get("MemTotal")
        available = values.get("MemAvailable")
        if total is None or available is None:
            return None, None
        return total - available, total
    except OSError:
        return None, None


def read_thermal_max_celsius():
    temps = []
    thermal_root = "/sys/class/thermal"
    if not os.path.isdir(thermal_root):
        return None

    for name in os.listdir(thermal_root):
        temp_path = os.path.join(thermal_root, name, "temp")
        try:
            with open(temp_path, "r", encoding="utf-8") as f:
                raw = float(f.read().strip())
            temps.append(raw / 1000.0 if raw > 1000.0 else raw)
        except (OSError, ValueError, TypeError, UnicodeError):
            continue
    return _safe_max(temps)


def parse_tegrastats_line(line):
    sample = {}

    ram_match = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
    if ram_match:
        sample["tegrastats_ram_used_mb"] = float(ram_match.group(1))
        sample["tegrastats_ram_total_mb"] = float(ram_match.group(2))

    gpu_match = re.search(r"GR3D_FREQ\s+(\d+)%", line)
    if gpu_match:
        sample["gpu_util_percent"] = float(gpu_match.group(1))

    cpu_match = re.search(r"CPU\s+\[([^\]]+)\]", line)
    if cpu_match:
        cpu_values = [float(value) for value in re.findall(r"(\d+)%@", cpu_match.group(1))]
        sample["tegrastats_cpu_percent"] = _safe_mean(cpu_values)

    temps = [float(value) for _, value in re.findall(r"([A-Za-z0-9_]+)@([\d.]+)C", line)]
    if temps:
        sample["tegrastats_temp_c"] = _safe_max(temps)

    return sample


class BenchmarkResourceSampler:
    def __init__(self, interval_sec=1.0, use_tegrastats=True):
        self.interval_sec = max(float(interval_sec), 0.2)
        self.use_tegrastats = use_tegrastats
        self.samples = []
        self.tegrastats_samples = []
        self._stop = threading.Event()
        self._thread = None
        self._tegrastats_process = None
        self._tegrastats_thread = None
        self._psutil = None
        self._previous_cpu = None

        try:
            import psutil  # type: ignore
            self._psutil = psutil
        except ImportError:
            self._psutil = None

    def start(self):
        self._previous_cpu = read_proc_cpu_snapshot()
        if self._psutil is not None:
            self._psutil.cpu_percent(interval=None)

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        self._start_tegrastats()

    def stop(self):
        self.samples.append(self._collect_system_sample())
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_sec + 0.5)
        self._stop_tegrastats()

    def _sample_loop(self):
        while not self._stop.is_set():
            self.samples.append(self._collect_system_sample())
            self._stop.wait(self.interval_sec)

    def _collect_system_sample(self):
        sample = {"time": time.time()}

        if self._psutil is not None:
            vm = self._psutil.virtual_memory()
            sample["cpu_percent"] = float(self._psutil.cpu_percent(interval=None))
            sample["memory_used_mb"] = float((vm.total - vm.available) / (1024 * 1024))
            sample["memory_total_mb"] = float(vm.total / (1024 * 1024))
        else:
            current_cpu = read_proc_cpu_snapshot()
            sample["cpu_percent"] = proc_cpu_percent(self._previous_cpu, current_cpu)
            self._previous_cpu = current_cpu
            used_mb, total_mb = read_proc_memory_mb()
            sample["memory_used_mb"] = used_mb
            sample["memory_total_mb"] = total_mb

        sample["thermal_max_c"] = read_thermal_max_celsius()
        return sample

    def _start_tegrastats(self):
        if not self.use_tegrastats or shutil.which("tegrastats") is None:
            return
        try:
            self._tegrastats_process = subprocess.Popen(
                ["tegrastats", "--interval", str(int(self.interval_sec * 1000))],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            self._tegrastats_process = None
            return

        self._tegrastats_thread = threading.Thread(target=self._read_tegrastats_loop, daemon=True)
        self._tegrastats_thread.start()

    def _read_tegrastats_loop(self):
        process = self._tegrastats_process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            if self._stop.is_set():
                break
            parsed = parse_tegrastats_line(line)
            if parsed:
                parsed["time"] = time.time()
                self.tegrastats_samples.append(parsed)

    def _stop_tegrastats(self):
        process = self._tegrastats_process
        if process is None:
            return
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
        if self._tegrastats_thread is not None:
            self._tegrastats_thread.join(timeout=1.0)

    def summary(self):
        cpu_values = [sample.get("cpu_percent") for sample in self.samples]
        mem_used_values = [sample.get("memory_used_mb") for sample in self.samples]
        mem_total_values = [sample.get("memory_total_mb") for sample in self.samples]
        thermal_values = [sample.get("thermal_max_c") for sample in self.samples]

        summary = {
            "resource_samples": len(self.samples),
            "cpu_percent_avg": _safe_mean(cpu_values),
            "cpu_percent_max": _safe_max(cpu_values),
            "memory_used_mb_avg": _safe_mean(mem_used_values),
            "memory_used_mb_max": _safe_max(mem_used_values),
            "memory_total_mb": _safe_max(mem_total_values),
            "thermal_max_c": _safe_max(thermal_values),
            "tegrastats_samples": len(self.tegrastats_samples),
        }

        if self.tegrastats_samples:
            gpu_values = [sample.get("gpu_util_percent") for sample in self.tegrastats_samples]
            ts_cpu_values = [sample.get("tegrastats_cpu_percent") for sample in self.tegrastats_samples]
            ts_mem_values = [sample.get("tegrastats_ram_used_mb") for sample in self.tegrastats_samples]
            ts_mem_total_values = [sample.get("tegrastats_ram_total_mb") for sample in self.tegrastats_samples]
            ts_temp_values = [sample.get("tegrastats_temp_c") for sample in self.tegrastats_samples]
            summary.update({
                "gpu_util_percent_avg": _safe_mean(gpu_values),
                "gpu_util_percent_max": _safe_max(gpu_values),
                "tegrastats_cpu_percent_avg": _safe_mean(ts_cpu_values),
                "tegrastats_ram_used_mb_avg": _safe_mean(ts_mem_values),
                "tegrastats_ram_used_mb_max": _safe_max(ts_mem_values),
                "tegrastats_ram_total_mb": _safe_max(ts_mem_total_values),
                "tegrastats_temp_c_max": _safe_max(ts_temp_values),
            })

        return summary


def create_debug_writer():
    # Debug only: record per-frame tracker state so jump causes can be inspected later.
    log_path = os.path.join(DEBUG_LOG_DIR, f"tracking_debug_{run_tag}.csv")
    csv_file = open(log_path, "w", newline="", encoding="utf-8")
    fieldnames = [
        "frame",
        "state",
        "redetect_reason",
        "feature_count",
        "low_feature_count",
        "flow_residual",
        "feature_center_offset",
        "area_change",
        "center_shift",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "iou",
        "is_failure",
        "yolo_action",
        "yolo_mode",
        "yolo_score",
        "yolo_calls",
        "fps",
        "capture_path",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    return csv_file, writer, log_path


def should_capture_debug_frame(state_text, redetect_reason, yolo_action, is_failure):
    # Debug only: save sparse images for frames that explain visible jumps.
    if is_failure:
        return True
    if yolo_action in {"hold", "fail"}:
        return True
    if state_text in {"Recovery", "Recovery-Fail"}:
        return True
    return redetect_reason in {"center_jump", "area_jump", "low_features"}


def evaluate_tracker(params, video_path=VIDEO_PATH, model_path=MODEL_PATH, gt_json_path=None, model=None, max_frames=None):
    """
    GA evaluation entry point.
    Runs one video with a candidate parameter set and returns tracking metrics.
    """
    old_params, applied_params = apply_tracker_params(params)
    cap = None
    start_time = None
    executor = None

    try:
        gt_rect, exist = load_gt_annotations(gt_json_path or default_gt_json_path(video_path))
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read the first frame for evaluation.")

        eval_model = model if model is not None else YOLO(model_path, task="detect")
        start_time = time.time()

        # ==================================================
        # v18 SEARCH -> TRACK
        # ==================================================
        search_frame = first_frame
        search_frame_index = 0
        search_yolo_calls = 0

        # Real-time input pacing for SEARCH state
        search_replay_start = time.perf_counter()

        search_confirm_hits = 0
        search_candidate_detections = 0
        search_candidate_resets = 0

        bbox = None
        score = 0.0
        mode = None

        while True:

            # --------------------------------------------------
            # SEARCH도 실제 20 FPS 입력 속도로 재생
            # --------------------------------------------------
            search_target_time = (
                search_replay_start
                + search_frame_index / REALTIME_INPUT_FPS
            )

            search_sleep_time = (
                search_target_time - time.perf_counter()
            )

            if search_sleep_time > 0:
                time.sleep(search_sleep_time)

            # frame 0 및 이후 INITIAL_SEARCH_INTERVAL마다
            # full-frame YOLO 검색
            if (
                search_frame_index == 0
                or search_frame_index % INITIAL_SEARCH_INTERVAL == 0
            ):
                bbox, score, mode = yolo_detect(
                    search_frame,
                    eval_model,
                    roi_bbox=None,
                )
                search_yolo_calls += 1

                if bbox is not None:
                    search_candidate_detections += 1
                    search_confirm_hits += 1

                    if search_confirm_hits >= SEARCH_CONFIRM_HITS:
                        break

                else:
                    # 중간 검색 시도에서 탐지가 끊기면
                    # 이전 후보는 연속 검출로 인정하지 않는다.
                    if search_confirm_hits > 0:
                        search_candidate_resets += 1

                    search_confirm_hits = 0

            # 다음 프레임으로 진행
            ret, next_frame = cap.read()

            if not ret:
                return {
                    "fitness": -1.0,
                    "params": applied_params,
                    "frames": 0,
                    "valid_gt_frames": 0,
                    "mean_iou": 0.0,
                    "failure_frame_ratio": 1.0,
                    "reacquisition_time": 0.0,
                    "reacquisition_time_normalized": 0.0,
                    "reacquisition_count": 0,
                    "recovery_count": 0,
                    "recovery_fail_count": 0,
                    "yolo_calls": search_yolo_calls,
                    "yolo_calls_ratio": 0.0,
                    "fps": 0.0,
                    "average_fps": 0.0,
                    "sys_fps": 0.0,
                    "end_to_end_fps": 0.0,
                    "fps_normalized": 0.0,
                    "initial_detection": False,
                    "acquisition_success": False,
                    "initial_search_interval": INITIAL_SEARCH_INTERVAL,
                    "initial_search_yolo_calls": search_yolo_calls,
                }

            search_frame_index += 1
            search_frame = next_frame

            if (
                max_frames is not None
                and search_frame_index >= max_frames
            ):
                return {
                    "fitness": -1.0,
                    "params": applied_params,
                    "frames": 0,
                    "valid_gt_frames": 0,
                    "mean_iou": 0.0,
                    "failure_frame_ratio": 1.0,
                    "reacquisition_time": 0.0,
                    "reacquisition_time_normalized": 0.0,
                    "reacquisition_count": 0,
                    "recovery_count": 0,
                    "recovery_fail_count": 0,
                    "yolo_calls": search_yolo_calls,
                    "yolo_calls_ratio": 0.0,
                    "fps": 0.0,
                    "average_fps": 0.0,
                    "sys_fps": 0.0,
                    "end_to_end_fps": 0.0,
                    "fps_normalized": 0.0,
                    "initial_detection": False,
                    "acquisition_success": False,
                    "initial_search_interval": INITIAL_SEARCH_INTERVAL,
                    "initial_search_yolo_calls": search_yolo_calls,
                }

        # ----------------------------------------------
        # 탐지 성공 → TRACK 상태 초기화
        # ----------------------------------------------
        initial_acquisition_frame = search_frame_index

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if src_fps <= 0:
            src_fps = REALTIME_INPUT_FPS

        initial_acquisition_time_s = (
            initial_acquisition_frame / float(src_fps)
        )

        # GT는 알고리즘 제어에는 사용하지 않고 평가용으로만 기록
        acquisition_gt_present = None

        if (
            exist is not None
            and initial_acquisition_frame < len(exist)
        ):
            acquisition_gt_present = bool(
                exist[initial_acquisition_frame]
            )

        # Optical Flow는 실제 탐지에 성공한 프레임에서 시작
        first_frame = search_frame

        x, y, w, h = bbox
        base_w, base_h = w, h
        smooth_bbox = (x, y, w, h)

        old_gray, p0, feature_params, lk_params = init_follower(
            first_frame,
            smooth_bbox
        )

        # frame_index는 원본 영상의 실제 frame 번호 유지
        frame_index = initial_acquisition_frame
        processed_frames = 0

        # --------------------------------------------------
        # Metric bookkeeping
        # SEARCH와 TRACK 시간을 분리해서 측정한다.
        # --------------------------------------------------
        track_start_time = time.time()

        low_feature_count = 0

        # SEARCH 단계의 실제 YOLO 호출도 전체 호출량에 포함
        yolo_call_count = search_yolo_calls

        prev_area = max(w * h, 1)
        prev_center = (x + w // 2, y + h // 2)
        pending_yolo_bbox = None
        iou_values = []
        failure_flags = []
        recovery_count = 0
        recovery_fail_count = 0

        # CPU Optical Flow와 GPU YOLO를 겹쳐 실행하는 단일 GPU worker.
        # YOLO 모델에는 한 worker만 접근시켜 thread-safety 문제를 최소화한다.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo-gpu")
        yolo_future = None
        yolo_request = None
        async_submit_count = 0
        async_apply_count = 0
        async_busy_skip_count = 0
        async_result_ages = []
        async_yolo_latencies_ms = []
        async_yolo_roi_latencies_ms = []
        async_yolo_full_latencies_ms = []
        async_yolo_roi_results = 0
        async_yolo_full_results = 0

        # 실제 20 FPS 카메라 입력을 모사
        replay_start = time.perf_counter()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and processed_frames >= max_frames:
                break

            frame_index += 1
            processed_frames += 1

            # Frame N은 실제 카메라 주기(20 FPS = 50 ms)에 맞춰 도착한다고 가정.
            # TRACK 시작점을 0으로 하는 상대 frame pacing
            track_relative_frame = (
                frame_index - initial_acquisition_frame
            )

            target_time = (
                replay_start
                + track_relative_frame / REALTIME_INPUT_FPS
            )
            sleep_time = target_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

            raw_frame = frame
            frame_gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)

            feature_count = 0
            flow_residual = 0.0
            feature_center_offset = 0.0
            x, y, w, h = smooth_bbox

            # CPU Optical Flow follower: same structure as main(), without visualization.
            if p0 is not None and len(p0) >= 3:
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                if p1 is not None and st is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]

                    if len(good_new) >= 3:
                        flow_vectors = good_new - good_old
                        median_flow = np.median(flow_vectors, axis=0)
                        distances = np.linalg.norm(flow_vectors - median_flow, axis=1)
                        flow_residual = float(np.percentile(distances, 75))
                        keep = distances < FLOW_KEEP_RESIDUAL
                        good_new = good_new[keep]
                        good_old = good_old[keep]
                        feature_count = len(good_new)

                        if flow_residual > FLOW_RESIDUAL_THRES:
                            old_gray = frame_gray.copy()
                            p0 = keep_points_near_bbox(good_new.reshape(-1, 1, 2), smooth_bbox, raw_frame.shape)
                            low_feature_count = LOW_FEATURE_LIMIT
                        elif feature_count >= MIN_FLOW_FEATURES:
                            px, py, pw, ph = smooth_bbox
                            prev_cx = px + pw / 2
                            prev_cy = py + ph / 2
                            dx, dy = np.median(good_new - good_old, axis=0)
                            dx, dy = damp_upward_flow(dx, dy)

                            smooth_cx = prev_cx + dx
                            smooth_cy = prev_cy + dy
                            smooth_w = BBOX_SIZE_ALPHA * pw + (1 - BBOX_SIZE_ALPHA) * base_w
                            smooth_h = BBOX_SIZE_ALPHA * ph + (1 - BBOX_SIZE_ALPHA) * base_h

                            x = int(smooth_cx - smooth_w / 2)
                            y = int(smooth_cy - smooth_h / 2)
                            w = int(smooth_w)
                            h = int(smooth_h)
                            smooth_bbox = clip_bbox((x, y, w, h), raw_frame.shape)
                            feature_center_offset = feature_center_offset_score(good_new, smooth_bbox)

                            old_gray = frame_gray.copy()
                            p0 = keep_points_near_bbox(good_new.reshape(-1, 1, 2), smooth_bbox, raw_frame.shape)
                            if p0 is not None and len(p0) < REFRESH_FEATURE_LIMIT:
                                old_gray, p0, feature_params, lk_params = init_follower(raw_frame, smooth_bbox)
                            low_feature_count = 0
                        else:
                            low_feature_count += 1
                    else:
                        low_feature_count += 1
                else:
                    low_feature_count += 1
            else:
                low_feature_count += 1

            # 이전 프레임에서 비동기로 요청한 GPU YOLO 결과 확인.
            async_applied_this_frame = False
            if yolo_future is not None and yolo_future.done():
                request = yolo_request

                try:
                    yolo_bbox, yolo_score, yolo_mode, yolo_latency_ms = yolo_future.result()

                    async_yolo_latencies_ms.append(yolo_latency_ms)

                    if yolo_mode == "roi":
                        async_yolo_roi_results += 1
                        async_yolo_roi_latencies_ms.append(yolo_latency_ms)

                    elif yolo_mode.startswith("full"):
                        async_yolo_full_results += 1
                        async_yolo_full_latencies_ms.append(yolo_latency_ms)

                except Exception as exc:
                    print(f"[ASYNC YOLO ERROR] {exc}")
                    yolo_bbox, yolo_score, yolo_mode = None, 0.0, "error"
                    yolo_latency_ms = None

                yolo_future = None
                yolo_request = None

                if request is not None:
                    result_age = frame_index - request["frame_index"]
                    async_result_ages.append(result_age)

                if yolo_bbox is not None and request is not None:
                    # YOLO는 과거 프레임을 본 결과이므로 그 사이 CPU follower가
                    # 이동한 만큼 bbox를 현재 프레임 좌표로 보상한다.
                    sx, sy, sw, sh = request["submit_bbox"]
                    cx, cy, cw, ch = smooth_bbox

                    submit_cx = sx + sw / 2.0
                    submit_cy = sy + sh / 2.0
                    current_cx = cx + cw / 2.0
                    current_cy = cy + ch / 2.0

                    shift_x = current_cx - submit_cx
                    shift_y = current_cy - submit_cy

                    yx, yy, yw, yh = yolo_bbox
                    compensated_bbox = clip_bbox(
                        (
                            int(round(yx + shift_x)),
                            int(round(yy + shift_y)),
                            int(yw),
                            int(yh),
                        ),
                        raw_frame.shape,
                    )

                    reason = request["reason"]
                    smooth_bbox = apply_redetection_bbox(
                        smooth_bbox,
                        compensated_bbox,
                        raw_frame.shape,
                        reason,
                    )

                    x, y, w, h = smooth_bbox
                    base_w, base_h = w, h

                    # YOLO 보정 결과가 반영된 현재 프레임에서 CPU follower 재초기화.
                    old_gray, p0, feature_params, lk_params = init_follower(
                        raw_frame, smooth_bbox
                    )
                    low_feature_count = 0
                    pending_yolo_bbox = None
                    async_apply_count += 1
                    async_applied_this_frame = True

                elif request is not None:
                    recovery_fail_count += 1

            x, y, w, h = smooth_bbox
            current_area = max(w * h, 1)
            center = (x + w // 2, y + h // 2)
            area_change = abs(current_area - prev_area) / max(prev_area, 1)
            center_shift = float(np.linalg.norm(np.array(center) - np.array(prev_center)))

            need_redetect = False
            redetect_reason = ""
            if low_feature_count >= LOW_FEATURE_LIMIT:
                need_redetect = True
                redetect_reason = "low_features"
            elif center_shift > CENTER_SHIFT_THRES:
                need_redetect = True
                redetect_reason = "center_jump"
            elif area_change > AREA_CHANGE_THRES:
                need_redetect = True
                redetect_reason = "area_jump"
            elif feature_center_offset > FEATURE_CENTER_OFFSET_THRES:
                need_redetect = True
                redetect_reason = "feature_offset"
            elif (frame_index - initial_acquisition_frame) % PERIODIC_DETECT_INTERVAL == 0:
                need_redetect = True
                redetect_reason = "periodic"

            # GPU YOLO 요청은 blocking하지 않는다.
            # CPU follower는 다음 프레임으로 즉시 진행한다.
            if need_redetect and not async_applied_this_frame:
                if yolo_future is None:
                    submit_bbox = tuple(smooth_bbox)

                    if redetect_reason != "periodic":
                        recovery_count += 1

                    yolo_future = executor.submit(
                        timed_yolo_detect,
                        raw_frame.copy(),
                        eval_model,
                        submit_bbox,
                        submit_bbox,
                    )
                    yolo_request = {
                        "frame_index": frame_index,
                        "reason": redetect_reason,
                        "submit_bbox": submit_bbox,
                    }

                    yolo_call_count += 1
                    async_submit_count += 1
                else:
                    # GPU가 이미 탐지 중이면 중복 요청을 쌓지 않는다.
                    async_busy_skip_count += 1

            x, y, w, h = smooth_bbox
            prev_area = max(w * h, 1)
            prev_center = (x + w // 2, y + h // 2)

            if gt_rect is not None and frame_index < len(gt_rect):
                gt_exists = True if exist is None or frame_index >= len(exist) else bool(exist[frame_index])
                gt_bbox = tuple(gt_rect[frame_index]) if gt_exists else None
                iou = bbox_iou(smooth_bbox, gt_bbox) if gt_bbox is not None else 0.0
                iou_values.append(iou)
                failure_flags.append(iou < 0.10 if gt_exists else False)
            else:
                failure_flags.append(False)

        # 마지막 프레임에서 GPU 작업이 남아 있으면 완료까지 기다린다.
        # 마지막 결과는 평가 프레임이 이미 끝났으므로 bbox에는 반영하지 않지만,
        # 처리시간에는 포함하여 FPS를 과대평가하지 않는다.
        if yolo_future is not None:
            try:
                yolo_future.result()
            except Exception as exc:
                print(f"[ASYNC YOLO FINAL ERROR] {exc}")
            yolo_future = None
            yolo_request = None

        # ==================================================
        # Correct SEARCH / TRACK / overall FPS bookkeeping
        # ==================================================

        end_time = time.time()

        overall_elapsed = max(end_time - start_time, 1e-6)
        track_elapsed = max(end_time - track_start_time, 1e-6)

        # acquisition frame도 SEARCH에서 실제 소비한 프레임이다.
        search_frames = int(initial_acquisition_frame + 1)

        # TRACK loop에서 실제 처리된 프레임 수
        track_frames = int(processed_frames)

        # 전체 영상에서 실제 소비한 프레임 수
        overall_frames = int(search_frames + track_frames)

        track_fps = track_frames / track_elapsed
        overall_fps = overall_frames / overall_elapsed

        # 이전 코드의 FPS 값도 비교/검증을 위해 보존
        legacy_fps = track_frames / overall_elapsed

        # --------------------------------------------------
        # GT-based acquisition latency
        # 알고리즘 제어에는 사용하지 않고 평가용으로만 사용.
        #
        # acquisition 시점이 속한 PRESENT episode의 시작점을
        # 찾아 target appearance -> confirmed acquisition 지연 측정.
        # --------------------------------------------------
        acquisition_episode_start_frame = None
        acquisition_delay_frames = None
        acquisition_delay_s = None

        if (
            acquisition_gt_present
            and exist is not None
            and initial_acquisition_frame < len(exist)
        ):
            episode_start = int(initial_acquisition_frame)

            while (
                episode_start > 0
                and bool(exist[episode_start - 1])
            ):
                episode_start -= 1

            acquisition_episode_start_frame = episode_start
            acquisition_delay_frames = int(
                initial_acquisition_frame - episode_start
            )

            acquisition_delay_s = float(
                acquisition_delay_frames / float(src_fps)
            )

        # 기존 변수명을 사용하는 아래 코드와의 호환성 유지.
        # 이제 fps는 올바른 전체 시스템 FPS를 의미한다.
        elapsed = overall_elapsed
        fps = overall_fps
        mean_iou = float(np.mean(iou_values)) if iou_values else None
        failure_frame_ratio = float(np.mean(failure_flags)) if failure_flags else 0.0
        failure_frames = int(np.sum(failure_flags)) if failure_flags else 0
        valid_gt_frames = len(iou_values)

        # Requested tracking metrics
        precision_iou_threshold = 0.50
        precision = float(np.mean([iou >= precision_iou_threshold for iou in iou_values])) if iou_values else 0.0

        success_thresholds = np.linspace(0.0, 1.0, 21)
        success_auc = float(np.mean([
            np.mean([iou >= th for iou in iou_values]) for th in success_thresholds
        ])) if iou_values else 0.0

        # Single-object optical-flow follower has no explicit ID assignment.
        # Therefore ID switch is defined as 0 unless a separate ID-based tracker is used.
        id_switch = 0

        sys_fps = fps
        end_to_end_fps = fps
        gpu_saved_ratio = 1.0 - (yolo_call_count / max(processed_frames, 1))

        reacquisition_time = compute_reacquisition_time(failure_flags)

        if mean_iou is None and processed_frames > 0:
            failure_frame_ratio = recovery_fail_count / processed_frames

        metrics = {
            "params": applied_params,
            "frames": processed_frames,
            "valid_gt_frames": valid_gt_frames,
            "sys_fps": sys_fps,
            "end_to_end_fps": end_to_end_fps,
            "precision": precision,
            "precision_percent": precision * 100.0,
            "success_auc": success_auc,
            "success_auc_percent": success_auc * 100.0,
            "mean_iou": mean_iou,
            "mean_iou_percent": mean_iou * 100.0 if mean_iou is not None else 0.0,
            "failure_frames": failure_frames,
            "failure_frame_ratio": failure_frame_ratio,
            "id_switch": id_switch,
            "gpu_saved_ratio": gpu_saved_ratio,
            "gpu_saved_ratio_percent": gpu_saved_ratio * 100.0,
            "reacquisition_time": reacquisition_time,
            "reacquisition_time_normalized": min(reacquisition_time / 30.0, 1.0),
            "reacquisition_count": recovery_count,
            "recovery_count": recovery_count,
            "recovery_fail_count": recovery_fail_count,
            "yolo_calls": yolo_call_count,
            "yolo_calls_ratio": yolo_call_count / max(processed_frames, 1),
            "async_cpu_gpu": True,
            "async_yolo_submissions": async_submit_count,
            "async_yolo_applied": async_apply_count,
            "async_yolo_busy_skips": async_busy_skip_count,
            "async_result_age_frames_avg": (
                float(np.mean(async_result_ages)) if async_result_ages else 0.0
            ),
            "async_result_age_frames_max": (
                int(max(async_result_ages)) if async_result_ages else 0
            ),
            "async_yolo_latency_ms_avg": (
                float(np.mean(async_yolo_latencies_ms))
                if async_yolo_latencies_ms else 0.0
            ),
            "async_yolo_latency_ms_min": (
                float(np.min(async_yolo_latencies_ms))
                if async_yolo_latencies_ms else 0.0
            ),
            "async_yolo_latency_ms_max": (
                float(np.max(async_yolo_latencies_ms))
                if async_yolo_latencies_ms else 0.0
            ),
            "async_yolo_latency_ms_p50": (
                float(np.percentile(async_yolo_latencies_ms, 50))
                if async_yolo_latencies_ms else 0.0
            ),
            "async_yolo_latency_ms_p95": (
                float(np.percentile(async_yolo_latencies_ms, 95))
                if async_yolo_latencies_ms else 0.0
            ),
            "async_yolo_roi_results": async_yolo_roi_results,
            "async_yolo_full_results": async_yolo_full_results,

            "async_yolo_roi_latency_ms_avg": (
                float(np.mean(async_yolo_roi_latencies_ms))
                if async_yolo_roi_latencies_ms else 0.0
            ),
            "async_yolo_roi_latency_ms_p50": (
                float(np.percentile(async_yolo_roi_latencies_ms, 50))
                if async_yolo_roi_latencies_ms else 0.0
            ),
            "async_yolo_roi_latency_ms_p95": (
                float(np.percentile(async_yolo_roi_latencies_ms, 95))
                if async_yolo_roi_latencies_ms else 0.0
            ),

            "async_yolo_full_latency_ms_avg": (
                float(np.mean(async_yolo_full_latencies_ms))
                if async_yolo_full_latencies_ms else 0.0
            ),
            "async_yolo_full_latency_ms_p50": (
                float(np.percentile(async_yolo_full_latencies_ms, 50))
                if async_yolo_full_latencies_ms else 0.0
            ),
            "async_yolo_full_latency_ms_p95": (
                float(np.percentile(async_yolo_full_latencies_ms, 95))
                if async_yolo_full_latencies_ms else 0.0
            ),

            # Corrected performance metrics
            "fps": overall_fps,
            "average_fps": overall_fps,
            "sys_fps": overall_fps,
            "end_to_end_fps": overall_fps,
            "fps_normalized": min(overall_fps / 30.0, 1.0),

            "overall_fps": float(overall_fps),
            "track_fps": float(track_fps),
            "legacy_fps": float(legacy_fps),

            "overall_elapsed_s": float(overall_elapsed),
            "track_elapsed_s": float(track_elapsed),

            "search_frames": int(search_frames),
            "track_frames": int(track_frames),
            "overall_frames": int(overall_frames),

            "acquisition_episode_start_frame": (
                int(acquisition_episode_start_frame)
                if acquisition_episode_start_frame is not None
                else None
            ),
            "acquisition_delay_frames": acquisition_delay_frames,
            "acquisition_delay_s": acquisition_delay_s,
            "initial_detection": bool(initial_acquisition_frame == 0),
            "acquisition_success": True,
            "initial_acquisition_frame": int(initial_acquisition_frame),
            "initial_acquisition_time_s": float(initial_acquisition_time_s),
            "initial_search_interval": INITIAL_SEARCH_INTERVAL,
            "initial_search_yolo_calls": int(search_yolo_calls),
            "search_confirm_required_hits": int(SEARCH_CONFIRM_HITS),
            "search_confirm_hits": int(search_confirm_hits),
            "search_candidate_detections": int(search_candidate_detections),
            "search_candidate_resets": int(search_candidate_resets),
            "acquisition_gt_present": acquisition_gt_present,
        }
        metrics["fitness"] = compute_fitness(metrics)
        return metrics
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        if cap is not None:
            cap.release()
        restore_tracker_params(old_params)


def evaluate_params(params, video_path=VIDEO_PATH, model_path=MODEL_PATH, gt_json_path=None, model=None, max_frames=None):
    """
    Evaluate one parameter set on one video using visible.json ground truth.
    This is the stable entry point for GA and manual parameter experiments.
    """
    metrics = evaluate_tracker(
        params,
        video_path=video_path,
        model_path=model_path,
        gt_json_path=gt_json_path,
        model=model,
        max_frames=max_frames,
    )

    # GA/reporting aliases: keep evaluate_tracker compatibility while exposing requested names.
    metrics["mean_iou"] = 0.0 if metrics["mean_iou"] is None else metrics["mean_iou"]
    metrics["average_fps"] = metrics.get("average_fps", metrics.get("fps", 0.0))
    metrics["recovery_count"] = metrics.get("recovery_count", metrics.get("reacquisition_count", 0))
    metrics["reacquisition_count"] = metrics.get("reacquisition_count", metrics["recovery_count"])
    metrics["fitness"] = compute_fitness(metrics)
    return metrics


def random_ga_candidate(rng):
    candidate = {}
    for key, (low, high) in GA_PARAM_BOUNDS.items():
        if key in INT_PARAMS:
            candidate[key] = int(rng.integers(low, high + 1))
        else:
            candidate[key] = float(rng.uniform(low, high))
    return normalize_tracker_params(candidate)


def mutate_ga_candidate(candidate, rng, mutation_rate=0.30):
    child = dict(candidate)
    for key, (low, high) in GA_PARAM_BOUNDS.items():
        if rng.random() > mutation_rate:
            continue
        span = high - low
        if key in INT_PARAMS:
            child[key] = int(round(child[key] + rng.normal(0, span * 0.10)))
        else:
            child[key] = float(child[key] + rng.normal(0, span * 0.10))
    return normalize_tracker_params(child)


def crossover_ga_candidates(parent_a, parent_b, rng):
    child = {}
    for key in GA_PARAM_BOUNDS:
        child[key] = parent_a[key] if rng.random() < 0.5 else parent_b[key]
    return normalize_tracker_params(child)


def ga_optimize(population_size=20, generations=16, elite_count=4, max_frames=None, seed=42,
                video_path=VIDEO_PATH, model_path=MODEL_PATH, gt_json_path=None):
    """
    GA scaffold. It repeatedly calls evaluate_params() and returns the best candidate.
    Keep population/generation small while iterating because each evaluation runs YOLO.
    """
    rng = np.random.default_rng(seed)
    model = YOLO(model_path, task="detect")
    population = [random_ga_candidate(rng) for _ in range(population_size)]
    history = []

    for generation in range(generations):
        scored = []
        for candidate in population:
            metrics = evaluate_params(
                candidate,
                video_path=video_path,
                model_path=model_path,
                gt_json_path=gt_json_path,
                model=model,
                max_frames=max_frames,
            )
            scored.append((metrics["fitness"], candidate, metrics))

        scored.sort(key=lambda item: item[0], reverse=True)
        history.append({
            "generation": generation,
            "best_fitness": scored[0][0],
            "best_params": scored[0][1],
            "best_metrics": scored[0][2],
        })
        print(f"[GA] generation={generation}, best_fitness={scored[0][0]:.4f}, metrics={scored[0][2]}")

        elites = [item[1] for item in scored[:elite_count]]
        next_population = elites.copy()
        while len(next_population) < population_size:
            parent_indices = rng.choice(len(elites), size=2, replace=True)
            child = crossover_ga_candidates(elites[parent_indices[0]], elites[parent_indices[1]], rng)
            child = mutate_ga_candidate(child, rng)
            next_population.append(child)
        population = next_population

    best = history[-1]
    return {
        "best_params": best["best_params"],
        "best_metrics": best["best_metrics"],
        "history": history,
    }


def parse_device_arg(device_value):
    if device_value is None:
        return DEVICE
    try:
        return int(device_value)
    except ValueError:
        return device_value


def run_benchmark(video_path=VIDEO_PATH, model_path=MODEL_PATH, gt_json_path=None, max_frames=None,
                  sample_interval=1.0, use_tegrastats=True):
    sampler = BenchmarkResourceSampler(interval_sec=sample_interval, use_tegrastats=use_tegrastats)
    started_at = datetime.now().isoformat(timespec="seconds")

    sampler.start()
    try:
        metrics = evaluate_params(
            DEFAULT_TRACKER_PARAMS,
            video_path=video_path,
            model_path=model_path,
            gt_json_path=gt_json_path,
            max_frames=max_frames,
        )
    finally:
        sampler.stop()

    return {
        "started_at": started_at,
        "video_path": video_path,
        "model_path": model_path,
        "gt_json_path": gt_json_path or default_gt_json_path(video_path),
        "device": DEVICE,
        "tracker_metrics": metrics,
        "resource_metrics": sampler.summary(),
    }


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, first_frame = cap.read()

    if not ret:
        print("영상을 읽을 수 없습니다.")
        return

    model = YOLO(MODEL_PATH, task="detect")

    # 1) YOLO 초기 탐지
    bbox, score, mode = yolo_detect(first_frame, model, roi_bbox=None)
    if bbox is None:
        print("YOLO 초기 탐지 실패")
        return

    x, y, w, h = bbox
    base_w, base_h = w, h
    smooth_bbox = (x, y, w, h)
    old_gray, p0, feature_params, lk_params = init_follower(first_frame, smooth_bbox)

    print(f"YOLO 초기 탐지 성공: bbox={smooth_bbox}, conf={score:.3f}, mode={mode}")

    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if src_fps <= 0:
            src_fps = 30
        frame_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        out = cv2.VideoWriter(SAVE_PATH, fourcc, src_fps, frame_size)
        print(f"저장 경로: {SAVE_PATH}")
    else:
        out = None

    mask = np.zeros_like(first_frame)
    color = np.random.randint(0, 255, (FEATURE_MAX_CORNERS, 3))

    frame_count = 0
    low_feature_count = 0
    yolo_call_count = 1
    last_time = time.time()

    prev_area = max(w * h, 1)
    prev_center = (x + w // 2, y + h // 2)
    pending_yolo_bbox = None
    pending_yolo_score = 0.0
    gt_rect, exist = load_gt_annotations(default_gt_json_path(VIDEO_PATH))
    debug_csv_file = None
    debug_writer = None

    if DEBUG_TRACKING_LOG:
        debug_csv_file, debug_writer, debug_log_path = create_debug_writer()
        print(f"Debug log path: {debug_log_path}")
        if DEBUG_CAPTURE_FRAMES:
            print(f"Debug frame dir: {DEBUG_FRAME_DIR}")

    cv2.namedWindow("Drone Tracking", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Modified: keep a clean frame for tracking/YOLO and draw overlays only on vis_frame.
        raw_frame = frame.copy()
        vis_frame = frame.copy()
        if DRAW_FLOW:
            # Modified: fade old flow trails every frame so they do not visually accumulate.
            mask = cv2.addWeighted(mask, FLOW_TRAIL_DECAY, np.zeros_like(mask), 0.0, 0.0)

        frame_count += 1
        now = time.time()
        fps = 1.0 / max(now - last_time, 1e-6)
        last_time = now

        frame_gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)

        feature_count = 0
        flow_residual = 0.0
        feature_center_offset = 0.0
        area_change = 0.0
        center_shift = 0.0
        state_text = "Stable"
        redetect_reason = ""
        yolo_action = "none"
        yolo_mode_text = ""
        yolo_score_value = ""

        x, y, w, h = smooth_bbox

        # 2) CPU optical flow follower
        if p0 is not None and len(p0) >= 3:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            if p1 is not None and st is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                if len(good_new) >= 3:
                    # Modified: reject points whose motion differs from the median motion.
                    flow_vectors = good_new - good_old
                    median_flow = np.median(flow_vectors, axis=0)
                    distances = np.linalg.norm(flow_vectors - median_flow, axis=1)
                    flow_residual = float(np.percentile(distances, 75))
                    keep = distances < FLOW_KEEP_RESIDUAL
                    good_new = good_new[keep]
                    good_old = good_old[keep]

                    feature_count = len(good_new)

                    if flow_residual > FLOW_RESIDUAL_THRES:
                        # Modified: many features are moving inconsistently, so do not drag the bbox.
                        old_gray = frame_gray.copy()
                        p0 = keep_points_near_bbox(good_new.reshape(-1, 1, 2), smooth_bbox, raw_frame.shape)
                        low_feature_count = LOW_FEATURE_LIMIT
                    elif feature_count >= MIN_FLOW_FEATURES:
                        px, py, pw, ph = smooth_bbox
                        prev_cx = px + pw / 2
                        prev_cy = py + ph / 2
                        dx, dy = np.median(good_new - good_old, axis=0)
                        dx, dy = damp_upward_flow(dx, dy)

                        # Modified: apply median LK displacement after upward drift damping.
                        smooth_cx = prev_cx + dx
                        smooth_cy = prev_cy + dy

                        # YOLO 재탐지 전까지는 크기를 급격히 바꾸지 않음
                        smooth_w = BBOX_SIZE_ALPHA * pw + (1 - BBOX_SIZE_ALPHA) * base_w
                        smooth_h = BBOX_SIZE_ALPHA * ph + (1 - BBOX_SIZE_ALPHA) * base_h

                        x = int(smooth_cx - smooth_w / 2)
                        y = int(smooth_cy - smooth_h / 2)
                        w = int(smooth_w)
                        h = int(smooth_h)
                        smooth_bbox = clip_bbox((x, y, w, h), raw_frame.shape)
                        feature_center_offset = feature_center_offset_score(good_new, smooth_bbox)

                        if DRAW_FLOW:
                            # Modified: draw flow only on visualization buffers, never on raw_frame.
                            for i, (new, old) in enumerate(zip(good_new, good_old)):
                                a, b = new.ravel()
                                c, d = old.ravel()
                                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i % len(color)].tolist(), 2)
                                vis_frame = cv2.circle(vis_frame, (int(a), int(b)), 2, color[i % len(color)].tolist(), -1)

                        old_gray = frame_gray.copy()
                        p0 = keep_points_near_bbox(good_new.reshape(-1, 1, 2), smooth_bbox, raw_frame.shape)
                        if p0 is not None and len(p0) < REFRESH_FEATURE_LIMIT:
                            # Modified: refresh features inside the current bbox, similar
                            # to the original follower's frequent feature re-capture.
                            old_gray, p0, feature_params, lk_params = init_follower(raw_frame, smooth_bbox)
                        low_feature_count = 0
                    else:
                        low_feature_count += 1
                else:
                    low_feature_count += 1
            else:
                low_feature_count += 1
        else:
            low_feature_count += 1

        x, y, w, h = smooth_bbox
        current_area = max(w * h, 1)
        center = (x + w // 2, y + h // 2)
        area_change = abs(current_area - prev_area) / max(prev_area, 1)
        center_shift = float(np.linalg.norm(np.array(center) - np.array(prev_center)))

        # 3) follower 상태 판단
        need_redetect = False
        if low_feature_count >= LOW_FEATURE_LIMIT:
            need_redetect = True
            state_text = "Recovery"
            redetect_reason = "low_features"
        elif center_shift > CENTER_SHIFT_THRES:
            need_redetect = True
            state_text = "Recovery"
            redetect_reason = "center_jump"
        elif area_change > AREA_CHANGE_THRES:
            need_redetect = True
            state_text = "Recovery"
            redetect_reason = "area_jump"
        elif feature_center_offset > FEATURE_CENTER_OFFSET_THRES:
            need_redetect = True
            state_text = "Refine"
            redetect_reason = "feature_offset"
        elif frame_count % PERIODIC_DETECT_INTERVAL == 0:
            need_redetect = True
            state_text = "Refine"
            redetect_reason = "periodic"
        else:
            state_text = "Stable" if feature_count >= 20 else "Unstable"

        # 4) 필요 시 YOLO 재탐지
        if need_redetect:
            yolo_bbox, yolo_score, yolo_mode = yolo_detect(
                raw_frame,
                model,
                roi_bbox=smooth_bbox,
                prev_bbox=smooth_bbox,
            )
            yolo_call_count += 1

            if yolo_bbox is not None:
                yolo_mode_text = yolo_mode
                yolo_score_value = yolo_score
                # Modified: periodic re-detection is smoothed to reduce visible bbox snapping.
                smooth_bbox = apply_redetection_bbox(smooth_bbox, yolo_bbox, raw_frame.shape, redetect_reason)
                x, y, w, h = smooth_bbox
                base_w, base_h = w, h
                old_gray, p0, feature_params, lk_params = init_follower(raw_frame, smooth_bbox)
                mask = np.zeros_like(raw_frame)
                low_feature_count = 0
                pending_yolo_bbox = None
                pending_yolo_score = 0.0
                yolo_action = "accept"
                state_text = f"YOLO-{yolo_mode}"
                print(f"[YOLO 재탐지] frame={frame_count}, reason={redetect_reason}, bbox={smooth_bbox}, conf={yolo_score:.3f}, mode={yolo_mode}")
            else:
                pending_yolo_bbox = None
                pending_yolo_score = 0.0
                yolo_action = "fail"
                yolo_mode_text = yolo_mode
                state_text = "Recovery-Fail"

        # 5) 상태 업데이트
        x, y, w, h = smooth_bbox
        prev_area = max(w * h, 1)
        prev_center = (x + w // 2, y + h // 2)
        current_iou = ""
        is_failure = False
        if gt_rect is not None and frame_count < len(gt_rect):
            gt_exists = True if exist is None or frame_count >= len(exist) else bool(exist[frame_count])
            if gt_exists:
                current_iou = bbox_iou(smooth_bbox, tuple(gt_rect[frame_count]))
                is_failure = current_iou < DEBUG_IOU_FAIL_THRES

        # 6) 시각화
        img = cv2.add(vis_frame, mask)

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(img, f"FPS: {fps:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"Features: {feature_count}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"Area Change: {area_change:.2f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"Center Shift: {center_shift:.1f}", (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"Low Features: {low_feature_count}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"YOLO Calls: {yolo_call_count}", (20, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"State: {state_text}", (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(img, "ESC: Exit", (20, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        capture_path = ""
        if DEBUG_TRACKING_LOG:
            should_capture = DEBUG_CAPTURE_FRAMES and should_capture_debug_frame(
                state_text,
                redetect_reason,
                yolo_action,
                is_failure,
            )
            if should_capture:
                # Debug only: save visual evidence for jump/failure frames.
                safe_reason = redetect_reason if redetect_reason else state_text
                capture_name = f"frame_{frame_count:06d}_{safe_reason}_{yolo_action}.jpg"
                capture_path = os.path.join(DEBUG_FRAME_DIR, capture_name)
                cv2.imwrite(capture_path, img)

            debug_writer.writerow({
                "frame": frame_count,
                "state": state_text,
                "redetect_reason": redetect_reason,
                "feature_count": feature_count,
                "low_feature_count": low_feature_count,
                "flow_residual": f"{flow_residual:.3f}",
                "feature_center_offset": f"{feature_center_offset:.3f}",
                "area_change": f"{area_change:.6f}",
                "center_shift": f"{center_shift:.3f}",
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "iou": "" if current_iou == "" else f"{current_iou:.6f}",
                "is_failure": int(is_failure),
                "yolo_action": yolo_action,
                "yolo_mode": yolo_mode_text,
                "yolo_score": "" if yolo_score_value == "" else f"{yolo_score_value:.6f}",
                "yolo_calls": yolo_call_count,
                "fps": f"{fps:.3f}",
                "capture_path": capture_path,
            })
            debug_csv_file.flush()

        cv2.imshow("Drone Tracking", img)

        if SAVE_VIDEO:
            out.write(img)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    if SAVE_VIDEO and out is not None:
        out.release()
        print(f"저장 완료: {SAVE_PATH}")
    if debug_csv_file is not None:
        debug_csv_file.close()


def run_cli():
    # GA/evaluation entry point: default behavior still runs the tracker video.
    parser = argparse.ArgumentParser(description="YOLO Observer + CPU Follower tracker")
    parser.add_argument("--mode", choices=["run", "eval", "ga", "bench"], default="run")
    parser.add_argument("--video-path", default=VIDEO_PATH)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--gt-json-path", default=None)
    parser.add_argument("--device", default=None, help="Ultralytics device value, e.g. 0, cpu, cuda:0")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--generations", type=int, default=16)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--no-tegrastats", action="store_true")
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args()

    global DEVICE
    DEVICE = parse_device_arg(args.device)

    if args.mode == "eval":
        metrics = evaluate_params(
            DEFAULT_TRACKER_PARAMS,
            video_path=args.video_path,
            model_path=args.model_path,
            gt_json_path=args.gt_json_path,
            max_frames=args.max_frames,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    if args.mode == "bench":
        report = run_benchmark(
            video_path=args.video_path,
            model_path=args.model_path,
            gt_json_path=args.gt_json_path,
            max_frames=args.max_frames,
            sample_interval=args.sample_interval,
            use_tegrastats=not args.no_tegrastats,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.report_json:
            with open(args.report_json, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"Benchmark report saved: {args.report_json}")
        return

    if args.mode == "ga":
        result = ga_optimize(
            population_size=args.population_size,
            generations=args.generations,
            elite_count=args.elite_count,
            max_frames=args.max_frames,
            seed=args.seed,
            video_path=args.video_path,
            model_path=args.model_path,
            gt_json_path=args.gt_json_path,
        )
        print(json.dumps({
            "best_params": result["best_params"],
            "best_metrics": result["best_metrics"],
        }, ensure_ascii=False, indent=2))
        return

    main()


if __name__ == "__main__":
    run_cli()
