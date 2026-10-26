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

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
import time
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
# 35 m 표적은 약 24x10 px라 LK 특징점이 구름에 쉽게 고착된다. 20 Hz에서
# 10프레임(0.5초) 보정은 3 km/h 표적도 READY 범위 대부분을 이동하고 난 뒤다.
# 5프레임 보정은 안전장치 적용 뒤에도 명중률 이득이 없었으므로, 안정 기준인
# 10프레임 주기를 유지한다.
PERIODIC_DETECT_INTERVAL = 10
# 비동기 ROI 추론이 끝날 때까지 3 km/h / 35 m 표적이 움직이는 양은 보통
# 수 픽셀이다. LK가 배경에 붙어 만든 큰 displacement를 YOLO 결과에 그대로
# 더하면 정확한 detector anchor까지 화면 밖으로 보내는 양의 피드백이 된다.
ASYNC_TRACK_COMPENSATION_MAX_PX = 6.0
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

# 탐지기가 알아보는 표적 크기의 상한(모델 입력 화소 기준).
#
# 2026-09-19 실측(yolo11s_ga_final-3/best.pt). 화각 6.9도로 10.8 m 드론을 찍은 영상(드론이 온전히
# 보이는 것)을 크기만 줄여 가며 넣었다. 최고 신뢰도(0.25 이상이면 탐지):
#   카메라 영상에서 드론 폭    imgsz 640      imgsz 960
#        181 px                0.00           0.00
#        135 px                0.34           0.00
#         90 px                0.72           0.27
#         63 px                0.76           0.75
#         45 px                0.74           0.76
# 모델 입력 기준 약 100 px 까지는 안정적이고, 135 px 에서 약해지며, 그 위로는 알아보지 못한다.
# 학습 데이터가 작고 먼 드론 위주라 가까이서 크게 찍힌 드론은 놓친다. 그래서 화각을 좁혀
# (배율을 높여) 100 m 를 잡으면, 같은 설정에서 가까운 표적(6.9도 화각 기준 약 24 m 이내)이
# 아예 탐지되지 않았다. 입력 크기를 조절해 표적이 이 상한 아래로 들어가게 한다.
MAX_MODEL_TARGET_PX = 100
# 표적 크기를 모를 때(첫 포착) 프레임마다 돌아가며 쓰는 입력 크기. 한 프레임에 한 번만
# 추론하므로 탐색 중 부하는 그대로이고, 최대 3 프레임 안에 크기와 무관하게 포착한다.
# 960: 작고 먼 표적(카메라 약 100 px 까지), 640: 중간(약 130 px 까지), 320: 가까운 큰 표적.
ACQUIRE_IMGSZ_CYCLE = (960, 640, 320)
MIN_IMGSZ = 256


def imgsz_for_target(source_max_side, target_max_side, cap):
    """source 이미지를 넣을 때 표적이 MAX_MODEL_TARGET_PX 이하가 되는 입력 크기.

    Ultralytics 는 source 의 긴 변을 imgsz 로 맞추므로 표적은 imgsz / source 배가 된다.
    cap 보다 크게 키우지는 않는다(작은 표적은 cap 그대로 = 이전 동작). 32 의 배수로 내림.
    """

    if target_max_side <= 0 or source_max_side <= 0:
        return int(cap)
    limit = source_max_side * MAX_MODEL_TARGET_PX / float(target_max_side)
    size = int(min(float(cap), limit)) // 32 * 32
    return max(MIN_IMGSZ, size)


_NON_PERIODIC_REDETECT_REASONS = {"low_features", "center_jump", "area_jump", "feature_offset"}

# A completed detector result older than this is no longer allowed to move the
# live tracker. CUDA inference is normally well below this limit. CPU-only
# inference must not apply multi-second-old corrections to a moving target.
ASYNC_RESULT_MAX_AGE_S = 1.0

# 진단 전용(동작 불변). SMASH_ASYNC_DIAG=<파일> 이면 비동기 YOLO 결과가 적용/폐기될 때마다 JSON 한 줄을
# 덧붙인다: 결과 나이, 제출 이후 지난 프레임 수, 보정이 필요했던 이동량과 실제 보정량, bbox 점프.
_ASYNC_DIAG_PATH = os.environ.get("SMASH_ASYNC_DIAG") or None


def _async_diag(event: dict) -> None:
    if _ASYNC_DIAG_PATH is None:
        return
    try:
        with open(_ASYNC_DIAG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + chr(10))
    except OSError:
        pass
SEARCH_CONFIRM_HITS = 2
RECOVERY_CONFIRM_MISSES = 3
# 단일 표적 조준경의 최초 포착은 사수가 십자선 부근에 둔 후보만 허용한다.
# D3D12 하늘/구름 렌더에서 화면 가장자리 false-positive가 높은 confidence로
# 선택되어 포탑을 시야 밖으로 끌고 가는 현상을 막는다.
SEARCH_CENTER_GATE_FRACTION = 0.40


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


def bbox_inside_search_center_gate(bbox, frame_shape, fraction=SEARCH_CENTER_GATE_FRACTION):
    """Whether a first-acquisition bbox is near the operator's crosshair."""

    x, y, w, h = bbox
    height, width = frame_shape[:2]
    cx, cy = x + w / 2.0, y + h / 2.0
    distance = float(np.hypot(cx - width / 2.0, cy - height / 2.0))
    return distance <= max(0.0, float(fraction)) * min(width, height)


def select_initial_bbox_from_yolo(boxes, frame_shape):
    """Highest-confidence candidate inside the central acquisition gate."""

    best = None
    best_conf = -1.0
    for i in range(len(boxes)):
        conf = float(boxes.conf[i].cpu().item())
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
        bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        if bbox_inside_search_center_gate(bbox, frame_shape) and conf > best_conf:
            best, best_conf = bbox, conf
    if best is None:
        return None, 0.0
    return apply_yolo_bbox_bias(best, frame_shape), best_conf


def yolo_detect(frame, model, device, conf_thres, roi_bbox=None, prev_bbox=None,
                full_frame_imgsz=None, limit_target_size=False):
    """ROI-cropped YOLO detection with motion-consistent candidate
    selection; falls back to full-frame detection if the ROI crop fails.

    full_frame_imgsz 는 전체 프레임 경로에만 적용하는 추론 입력 크기다. 작고 먼
    표적을 처음 포착할 때 필요하다 — 근거는 아래.

    Ultralytics 는 입력을 imgsz 로 레터박스하므로, ROI 경로는 작은 crop 이 기본값
    640 으로 **확대**되어 이미 고해상도 효과를 얻는다. 반면 전체 프레임 경로는
    640x640 카메라 영상이 그대로 640 으로 들어가 확대가 전혀 없다. 그래서 화면에서
    수십 화소인 표적은 stride-32 격자에 해상되지 않는다.

    35 m 정지 호버링 표적(화면상 24x10 px)을 실제 캡처 프레임으로 측정한 결과
    (2026-09-12, yolo11s_ga_final-3/best.pt, GTX 1050 Ti):

        imgsz  최고 신뢰도   전체프레임 추론시간
          640      0.020            32.5 ms      <- conf_thres 0.25 미달, 미탐지
          960      0.585            55.1 ms
         1280      0.701            93.3 ms
         1920      0.720                 -

    즉 임계값이나 가중치 문제가 아니라 입력 해상도 문제다. conf_thres 를 0.02 까지
    낮추면 "잡히기"는 하지만 잡음 수준이라 오탐이 쏟아지므로 해법이 아니다.
    기본값 960 은 신뢰도 여유(0.585)와 비용(+22.6 ms)을 맞바꾼 값이며, 이 비용은
    표적을 포착하기 전(탐색 중)에만 든다. 한 번 물면 ROI 경로로 넘어가 640 을 쓴다.
    """

    if roi_bbox is not None:
        rx, ry, rw, rh = expand_bbox(roi_bbox, frame.shape, ROI_MARGIN)
        crop = frame[ry:ry + rh, rx:rx + rw]

        if crop.size != 0:
            # crop 은 bbox 의 ROI_MARGIN 배라 보통 640 으로 확대하면 표적이 약 108 px 이 된다.
            # 표적이 커서 crop 이 화면 끝에 잘리면 그 비율이 깨져 상한을 넘으므로 입력을 줄인다.
            roi_kwargs = {}
            if limit_target_size:
                roi_kwargs["imgsz"] = imgsz_for_target(max(rw, rh), max(roi_bbox[2], roi_bbox[3]), 640)
            results = model.predict(source=crop, conf=conf_thres, device=device, verbose=False, **roi_kwargs)
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

    full_imgsz = full_frame_imgsz
    if limit_target_size and prev_bbox is not None and full_imgsz is not None:
        # 재포착: 직전 표적 크기를 알므로 상한에 맞춰 입력 크기를 고른다
        full_imgsz = imgsz_for_target(max(frame.shape[:2]), max(prev_bbox[2], prev_bbox[3]), full_imgsz)
    full_kwargs = {} if full_imgsz is None else {"imgsz": int(full_imgsz)}
    results = model.predict(source=frame, conf=conf_thres, device=device, verbose=False,
                            **full_kwargs)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None, 0.0

    if prev_bbox is not None:
        bbox, score = select_motion_consistent_bbox_from_yolo(boxes, prev_bbox, frame.shape, offset=(0, 0))
        if bbox is not None:
            return bbox, score

    return select_initial_bbox_from_yolo(boxes, frame.shape)


def make_lk_params(w, h, feature_override=None):
    area = w * h
    if area < 200:
        feature_params = dict(maxCorners=FEATURE_MAX_CORNERS, qualityLevel=0.0005, minDistance=1, blockSize=3)
        lk_params = dict(winSize=(5, 5), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    else:
        feature_params = dict(maxCorners=FEATURE_MAX_CORNERS, qualityLevel=0.01, minDistance=1, blockSize=7)
        lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03))
    if feature_override:
        # 작은 표적(수십 px)은 기본 품질 문턱(0.01, blockSize 7)에서 특징점이 7~9개뿐이라 최소 기준(6)에
        # 걸려 low_features 복구가 반복된다. 문턱을 낮추거나 블록을 줄이면 특징점이 늘어난다.
        feature_params = dict(feature_params, **feature_override)
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


def init_follower(frame, bbox, feature_override=None):
    x, y, w, h = clip_bbox(bbox, frame.shape)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    feature_params, lk_params = make_lk_params(w, h, feature_override)
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
    used_yolo: bool = False  # a completed YOLO result was applied this frame -- for latency profiling


class OpticalFlowTracker:
    """Stateful single-target tracker: LK optical flow every frame, YOLO
    only on init / every PERIODIC_DETECT_INTERVAL frames / on an actual
    tracking failure. See module docstring for validation numbers."""

    # __init__ 을 거치지 않고 만든 시험용 객체에서도 기존 상수로 동작하도록 클래스 기본값을 둔다.
    _periodic_detect_interval = PERIODIC_DETECT_INTERVAL
    _async_compensation_max_px = ASYNC_TRACK_COMPENSATION_MAX_PX
    _feature_override = None

    def __init__(self, model_path: str, device: "int | str" = 0, conf_thres: float = 0.25,
                 acquire_imgsz: "int | None" = 960, multi_scale: bool = True,
                 *, warmup: bool = True, async_result_max_age_s: float = ASYNC_RESULT_MAX_AGE_S,
                 periodic_detect_interval: int = PERIODIC_DETECT_INTERVAL,
                 async_compensation_max_px: float = ASYNC_TRACK_COMPENSATION_MAX_PX,
                 feature_quality_level: float = 0.0, feature_block_size: int = 0,
                 executor_factory=ThreadPoolExecutor):
        """acquire_imgsz: 표적을 아직 물지 않았을 때(전체 프레임 탐색) 쓰는 추론
        입력 크기. 작고 먼 표적 포착에 필요하며 근거는 yolo_detect 참고.
        None 이면 Ultralytics 기본값(640)을 써 이전 동작으로 되돌아간다.

        multi_scale: 참이면 표적이 탐지기의 크기 상한(MAX_MODEL_TARGET_PX)을 넘지 않게
        입력 크기를 조절한다. 첫 포착은 ACQUIRE_IMGSZ_CYCLE 을 프레임마다 돌아가며 쓰고,
        추적·재포착은 직전 표적 크기로 입력을 고른다. 작은 표적에서는 이전과 같다."""

        from ultralytics import YOLO  # heavy import kept lazy -- only paid if this tracker is actually used

        self._model = YOLO(model_path)
        self._device = device
        self._acquire_imgsz = acquire_imgsz
        self._conf_thres = conf_thres
        self._multi_scale = bool(multi_scale) and acquire_imgsz is not None
        self._acquire_cycle_index = 0

        # YOLO must never run on the ROS image callback.  A single worker also
        # serializes Ultralytics model access, which is not guaranteed to be
        # thread-safe.  update() only polls completed futures and never waits.
        self._executor = executor_factory(max_workers=1, thread_name_prefix="smash-yolo")
        self._future: Optional[Future] = None
        self._request = None
        self._generation = 0
        self._closed = False
        self._async_result_max_age_s = max(0.0, float(async_result_max_age_s))
        self._periodic_detect_interval = max(1, int(periodic_detect_interval))
        self._async_compensation_max_px = max(0.0, float(async_compensation_max_px))
        override = {}
        if float(feature_quality_level) > 0.0:
            override["qualityLevel"] = float(feature_quality_level)
        if int(feature_block_size) > 0:
            override["blockSize"] = int(feature_block_size)
        self._feature_override = override or None  # None = 기존 동작
        self._async_submitted = 0
        self._async_applied = 0
        self._async_stale = 0
        self._async_failed = 0
        self._search_hits = 0
        self._recovery_misses = 0

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

        if warmup:
            # Pay PyTorch/Ultralytics cold-start cost in the detector worker,
            # before the first live acquisition.  The largest acquisition size
            # warms all model layers without blocking ROS callbacks.
            warm_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            self._submit_detection(warm_frame, reason="warmup", roi_bbox=None,
                                   full_frame_imgsz=self._acquire_imgsz)

    def reset(self) -> None:
        """Drop all follower state so the next update() re-runs a full-frame
        YOLO acquisition, exactly as if this tracker had just been constructed.

        The loaded YOLO model is deliberately kept -- reset() is meant to be
        cheap enough to call the moment a target is judged lost, without
        paying a model reload.

        Why this exists: this follower never gives up on its own. Once
        _smooth_bbox is set, _track() always returns a FollowerResult, even
        when the target has left the frame entirely and the LK features are
        riding on background texture. A caller that needs a real
        "target lost -> reacquire from scratch" transition (e.g. the
        SEARCH state of a fire-control state machine) has to declare the
        loss itself and call this. See drone_sim/smash_fcs_node.py's
        _enter_search() for the reference caller.
        """

        self._generation += 1
        self._smooth_bbox = None
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
        self._search_hits = 0
        self._recovery_misses = 0
        # _acquire_cycle_index 는 일부러 초기화하지 않는다. smash_fcs_node 는 SEARCH 인 동안
        # 매 프레임 reset() 을 부르므로, 여기서 0 으로 되돌리면 입력 크기가 960 에 고정되어
        # 가까운 큰 표적을 영영 못 잡는다(2026-09-19 실측: 순환이 한 번도 돌지 않았다).

    def update(self, frame_bgr: np.ndarray) -> Optional[FollowerResult]:
        if self._closed:
            raise RuntimeError("OpticalFlowTracker is closed")
        if self._smooth_bbox is None:
            return self._initialize(frame_bgr)
        return self._track(frame_bgr)

    def _initialize(self, frame_bgr: np.ndarray) -> Optional[FollowerResult]:
        completed = self._take_completed_detection(frame_bgr)
        if completed is not None:
            request, bbox, score = completed
            if request["reason"] == "search" and bbox is not None:
                bbox = self._compensate_search_bbox(request, bbox, frame_bgr)
                self._search_hits += 1
                if self._search_hits >= SEARCH_CONFIRM_HITS:
                    x, y, w, h = bbox
                    self._smooth_bbox = (x, y, w, h)
                    self._base_w, self._base_h = w, h
                    self._old_gray, self._p0, self._lk_params = init_follower(frame_bgr, self._smooth_bbox, self._feature_override)
                    self._prev_area = max(w * h, 1)
                    self._prev_center = (x + w / 2, y + h / 2)
                    self._last_score = score
                    self._async_applied += 1
                    self._search_hits = 0
                    return FollowerResult(
                        bbox=self._smooth_bbox, score=score,
                        is_recovery_event=False, used_yolo=True,
                    )
            elif request["reason"] == "search":
                self._search_hits = 0

        # A warm-up or failed/stale search leaves the worker free.  Submit only
        # the newest frame; while the worker is busy, incoming frames are not
        # queued and the ROS callback remains non-blocking.
        if self._future is not None:
            return None
        imgsz = self._acquire_imgsz
        if self._multi_scale:
            cycle = (self._acquire_imgsz,) + tuple(s for s in ACQUIRE_IMGSZ_CYCLE[1:] if s < self._acquire_imgsz)
            imgsz = cycle[self._acquire_cycle_index % len(cycle)]
            self._acquire_cycle_index += 1
        self._submit_detection(frame_bgr, reason="search", roi_bbox=None,
                               full_frame_imgsz=imgsz)
        return None

    def _track(self, frame_bgr: np.ndarray) -> Optional[FollowerResult]:
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
                            self._old_gray, self._p0, self._lk_params = init_follower(frame_bgr, self._smooth_bbox, self._feature_override)
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
        elif self._frame_index % self._periodic_detect_interval == 0:
            redetect_reason = "periodic"

        completed = self._take_completed_detection(frame_bgr)
        lost = False
        applied_yolo = False
        applied_recovery = False
        if completed is not None:
            request, yolo_bbox, yolo_score = completed
            if request["reason"] not in ("search", "warmup"):
                if yolo_bbox is not None:
                    _before = self._smooth_bbox
                    _raw = yolo_bbox
                    yolo_bbox = self._compensate_track_bbox(
                        request, yolo_bbox, self._smooth_bbox, frame_bgr.shape,
                        self._async_compensation_max_px,
                    )
                    self._smooth_bbox = apply_redetection_bbox(
                        self._smooth_bbox, yolo_bbox, frame_bgr.shape, request["reason"]
                    )
                    if _ASYNC_DIAG_PATH is not None and request["submit_bbox"] is not None:
                        _sx, _sy, _sw, _sh = request["submit_bbox"]
                        _cx, _cy, _cw, _ch = _before
                        _need_x = (_cx + _cw / 2.0) - (_sx + _sw / 2.0)
                        _need_y = (_cy + _ch / 2.0) - (_sy + _sh / 2.0)
                        _ax, _ay = yolo_bbox[0] - _raw[0], yolo_bbox[1] - _raw[1]
                        _nx, _ny, _nw, _nh = self._smooth_bbox
                        _async_diag({
                            "ev": "apply", "wall": time.time(), "reason": request["reason"],
                            "age_s": time.perf_counter() - request["submitted_at"],
                            "frames": self._frame_index - request["frame_index"],
                            "need_shift": [_need_x, _need_y], "applied_shift": [_ax, _ay],
                            "cap_px": self._async_compensation_max_px,
                            "snap": [(_nx + _nw / 2.0) - (_cx + _cw / 2.0), (_ny + _nh / 2.0) - (_cy + _ch / 2.0)],
                            "bbox_before": list(_before), "bbox_after": [_nx, _ny, _nw, _nh],
                        })
                    self._async_applied += 1
                    applied_yolo = True
                    applied_recovery = request["reason"] in _NON_PERIODIC_REDETECT_REASONS
                    self._recovery_misses = 0
                    x, y, w, h = self._smooth_bbox
                    self._base_w, self._base_h = w, h
                    self._old_gray, self._p0, self._lk_params = init_follower(frame_bgr, self._smooth_bbox, self._feature_override)
                    self._low_feature_count = 0
                    self._last_score = yolo_score
                    redetect_reason = ""
                elif request["reason"] in _NON_PERIODIC_REDETECT_REASONS:
                    # The result describes an older frame. One miss cannot
                    # prove that the current target is gone; reliability will
                    # remain unrefreshed and block aim/fire while retries run.
                    self._recovery_misses += 1
                    lost = self._recovery_misses >= RECOVERY_CONFIRM_MISSES

        if _ASYNC_DIAG_PATH is not None and completed is not None and completed[1] is None:
            _async_diag({"ev": "miss", "wall": time.time(), "reason": completed[0]["reason"],
                         "recovery_misses": self._recovery_misses, "lost": bool(lost),
                         "frames": self._frame_index - completed[0]["frame_index"]})
        if redetect_reason and not lost and self._future is None:
            submit_bbox = tuple(self._smooth_bbox)
            self._submit_detection(
                frame_bgr, reason=redetect_reason, roi_bbox=submit_bbox,
                prev_bbox=submit_bbox, full_frame_imgsz=self._acquire_imgsz,
                limit_target_size=self._multi_scale,
            )

        if _ASYNC_DIAG_PATH is not None:
            _async_diag({"ev": "frame", "wall": time.time(), "idx": self._frame_index,
                         "pts": 0 if self._p0 is None else int(len(self._p0)),
                         "low": int(self._low_feature_count), "reason": redetect_reason,
                         "pending": self._future is not None, "lost": bool(lost),
                         "bbox": None if self._smooth_bbox is None else list(self._smooth_bbox)})
        if lost:
            self._smooth_bbox = None
            self._p0 = None
            self._old_gray = None
            self._low_feature_count = 0
            return None

        if self._smooth_bbox is None:
            return None

        x, y, w, h = self._smooth_bbox
        self._prev_area = max(w * h, 1)
        self._prev_center = (x + w / 2, y + h / 2)

        return FollowerResult(
            bbox=self._smooth_bbox, score=self._last_score,
            is_recovery_event=applied_recovery, used_yolo=applied_yolo,
        )

    def _submit_detection(self, frame_bgr, *, reason, roi_bbox,
                          prev_bbox=None, full_frame_imgsz=None,
                          limit_target_size=False) -> bool:
        if self._future is not None or self._closed:
            return False
        frame_copy = np.ascontiguousarray(frame_bgr.copy())
        request = {
            "reason": reason,
            "generation": self._generation,
            "submitted_at": time.perf_counter(),
            "submitted_wall": time.time(),
            "frame_index": self._frame_index,
            "frame": frame_copy,
            "submit_bbox": None if roi_bbox is None else tuple(roi_bbox),
        }
        self._future = self._executor.submit(
            yolo_detect, frame_copy, self._model, self._device, self._conf_thres,
            roi_bbox, prev_bbox, full_frame_imgsz, limit_target_size,
        )
        self._request = request
        self._async_submitted += 1
        return True

    def _take_completed_detection(self, current_frame):
        if self._future is None or not self._future.done():
            return None
        future, request = self._future, self._request
        self._future = self._request = None
        try:
            bbox, score = future.result()
        except Exception:
            self._async_failed += 1
            return None
        age_s = time.perf_counter() - request["submitted_at"]
        # Acquisition is allowed to be old: SEARCH has no live box to corrupt,
        # and _compensate_search_bbox advances it to the current frame.  The
        # age gate protects an already-running track from late corrections.
        too_old_for_live_track = (
            request["reason"] not in ("search", "warmup")
            and self._async_result_max_age_s > 0.0
            and age_s > self._async_result_max_age_s
        )
        if request["generation"] != self._generation or too_old_for_live_track:
            self._async_stale += 1
            _async_diag({"ev": "stale", "wall": time.time(), "reason": request["reason"], "age_s": age_s,
                         "frames": self._frame_index - request.get("frame_index", self._frame_index),
                         "generation_changed": request["generation"] != self._generation})
            return None
        return request, bbox, score

    @staticmethod
    def _compensate_track_bbox(request, detected_bbox, current_bbox, frame_shape,
                               max_shift_px: float = ASYNC_TRACK_COMPENSATION_MAX_PX):
        submit_bbox = request["submit_bbox"]
        if submit_bbox is None:
            return clip_bbox(detected_bbox, frame_shape)
        # The live LK box has advanced while inference ran.  Move the detector
        # result by the same center displacement before applying it.
        sx, sy, sw, sh = submit_bbox
        cx, cy, cw, ch = current_bbox
        shift_x = (cx + cw / 2.0) - (sx + sw / 2.0)
        shift_y = (cy + ch / 2.0) - (sy + sh / 2.0)
        shift_norm = float(np.hypot(shift_x, shift_y))
        if shift_norm > max_shift_px:
            scale = max_shift_px / shift_norm
            shift_x *= scale
            shift_y *= scale
        dx, dy, dw, dh = detected_bbox
        return clip_bbox(
            (int(round(dx + shift_x)), int(round(dy + shift_y)), dw, dh), frame_shape
        )

    @staticmethod
    def _compensate_search_bbox(request, detected_bbox, current_frame):
        """Advance an acquisition bbox from its captured frame to now with LK."""
        submitted = request["frame"]
        old_gray, points, lk = init_follower(submitted, detected_bbox)
        if points is None or len(points) < 3:
            return clip_bbox(detected_bbox, current_frame.shape)
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        moved, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, points, None, **lk)
        if moved is None or status is None:
            return clip_bbox(detected_bbox, current_frame.shape)
        good_new, good_old = moved[status == 1], points[status == 1]
        if len(good_new) < MIN_FLOW_FEATURES:
            return clip_bbox(detected_bbox, current_frame.shape)
        dx, dy = np.median(good_new - good_old, axis=0)
        x, y, w, h = detected_bbox
        return clip_bbox((int(round(x + dx)), int(round(y + dy)), w, h), current_frame.shape)

    def close(self) -> None:
        if not self._closed:
            self._executor.shutdown(wait=True)
            self._future = self._request = None
            self._closed = True
