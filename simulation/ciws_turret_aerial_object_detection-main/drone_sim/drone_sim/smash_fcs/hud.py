"""hud.py — SMASH 스마트 스코프 시연용 HUD 렌더러.

smash_fcs_node._render 가 이번 프레임의 상태를 HudFrame 으로 모아 넘기면, 여기서
영상 위에 그린다. 노드 로직과 그리기를 분리해 둔 것이다.

구성 요소
    - 스코프 비네팅과 듀플렉스 레티클(실제 소총 조준경 모양)
    - 표적 잠금 괄호. 포착 직후 크게 벌어졌다가 조여 드는 애니메이션
    - 상태 표시줄(SEARCH / TRACK / ALIGN / READY 색상 구분)
    - 리드 레티클과 정렬 유도선
    - 탄착 과녁 패널: 표적 중심 기준 탄착 분포(미터). 명중 반경 원을 함께 그려
      "얼마나 정확한가"를 한눈에 보인다.
    - 명중 히트마커, 격추 배너
    - 추적 신뢰도 게이지, 정렬 허용 원, 표적 궤적·속도, 격추 요약 카드(일반 관객용)
    - 상세 모드(디버그 수치). 시연 중에는 끄고 필요할 때만 켠다.

부하 기준: 640x640 에서 한 프레임 약 3~5 ms. 비네팅 마스크는 해상도별로 한 번만
만든다. 반투명 패널은 해당 영역만 어둡게 하므로 전체 프레임 합성보다 싸다.

텍스트는 ASCII 만 쓴다. OpenCV Hershey 폰트에는 한글 글리프가 없다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR
C_WHITE = (255, 255, 255)
C_BLACK = (0, 0, 0)
C_GREEN = (80, 255, 80)
C_YELLOW = (0, 230, 255)
C_ORANGE = (0, 150, 255)
C_RED = (60, 60, 255)
C_GREY = (170, 170, 170)
C_CYAN = (255, 220, 0)

RELIABILITY_COLORS = {"STABLE": C_GREEN, "HOLD": C_YELLOW, "LOST": C_RED}

STATE_COLORS = {
    "SEARCH": C_GREY,
    "TRACK": C_YELLOW,
    "ALIGN": C_ORANGE,
    "READY": C_GREEN,
}


@dataclass
class ImpactMark:
    """탄착 과녁 패널의 점 하나. 표적 중심 기준, 시선에 수직인 평면(미터)."""

    right_m: float
    up_m: float
    hit: bool


@dataclass
class Engagement:
    """격추 한 건의 요약. 잠금 시점부터 격추까지."""

    index: int
    seconds: float
    shots: int
    hits: int


@dataclass
class HudFrame:
    """한 프레임을 그리는 데 필요한 상태 묶음."""

    now: float
    principal_point: tuple[int, int]
    state: str  # SEARCH | TRACK | ALIGN | READY
    state_detail: str = ""  # 예: "12px", "NO TF"
    bbox_xyxy: Optional[tuple[float, float, float, float]] = None
    lock_age_s: Optional[float] = None  # 표적을 문 뒤 경과 시간. 잠금 애니메이션용
    lead_pixel: Optional[tuple[int, int]] = None
    range_m: Optional[float] = None
    range_src: str = ""
    tof_ms: Optional[float] = None
    hit_probability: Optional[float] = None
    fire_enabled: bool = False
    shots: int = 0
    hits: int = 0
    misses: int = 0
    kills: int = 0
    target_down: bool = False
    in_flight_pixels: Sequence[tuple[int, int]] = field(default_factory=list)
    recent_impacts: Sequence[tuple[tuple[int, int], str]] = field(default_factory=list)
    last_hit_age_s: Optional[float] = None
    kill_banner_age_s: Optional[float] = None
    impacts: Sequence[ImpactMark] = field(default_factory=list)
    hit_radius_m: float = 0.2
    debug: bool = False
    debug_lines: Sequence[str] = field(default_factory=list)
    reliability: str = ""  # STABLE | HOLD | LOST. 빈 문자열이면 게이지를 그리지 않는다
    quality: Optional[float] = None  # 추적 품질 0~1
    align_tol_px: Optional[float] = None  # 리드점이 이 반경 안에 들어오면 정렬 완료
    target_kmh: Optional[float] = None
    trail: Sequence[tuple[int, int]] = field(default_factory=list)  # 최근 표적 중심(오래된 것부터)
    engagement: Optional[Engagement] = None
    engagement_age_s: Optional[float] = None


class HudRenderer:
    def __init__(self) -> None:
        self._vignette_shape: Optional[tuple[int, int]] = None
        self._vignette: Optional[np.ndarray] = None

    # ── 공용 도형 ──────────────────────────────────────────────────────

    @staticmethod
    def _text(img, text, org, scale=0.5, color=C_WHITE, thick=1, outline=True):
        if outline:
            cv2.putText(img, text, org, FONT, scale, C_BLACK, thick + 2, cv2.LINE_AA)
        cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)

    @staticmethod
    def _darken(img, x1, y1, x2, y2, alpha=0.55):
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return
        roi = img[y1:y2, x1:x2]
        cv2.convertScaleAbs(roi, dst=roi, alpha=1.0 - alpha)

    def _apply_vignette(self, img) -> None:
        h, w = img.shape[:2]
        if self._vignette_shape != (h, w):
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            r = np.sqrt((xx - w / 2.0) ** 2 + (yy - h / 2.0) ** 2) / (min(w, h) / 2.0)
            # 반경 0.88 까지 원본, 1.0 에서 40 %, 그 밖은 25 %
            gain = np.clip(1.0 - (r - 0.88) / 0.12 * 0.6, 0.25, 1.0)
            gain[r > 1.0] = 0.25
            mask = (gain * 255.0).astype(np.uint8)
            self._vignette = cv2.merge([mask, mask, mask])
            self._vignette_shape = (h, w)
        cv2.multiply(img, self._vignette, dst=img, scale=1.0 / 255.0)

    # ── 구성 요소 ──────────────────────────────────────────────────────

    def _reticle(self, img, cx, cy, color) -> None:
        """듀플렉스 레티클: 바깥은 굵고 중앙은 가늘다. 중심점은 상태 색."""

        h, w = img.shape[:2]
        outer = int(min(w, h) * 0.47)
        inner = int(min(w, h) * 0.16)
        gap = 7
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cv2.line(img, (cx + dx * outer, cy + dy * outer), (cx + dx * inner, cy + dy * inner), C_BLACK, 3)
            cv2.line(img, (cx + dx * inner, cy + dy * inner), (cx + dx * gap, cy + dy * gap), C_BLACK, 1, cv2.LINE_AA)
        # 가는 선 위 밀도트(10 mrad 간격 근사 없이 균등 눈금)
        step = max(8, (inner - gap) // 4)
        for k in range(1, 4):
            d = gap + k * step
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                px, py = cx + dx * d, cy + dy * d
                cv2.circle(img, (px, py), 1, C_BLACK, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 2, color, -1, cv2.LINE_AA)

    def _brackets(self, img, bbox, color, lock_age_s) -> None:
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # 포착 직후 0.35 초 동안 2.5배에서 1배로 조여 든다
        scale = 1.0
        if lock_age_s is not None and lock_age_s < 0.35:
            scale = 1.0 + 1.5 * (1.0 - lock_age_s / 0.35)
        pad = 6
        hw = (bw / 2.0 + pad) * scale
        hh = (bh / 2.0 + pad) * scale
        l = max(6, int(0.35 * min(hw, hh) + 4))
        xa, xb = int(mx - hw), int(mx + hw)
        ya, yb = int(my - hh), int(my + hh)
        for (px, py, sx, sy) in ((xa, ya, 1, 1), (xb, ya, -1, 1), (xa, yb, 1, -1), (xb, yb, -1, -1)):
            for c, t in ((C_BLACK, 4), (color, 2)):
                cv2.line(img, (px, py), (px + sx * l, py), c, t, cv2.LINE_AA)
                cv2.line(img, (px, py), (px, py + sy * l), c, t, cv2.LINE_AA)

    def _lead(self, img, cx, cy, pixel, color, ready) -> None:
        px, py = pixel
        if not ready:
            # 점선 유도선
            dist = math.hypot(px - cx, py - cy)
            n = int(dist // 8)
            for i in range(0, n, 2):
                a = (int(cx + (px - cx) * i / max(n, 1)), int(cy + (py - cy) * i / max(n, 1)))
                b = (int(cx + (px - cx) * (i + 1) / max(n, 1)), int(cy + (py - cy) * (i + 1) / max(n, 1)))
                cv2.line(img, a, b, color, 1, cv2.LINE_AA)
        pts = np.array([[px, py - 9], [px + 9, py], [px, py + 9], [px - 9, py]], dtype=np.int32)
        cv2.polylines(img, [pts], True, C_BLACK, 3, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, color, 1 if not ready else 2, cv2.LINE_AA)
        self._text(img, "LEAD", (px + 12, py - 14), 0.36, color, 1)

    @staticmethod
    def _range_label(f: HudFrame) -> str:
        """우상단 거리. 관객용 기본 화면은 정수 미터로 보여 준다(소수점 이하는 추정 오차 안이라 흔들려 보인다). 상세 모드는 원래 값."""

        return f"RNG {f.range_m:5.1f} m" if f.debug else f"RNG {f.range_m:.0f} m"

    @staticmethod
    def _speed_label(f: HudFrame) -> Optional[str]:
        """표적 옆 추정 속도. 상세 모드(H 키)에서만 보인다.

        추정 속도는 표적이 일정하게 움직여도 3~24 km/h 로 흔들리고 방향을 바꿀 때 한동안 튄다(2026-10-18 시험).
        외부 관객에게는 시스템이 속도를 못 재는 것처럼 보이고 조준 결과를 이해하는 데 필요하지도 않아 기본 화면에서는 숨긴다.
        속도가 안정될 때까지 발사를 보류하는 동작은 상태 줄("SETTLING SPEED")과 한글 자막으로 계속 보여 준다.
        """

        if not f.debug or f.target_kmh is None or f.state == "SEARCH":
            return None
        return "HOVERING" if f.target_kmh < 0.8 else f"{f.target_kmh:.1f} km/h"

    @staticmethod
    def _friendly_detail(f: HudFrame) -> str:
        """상태 줄 부가 문구. 상세 모드가 아니면 내부 용어·픽셀 수치를 풀어 쓴다."""

        d = f.state_detail
        if f.debug or not d:
            return d
        if d.startswith("READY BLOCK"):
            return "SETTLING SPEED" if "MOTION" in d else "LOW CONFIDENCE"
        if d == "NO RANGE":
            return "RANGING"
        if d == "NO TF":
            return "SYNCING"
        if d.startswith("REL"):
            return "VERIFYING"
        if d.endswith("px"):
            return ""
        return d

    def _status_bar(self, img, f: HudFrame, color) -> None:
        label = {
            "SEARCH": "SEARCHING",
            "TRACK": "TRACKING",
            "ALIGN": "ALIGN",
            "READY": "FIRE READY",
        }.get(f.state, f.state)
        detail = self._friendly_detail(f)
        if detail:
            label = f"{label}  {detail}"
        if f.target_down:
            label, color = "TARGET DOWN", C_GREEN
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.62, 2)
        x, y = 12, 12
        self._darken(img, x, y, x + tw + 34, y + th + 16, 0.6)
        cv2.rectangle(img, (x, y), (x + tw + 34, y + th + 16), color, 1)
        blink = f.state == "READY" and int(f.now * 4) % 2 == 0
        cv2.circle(img, (x + 13, y + (th + 16) // 2), 5, color, -1 if not blink else 1, cv2.LINE_AA)
        self._text(img, label, (x + 26, y + th + 8), 0.62, color, 2, outline=False)

        # 우측 상단: 거리
        if f.range_m is not None:
            rng = HudRenderer._range_label(f)
            (rw, rh), _ = cv2.getTextSize(rng, FONT, 0.62, 2)
            w = img.shape[1]
            self._darken(img, w - rw - 28, y, w - 12, y + rh + 16, 0.6)
            self._text(img, rng, (w - rw - 20, y + rh + 8), 0.62, C_WHITE, 2, outline=False)
            if f.range_src and f.debug:
                self._text(img, f.range_src, (w - rw - 20, y + rh + 26), 0.38, C_GREY, 1)

    def _data_block(self, img, f: HudFrame) -> None:
        h = img.shape[0]
        lines = []
        tof = f"TOF {f.tof_ms:4.0f} ms" if f.tof_ms is not None else "TOF  -- ms"
        phit = f"P(HIT) {f.hit_probability:4.2f}" if f.hit_probability is not None else "P(HIT)  --"
        lines.append(f"{tof}   {phit}")
        if f.fire_enabled:
            judged = f.hits + f.misses
            acc = f"{100.0 * f.hits / judged:5.1f}%" if judged else "  --.-%"
            lines.append(f"SHOTS {f.shots:3d}  HIT {f.hits:3d}  ACC {acc}")
            lines.append(f"KILLS {f.kills:2d}")
        # 하단 14 px 는 윈도우 뷰어의 한글 자막 띠가 덮는 자리라 비워 둔다
        y0 = h - 58 - 18 * (len(lines) - 1)
        self._darken(img, 8, y0 - 16, 262, h - 48, 0.55)
        for i, s in enumerate(lines):
            self._text(img, s, (14, y0 + 18 * i), 0.45, C_WHITE, 1, outline=False)

    def _impact_panel(self, img, f: HudFrame) -> None:
        """탄착 과녁. 원 = 명중 반경. 최근 탄일수록 진하게."""

        w = img.shape[1]
        size = 132
        x1, y1 = w - size - 12, 58
        cx, cy = x1 + size // 2, y1 + size // 2 + 4
        self._darken(img, x1, y1, x1 + size, y1 + size + 26, 0.65)
        cv2.rectangle(img, (x1, y1), (x1 + size, y1 + size + 26), C_GREY, 1)
        self._text(img, "IMPACT (m)", (x1 + 6, y1 + 14), 0.38, C_GREY, 1, outline=False)
        ring_px = 34.0
        px_per_m = ring_px / max(f.hit_radius_m, 1e-3)
        cv2.circle(img, (cx, cy), int(ring_px), C_GREEN, 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), int(ring_px * 1.75), (90, 90, 90), 1, cv2.LINE_AA)
        cv2.line(img, (cx - 58, cy), (cx + 58, cy), (90, 90, 90), 1)
        cv2.line(img, (cx, cy - 58), (cx, cy + 58), (90, 90, 90), 1)
        # 표적(드론) 크기 감을 주는 작은 몸체 표시
        cv2.rectangle(img, (cx - 4, cy - 2), (cx + 4, cy + 2), C_GREY, -1)
        marks = list(f.impacts)[-40:]
        n = len(marks)
        for i, m in enumerate(marks):
            px = int(round(cx + m.right_m * px_per_m))
            py = int(round(cy - m.up_m * px_per_m))
            px = min(max(px, x1 + 3), x1 + size - 3)
            py = min(max(py, y1 + 18), y1 + size + 2)
            color = C_GREEN if m.hit else C_RED
            newest = i == n - 1
            cv2.circle(img, (px, py), 4 if newest else 2, color, -1, cv2.LINE_AA)
            if newest:
                cv2.circle(img, (px, py), 6, C_WHITE, 1, cv2.LINE_AA)
        # 요약: 평균 편차와 산포
        if n:
            rs = np.array([[m.right_m, m.up_m] for m in marks])
            mean = rs.mean(axis=0)
            spread = float(np.sqrt(((rs - mean) ** 2).sum(axis=1).mean()))
            off = float(np.hypot(*mean))
            self._text(img, f"OFF {off:.2f}  SPR {spread:.2f}", (x1 + 6, y1 + size + 20), 0.36, C_WHITE, 1, outline=False)
        else:
            self._text(img, "NO SHOTS", (x1 + 6, y1 + size + 20), 0.36, C_GREY, 1, outline=False)

    def _tolerance_ring(self, img, cx, cy, tol_px, color, ready) -> None:
        """리드 마름모가 이 원 안에 들어오면 정렬 완료. 숫자 대신 "넣으면 된다"를 보인다."""

        r = int(min(max(tol_px, 10.0), 70.0))
        if ready:
            cv2.circle(img, (cx, cy), r, C_BLACK, 4, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), r, color, 2, cv2.LINE_AA)
            return
        for k in range(0, 24, 2):  # 점선
            a0, a1 = k * 15, k * 15 + 9
            cv2.ellipse(img, (cx, cy), (r, r), 0, a0, a1, C_BLACK, 3, cv2.LINE_AA)
            cv2.ellipse(img, (cx, cy), (r, r), 0, a0, a1, color, 1, cv2.LINE_AA)

    @staticmethod
    def _trail(img, points, color) -> None:
        """표적 중심의 최근 궤적. 오래될수록 어둡고 가늘다."""

        n = len(points)
        if n < 2:
            return
        for i in range(1, n):
            k = i / (n - 1)
            c = tuple(int(v * (0.25 + 0.75 * k)) for v in color)
            cv2.line(img, tuple(points[i - 1]), tuple(points[i]), c, 1 if k < 0.6 else 2, cv2.LINE_AA)

    def _quality_gauge(self, img, f: HudFrame) -> None:
        """추적 신뢰도. SMASH 는 신뢰도가 충분할 때만 조준해를 믿고 발사한다."""

        if not f.reliability or f.state == "SEARCH" or f.target_down:
            return
        color = RELIABILITY_COLORS.get(f.reliability, C_GREY)
        q = f.quality
        if q is None:
            q = {"STABLE": 1.0, "HOLD": 0.45, "LOST": 0.0}.get(f.reliability, 0.0)
        q = float(min(max(q, 0.0), 1.0))
        x, y, w, bh = 12, 56, 150, 8
        self._darken(img, x - 4, y - 4, x + w + 4, y + bh + 30, 0.6)
        self._text(img, "CONFIDENCE", (x, y + 12), 0.36, C_GREY, 1, outline=False)
        by = y + 18
        cv2.rectangle(img, (x, by), (x + w, by + bh), C_GREY, 1)
        fill = int((w - 2) * q)
        if fill > 0:
            cv2.rectangle(img, (x + 1, by + 1), (x + 1 + fill, by + bh - 1), color, -1)
        (rw, _), _ = cv2.getTextSize(f.reliability, FONT, 0.36, 1)
        self._text(img, f.reliability, (x + w - rw, y + 12), 0.36, color, 1, outline=False)

    def _engagement_card(self, img, f: HudFrame) -> None:
        e, age = f.engagement, f.engagement_age_s
        if e is None or age is None or age > 5.0 or not f.target_down:
            return  # 새 표적이 나타나면 이전 교전 카드는 치운다
        h, w = img.shape[:2]
        acc = f"{100.0 * e.hits / e.shots:.0f}%" if e.shots else "--"
        lines = (
            f"ENGAGEMENT #{e.index}",
            f"TIME TO KILL {e.seconds:4.1f} s",
            f"SHOTS {e.shots}   HITS {e.hits}   ACC {acc}",
        )
        y0 = int(h * 0.74) - 92  # 격추 배너 위. 좌하단 데이터 블록과 겹치지 않게
        bw = 300
        x = (w - bw) // 2
        self._darken(img, x, y0 - 18, x + bw, y0 + 18 * len(lines) - 6, 0.6)
        cv2.rectangle(img, (x, y0 - 18), (x + bw, y0 + 18 * len(lines) - 6), C_GREEN, 1)
        for i, text in enumerate(lines):
            scale, color = (0.45, C_GREEN) if i == 0 else (0.5, C_WHITE)
            (tw, _), _ = cv2.getTextSize(text, FONT, scale, 1)
            self._text(img, text, ((w - tw) // 2, y0 + 18 * i), scale, color, 1, outline=False)

    @staticmethod
    def _hitmarker(img, cx, cy, age_s) -> None:
        if age_s is None or age_s > 0.3:
            return
        a, b = 10, 22
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            for c, t in ((C_BLACK, 5), (C_WHITE, 2)):
                cv2.line(img, (cx + sx * a, cy + sy * a), (cx + sx * b, cy + sy * b), c, t, cv2.LINE_AA)

    def _kill_banner(self, img, age_s) -> None:
        if age_s is None or age_s > 2.2:
            return
        h, w = img.shape[:2]
        text = "TARGET DESTROYED"
        scale = 1.0
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, 3)
        x, y = (w - tw) // 2, int(h * 0.74)
        self._darken(img, 0, y - th - 14, w, y + 14, 0.55)
        self._text(img, text, (x, y), scale, C_GREEN, 3)

    # ── 진입점 ─────────────────────────────────────────────────────────

    def render(self, frame_bgr: np.ndarray, f: HudFrame) -> np.ndarray:
        img = frame_bgr.copy()
        self._apply_vignette(img)
        cx, cy = f.principal_point
        color = STATE_COLORS.get(f.state, C_GREY)
        ready = f.state == "READY"

        self._reticle(img, cx, cy, color)

        tracking = f.bbox_xyxy is not None and not f.target_down
        if tracking and f.trail:
            self._trail(img, f.trail, C_CYAN)
        if f.lead_pixel is not None and not f.target_down and f.align_tol_px is not None:
            self._tolerance_ring(img, cx, cy, f.align_tol_px, color, ready)
        if tracking:
            self._brackets(img, f.bbox_xyxy, color, f.lock_age_s)
            speed = self._speed_label(f)
            if speed is not None:
                x1, y1, x2, y2 = f.bbox_xyxy
                self._text(img, speed, (int(x1), int(y2) + 20), 0.42, C_CYAN, 1)
        if f.lead_pixel is not None and not f.target_down:
            self._lead(img, cx, cy, f.lead_pixel, color, ready)

        for px in f.in_flight_pixels:
            cv2.circle(img, px, 2, C_YELLOW, -1, cv2.LINE_AA)
        for px, verdict in f.recent_impacts:
            if verdict == "HIT":
                cv2.circle(img, px, 10, C_GREEN, 2, cv2.LINE_AA)
            elif verdict == "MISS":
                cv2.line(img, (px[0] - 6, px[1] - 6), (px[0] + 6, px[1] + 6), C_RED, 2)
                cv2.line(img, (px[0] - 6, px[1] + 6), (px[0] + 6, px[1] - 6), C_RED, 2)

        self._hitmarker(img, cx, cy, f.last_hit_age_s)
        self._status_bar(img, f, color)
        self._quality_gauge(img, f)
        if f.fire_enabled:
            self._impact_panel(img, f)
        self._data_block(img, f)
        self._kill_banner(img, f.kill_banner_age_s)
        self._engagement_card(img, f)
        if f.debug:
            h = img.shape[0]
            y0 = 112
            self._darken(img, 8, y0 - 14, 330, y0 + 16 * len(f.debug_lines), 0.55)
            for i, s in enumerate(f.debug_lines):
                self._text(img, s, (12, y0 + 16 * i), 0.4, C_WHITE, 1, outline=False)
        return img


def impact_mark(
    projectile_point: np.ndarray, target_point: np.ndarray, launch_point: np.ndarray, hit: bool
) -> Optional[ImpactMark]:
    """탄착점을 표적 중심 기준, 시선에 수직인 평면의 (우, 상) 미터 좌표로 바꾼다.

    좌표계는 base_footprint(Z 위). 시선 u 에 대해 right = u x z, up = right x u.
    """

    u = np.asarray(target_point, dtype=np.float64) - np.asarray(launch_point, dtype=np.float64)
    n = float(np.linalg.norm(u))
    if n < 1e-6:
        return None
    u /= n
    right = np.cross(u, np.array([0.0, 0.0, 1.0]))
    rn = float(np.linalg.norm(right))
    if rn < 1e-6:
        return None
    right /= rn
    up = np.cross(right, u)
    d = np.asarray(projectile_point, dtype=np.float64) - np.asarray(target_point, dtype=np.float64)
    return ImpactMark(right_m=float(d @ right), up_m=float(d @ up), hit=hit)
