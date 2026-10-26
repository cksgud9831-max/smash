"""scope_panel.py — 스코프 영상 오른쪽의 관객용 상태 패널.

FCS 상태 배지, 표적 정보, 최근 30초 추적 이력 띠, 이벤트 기록, 격추·발사 준비 횟수를 보여 준다.
데이터는 smash_scope_viewer 의 /status(fcs_state·reliability·event·motion·kmh·range_m)뿐이라
WSL 쪽 노드를 바꾸지 않아도 된다.

StatusHistory 는 순수 파이썬이라 GUI 없이 시험할 수 있다. PanelRenderer 만 PIL 을 쓴다.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Optional

try:
    from .scope_caption import TONE_COLORS, caption_for, find_font
    from .scope_controls import FOOTER_H, FOOTER_HELP_H, PAD, SPAWN_MOTIONS, control_layout
except ImportError:  # 스크립트 폴더에서 직접 실행할 때
    from scope_caption import TONE_COLORS, caption_for, find_font
    from scope_controls import FOOTER_H, FOOTER_HELP_H, PAD, SPAWN_MOTIONS, control_layout

HISTORY_S = 30.0          # 추적 이력 띠가 보여 주는 시간
SAMPLE_KEEP_S = 40.0      # 표본 보관 시간(띠 창보다 조금 길게)
# 격추는 발사 준비와 같은 초록이라 구분이 안 된다. 패널에서는 하늘색으로 따로 쓴다.
PHASE_COLORS = {**TONE_COLORS, "kill": (255, 220, 110)}
PHASE_CONFIRM_S = 0.4     # 이 시간보다 짧은 상태 깜빡임은 이벤트로 남기지 않는다
EVENT_KEEP = 40

# 띠·범례에 쓰는 단계. tone(caption_for) 과 같은 키를 쓴다.
PHASE_LABELS = {
    "idle": "탐색",
    "track": "추적",
    "align": "정렬",
    "ready": "발사 준비",
    "warn": "신뢰도 저하",
    "kill": "격추",
}
# 이벤트 기록 문구. 탐색(idle)은 "복귀"일 때만 남긴다.
PHASE_EVENTS = {
    "track": "표적 포착",
    "align": "조준선 정렬",
    "ready": "발사 준비 완료",
    "warn": "추적 신뢰도 저하 — 발사 보류",
    "kill": "표적 격추",
}
MOTION_NAMES = {
    "none": "정지 (호버링)",
    "straight": "직선 통과",
    "cross": "좌우 왕복",
    "maneuver": "8자 기동",
    "approach": "접근",
}
TARGET_STATE_NAMES = {
    "MOVE": "이동 중",
    "WAIT LOCK": "잠금 대기",
    "WAIT": "출발 대기",
    "HOLD": "정지",
    "DOWN": "격추됨",
}


def range_text(status: dict) -> str:
    """HUD 우상단 RNG 와 같은 값(FCS 측정 거리, 정수 m). 측정이 없으면 "--".

    status["range_m"] 은 target_mover 의 시나리오 값이라 HUD 와 달라 쓰지 않는다.
    옛 WSL 노드(hud_range_m 키 없음)면 시나리오 값을 "설정" 으로 표시해 HUD 와 혼동하지 않게 한다.
    """

    if "hud_range_m" in status:
        value = status["hud_range_m"]
        return "--" if value is None else f"{value:.0f} m"
    value = status.get("range_m")
    return f"설정 {value:g} m" if value else "--"


def phase_of(status: dict) -> str:
    """FCS 상태 -> 단계 키. 자막과 같은 판정(caption_for)을 따른다."""

    cap = caption_for(status)
    return cap[1] if cap else "idle"


class StatusHistory:
    """/status 표본을 쌓아 추적 이력 띠·이벤트 기록·횟수를 만든다."""

    def __init__(self) -> None:
        self.samples: deque[tuple[float, str]] = deque()
        self.events: deque[tuple[float, str, str]] = deque(maxlen=EVENT_KEEP)  # (시각, 단계, 문구)
        self.kills = 0
        self.ready_count = 0
        self._origin: Optional[float] = None
        self._confirmed = "idle"          # 이벤트 기준으로 확정된 단계
        self._pending: Optional[tuple[str, float]] = None
        self._last_kill_age: Optional[float] = None
        self.lock = threading.Lock()      # 그리는 스레드와 갱신하는 스레드가 같은 자료를 본다

    def update(self, status: dict, now: float) -> None:
        with self.lock:
            self._update(status, now)

    def _update(self, status: dict, now: float) -> None:
        if self._origin is None:
            self._origin = now
        if not status:  # 상태를 못 받은 동안은 탐색으로 기록하되 이벤트는 만들지 않는다
            self.samples.append((now, "idle"))
            return
        phase = phase_of(status)
        self.samples.append((now, phase))
        while self.samples and now - self.samples[0][0] > SAMPLE_KEEP_S:
            self.samples.popleft()

        # 격추 횟수: HUD 의 KILLS 와 같은 숫자(hud_kills)가 오면 그것을 쓴다. 뷰어를 늦게 켜도, 통계를 초기화해도 HUD 와 맞는다.
        # 없으면(옛 WSL 노드) 이벤트 나이가 줄어들었을 때를 새 격추로 센다.
        age = status.get("event_age_s")
        if status.get("event") == "destroyed" and age is not None:
            if self._last_kill_age is None or age < self._last_kill_age - 0.01:
                self.kills += 1
            self._last_kill_age = age
        if status.get("hud_kills") is not None:
            self.kills = int(status["hud_kills"])

        # 단계가 PHASE_CONFIRM_S 이상 유지됐을 때만 이벤트로 확정한다(READY 깜빡임 방지)
        if phase == self._confirmed:
            self._pending = None
            return
        if self._pending is None or self._pending[0] != phase:
            self._pending = (phase, now)
            return
        if now - self._pending[1] < PHASE_CONFIRM_S:
            return
        previous, self._confirmed, self._pending = self._confirmed, phase, None
        text = PHASE_EVENTS.get(phase)
        if phase == "idle" and previous != "kill":
            text = "탐색으로 복귀"
        if text:
            self.events.append((now, phase, text))
        if phase == "ready":
            self.ready_count += 1

    def since_start(self, t: float) -> float:
        return 0.0 if self._origin is None else max(0.0, t - self._origin)

    def timeline(self, now: float, window_s: float = HISTORY_S):
        """(시작 비율, 끝 비율, 단계) 구간 목록. 0 = window_s 전, 1 = 지금."""

        segs: list[tuple[float, float, str]] = []
        items = list(self.samples)
        for i, (t, phase) in enumerate(items):
            end = items[i + 1][0] if i + 1 < len(items) else now
            lo, hi = max(t, now - window_s), min(end, now)
            if hi <= lo:
                continue
            a, b = (lo - (now - window_s)) / window_s, (hi - (now - window_s)) / window_s
            if segs and segs[-1][2] == phase and abs(segs[-1][1] - a) < 1e-6:
                segs[-1] = (segs[-1][0], b, phase)
            else:
                segs.append((a, b, phase))
        return segs


class PanelRenderer:
    """PIL 로 패널을 그린다. 글꼴이 없으면 available 이 False 라 호출 쪽이 옛 패널로 돌아간다."""

    WIDTH = 290
    REDRAW_S = 0.2

    def __init__(self) -> None:
        self._font_path = find_font()
        self.available = self._font_path is not None
        self._fonts: dict[int, object] = {}
        self._cache = None
        self._busy = False
        self._cache_t = -1e9

    def _font(self, size: int):
        font = self._fonts.get(size)
        if font is None:
            from PIL import ImageFont

            font = ImageFont.truetype(self._font_path, size)
            self._fonts[size] = font
        return font

    def render(self, height: int, status: dict, history: StatusHistory, now: float,
               link_state: str, fps: float, recording: bool, rec_elapsed: float, notice: str,
               show_help: bool, controls=None):
        """(height, WIDTH, 3) BGR 배열. 그리는 데 십수 ms 가 걸려 영상 표시 박자가 깨지므로,
        그림은 별도 스레드에서 0.2 초마다 새로 만들고 여기서는 마지막 그림을 바로 돌려준다."""

        args = (height, dict(status), history, now, link_state, fps, recording, rec_elapsed, notice, show_help,
                None if controls is None else (controls.motion, controls.kmh, now < controls.flash_until))
        if self._cache is None:
            self._cache, self._cache_t = self._draw(*args), now  # 첫 그림만 동기로
            return self._cache
        if now - self._cache_t >= self.REDRAW_S and not self._busy:
            self._busy = True
            self._cache_t = now

            def work():
                try:
                    self._cache = self._draw(*args)
                finally:
                    self._busy = False

            threading.Thread(target=work, daemon=True).start()
        return self._cache

    def _draw_controls(self, d, text, width_of, layout, controls, status, w) -> None:
        motion, kmh, flashing = controls
        enabled = status.get("spawn_enabled", True)
        pad = PAD
        d.line((pad, layout["top"] - 4, w - pad, layout["top"] - 4), fill=(48, 54, 60))
        text((pad, layout["top"]), "표적 생성", 12, (130, 138, 148))

        def button(rect, label, size, fill, fg=(235, 238, 242)):
            d.rounded_rectangle(rect, radius=6, fill=fill)
            x0, y0, x1, y1 = rect
            lw = width_of(label, size)
            text(((x0 + x1 - lw) // 2, (y0 + y1) // 2 - size // 2 - 2), label, size, fg)

        for name, label in SPAWN_MOTIONS:
            on = name == motion
            button(layout["motion:" + name], label, 13, (46, 108, 190) if on else (36, 42, 48),
                   (255, 255, 255) if on else (170, 178, 188))
        button(layout["speed:-"], "-", 18, (36, 42, 48))
        button(layout["speed:+"], "+", 18, (36, 42, 48))
        x0, y0, x1, y1 = layout["speed:value"]
        label = f"{kmh:g} km/h"
        text(((x0 + x1 - width_of(label, 15)) // 2, y0 + 4), label, 15, (235, 238, 242))
        down = status.get("state") == "DOWN"
        if not enabled:
            button(layout["spawn"], "생성 불가 (격추 연출 꺼짐)", 13, (40, 44, 48), (130, 138, 148))
        else:
            fill = (90, 200, 120) if flashing else (40, 150, 80) if down else (46, 108, 190)
            button(layout["spawn"], "표적 즉시 생성" if down else "표적 새로 생성", 15, fill, (255, 255, 255))

    def _draw(self, height: int, status: dict, history: StatusHistory, now: float,
              link_state: str, fps: float, recording: bool, rec_elapsed: float, notice: str,
              show_help: bool, controls=None):
        import numpy as np

        with history.lock:
            events = list(history.events)
            segments = history.timeline(now)
            kills, ready_count = history.kills, history.ready_count
            origin = history.since_start
        from PIL import Image, ImageDraw

        w = self.WIDTH
        img = Image.new("RGB", (w, height), (18, 21, 24))
        d = ImageDraw.Draw(img)

        def rgb(bgr):
            return (bgr[2], bgr[1], bgr[0])

        def text(xy, s, size, fill=(225, 228, 232)):
            d.text(xy, s, font=self._font(size), fill=fill)

        def width_of(s, size):
            l, _, r, _ = d.textbbox((0, 0), s, font=self._font(size))
            return r - l

        pad = PAD
        # ── 제목 ──
        text((pad, 12), "SMASH", 24, (245, 247, 250))
        text((pad + 92, 21), "SMART SCOPE FCS", 11, (130, 138, 148))
        d.line((pad, 48, w - pad, 48), fill=(48, 54, 60))

        # ── 상태 배지 ──
        cap = caption_for(status)
        label, tone = cap if cap else ("연결 대기 중", "idle")
        accent = rgb(PHASE_COLORS.get(tone, PHASE_COLORS["idle"]))
        d.rounded_rectangle((pad, 56, w - pad, 120), radius=8, fill=tuple(int(c * 0.22) + 14 for c in accent))
        d.rounded_rectangle((pad, 56, pad + 6, 120), radius=3, fill=accent)
        text((pad + 18, 61), "FCS 상태", 11, tuple(int(c * 0.7) + 40 for c in accent))
        lines, cur = [], ""
        for ch in label:
            if width_of(cur + ch, 18) > w - 2 * pad - 30 and cur:
                lines.append(cur)
                cur = ch
            else:
                cur += ch
        lines.append(cur)
        for i, ln in enumerate(lines[:2]):
            text((pad + 18, 77 + i * 22), ln.strip(), 18, (255, 255, 255))

        # ── 표적 정보 ──
        y = 130
        text((pad, y), "표적", 12, (130, 138, 148))
        y += 20
        motion = status.get("motion")
        rows = [
            ("기동", MOTION_NAMES.get(motion, "--") if motion else "--"),
            ("설정 속도", f"{status['kmh']:g} km/h" if motion and "kmh" in status else "--"),
            ("거리", range_text(status)),
            ("상태", TARGET_STATE_NAMES.get(status.get("state"), "--") if motion else "--"),
        ]
        for name, value in rows:
            text((pad, y), name, 13, (150, 158, 168))
            text((pad + 62, y), value, 14, (235, 238, 242))
            y += 21

        # ── 횟수 카드 ──
        y += 4
        card_w = (w - 2 * pad - 10) // 2
        for i, (name, value, color) in enumerate((("격추", kills, (110, 220, 255)),
                                                  ("발사 준비", ready_count, (110, 235, 120)))):
            x0 = pad + i * (card_w + 10)
            d.rounded_rectangle((x0, y, x0 + card_w, y + 46), radius=6, fill=(28, 33, 38))
            text((x0 + 10, y + 4), name, 11, (130, 138, 148))
            text((x0 + 10, y + 17), str(value), 22, color)
        y += 56

        # ── 추적 이력 띠 ──
        text((pad, y), f"추적 이력 · 최근 {int(HISTORY_S)}초", 12, (130, 138, 148))
        y += 20
        bar_x0, bar_x1, bar_h = pad, w - pad, 20
        d.rectangle((bar_x0, y, bar_x1, y + bar_h), fill=(30, 35, 40))
        span = bar_x1 - bar_x0
        for a, b, phase in segments:
            x0 = bar_x0 + int(a * span)
            x1 = max(x0 + 1, bar_x0 + int(b * span))
            d.rectangle((x0, y, x1, y + bar_h), fill=rgb(PHASE_COLORS.get(phase, PHASE_COLORS["idle"])))
        text((bar_x0, y + bar_h + 3), f"-{int(HISTORY_S)}초", 10, (110, 118, 128))
        text((bar_x1 - width_of("지금", 10), y + bar_h + 3), "지금", 10, (110, 118, 128))
        y += bar_h + 20
        lx, ly = pad, y
        for phase in ("idle", "track", "align", "ready", "warn", "kill"):
            name = PHASE_LABELS[phase]
            need = width_of(name, 10) + 20
            if lx + need > w - pad:
                lx, ly = pad, ly + 16
            d.rectangle((lx, ly + 3, lx + 8, ly + 11), fill=rgb(PHASE_COLORS.get(phase, PHASE_COLORS["idle"])))
            text((lx + 13, ly), name, 10, (150, 158, 168))
            lx += need + 6
        y = ly + 22

        # ── 표적 생성 버튼 ──(이벤트 기록은 그 위 남는 자리만 쓴다)
        footer_h = FOOTER_HELP_H if show_help else FOOTER_H
        layout = control_layout(height, show_help) if controls is not None else None
        limit = layout["top"] - 6 if layout else height - footer_h
        if layout:
            self._draw_controls(d, text, width_of, layout, controls, status, w)

        # ── 이벤트 기록 ──
        text((pad, y), "이벤트", 12, (130, 138, 148))
        y += 20
        room = max(0, (limit - y) // 20)
        for t, phase, msg in events[-room:] if room else []:
            sec = int(origin(t))
            col = rgb(PHASE_COLORS.get(phase, PHASE_COLORS["idle"]))
            d.ellipse((pad, y + 5, pad + 8, y + 13), fill=col)
            text((pad + 14, y + 2), f"{sec // 60:02d}:{sec % 60:02d}", 11, (110, 118, 128))
            text((pad + 56, y), msg, 12, (225, 228, 232))
            y += 20

        # ── 하단: 연결·녹화·조작 안내 ──
        fy = height - footer_h
        d.line((pad, fy - 6, w - pad, fy - 6), fill=(48, 54, 60))
        live = link_state == "LIVE"
        dot = (90, 220, 120) if live else (255, 180, 70)
        d.ellipse((pad, fy + 4, pad + 9, fy + 13), fill=dot)
        text((pad + 16, fy), f"LIVE  {fps:.0f} FPS" if live else link_state, 12, (200, 205, 212))
        if recording:
            m, s = int(rec_elapsed) // 60, int(rec_elapsed) % 60
            d.ellipse((pad + 150, fy + 4, pad + 159, fy + 13), fill=(255, 70, 70))
            text((pad + 166, fy), f"REC {m:02d}:{s:02d}", 12, (255, 120, 120))
        elif notice:
            text((pad + 110, fy), notice[:20], 10, (255, 190, 90))
        if show_help:
            for i, ln in enumerate(("클릭/Space 발사(수동)   V 녹화", "F 전체화면   H 상세   R 탐색 자세   N 표적 생성",
                                    "0~4 기동   +/- 속도   M 출발·정지   Q 종료")):
                text((pad, fy + 26 + i * 18), ln, 11, (150, 158, 168))

        return np.asarray(img)[:, :, ::-1].copy()
