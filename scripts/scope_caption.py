"""scope_caption.py — 스코프 영상 하단 한글 자막.

HUD(WSL2, OpenCV Hershey 폰트)는 영문 약어만 그릴 수 있어 일반 관객에게는 "지금 무슨 일이
일어나는지"가 전달되지 않는다. Windows 뷰어에서 FCS 상태를 한글 한 줄로 풀어 얹는다.

caption_for() 는 순수 함수라 GUI 없이 시험할 수 있다. draw_caption() 만 PIL/OpenCV 를 쓴다.
"""

from __future__ import annotations

import os
from typing import Optional

KILL_CAPTION_S = 2.5
RESPAWN_CAPTION_S = 4.0

# tone -> BGR 강조색. HUD 상태 색(회색/노랑/주황/초록)과 맞춘다.
TONE_COLORS = {
    "idle": (170, 170, 170),
    "track": (0, 230, 255),
    "align": (0, 150, 255),
    "ready": (80, 255, 80),
    "kill": (80, 255, 80),
    "warn": (60, 60, 255),
}


def caption_for(status: dict) -> Optional[tuple[str, str]]:
    """FCS 상태 -> (자막, tone). 표시할 내용이 없으면 None.

    status 키(smash_scope_viewer 의 /status): fcs_state, reliability, event, event_age_s, trigger_mode.
    """

    state = str(status.get("fcs_state") or "")
    rel = str(status.get("reliability") or "")
    event = status.get("event")
    age = status.get("event_age_s")
    if not state and not event:
        return None

    if event == "destroyed":
        # 재출현 전까지는 추적 상태 전이(놓침 등)를 보여 주지 않는다. 격추 뒤 정리 동작일 뿐이다.
        if age is not None and age <= KILL_CAPTION_S:
            return "표적 격추!", "kill"
        return "표적 격추 완료 — 다음 표적 대기 중", "idle"
    if event == "respawn" and age is not None and age <= RESPAWN_CAPTION_S and state in ("", "SEARCH"):
        return "새 표적 출현 — 포를 돌려 탐색 중", "idle"

    if state == "READY":
        if str(status.get("trigger_mode") or "") == "manual":
            return "발사 준비 완료 — 클릭 또는 스페이스로 발사", "ready"
        return "발사 준비 완료 — 자동 발사", "ready"
    if state.startswith("READY BLOCK"):
        if "MOTION" in state:
            return "표적 속도 안정화 대기 — 발사 보류", "align"
        return "추적 신뢰도 부족 — 발사 보류", "warn"
    if state.startswith("ALIGN"):
        return "조준선 정렬 중 — 예측 위치로 포 이동", "align"
    if state == "SEARCH":
        return "표적 탐색 중", "idle"
    if state.startswith("TRACK("):
        reason = state[6:-1] if state.endswith(")") else state[6:]
        if reason == "NO RANGE":
            return "표적 포착 — 거리 측정 중", "track"
        if reason == "NO TF":
            return "표적 포착 — 포 자세 동기화 중", "track"
        if reason.startswith("REL"):
            if rel == "LOST":
                return "표적을 놓침 — 다시 찾는 중", "warn"
            return "추적 불안정 — 신뢰도 회복 대기", "warn"
        return "표적 추적 중", "track"
    if state in ("TRACK", "AIM"):
        return "표적 추적 중 — 이동 궤적 분석", "track"
    return None


_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\malgunbd.ttf",
    r"C:\Windows\Fonts\malgun.ttf",
    "/mnt/c/Windows/Fonts/malgun.ttf",
)


def find_font() -> Optional[str]:
    """한글을 그릴 수 있는 글꼴 경로. 없으면 None."""

    return next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)


class CaptionRenderer:
    """프레임 하단에 자막 띠를 그린다. 글꼴이 없으면 아무것도 하지 않는다."""

    def __init__(self) -> None:
        self._fonts: dict[int, object] = {}
        self._font_path: Optional[str] = find_font()
        self.available = self._font_path is not None

    def _font(self, size: int):
        font = self._fonts.get(size)
        if font is None:
            from PIL import ImageFont

            font = ImageFont.truetype(self._font_path, size)
            self._fonts[size] = font
        return font

    def draw(self, frame, caption: Optional[tuple[str, str]]) -> None:
        """frame(BGR ndarray)을 제자리에서 수정한다."""

        if caption is None or not self.available:
            return
        import numpy as np
        from PIL import Image, ImageDraw

        text, tone = caption
        h, w = frame.shape[:2]
        size = max(16, int(w * 0.034))
        bar_h = size + 22
        strip = frame[h - bar_h:h]
        strip[:] = (strip * 0.30).astype(np.uint8)

        accent = TONE_COLORS.get(tone, TONE_COLORS["idle"])
        strip[:2, :] = accent  # 윗선

        img = Image.fromarray(strip[:, :, ::-1])
        draw = ImageDraw.Draw(img)
        font = self._font(size)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw = right - left
        x = max(10, (w - tw) // 2)
        y = (bar_h - (bottom - top)) // 2 - top + 1
        rgb = accent[::-1]
        draw.ellipse((x - 22, bar_h // 2 - 6, x - 10, bar_h // 2 + 6), fill=rgb)
        draw.text((x, y), text, font=font, fill=(255, 255, 255))
        strip[:] = np.asarray(img)[:, :, ::-1]


def letterbox(frame, width: int, height: int):
    """종횡비를 유지하며 (width, height) 안에 맞추고 남는 곳은 검게 채운다.

    전체화면에서 640x640 영상을 16:9 화면에 그대로 늘리면 원이 타원이 된다.
    """

    import cv2
    import numpy as np

    fh, fw = frame.shape[:2]
    if width <= 0 or height <= 0 or (fw == width and fh == height):
        return frame
    scale = min(width / fw, height / fh)
    nw, nh = max(1, int(round(fw * scale))), max(1, int(round(fh * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - nw) // 2, (height - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas
