"""Presentation-only video connection and recording panel (no ROS dependencies)."""
from collections import deque
import os
import cv2
import numpy as np


class StreamHealth:
    def __init__(self):
        self.samples = deque(maxlen=120)
        self.error = ""
        self.retries = 0

    def received(self, now):
        self.samples.append(now)
        self.error = ""

    def snapshot(self, now):
        samples = tuple(self.samples)
        age = max(0.0, now - samples[-1]) if samples else None
        recent = [t for t in samples if now - t <= 3.0]
        fps = ((len(recent) - 1) / (recent[-1] - recent[0])
               if len(recent) > 1 and recent[-1] > recent[0] else 0.0)
        state = "CONNECTING" if age is None else ("LIVE" if age < 2.0 else "STALE")
        return state, age, fps


def compose_view(frame, health, now, recording=False, elapsed=0.0, path=None, notice="",
                 audience=None, show_help=False):
    """Keep source pixels separate from local media diagnostics and controls.

    audience = (PanelRenderer, StatusHistory, status dict) 이면 오른쪽에 관객용 패널을 그린다.
    없거나 글꼴이 없으면 영상 모니터(연결·녹화 진단) 패널을 그린다.
    """
    state, age, fps = health.snapshot(now)
    if frame is None:
        frame = np.zeros((640, 640, 3), np.uint8)
    h, w = frame.shape[:2]
    panel = 290
    canvas = np.full((max(h, 540) + 42, w + panel, 3), (24, 21, 18), np.uint8)
    canvas[:h, :w] = frame
    color = (120, 220, 110) if state == "LIVE" else (70, 180, 255)

    if audience is not None and audience[0].available:
        renderer, history, status = audience[:3]
        controls = audience[3] if len(audience) > 3 else None  # 표적 생성 버튼 상태(SpawnControls)
        canvas[:, w:] = renderer.render(canvas.shape[0], status, history, now, state, fps,
                                        recording, elapsed, notice or health.error, show_help, controls)
        if state != "LIVE":
            cv2.rectangle(canvas, (0, 0), (w-1, 42), (35, 35, 70), -1)
            cv2.putText(canvas, "WAITING FOR VIDEO" if age is None else "STALE VIDEO - RECONNECTING",
                        (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 255), 1, cv2.LINE_AA)
        return canvas

    def line(text, y, tint=(205, 205, 205), scale=0.48):
        cv2.putText(canvas, text, (w + 18, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, tint, 1, cv2.LINE_AA)

    line("VIDEO MONITOR", 34, (240, 240, 240), 0.65)
    line(state, 77, color, 0.65)
    line(f"Receive rate: {fps:.1f} FPS", 111)
    line("Last frame: " + ("--" if age is None else f"{age:.1f}s ago"), 141)
    line(f"Reconnects: {health.retries}", 171)
    line("RECORDING" if recording else "RECORDING OFF", 228,
         (80, 80, 255) if recording else (180, 180, 180))
    line(f"Duration: {int(elapsed)//60:02d}:{int(elapsed)%60:02d}", 260)
    if path:
        name = os.path.basename(path)
        for i in range(0, len(name), 27):
            line(name[i:i+27], 292 + 22*(i//27), scale=0.40)
    for text, y in [("V   Start / stop recording", 380), ("F   Fullscreen", 410),
                    ("F1  Existing help overlay", 440), ("Q / ESC   Close", 470)]:
        line(text, y, scale=0.43)
    message = notice or health.error
    if message:
        line(message[:32], 515, (70, 180, 255), 0.38)
    cv2.putText(canvas, "127.0.0.1:9999  |  Local video viewer", (14, canvas.shape[0]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (175, 175, 175), 1, cv2.LINE_AA)
    if state != "LIVE":
        cv2.rectangle(canvas, (0, 0), (w-1, 42), (35, 35, 70), -1)
        cv2.putText(canvas, "WAITING FOR VIDEO" if age is None else "STALE VIDEO - RECONNECTING",
                    (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 255), 1, cv2.LINE_AA)
    return canvas
