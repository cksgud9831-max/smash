#!/usr/bin/env python3
"""win_scope_viewer.py — Windows 네이티브 SMASH 스코프 뷰어
WSL2의 가상화 GUI(WSLg) 버그를 100% 우회하여,
Windows 데스크탑에서 직접 OpenCV 창을 띄우고 격발 및 수동 조준을 수행합니다.
GetAsyncKeyState + cv2.waitKeyEx 이중 키 감지로 방향키/WASD 조작 보장.
"""

import sys
import time
import threading
import urllib.request
import urllib.parse
import ctypes
import numpy as np
import cv2


STREAM_URL = "http://localhost:9999/stream"
BASE_URL = "http://localhost:9999"
WINDOW_NAME = "SMASH Smart Scope FCS (Windows Native)"

user32 = ctypes.windll.user32

# 가상 키코드
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_SPACE = 0x20
VK_W = 0x57
VK_A = 0x41
VK_S = 0x53
VK_D = 0x44
VK_R = 0x52
VK_T = 0x54
VK_Q = 0x51
VK_ESCAPE = 0x1B


def is_key_down(vk):
    return (user32.GetAsyncKeyState(vk) & 0x8000) != 0


def send_async(endpoint):
    def _req():
        try:
            urllib.request.urlopen(BASE_URL + endpoint, timeout=0.5)
        except Exception:
            pass
    threading.Thread(target=_req, daemon=True).start()


def fire():
    print("[BANG!] 사수 격발 명령 전송")
    send_async("/fire")


def slew(pan, tilt):
    send_async(f"/slew?pan={pan}&tilt={tilt}")


def is_inside(pt, rect):
    x, y = pt
    rx, ry, rw, rh = rect
    return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)


def main():
    print("============================================================")
    print("  SMASH Windows 네이티브 스코프 뷰어 시작")
    print("  연결 대상: " + STREAM_URL)
    print("  조작: 마우스 좌클릭 / 스페이스바 (격발)")
    print("        방향키(↑,↓,←,→) / WASD / 화면 D_pad 버튼 (수동 조준)")
    print("        T (대공 경계 앙각 프리셋: 드론 즉시 포착) / R (수평 리셋)")
    print("        Q 또는 ESC (종료)")
    print("============================================================")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 720, 720)
    cv2.moveWindow(WINDOW_NAME, 100, 100)
    try:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 0)
    except Exception:
        pass

    btn_up = (0, 0, 0, 0)
    btn_down = (0, 0, 0, 0)
    btn_left = (0, 0, 0, 0)
    btn_right = (0, 0, 0, 0)
    btn_reset = (0, 0, 0, 0)
    btn_air = (0, 0, 0, 0)

    step_rad = 0.070  # 약 4.0도 (신속하고 시원한 조준경 회전)
    last_key_time = 0.0

    def on_mouse(event, x, y, flags, param):
        nonlocal btn_up, btn_down, btn_left, btn_right, btn_reset, btn_air
        if event == cv2.EVENT_LBUTTONDOWN:
            pt = (x, y)
            if is_inside(pt, btn_up):
                print("[BUTTON] 상방(TILT UP) 조준")
                slew(0.0, step_rad)
            elif is_inside(pt, btn_down):
                print("[BUTTON] 하방(TILT DOWN) 조준")
                slew(0.0, -step_rad)
            elif is_inside(pt, btn_left):
                print("[BUTTON] 좌측(PAN LEFT) 조준")
                slew(-step_rad, 0.0)
            elif is_inside(pt, btn_right):
                print("[BUTTON] 우측(PAN RIGHT) 조준")
                slew(step_rad, 0.0)
            elif is_inside(pt, btn_reset):
                print("[BUTTON] 조준 수평 리셋")
                slew(999.0, 0.0)
            elif is_inside(pt, btn_air):
                print("[BUTTON] 대공 경계 프리셋 (TILT +18.3deg)")
                slew(888.0, 0.0)
            else:
                fire()

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    stream = None
    stream_bytes = b""

    while True:
        if stream is None:
            loading = np.zeros((720, 720, 3), dtype=np.uint8)
            cv2.putText(loading, "SMASH Smart Scope FCS", (140, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 120), 2)
            cv2.putText(loading, "Connecting to WSL2 SMASH FCS (http://localhost:9999)...", (60, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
            cv2.imshow(WINDOW_NAME, loading)
            cv2.waitKey(100)

            try:
                req = urllib.request.Request(STREAM_URL)
                stream = urllib.request.urlopen(req, timeout=2.0)
                stream_bytes = b""
                print("[OK] WSL2 스코프 영상 스트림에 성공적으로 연결되었습니다.")
            except Exception:
                stream = None
                time.sleep(0.5)
                continue

        try:
            chunk = stream.read(4096)
            if not chunk:
                stream = None
                continue
            stream_bytes += chunk
            a = stream_bytes.find(b"\xff\xd8")
            b = stream_bytes.find(b"\xff\xd9")
            if a != -1 and b != -1:
                jpg = stream_bytes[a : b + 2]
                stream_bytes = stream_bytes[b + 2 :]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    h, w = frame.shape[:2]
                    bw, bh = 38, 28
                    cx = w - 85
                    cy = h - 110
                    btn_air = (cx, cy - (bh + 4) * 2, bw, bh)
                    btn_up = (cx, cy - bh - 4, bw, bh)
                    btn_down = (cx, cy + bh + 4, bw, bh)
                    btn_left = (cx - bw - 4, cy, bw, bh)
                    btn_right = (cx + bw + 4, cy, bw, bh)
                    btn_reset = (cx, cy, bw, bh)

                    # D_pad 반투명 버튼 박스 및 텍스트 렌더링
                    for b_rect, b_txt, b_col in [
                        (btn_air, "AIR", (0, 255, 255)),
                        (btn_up, "UP", (0, 255, 136)),
                        (btn_down, "DN", (0, 255, 136)),
                        (btn_left, "LT", (0, 255, 136)),
                        (btn_right, "RT", (0, 255, 136)),
                        (btn_reset, "RST", (180, 180, 180)),
                    ]:
                        bx, by, bw_i, bh_i = b_rect
                        cv2.rectangle(frame, (bx, by), (bx + bw_i, by + bh_i), (25, 30, 45), -1)
                        cv2.rectangle(frame, (bx, by), (bx + bw_i, by + bh_i), b_col, 1)
                        tw, th = cv2.getTextSize(b_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
                        tx = bx + (bw_i - tw) // 2
                        ty = by + (bh_i + th) // 2
                        cv2.putText(frame, b_txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, b_col, 1, cv2.LINE_AA)

                    # 하단 조작 가이드 안내문
                    guide_txt = "AIM: ARROWS / WASD | AIR PRESET: T | RESET: R | FIRE: SPACE"
                    cv2.putText(frame, guide_txt, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, guide_txt, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1, cv2.LINE_AA)

                    cv2.imshow(WINDOW_NAME, frame)

            key_ex = cv2.waitKeyEx(1)
            key = key_ex & 0xFF
            now = time.time()

            # 키보드 방향키 / WASD / T(대공) / R(리셋) / 스페이스바 판별
            is_up = is_key_down(VK_UP) or is_key_down(VK_W) or (key in (ord("w"), ord("W"))) or (key_ex in (2490368, 0x260000, 38))
            is_down = is_key_down(VK_DOWN) or is_key_down(VK_S) or (key in (ord("s"), ord("S"))) or (key_ex in (2621440, 0x280000, 40))
            is_left = is_key_down(VK_LEFT) or is_key_down(VK_A) or (key in (ord("a"), ord("A"))) or (key_ex in (2424832, 0x250000, 37))
            is_right = is_key_down(VK_RIGHT) or is_key_down(VK_D) or (key in (ord("d"), ord("D"))) or (key_ex in (2555904, 0x270000, 39))
            is_air = is_key_down(VK_T) or (key in (ord("t"), ord("T")))
            is_reset = is_key_down(VK_R) or (key in (ord("r"), ord("R")))
            is_fire = is_key_down(VK_SPACE) or (key == 32)
            is_quit = (key == 27) or (cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1)

            if now - last_key_time > 0.05:  # 50ms 쿨다운으로 신속하고 부드러운 연속 조작
                if is_air:
                    print("[KEY] 대공 경계 자세 프리셋 (TILT +18.3deg)")
                    slew(888.0, 0.0)
                    last_key_time = now + 0.15
                elif is_up:
                    print("[KEY] 상방(TILT UP) 조준")
                    slew(0.0, step_rad)
                    last_key_time = now
                elif is_down:
                    print("[KEY] 하방(TILT DOWN) 조준")
                    slew(0.0, -step_rad)
                    last_key_time = now
                elif is_left:
                    print("[KEY] 좌측(PAN LEFT) 조준")
                    slew(-step_rad, 0.0)
                    last_key_time = now
                elif is_right:
                    print("[KEY] 우측(PAN RIGHT) 조준")
                    slew(step_rad, 0.0)
                    last_key_time = now
                elif is_reset:
                    print("[KEY] 조준 수평 리셋")
                    slew(999.0, 0.0)
                    last_key_time = now
                elif is_fire:
                    fire()
                    last_key_time = now + 0.2  # 단발 방아쇠 쿨다운

            if is_quit:
                break

        except Exception as e:
            stream = None
            time.sleep(0.2)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
