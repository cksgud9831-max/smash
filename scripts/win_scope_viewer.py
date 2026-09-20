#!/usr/bin/env python3
"""win_scope_viewer.py — Windows 네이티브 SMASH 스코프 뷰어
WSL2의 가상화 GUI(WSLg) 버그를 100% 우회하여,
Windows 데스크탑에서 직접 OpenCV 창을 띄우고 격발 및 수동 조준을 수행합니다.
GetAsyncKeyState + cv2.waitKeyEx 이중 키 감지로 방향키/WASD 조작 보장.
"""

import os
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
VK_SHIFT = 0x10
VK_H = 0x48
VK_C = 0x43
VK_V = 0x56

# 녹화: 화면에 들어온 HUD 영상을 그대로(D_pad 버튼·안내문 제외) MP4 로 저장한다.
# 영상 도착 간격이 고르지 않으므로 벽시계 기준 고정 FPS 로 마지막 프레임을 채워
# 넣는다 — 재생 속도가 실제 시간과 같아진다.
REC_FPS = 20.0
REC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recordings")

# 조준 스텝. 카메라 화각 0.4 rad / 640 px 이므로 1 mrad 가 약 1.6 px 이다.
# READY 정렬 허용오차는 표적 거리에서 유도된다(명중 반경의 각크기, smash_fcs_node
# _align_tolerance_for_range): 10.8 m 약 29 px, 20 m 약 16 px, 35 m 약 9 px.
# 이전 정밀 스텝 0.010 rad(약 16 px)은 허용오차가 35 px 고정이던 시절 값이라, 20 m 이상에서는
# 한 번 누를 때마다 허용오차를 통째로 건너뛰어 키로는 READY 에 들어가기 어려웠다
# (2026-09-18 실측). 35 m 허용오차의 약 3분의 2인 0.004 rad(약 6 px)로 줄인다.
# 크게 돌릴 때는 Shift(표적 탐색용 큰 이동)를 쓴다.
STEP_FINE_RAD = 0.004   # 약 0.23도, 약 6 px
STEP_COARSE_RAD = 0.070  # 약 4.0도 (Shift: 표적 탐색용 큰 이동)


_viewer_hwnd = {"h": 0}


def viewer_focused():
    """뷰어 창이 맨 앞일 때만 참.

    GetAsyncKeyState 는 어느 창에 입력하든 키 상태를 돌려준다. 이 검사가 없으면 다른
    창에서 타이핑한 W/A/S/D·Space 가 그대로 조준·격발로 들어간다(2026-09-18 실측:
    뷰어를 띄워 둔 채 다른 창을 쓰는 동안 포탑이 돌고 한 발이 나갔다).
    """
    if not _viewer_hwnd["h"]:
        _viewer_hwnd["h"] = user32.FindWindowW(None, WINDOW_NAME)
    return bool(_viewer_hwnd["h"]) and user32.GetForegroundWindow() == _viewer_hwnd["h"]


def is_key_down(vk):
    return viewer_focused() and (user32.GetAsyncKeyState(vk) & 0x8000) != 0


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
    # pan 은 ROS 관례(Z 축 기준 반시계 = 왼쪽이 +)다. 왼쪽 키는 +, 오른쪽 키는 - 를 보낸다.
    # 이전에는 반대로 보내 왼쪽 키가 포탑을 오른쪽으로 돌렸다(2026-09-18 실측: 화면 왼쪽
    # 표적을 향해 A 를 누르자 표적이 화면 밖으로 밀려났다).
    send_async(f"/slew?pan={pan}&tilt={tilt}")


def ui_command(name):
    send_async(f"/cmd?name={name}")


class Recorder:
    def __init__(self):
        self.writer = None
        self.path = None
        self.size = None
        self.started = 0.0
        self.next_t = 0.0

    @property
    def active(self):
        return self.writer is not None

    def start(self, frame):
        os.makedirs(REC_DIR, exist_ok=True)
        self.path = os.path.join(REC_DIR, time.strftime("smash_%Y%m%d_%H%M%S.mp4"))
        h, w = frame.shape[:2]
        self.size = (w, h)
        self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), REC_FPS, self.size)
        if not self.writer.isOpened():
            print("[REC] 녹화 파일을 열지 못했습니다: " + self.path)
            self.writer = None
            return
        self.started = self.next_t = time.time()
        print("[REC] 녹화 시작: " + self.path)

    def feed(self, frame, now):
        if self.writer is None or frame is None:
            return
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        while self.next_t <= now:
            self.writer.write(frame)
            self.next_t += 1.0 / REC_FPS

    def stop(self):
        if self.writer is not None:
            self.writer.release()
            print(f"[REC] 녹화 종료 ({time.time() - self.started:.0f}초): {self.path}")
        self.writer = None


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
    print("        V (녹화 시작/종료, recordings 폴더) / H (HUD 상세 수치) / C (사격 통계 초기화)")
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

    step_rad = STEP_FINE_RAD  # 매 루프에서 Shift 여부로 갱신한다
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
                slew(step_rad, 0.0)
            elif is_inside(pt, btn_right):
                print("[BUTTON] 우측(PAN RIGHT) 조준")
                slew(-step_rad, 0.0)
            elif is_inside(pt, btn_reset):
                print("[BUTTON] 조준 수평 리셋")
                slew(999.0, 0.0)
            elif is_inside(pt, btn_air):
                print("[BUTTON] 대공 경계 프리셋 (TILT +18.3deg)")
                slew(888.0, 0.0)
            else:
                fire()

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    # ── 영상 수신 (별도 스레드) ────────────────────────────────────────
    # 이전 구현은 메인 루프에서 4096바이트씩 읽으면서 매 반복 cv2.waitKeyEx(1)
    # 을 호출했다. waitKey 한 번이 최소 1~5 ms 를 쓰므로 읽기 속도가 초당 1 MB
    # 안팎으로 묶이는데, 서버는 그보다 빠르게 프레임을 밀어낸다. 소비가 생산을
    # 못 따라가면 TCP 버퍼에 프레임이 계속 쌓여 화면 지연이 무한정 늘어난다
    # (사용자가 겪은 "버퍼링" 증상). 수신을 전용 스레드로 분리하고, 버퍼에 밀린
    # 프레임은 버린 뒤 항상 최신 한 장만 남긴다.
    latest = {"jpeg": None}

    def stream_reader():
        while True:
            try:
                req = urllib.request.Request(STREAM_URL)
                with urllib.request.urlopen(req, timeout=2.0) as stream:
                    print("[OK] WSL2 스코프 영상 스트림에 성공적으로 연결되었습니다.")
                    buf = b""
                    while True:
                        chunk = stream.read(65536)
                        if not chunk:
                            break
                        buf += chunk
                        # 버퍼에 여러 장이 밀려 있으면 마지막(최신) 한 장만 취한다.
                        end = buf.rfind(b"\xff\xd9")
                        if end != -1:
                            start = buf.rfind(b"\xff\xd8", 0, end)
                            if start != -1:
                                latest["jpeg"] = buf[start : end + 2]
                            buf = buf[end + 2 :]
                        if len(buf) > 4_000_000:  # 경계 손상 시 폭주 방지
                            buf = b""
            except Exception:
                pass
            latest["jpeg"] = None
            time.sleep(0.5)

    threading.Thread(target=stream_reader, daemon=True).start()

    shown_jpeg = None
    recorder = Recorder()
    clean_frame = None
    prev_toggle = {VK_H: False, VK_C: False, VK_V: False}

    def toggled(vk, key, chars):
        """누르는 순간 한 번만 참. GetAsyncKeyState 는 누르고 있는 동안 계속 참이다."""
        down = is_key_down(vk) or key in chars
        edge = down and not prev_toggle[vk]
        prev_toggle[vk] = down
        return edge

    while True:
        jpg = latest["jpeg"]

        if jpg is None:
            loading = np.zeros((720, 720, 3), dtype=np.uint8)
            cv2.putText(loading, "SMASH Smart Scope FCS", (140, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 120), 2)
            cv2.putText(loading, "Connecting to WSL2 SMASH FCS (http://localhost:9999)...", (60, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
            cv2.imshow(WINDOW_NAME, loading)

        try:
            # 같은 프레임을 다시 디코딩하지 않는다. 새 프레임일 때만 그린다.
            if jpg is not None and jpg is not shown_jpeg:
                shown_jpeg = jpg
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    clean_frame = frame.copy()
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
                    guide_txt = "AIM: WASD (SHIFT=FAST) | AIR: T | FIRE: SPACE | REC: V | INFO: H"
                    cv2.putText(frame, guide_txt, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, guide_txt, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1, cv2.LINE_AA)

                    if recorder.active:
                        el = int(time.time() - recorder.started)
                        # 상단 가운데: 좌상단 상태 표시줄과 우상단 거리·과녁 패널을 피한다
                        if el % 2 == 0:
                            cv2.circle(frame, (w // 2 - 52, 26), 7, (0, 0, 255), -1, cv2.LINE_AA)
                        cv2.putText(frame, f"REC {el // 60:02d}:{el % 60:02d}", (w // 2 - 40, 32),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

                    cv2.imshow(WINDOW_NAME, frame)

            key_ex = cv2.waitKeyEx(1)
            key = key_ex & 0xFF
            now = time.time()
            recorder.feed(clean_frame, now)

            if toggled(VK_V, key, (ord("v"), ord("V"))) and clean_frame is not None:
                if recorder.active:
                    recorder.stop()
                else:
                    recorder.start(clean_frame)
            if toggled(VK_H, key, (ord("h"), ord("H"))):
                ui_command("toggle_debug")
            if toggled(VK_C, key, (ord("c"), ord("C"))):
                print("[KEY] 사격 통계 초기화")
                ui_command("reset_stats")

            # Shift 를 누르고 있으면 큰 스텝(표적 탐색), 아니면 정밀 스텝(정렬).
            step_rad = STEP_COARSE_RAD if is_key_down(VK_SHIFT) else STEP_FINE_RAD

            # 키보드 방향키 / WASD / T(대공) / R(리셋) / 스페이스바 판별
            is_up = is_key_down(VK_UP) or is_key_down(VK_W) or (key in (ord("w"), ord("W"))) or (key_ex in (2490368, 0x260000, 38))
            is_down = is_key_down(VK_DOWN) or is_key_down(VK_S) or (key in (ord("s"), ord("S"))) or (key_ex in (2621440, 0x280000, 40))
            is_left = is_key_down(VK_LEFT) or is_key_down(VK_A) or (key in (ord("a"), ord("A"))) or (key_ex in (2424832, 0x250000, 37))
            is_right = is_key_down(VK_RIGHT) or is_key_down(VK_D) or (key in (ord("d"), ord("D"))) or (key_ex in (2555904, 0x270000, 39))
            is_air = is_key_down(VK_T) or (key in (ord("t"), ord("T")))
            is_reset = is_key_down(VK_R) or (key in (ord("r"), ord("R")))
            is_fire = is_key_down(VK_SPACE) or (key == 32)
            # 안내문대로 Q 도 종료로 받는다(이전에는 ESC 만 받아 Q 를 눌러도 창이 남았다)
            # 다른 키와 마찬가지로 GetAsyncKeyState 로도 받는다. waitKeyEx 로만 받으면 입력이
            # OpenCV 창에 전달되지 않는 경우 ESC 가 먹지 않았다(2026-09-18 실측).
            is_quit = (key in (27, ord("q"), ord("Q")) or is_key_down(VK_ESCAPE) or is_key_down(VK_Q)
                       or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1)

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
                    slew(step_rad, 0.0)
                    last_key_time = now
                elif is_right:
                    print("[KEY] 우측(PAN RIGHT) 조준")
                    slew(-step_rad, 0.0)
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

    recorder.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
