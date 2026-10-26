#!/usr/bin/env python3
"""win_scope_viewer.py — Windows 네이티브 SMASH 스코프 뷰어

WSL2 의 smash_scope_viewer 가 내보내는 MJPEG 스트림(HUD 영상)을 Windows 창에 띄우고
녹화한다. 격발은 FCS 가 READY 에서 자동으로 하므로 조준·격발 입력은 없다.
키 입력은 GetAsyncKeyState(창이 맨 앞일 때만)와 cv2.waitKeyEx 로 함께 받는다.

  Space / 마우스 왼쪽 클릭   수동 격발 모드(SMASH_FIRE_TRIGGER=manual)에서 READY 일 때 한 번에 한 발
  R            탐색 자세 복귀(표적을 놓쳐 포가 엉뚱한 곳을 볼 때)
  패널 버튼     표적 생성: 방식(직선·왕복·8자·접근)·속도를 고르고 "표적 즉시 생성"(키 N). 격추 뒤 대기 없이,
               살아 있으면 교체한다. 영상 위 클릭만 방아쇠이고 패널 클릭은 방아쇠가 아니다.
  0~4, +/-, M  이동 표적 기동 방식 / 속도 / 출발·정지
  H            HUD 상세 수치     C  사격 통계 초기화
  F            전체화면 켬/끔(종횡비 유지)
  F1           조작 안내·표적 상태 표시 켬/끔(기본 끔: 관객용 화면)
  V            녹화 시작/종료(recordings 폴더)
  Q / ESC      종료
"""

import json
import os
import threading
import time
import traceback
import urllib.parse
import urllib.request
import ctypes
import sys
from datetime import datetime

import cv2
import numpy as np

from scope_caption import CaptionRenderer, caption_for, letterbox
from scope_stream import PlayoutBuffer, iter_latest_jpegs
from scope_monitor import StreamHealth, compose_view
from scope_panel import PanelRenderer, StatusHistory
from scope_controls import SpawnControls, control_layout, hit_test, window_to_view


# WSL 서버는 IPv4 로 listen 한다. Windows 의 localhost 는 ::1 을 먼저
# 시도해 요청마다 IPv6 연결 timeout 뒤 IPv4 로 돌아갈 수 있다. 직접 연결한다.
BASE_URL = "http://127.0.0.1:9999"
STREAM_URL = BASE_URL + "/stream"
WINDOW_NAME = "SMASH Smart Scope FCS (Windows Native)"

user32 = ctypes.windll.user32

# 가상 키코드. 문자로 받으면 한글 입력 상태에서 키가 달라지므로 가상 키코드를 쓴다.
VK_Q = 0x51
VK_ESCAPE = 0x1B
VK_R = 0x52
VK_H = 0x48
VK_C = 0x43
VK_V = 0x56
VK_M = 0x4D
VK_F1 = 0x70
VK_SPACE = 0x20
VK_F = 0x46
VK_N = 0x4E
# 이동 표적 기동 방식: 숫자 0~4 (윗줄·숫자패드 모두)
MOTION_KEYS = {  # 방식 -> (윗줄 키, 숫자패드 키, 문자)
    "none": (0x30, 0x60, "0"),
    "straight": (0x31, 0x61, "1"),
    "cross": (0x32, 0x62, "2"),
    "maneuver": (0x33, 0x63, "3"),
    "approach": (0x34, 0x64, "4"),
}
MOTION_LABELS = {"none": "HOVER", "straight": "STRAIGHT", "cross": "BACK&FORTH",
                 "maneuver": "FIGURE-8", "approach": "APPROACH"}
VK_OEM_PLUS, VK_ADD = 0xBB, 0x6B
VK_OEM_MINUS, VK_SUBTRACT = 0xBD, 0x6D

# turret_command 의 탐색 자세 복귀 프리셋 값(pan 증분 자리에 888 을 보낸다)
SEARCH_POSE_PRESET = 888.0

# 녹화: 화면에 들어온 HUD 영상을 그대로 MP4 로 저장한다. 영상 도착 간격이 고르지 않으므로
# 벽시계 기준 고정 FPS 로 마지막 프레임을 채워 넣는다 — 재생 속도가 실제 시간과 같아진다.
REC_FPS = 20.0
REC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recordings")

# 화면 끊김 완화: 도착 간격이 고르지 않은 프레임을 이만큼 쌓았다가 일정한 박자로 보여 준다.
# 프레임 수 단위(약 1프레임 = 45~60 ms 지연). 0 이면 도착 즉시 표시(지연 최소, 끊김 그대로).
try:
    JITTER_FRAMES = max(0, int(os.environ.get("SMASH_VIEW_BUFFER", "2")))
except ValueError:
    JITTER_FRAMES = 2

_viewer_hwnd = {"h": 0}


def viewer_focused():
    """뷰어 창이 맨 앞일 때만 참.

    GetAsyncKeyState 는 어느 창에 입력하든 키 상태를 돌려준다. 이 검사가 없으면 다른
    창에서 타이핑한 키가 그대로 뷰어 명령(표적 기동 변경 등)으로 들어간다.
    """
    if not _viewer_hwnd["h"]:
        _viewer_hwnd["h"] = user32.FindWindowW(None, WINDOW_NAME)
    return bool(_viewer_hwnd["h"]) and user32.GetForegroundWindow() == _viewer_hwnd["h"]


def is_key_down(vk):
    return viewer_focused() and (user32.GetAsyncKeyState(vk) & 0x8000) != 0


def send_async(endpoint):
    def _req():
        try:
            with urllib.request.urlopen(BASE_URL + endpoint, timeout=0.5) as response:
                response.read()
        except Exception:
            pass
    threading.Thread(target=_req, daemon=True).start()


def ui_command(name):
    # "speed:+5" 의 + 가 URL 에서 공백으로 바뀌지 않도록 인코딩한다
    send_async("/cmd?name=" + urllib.parse.quote(name, safe=""))


def search_pose():
    send_async(f"/slew?pan={SEARCH_POSE_PRESET}&tilt=0.0")


class Recorder:
    """HUD 영상을 MP4 로 남긴다."""

    def __init__(self):
        self.writer = None
        self.path = None
        self.size = None
        self.started = 0.0
        self.next_t = 0.0
        self.notice = ""

    @property
    def active(self):
        return self.writer is not None

    def start(self, frame):
        os.makedirs(REC_DIR, exist_ok=True)
        self.path = os.path.join(REC_DIR, datetime.now().strftime("smash_%Y%m%d_%H%M%S_%f.mp4"))
        h, w = frame.shape[:2]
        self.size = (w, h)
        self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), REC_FPS, self.size)
        if not self.writer.isOpened():
            print("[REC] 녹화 파일을 열지 못했습니다: " + self.path)
            self.writer.release()
            self.writer = None
            self.notice = "Cannot open recording file"
            return
        self.started = self.next_t = time.monotonic()
        self.notice = ""
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
            print(f"[REC] 녹화 종료 ({time.monotonic() - self.started:.0f}초): {self.path}")
            self.notice = "Saved to recordings folder"
        self.writer = None


def put_text(frame, text, org, color):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def main():
    # Hidden launches redirect stdout to a file; Windows otherwise uses cp949,
    # which cannot encode punctuation in the startup message.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    print("============================================================")
    print("  SMASH Windows 네이티브 스코프 뷰어 시작")
    print("  연결 대상: " + STREAM_URL)
    print("  격발은 READY 에서 자동입니다. 조준 입력은 없습니다.")
    print("  Space 또는 마우스 왼쪽 클릭 (수동 격발 모드에서 READY 일 때 한 번에 한 발)")
    print("  R (탐색 자세 복귀: 표적을 놓쳤을 때)")
    print("  0~4 / +,- / M (이동 표적 기동 방식 / 속도 / 출발·정지)")
    print("  패널 버튼 또는 N (표적 즉시 생성: 방식·속도를 고르고 누름. 격추 뒤 기다리지 않고 새 표적)")
    print("  H (HUD 상세 수치) / C (사격 통계 초기화) / V (녹화 시작·종료, recordings 폴더)")
    print("  F (전체화면) / F1 (조작 안내 표시 켬/끔. 기본은 끔 — 관객용 화면)")
    print("  Q 또는 ESC (종료)")
    print(f"  화면 지터 버퍼: {JITTER_FRAMES} 프레임 (환경변수 SMASH_VIEW_BUFFER, 0=끔)")
    print("============================================================")

    # Windows 기본 타이머 해상도는 약 15.6 ms 라 프레임 표시 시각이 그 단위로 튄다. 1 ms 로 올린다.
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1040, 740)
    cv2.moveWindow(WINDOW_NAME, 100, 100)
    # 마우스 왼쪽 클릭: 영상 위면 방아쇠, 패널 위면 표적 생성 버튼. 콜백은 GUI 스레드에서 불리므로
    # 좌표만 쌓고 메인 루프가 처리한다(명령 전송·상태 갱신은 메인 루프에서).
    clicks = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    try:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 0)
    except Exception:
        pass

    # ── 영상 수신 (별도 스레드) ────────────────────────────────────────
    # 메인 루프에서 읽으면 waitKey 한 번이 최소 1~5 ms 를 써 읽기 속도가 묶이고, 서버가 더 빠르게
    # 밀어내면 TCP 버퍼에 프레임이 쌓여 화면 지연이 무한정 늘어난다. 수신을 전용 스레드로 분리한다.
    # 받은 프레임은 PlayoutBuffer 에 넣어 일정한 박자로 꺼내 보여 준다(밀리면 오래된 것부터 버림).
    playout = PlayoutBuffer(target_depth=JITTER_FRAMES)
    playout_lock = threading.Lock()
    health = StreamHealth()
    stop_event = threading.Event()

    def stream_reader():
        while not stop_event.is_set():
            try:
                req = urllib.request.Request(STREAM_URL)
                with urllib.request.urlopen(req, timeout=2.0) as stream:
                    print("[OK] WSL2 스코프 영상 스트림에 성공적으로 연결되었습니다.")
                    for jpeg in iter_latest_jpegs(stream):
                        if stop_event.is_set():
                            return
                        # time.monotonic 은 Windows 에서 해상도가 15.6 ms 라 박자 계산에 쓰지 못한다
                        with playout_lock:
                            playout.push(jpeg, time.perf_counter())
                        health.received(time.monotonic())
            except Exception as exc:
                health.error = type(exc).__name__
                print("[STREAM] " + str(exc))
            health.retries += 1
            stop_event.wait(0.5)

    stream_thread = threading.Thread(target=stream_reader, daemon=True)
    stream_thread.start()

    # 이동 표적 상태(방식·속도). 화면 아래에 표시한다. target_mover 가 없으면 비어 있다.
    target_status = {"text": ""}
    fcs_status = {}  # /status 전체. 한글 자막이 FCS 상태·이벤트를 읽는다
    status_seq = {"n": 0}  # 새로 받을 때마다 증가. 메인 루프가 이력(StatusHistory)에 한 번씩만 넣는다

    def status_reader():
        while not stop_event.is_set():
            try:
                with urllib.request.urlopen(BASE_URL + "/status", timeout=1.0) as r:
                    st = json.loads(r.read().decode("utf-8") or "{}")
                fcs_status.clear()
                fcs_status.update(st)
                status_seq["n"] += 1
                if st.get("motion"):
                    target_status["text"] = (f"TARGET: {MOTION_LABELS.get(st['motion'], st['motion'])} "
                                             f"{st.get('kmh', 0)} km/h  [{st.get('state', '')}]")
                else:
                    target_status["text"] = ""
            except Exception:
                target_status["text"] = ""
                fcs_status.clear()
                status_seq["n"] += 1
            stop_event.wait(0.25)

    status_thread = threading.Thread(target=status_reader, daemon=True)
    status_thread.start()

    display_frame = None
    last_draw = 0.0
    last_error = None
    recorder = Recorder()
    clean_frame = None
    captions = CaptionRenderer()
    history = StatusHistory()
    panel = PanelRenderer()
    seen_seq = -1
    if not captions.available:
        print("[WARN] 한글 글꼴(malgun.ttf)을 찾지 못해 자막을 표시하지 않습니다.")
    show_help = False
    fullscreen = False
    controls = SpawnControls()
    # 마지막으로 보여 준 합성 화면의 크기와 창 크기. 클릭 좌표를 합성 화면 좌표로 되돌리는 데 쓴다.
    geom = {"view_w": 0, "view_h": 0, "win_w": 0, "win_h": 0, "video_w": 640, "video_h": 640}
    prev_toggle = {VK_N: False, VK_SPACE: False, VK_H: False, VK_C: False, VK_V: False, VK_M: False, VK_R: False, VK_F1: False, VK_F: False,
                   "plus": False, "minus": False}
    prev_toggle.update({m: False for m in MOTION_KEYS})

    def toggled(vk, key, chars, name=None):
        """누르는 순간 한 번만 참. GetAsyncKeyState 는 누르고 있는 동안 계속 참이다.
        vk 는 가상 키코드 하나 또는 여러 개(윗줄·숫자패드), name 은 상태 저장용 이름."""
        vks = vk if isinstance(vk, tuple) else (vk,)
        slot = name if name is not None else vk
        down = any(is_key_down(v) for v in vks) or key in chars
        edge = down and not prev_toggle[slot]
        prev_toggle[slot] = down
        return edge

    while True:
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            # 재생 박자가 된 프레임이 있을 때만 디코딩해 그린다.
            with playout_lock:
                tick = time.perf_counter()
                jpg = playout.pop(tick)
                wait_s = playout.wait_s(tick)
            new_frame = jpg is not None
            if new_frame:
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    h = frame.shape[0]
                    w = frame.shape[1]
                    # 자막은 관객용 정보라 녹화에도 남긴다. 조작 안내·REC 표시는 녹화에서 뺀다.
                    captions.draw(frame, caption_for(dict(fcs_status)))
                    clean_frame = frame.copy()
                    if show_help:
                        lines = ["SPACE/CLICK: FIRE   R: SEARCH POSE   H: DEBUG   C: RESET   V: REC   F1: HIDE"]
                        tgt_txt = target_status["text"]
                        if tgt_txt:
                            lines.append(tgt_txt)
                            lines.append("0 HOVER 1 STRAIGHT 2 BACK&FORTH 3 FIGURE-8 4 APPROACH  +/- SPEED  M START")
                        y0 = h // 2 - 10
                        roi = frame[y0 - 16:y0 + 18 * len(lines) - 6, 0:min(w, 470)]
                        cv2.convertScaleAbs(roi, dst=roi, alpha=0.4)
                        for i, line in enumerate(lines):
                            put_text(frame, line, (10, y0 + 18 * i), (0, 255, 200) if i != 1 else (0, 200, 255))

                    if recorder.active:
                        el = int(time.monotonic() - recorder.started)
                        # 상단 가운데: 좌상단 상태 표시줄과 우상단 거리·과녁 패널을 피한다
                        if el % 2 == 0:
                            cv2.circle(frame, (w // 2 - 52, 26), 7, (0, 0, 255), -1, cv2.LINE_AA)
                        cv2.putText(frame, f"REC {el // 60:02d}:{el % 60:02d}", (w // 2 - 40, 32),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

                    # 창 크기가 바뀌었거나 전체화면이면 종횡비를 유지해 맞춘다
                    display_frame = frame
                else:
                    health.error = "Invalid JPEG frame"

            # 새 프레임이 없으면 다시 합성하지 않는다(상태 표시·REC 깜빡임용으로 0.1 초마다만 갱신).
            if new_frame or time.monotonic() - last_draw >= 0.1:
                last_draw = time.monotonic()
                if status_seq["n"] != seen_seq:
                    seen_seq = status_seq["n"]
                    history.update(dict(fcs_status), last_draw)
                _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
                controls.sync(fcs_status)
                media_view = compose_view(
                    display_frame, health, last_draw, recorder.active,
                    max(0.0, last_draw - recorder.started) if recorder.active else 0.0,
                    recorder.path, recorder.notice,
                    audience=(panel, history, dict(fcs_status), controls), show_help=show_help,
                )
                geom.update(view_w=media_view.shape[1], view_h=media_view.shape[0], win_w=win_w, win_h=win_h)
                if display_frame is not None:
                    geom.update(video_h=display_frame.shape[0], video_w=display_frame.shape[1])
                cv2.imshow(WINDOW_NAME, letterbox(media_view, win_w, win_h))
            key_ex = cv2.waitKeyEx(max(1, int(wait_s * 1000)))
            key = key_ex & 0xFF
            now = time.monotonic()
            if health.snapshot(time.monotonic())[0] == "LIVE":
                recorder.feed(clean_frame, now)
            elif recorder.active:
                recorder.stop()
                recorder.notice = "Stopped: video interrupted"
                print("[REC] 영상 수신 중단으로 녹화를 종료했습니다.")

            if toggled(VK_V, key, (ord("v"), ord("V"))) and clean_frame is not None:
                if recorder.active:
                    recorder.stop()
                elif health.snapshot(time.monotonic())[0] == "LIVE":
                    recorder.start(clean_frame)
            if toggled(VK_F, key, (ord("f"), ord("F"))):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )
            # 방아쇠: 누른 한 번(클릭 또는 스페이스 눌림 순간)에 명령 한 번. 수동 모드가 아니면 FCS 가 무시한다.
            fired = False
            while clicks:
                cx, cy = clicks.pop(0)
                pos = window_to_view(cx, cy, geom["win_w"], geom["win_h"], geom["view_w"], geom["view_h"])
                if pos is None:
                    continue  # 검은 여백
                vx, vy = pos
                if vx < geom["video_w"] and vy < geom["video_h"]:
                    fired = True
                elif vx >= geom["video_w"]:
                    layout = control_layout(geom["view_h"], show_help)
                    button = hit_test(layout, vx - geom["video_w"], vy)
                    if button is not None:
                        spawn_cmd = controls.press(button)
                        if spawn_cmd is not None:
                            controls.flash_until = time.monotonic() + 0.4
                            print("[UI] 표적 생성: " + spawn_cmd)
                            ui_command(spawn_cmd)
            if toggled(VK_SPACE, key, (32,)):
                fired = True
            if fired:
                print("[KEY] 방아쇠")
                ui_command("fire")
            if toggled(VK_N, key, (ord("n"), ord("N"))):
                controls.flash_until = time.monotonic() + 0.4
                print("[KEY] 표적 생성: " + controls.command())
                ui_command(controls.command())
            if toggled(VK_F1, -1, ()):
                show_help = not show_help
            if toggled(VK_H, key, (ord("h"), ord("H"))):
                ui_command("toggle_debug")
            if toggled(VK_C, key, (ord("c"), ord("C"))):
                print("[KEY] 사격 통계 초기화")
                ui_command("reset_stats")
            if toggled(VK_R, key, (ord("r"), ord("R"))):
                print("[KEY] 탐색 자세 복귀")
                search_pose()
            if toggled(VK_M, key, (ord("m"), ord("M"))):
                print("[KEY] 표적 기동 시작/정지")
                ui_command("toggle_motion")
            for motion, (vk_top, vk_pad, ch) in MOTION_KEYS.items():
                if toggled((vk_top, vk_pad), key, (ord(ch),), name=motion):
                    print(f"[KEY] 표적 기동 방식: {MOTION_LABELS[motion]}")
                    ui_command(f"motion:{motion}")
            if toggled((VK_OEM_PLUS, VK_ADD), key, (ord("+"), ord("=")), name="plus"):
                print("[KEY] 표적 속도 +5 km/h")
                ui_command("speed:+5")
            if toggled((VK_OEM_MINUS, VK_SUBTRACT), key, (ord("-"),), name="minus"):
                print("[KEY] 표적 속도 -5 km/h")
                ui_command("speed:-5")

            # Q 도 ESC 와 같이 종료로 받는다. waitKeyEx 로만 받으면 입력이 OpenCV 창에
            # 전달되지 않는 경우 ESC 가 먹지 않아 GetAsyncKeyState 로도 받는다.
            if (key in (27, ord("q"), ord("Q")) or is_key_down(VK_ESCAPE) or is_key_down(VK_Q)
                    or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1):
                break

        except Exception:
            # 예외를 삼키면 화면이 멈춰도 원인을 알 수 없다. 같은 예외는 한 번만 출력한다.
            err = traceback.format_exc()
            if err != last_error:
                last_error = err
                print("[ERR] 뷰어 루프 예외:\n" + err)
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
            time.sleep(0.2)

    stop_event.set()
    try:
        ctypes.windll.winmm.timeEndPeriod(1)
    except Exception:
        pass
    recorder.stop()
    stream_thread.join(timeout=2.5)
    status_thread.join(timeout=1.5)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
