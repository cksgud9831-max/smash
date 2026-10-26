"""Local recording player. Reads files only; never connects to the simulator."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

RECORDINGS = Path(__file__).resolve().parents[1] / "recordings"


def timestamp(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class VideoFile:
    def __init__(self, path):
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError("영상을 열 수 없습니다. 파일 또는 코덱을 확인하세요.")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(self.fps) or self.fps <= 0 or self.count <= 0:
            self.close()
            raise ValueError("재생 시간 정보를 읽을 수 없습니다.")
        self.index = -1
        self.duration = self.count / self.fps

    def frame_at(self, seconds):
        index = min(self.count - 1, max(0, int(seconds * self.fps)))
        if index == self.index:
            return None
        if index != self.index + 1:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        if not ok:
            raise ValueError("영상 프레임이 손상되었거나 읽을 수 없습니다.")
        self.index = index
        return frame

    def close(self):
        self.capture.release()


class RecordingPlayer:
    def __init__(self, root):
        self.root = root
        root.title("녹화 재생기 · 로컬 파일")
        root.geometry("1080x800")
        root.minsize(640, 480)
        root.configure(bg="#151b23")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#151b23")
        style.configure("TLabel", background="#151b23", foreground="#dce5ef")
        style.configure("TButton", padding=(12, 8))
        self.video = None
        self.playing = False
        self.position = 0.0
        self.anchor = time.monotonic()
        self.anchor_position = 0.0
        self.rate = 1.0
        self.scrubbing = False
        self.frame = None
        self.photo = None
        self.last_size = None
        header = ttk.Frame(root, padding=12)
        header.pack(fill="x")
        ttk.Button(header, text="파일 열기", command=self.choose).pack(side="left")
        ttk.Button(header, text="녹화 폴더", command=self.choose_recording).pack(side="left", padx=8)
        self.filename = tk.StringVar(value="MP4 / AVI / MKV 파일을 선택하세요")
        ttk.Label(header, textvariable=self.filename).pack(side="left", padx=12)
        self.canvas = tk.Canvas(root, bg="#0c1118", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=12)
        controls = ttk.Frame(root, padding=12)
        controls.pack(fill="x")
        self.timeline = ttk.Scale(controls, from_=0, to=1)
        self.timeline.pack(fill="x", pady=(0, 10))
        self.timeline.bind("<ButtonPress-1>", self.begin_scrub)
        self.timeline.bind("<ButtonRelease-1>", self.end_scrub)
        self.play_button = ttk.Button(controls, text="재생", command=self.toggle)
        self.play_button.pack(side="left")
        ttk.Button(controls, text="−5초", command=lambda: self.seek(self.position - 5)).pack(side="left", padx=6)
        ttk.Button(controls, text="+5초", command=lambda: self.seek(self.position + 5)).pack(side="left")
        self.speed = tk.StringVar(value="1.0x")
        speed = ttk.Combobox(controls, textvariable=self.speed, values=("0.25x", "0.5x", "1.0x", "1.5x", "2.0x"),
                             state="readonly", width=7)
        speed.pack(side="left", padx=12)
        speed.bind("<<ComboboxSelected>>", self.change_speed)
        self.clock = tk.StringVar(value="00:00:00 / 00:00:00")
        ttk.Label(controls, textvariable=self.clock).pack(side="right")
        self.status = tk.StringVar(value="Space 재생·일시정지   ←/→ 5초 이동   F 전체화면   Esc 전체화면 해제")
        ttk.Label(root, textvariable=self.status, padding=(12, 4, 12, 12)).pack(fill="x")
        root.bind("<space>", lambda e: self.toggle())
        root.bind("<Left>", lambda e: self.seek(self.position - 5))
        root.bind("<Right>", lambda e: self.seek(self.position + 5))
        root.bind("<f>", lambda e: root.attributes("-fullscreen", not root.attributes("-fullscreen")))
        root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.timer = root.after(30, self.tick)

    def choose_recording(self):
        RECORDINGS.mkdir(exist_ok=True)
        self.choose(RECORDINGS)

    def choose(self, directory=None):
        path = filedialog.askopenfilename(parent=self.root, initialdir=str(directory or RECORDINGS),
                                         filetypes=[("영상", "*.mp4 *.avi *.mkv *.mov"), ("모든 파일", "*.*")])
        if path:
            self.open(path)

    def open(self, path):
        candidate = None
        try:
            candidate = VideoFile(path)
            frame = candidate.frame_at(0)
        except (ValueError, cv2.error) as exc:
            if candidate:
                candidate.close()
            messagebox.showerror("파일 열기 실패", str(exc), parent=self.root)
            return False
        if self.video:
            self.video.close()
        self.video = candidate
        self.frame = frame
        self.last_size = None
        self.position = 0.0
        self.playing = False
        self.play_button.configure(text="재생")
        self.timeline.configure(to=self.video.duration)
        self.filename.set(Path(path).name)
        self.status.set(str(Path(path).resolve()))
        self.seek(0)
        return True

    def seek(self, seconds):
        if self.video:
            self.position = min(self.video.duration, max(0, seconds))
            self.anchor_position = self.position
            self.anchor = time.monotonic()

    def begin_scrub(self, event):
        self.scrubbing = True

    def end_scrub(self, event):
        self.seek(self.timeline.get())
        self.scrubbing = False

    def toggle(self):
        if not self.video:
            return
        if not self.playing and self.position >= self.video.duration:
            self.seek(0)
        self.playing = not self.playing
        self.seek(self.position)
        self.play_button.configure(text="일시정지" if self.playing else "재생")

    def change_speed(self, event=None):
        self.seek(self.position)
        self.rate = float(self.speed.get().rstrip("x"))

    def tick(self):
        if self.video:
            if self.playing and not self.scrubbing:
                self.position = min(self.video.duration, self.anchor_position +
                                    (time.monotonic() - self.anchor) * self.rate)
                if self.position >= self.video.duration:
                    self.playing = False
                    self.play_button.configure(text="다시 재생")
            try:
                frame = self.video.frame_at(self.position)
                if frame is not None:
                    self.frame = frame
                size = (max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height()))
                if self.frame is not None and (frame is not None or size != self.last_size):
                    image = Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
                    ratio = min(size[0] / image.width, size[1] / image.height)
                    image = image.resize((max(1, round(image.width * ratio)),
                                          max(1, round(image.height * ratio))), Image.Resampling.LANCZOS)
                    self.photo = ImageTk.PhotoImage(image)
                    self.canvas.delete("all")
                    self.canvas.create_image(size[0]//2, size[1]//2, image=self.photo)
                    self.last_size = size
            except (ValueError, cv2.error) as exc:
                self.playing = False
                self.play_button.configure(text="재생")
                self.status.set(str(exc))
            if not self.scrubbing:
                self.timeline.set(self.position)
            self.clock.set(f"{timestamp(self.position)} / {timestamp(self.video.duration)}")
        self.timer = self.root.after(30, self.tick)

    def close(self):
        self.root.after_cancel(self.timer)
        if self.video:
            self.video.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", help="Local video file")
    args = parser.parse_args()
    root = tk.Tk()
    player = RecordingPlayer(root)
    if args.file:
        player.open(args.file)
    root.mainloop()


if __name__ == "__main__":
    main()
