"""scope_controls.py — 뷰어 패널의 표적 생성 버튼(방식·속도 선택 + 즉시 생성).

GUI 없이 시험할 수 있게 상태(SpawnControls)·버튼 배치(control_layout)·좌표 변환을 순수 파이썬으로 둔다.
그리기는 scope_panel, 마우스 입력 처리는 win_scope_viewer 가 한다.

명령 형식은 WSL 쪽 drone_sim/smash_fcs/spawn_command.py 와 같다("spawn:<방식>:<km/h>").
"""

from __future__ import annotations

from typing import Optional

PANEL_WIDTH = 290
PAD = 16
CONTROL_H = 134          # 표적 생성 블록 전체 높이
FOOTER_H = 54            # 하단 연결·녹화 줄. 조작 안내를 켜면 FOOTER_HELP_H
FOOTER_HELP_H = 110

SPAWN_MOTIONS = (("straight", "직선"), ("cross", "왕복"), ("maneuver", "8자"), ("approach", "접근"))
# 속도 단계(km/h). 저속 수동 격발 연습(1~10)은 잘게, 그 위는 성기게.
SPEED_STEPS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150)
DEFAULT_MOTION = "cross"
DEFAULT_KMH = 10.0


def format_spawn(motion: str, kmh: float) -> str:
    return f"spawn:{motion}:{min(150.0, max(1.0, float(kmh))):g}"


def control_layout(height: int, show_help: bool = False) -> dict:
    """패널 좌표(원점 = 패널 왼쪽 위)의 버튼 사각형 (x0, y0, x1, y1). 그리기와 클릭 판정이 같이 쓴다."""

    footer = FOOTER_HELP_H if show_help else FOOTER_H
    top = height - footer - 6 - CONTROL_H
    inner = PANEL_WIDTH - 2 * PAD
    gap = 6
    rects = {"top": top}
    bw = (inner - 3 * gap) // 4
    y = top + 20
    for i, (name, _) in enumerate(SPAWN_MOTIONS):
        x = PAD + i * (bw + gap)
        rects["motion:" + name] = (x, y, x + bw, y + 28)
    y += 28 + 8
    rects["speed:-"] = (PAD, y, PAD + 48, y + 28)
    rects["speed:+"] = (PAD + inner - 48, y, PAD + inner, y + 28)
    rects["speed:value"] = (PAD + 48 + gap, y, PAD + inner - 48 - gap, y + 28)
    y += 28 + 8
    rects["spawn"] = (PAD, y, PAD + inner, y + 36)
    return rects


def hit_test(layout: dict, x: float, y: float) -> Optional[str]:
    """패널 좌표 (x, y) 에 있는 버튼 이름. 없으면 None. 속도 표시칸은 버튼이 아니다."""

    for name, rect in layout.items():
        if name in ("top", "speed:value"):
            continue
        x0, y0, x1, y1 = rect
        if x0 <= x < x1 and y0 <= y < y1:
            return name
    return None


def window_to_view(x: float, y: float, win_w: int, win_h: int, view_w: int, view_h: int):
    """창 안 좌표 -> letterbox 하기 전 합성 화면 좌표. 검은 여백이면 None.

    scope_caption.letterbox 가 종횡비를 유지해 가운데에 놓는 것을 거꾸로 푼다.
    """

    if min(win_w, win_h, view_w, view_h) <= 0:
        return None
    scale = min(win_w / view_w, win_h / view_h)
    nw, nh = max(1, int(round(view_w * scale))), max(1, int(round(view_h * scale)))
    ox, oy = (win_w - nw) // 2, (win_h - nh) // 2
    if not (ox <= x < ox + nw and oy <= y < oy + nh):
        return None
    return (x - ox) / scale, (y - oy) / scale


class SpawnControls:
    """다음에 생성할 표적의 방식·속도 선택 상태."""

    def __init__(self) -> None:
        self.motion = DEFAULT_MOTION
        self.kmh = DEFAULT_KMH
        self._synced = False
        self.flash_until = 0.0   # 생성 버튼을 누른 직후 잠깐 밝게 보이는 시각(monotonic)

    def sync(self, status: dict) -> None:
        """처음 표적 상태를 받았을 때 한 번만 지금 시나리오의 방식·속도로 맞춘다."""

        if self._synced:
            return
        names = [m for m, _ in SPAWN_MOTIONS]
        if status.get("motion") in names:
            self.motion = status["motion"]
            try:
                self.kmh = min(150.0, max(1.0, float(status["kmh"])))
            except (KeyError, TypeError, ValueError):
                pass
            self._synced = True

    def select_motion(self, motion: str) -> None:
        if motion in [m for m, _ in SPAWN_MOTIONS]:
            self.motion = motion

    def step_speed(self, direction: int) -> None:
        """direction +1 / -1: 속도 단계표에서 한 칸 위·아래(표에 없는 값에서도 다음 칸으로)."""

        if direction > 0:
            ups = [s for s in SPEED_STEPS if s > self.kmh + 1e-9]
            self.kmh = float(ups[0]) if ups else float(SPEED_STEPS[-1])
        else:
            downs = [s for s in SPEED_STEPS if s < self.kmh - 1e-9]
            self.kmh = float(downs[-1]) if downs else float(SPEED_STEPS[0])

    def command(self) -> str:
        return format_spawn(self.motion, self.kmh)

    def press(self, name: str) -> Optional[str]:
        """버튼 이름을 처리한다. 생성 버튼이면 보낼 명령 문자열, 아니면 None."""

        if name.startswith("motion:"):
            self.select_motion(name.split(":", 1)[1])
        elif name == "speed:-":
            self.step_speed(-1)
        elif name == "speed:+":
            self.step_speed(+1)
        elif name == "spawn":
            return self.command()
        return None
