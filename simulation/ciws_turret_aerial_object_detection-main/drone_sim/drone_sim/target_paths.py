"""target_paths.py — 이동 표적 궤적 (ROS 비의존, tests/test_target_paths.py 에서 시험).

모든 함수는 변위 (좌우 lateral: 시선에 수직인 수평 방향, 상하 vertical: 월드 수직) 를 돌려준다.
월드 좌표로 옮기는 것은 target_mover.py 가 한다.
"""

from __future__ import annotations

import bisect
import math


def cross_offset(t: float, speed: float, half_length: float) -> tuple[float, float]:
    """등속 왕복의 (변위, 속도 부호). 0 에서 출발해 +half_length 로 먼저 간다."""
    if speed <= 0.0 or half_length <= 0.0:
        return 0.0, 0.0
    period = 4.0 * half_length / speed
    phase = (t % period) / period  # 0 -> +L -> 0 -> -L -> 0
    if phase < 0.25:
        return 4.0 * half_length * phase, 1.0
    if phase < 0.75:
        return half_length - 4.0 * half_length * (phase - 0.25), -1.0
    return -half_length + 4.0 * half_length * (phase - 0.75), 1.0


def straight_offset(t: float, speed: float, half_length: float) -> float:
    """-half_length ~ +half_length 구간의 등속 직선. 0(중심)에서 + 쪽으로 출발하고, +끝에 닿으면
    -끝으로 옮겨 가 같은 방향으로 다시 지나간다(표적이 계속 새로 들어오는 반복 통과)."""
    if speed <= 0.0 or half_length <= 0.0:
        return 0.0
    return (speed * t + half_length) % (2.0 * half_length) - half_length


class Figure8Path:
    """좌우 x 상하 평면의 8자 기동 (리사주 1:2), 속력 일정.

        lateral  = A sin(th)
        vertical = B sin(2 th)

    th 를 호 길이로 다시 매개화해 궤적 위 속력이 정확히 speed 로 일정하다. 중심(스폰 위치)을
    한 주기에 두 번 지나고, 방향 전환은 곡선으로만 한다(가속도 유한). 가장 급한 선회는 좌우
    끝점에서 일어나며 곡률반경 4 B^2 / A, 구심가속도 speed^2 A / (4 B^2) 이다.
    """

    def __init__(self, speed: float, half_width: float, half_height: float, samples: int = 4096) -> None:
        self.speed = float(speed)
        self.a = float(half_width)
        self.b = float(half_height)
        self._th = [2.0 * math.pi * i / samples for i in range(samples + 1)]
        self._s = [0.0]
        px, py = self._xy(0.0)
        for th in self._th[1:]:
            x, y = self._xy(th)
            self._s.append(self._s[-1] + math.hypot(x - px, y - py))
            px, py = x, y
        self.length = self._s[-1]
        self.period = self.length / self.speed if self.speed > 0.0 else math.inf

    def _xy(self, th: float) -> tuple[float, float]:
        return self.a * math.sin(th), self.b * math.sin(2.0 * th)

    def offset(self, t: float) -> tuple[float, float]:
        """모의 시각 t (출발 0) 의 (lateral, vertical) 변위."""
        if self.speed <= 0.0 or self.length <= 0.0:
            return 0.0, 0.0
        s = (self.speed * t) % self.length
        i = min(max(bisect.bisect_right(self._s, s) - 1, 0), len(self._s) - 2)
        s0, s1 = self._s[i], self._s[i + 1]
        w = (s - s0) / (s1 - s0) if s1 > s0 else 0.0
        return self._xy(self._th[i] + w * (self._th[i + 1] - self._th[i]))

    def max_lateral_accel(self) -> float:
        """좌우 끝점의 구심가속도 (m/s^2). B = 0 이면 무한대(순간 반전)."""
        if self.b <= 0.0:
            return math.inf
        return self.speed ** 2 * self.a / (4.0 * self.b ** 2)
