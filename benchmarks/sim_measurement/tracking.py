"""auto_aim_probe.py --track 의 "모의 사수" 추종 제어 (ROS 없이 시험할 수 있게 분리).

FCS 가 발행하는 /smash_fcs/aim_track 의 리드 방향과 총구 방향을 보고, 포탑 수동 조작
명령(/smash_fcs/slew_cmd, 증분 라디안)으로 총구를 리드 레티클에 붙인다.

설계 (2026-09-21 실측으로 바꿈):
    pan-tilt 포탑에서 총구 방위각은 pan 명령과, 앙각은 tilt 명령과 상수 차이만 난다.
    그래서 "리드 방향 - 그 상수"를 명령으로 직접 내린다(개루프). 상수는 관측(총구 방향 - 그 영상
    때쯤의 명령)으로 천천히 배운다. 이 학습이 적분 동작 구실을 해서 포탑 지연 추정이 틀려도
    정상상태 오차가 남지 않는다.

    처음에는 "지금 총구 = 관측 총구 + 그 뒤에 내린 명령"으로 추정해 오차에 이득을 주는
    스미스 예측기 방식이었다. 실제 포탑 지연이 가정(0.1 s)보다 길자 추종이 1.8배 과속해
    표적을 놓쳤다. 예측기는 지연 모델이 틀리면 되먹임이 불안정해지기 쉽다.

    리드는 각속도를 추정해 "지금 + 서보 지연 + 명령 계단 반 칸" 시점까지 앞당겨 겨누고,
    영상 사이에는 추정 각속도만큼 명령을 계속 앞당긴다(피드포워드).
"""

from __future__ import annotations

import bisect
import math


def az_el(d) -> tuple[float, float]:
    x, y, z = (float(v) for v in d)
    return math.atan2(y, x), math.atan2(z, math.hypot(x, y))


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class LeadTracker:
    def __init__(self, servo_lag_s: float = 0.1, offset_alpha: float = 0.1, rate_alpha: float = 0.3,
                 max_rate_rad_s: float = 1.0, step_s: float = 1.0 / 30.0):
        self.servo_lag_s = float(servo_lag_s)
        # 명령은 계단으로 바뀐다(FCS 명령 타이머 30 Hz). 계단 한가운데가 리드에 오도록 반 칸 앞을 겨눈다.
        self.step_s = float(step_s)
        self.offset_alpha = float(offset_alpha)
        self.rate_alpha = float(rate_alpha)
        self.max_rate = float(max_rate_rad_s)
        self.cmd = [0.0, 0.0]
        self.rate = [0.0, 0.0]
        self.offset: list[float] | None = None  # 총구 각 - 명령 각
        self._hist_t: list[float] = []
        self._hist_cmd: list[tuple[float, float]] = []
        self._prev = None  # (stamp, lead_az, lead_el)
        self._target = None  # (기준 시각, 리드 방위각, 리드 앙각) — 피드포워드 기준

    def set_command(self, t: float, pan: float, tilt: float) -> None:
        self.cmd = [float(pan), float(tilt)]
        self._hist_t.append(float(t))
        self._hist_cmd.append((self.cmd[0], self.cmd[1]))
        if len(self._hist_t) > 2000:
            del self._hist_t[:1000]
            del self._hist_cmd[:1000]

    def command_at(self, t: float) -> tuple[float, float]:
        if not self._hist_t:
            return self.cmd[0], self.cmd[1]
        i = bisect.bisect_right(self._hist_t, t) - 1
        return self._hist_cmd[max(i, 0)]

    def _command_for(self, now: float) -> tuple[float, float]:
        t_ref, la, le = self._target
        h = now + self.servo_lag_s + 0.5 * self.step_s - t_ref
        return (la + self.rate[0] * h - self.offset[0], le + self.rate[1] * h - self.offset[1])

    def correction(self, now: float, stamp: float, fire_time: float, lead_dir, bore_dir) -> tuple[float, float]:
        """새 aim_track 메시지 하나로 증분 명령(dpan, dtilt)을 만들고 내부 명령값을 갱신한다."""
        la, le = az_el(lead_dir)
        ba, be = az_el(bore_dir)
        if self._prev is not None and stamp > self._prev[0]:
            dt = stamp - self._prev[0]
            inst = (wrap(la - self._prev[1]) / dt, (le - self._prev[2]) / dt)
            for k in (0, 1):
                r = (1.0 - self.rate_alpha) * self.rate[k] + self.rate_alpha * inst[k]
                self.rate[k] = max(-self.max_rate, min(self.max_rate, r))
        self._prev = (stamp, la, le)

        then = self.command_at(stamp - self.servo_lag_s)
        obs = (wrap(ba - then[0]), be - then[1])
        if self.offset is None:
            self.offset = [obs[0], obs[1]]
        else:
            a = self.offset_alpha
            self.offset = [(1 - a) * self.offset[0] + a * obs[0], (1 - a) * self.offset[1] + a * obs[1]]

        self._target = (fire_time, la, le)
        pan, tilt = self._command_for(now)
        inc = (wrap(pan - self.cmd[0]), tilt - self.cmd[1])
        self.set_command(now, self.cmd[0] + inc[0], self.cmd[1] + inc[1])
        return inc

    def coarse(self, now: float, target_dir, target_rate: tuple[float, float]) -> tuple[float, float]:
        """표적을 잃었을 때 실제 표적 방향(사람이 눈으로 다시 찾는 것에 해당)으로 대강 겨눈다.

        target_dir 은 총구 기준 표적 방향, target_rate 는 그 방위각·앙각 변화율(rad/s).
        리드는 없지만 표적이 화면 안에 들어오기만 하면 된다. 다시 STABLE 이 되면 correction 이 이어받는다.
        """
        az, el = az_el(target_dir)
        self.rate = [max(-self.max_rate, min(self.max_rate, float(r))) for r in target_rate]
        if self.offset is None:
            self.offset = [0.0, 0.0]
        self._target = (now, az, el)  # _command_for 가 오프셋을 빼므로 총구가 표적을 향한다
        self._prev = None  # 리드 각속도 추정은 새로 시작한다
        pan, tilt = self._command_for(now)
        inc = (wrap(pan - self.cmd[0]), tilt - self.cmd[1])
        self.set_command(now, self.cmd[0] + inc[0], self.cmd[1] + inc[1])
        return inc

    def feedforward(self, now: float) -> tuple[float, float]:
        """영상 사이: 마지막 리드와 각속도로 지금 명령을 다시 계산한다."""
        if self._target is None or self.offset is None:
            self.set_command(now, *self.cmd)
            return 0.0, 0.0
        pan, tilt = self._command_for(now)
        inc = (wrap(pan - self.cmd[0]), tilt - self.cmd[1])
        self.set_command(now, self.cmd[0] + inc[0], self.cmd[1] + inc[1])
        return inc
