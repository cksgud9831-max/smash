#!/usr/bin/env python3
"""기록된 측정값으로 표적 운동 추정기를 오프라인 재생해 비교한다. ROS·시뮬레이터 불필요.

입력: <label>.track_diag.jsonl (fcs_timed.py, est_diag 필드 포함). FCS 가 실제로 받아들인 측정
(admitted 이고 meas_pos 가 있는 프레임)만 추정기에 넣는다. FCS 는 STABLE 재포착 때 추정기를 새로
만드므로, 받아들인 측정 사이가 0.5 s 넘게 비면 추정기를 새로 시작한다.

평가: 각 영상 뒤에 "지금 쏘면 탄이 도착할 때"(촬영 시각 + 격발 지연 + 비행시간) 표적이 어디 있을지
예측하고, 그 시각의 실측과 비교한다. 오차 중 시선에 수직인 성분(가로·세로)만 명중을 좌우하므로 그것만 본다.
측정 자체에 있는 일정한 치우침(바운딩박스 중심과 모델 원점 차이로 추정)은 추정기와 무관하므로,
실행마다 정지 구간의 측정 치우침 중앙값을 빼고 "추정기 탓 오차"로 비교한다.

사용 예:
    python estimator_replay.py results/sim_runs/est_amb5_r1 results/sim_runs/est_amb5_r2
"""

from __future__ import annotations

import math
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from est_diag import REVERSAL_BINS, _axes, _interp, analyze, load  # noqa: E402

from aiming_engine.target_state import (  # noqa: E402
    ConstantAccelerationKalmanEstimator,
    ConstantVelocityEstimator,
)
from aiming_engine.types import Vector3  # noqa: E402

HIT_RADIUS_M = 0.1975
FIRE_LATENCY_S = 0.04  # 촬영 → 격발 (측정 중앙값 약 40 ms)
RESEED_GAP_S = 0.5


class ResettingCV:
    """[시험용 원형] 등속 OLS + 기동 감지 시 기록 비우기.

    새 측정이 직전 추정의 예측과 시선에 수직 방향으로 gate_m 넘게 어긋나는 일이 연속 n_hits 번이면
    표적이 방향을 바꿨다고 보고, 최근 keep 개 샘플만 남긴다. 시선 방향(거리) 성분은 거리 추정 잡음이
    커서 보지 않는다. 시선은 원점(총구 근처)에서 표적으로의 방향으로 근사한다.
    """

    def __init__(self, history_length: int = 20, gate_m: float = 0.25, n_hits: int = 2, keep: int = 3):
        self.history_length, self.gate_m, self.n_hits, self.keep = history_length, gate_m, n_hits, keep
        self._inner = ConstantVelocityEstimator(history_length, 1e-3)
        self._hits = 0
        self.resets = 0

    def update(self, position: Vector3, t: float) -> None:
        inner = self._inner
        if len(inner._times) >= 3:
            z = position.to_array()
            d = z - inner.predict(t).position.to_array()
            u = z / (np.linalg.norm(z) or 1.0)
            perp = d - (d @ u) * u
            self._hits = self._hits + 1 if np.linalg.norm(perp) > self.gate_m else 0
            if self._hits >= self.n_hits:
                times, pos = list(inner._times)[-self.keep:], list(inner._positions)[-self.keep:]
                inner._times = deque(times, maxlen=self.history_length)
                inner._positions = deque(pos, maxlen=self.history_length)
                inner._smoothed_at_t = None
                self._hits = 0
                self.resets += 1
        inner.update(position, t)

    def predict(self, t: float):
        return self._inner.predict(t)


CANDIDATES = {
    "CV20 (현재)": lambda: ConstantVelocityEstimator(20, 1e-3),
    "CV12": lambda: ConstantVelocityEstimator(12, 1e-3),
    "CV8": lambda: ConstantVelocityEstimator(8, 1e-3),
    "CV6": lambda: ConstantVelocityEstimator(6, 1e-3),
    "CV4": lambda: ConstantVelocityEstimator(4, 1e-3),
    "CA-KF q1 r0.1 (설정값)": lambda: ConstantAccelerationKalmanEstimator(1e-3, 1.0, 0.1),
    "CA-KF q5 r0.1": lambda: ConstantAccelerationKalmanEstimator(1e-3, 5.0, 0.1),
    "CA-KF q20 r0.1": lambda: ConstantAccelerationKalmanEstimator(1e-3, 20.0, 0.1),
    "CA-KF q20 r0.3": lambda: ConstantAccelerationKalmanEstimator(1e-3, 20.0, 0.3),
    "CV20+리셋 g0.25": lambda: ResettingCV(20, 0.25, 2, 3),
    "CV20+리셋 g0.15": lambda: ResettingCV(20, 0.15, 2, 3),
    "CV12+리셋 g0.25": lambda: ResettingCV(12, 0.25, 2, 3),
    # aiming_engine 에 넣은 구현 (각도 문턱, 발사점 기준 시선)
    "엔진 CV20+리셋 7mrad": lambda: ConstantVelocityEstimator(20, 1e-3, maneuver_reset=True, maneuver_gate_mrad=7.0),
    "엔진 CV20+리셋 5mrad": lambda: ConstantVelocityEstimator(20, 1e-3, maneuver_reset=True, maneuver_gate_mrad=5.0),
    "엔진 CV20+리셋 10mrad": lambda: ConstantVelocityEstimator(20, 1e-3, maneuver_reset=True, maneuver_gate_mrad=10.0),
}


def static_offset(recs: list[dict]) -> np.ndarray:
    """정지 구간 측정의 (가로, 세로) 치우침 중앙값."""
    rows = analyze(recs)["rows"]
    offs = [r["meas"][1:] for r in rows if "meas" in r and abs(r["v_lat"]) < 0.2]
    return np.median(np.array(offs), axis=0) if offs else np.zeros(2)


def replay(recs: list[dict], make, fire_latency_s: float = FIRE_LATENCY_S):
    recs = sorted(recs, key=lambda r: r["t"])
    ts = np.array([r["t"] for r in recs])
    gts = np.array([r["gt_pos"] for r in recs], dtype=float)
    context = {r["t"]: r for r in analyze(recs)["rows"]}
    est, last_meas_t, out = None, None, []
    for r in recs:
        if r.get("admitted") and r.get("meas_pos") is not None:
            if est is None or (last_meas_t is not None and r["t"] - last_meas_t > RESEED_GAP_S):
                est = make()
            z = Vector3.from_array(np.array(r["meas_pos"], dtype=float))
            if isinstance(est, ConstantVelocityEstimator):
                est.update(z, r["t"], observer=np.array(r["cam_pos"], dtype=float))
            else:
                est.update(z, r["t"])
            last_meas_t = r["t"]
        if est is None or r.get("tof") is None or last_meas_t is None or r["t"] - last_meas_t > RESEED_GAP_S:
            continue
        t_imp = r["t"] + fire_latency_s + r["tof"]
        gt_imp = _interp(ts, gts, t_imp)
        if gt_imp is None:
            continue
        pred = est.predict(t_imp).position.to_array()
        _, lat, ver = _axes(gt_imp, np.array(r["cam_pos"], dtype=float))
        e = pred - gt_imp
        c = context[r["t"]]
        out.append({"t": r["t"], "lat": float(e @ lat), "ver": float(e @ ver), "v_lat": c["v_lat"],
                    "since_rev": c["since_rev"]})
    return out, est


def score(rows: list[dict], offset: np.ndarray) -> dict:
    moving = [r for r in rows if abs(r["v_lat"]) >= 1.0]
    steady = [r for r in moving if r["since_rev"] is not None and r["since_rev"] >= 0.8]
    static = [r for r in rows if abs(r["v_lat"]) < 0.2]

    def perp(r):
        return math.hypot(r["lat"] - offset[0], r["ver"] - offset[1])

    def within(rs):
        return None if not rs else sum(perp(r) <= HIT_RADIUS_M for r in rs) / len(rs)

    res = {"moving_within": within(moving), "static_within": within(static),
           "steady_lat_std": float(np.std([r["lat"] for r in steady])) if len(steady) > 2 else None,
           "static_lat_std": float(np.std([r["lat"] for r in static])) if len(static) > 2 else None,
           "bins": []}
    for lo, hi in REVERSAL_BINS:
        rs = [r for r in moving if r["since_rev"] is not None and lo <= r["since_rev"] < hi]
        res["bins"].append(within(rs))
    return res


def main() -> None:
    prefixes = sys.argv[1:]
    runs = [(p, load(p)) for p in prefixes]
    offsets = {p: static_offset(recs) for p, recs in runs}
    for p, off in offsets.items():
        print(f"{os.path.basename(p)}: 측정 치우침(가로, 세로) {off[0]:+.3f}, {off[1]:+.3f} m — 추정기 비교에서 뺀다")
    bins_hdr = " ".join(f"{lo:.1f}-" if hi != math.inf else f"{lo:.1f}+" for lo, hi in REVERSAL_BINS[:5])
    print(f"\n{'추정기':24s} {'이동 중':>7} {'등속 흔들림':>10} {'정지 흔들림':>10} | 방향 전환 뒤 경과 s 별 명중 반경 안 비율: {bins_hdr}")
    for name, make in CANDIDATES.items():
        rows_all, resets = [], 0
        agg = {"moving": [], "bins": [[] for _ in REVERSAL_BINS]}
        for p, recs in runs:
            rows, est = replay(recs, make)
            resets += getattr(est, "resets", 0) + getattr(est, "maneuver_resets", 0)
            s = score(rows, offsets[p])
            rows_all.append(s)
        # 실행별 비율을 단순 평균한다(실행 길이가 비슷하다).
        def mean(xs):
            xs = [x for x in xs if x is not None]
            return None if not xs else sum(xs) / len(xs)
        mv = mean([s["moving_within"] for s in rows_all])
        sd = mean([s["steady_lat_std"] for s in rows_all])
        st = mean([s["static_lat_std"] for s in rows_all])
        bins = [mean([s["bins"][k] for s in rows_all]) for k in range(len(REVERSAL_BINS) - 1)]
        fmt = lambda x, f="{:.0%}": "-" if x is None else f.format(x)  # noqa: E731
        extra = f"  (리셋 {resets}회)" if resets else ""
        print(f"{name:24s} {fmt(mv):>7} {fmt(sd, '{:.3f} m'):>10} {fmt(st, '{:.3f} m'):>10} | "
              + " ".join(f"{fmt(b):>5}" for b in bins) + extra)


if __name__ == "__main__":
    main()
