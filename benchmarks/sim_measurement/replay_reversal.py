#!/usr/bin/env python3
"""왕복 표적 실행 기록(track_samples)으로 운동 추정기 변형을 오프라인 재생해 전환 직후 리드 오차를 비교한다.

입력: auto_aim_probe 결과 JSON(`meas_pos_world_m`, `observer_world_m` 가 기록된 실행). ROS·시뮬레이터 불필요.
각 프레임의 실제 FCS 입력(받아들인 측정)을 추정기에 다시 넣고, ``predict(stamp + lag)`` 의 시선 수직(횡) 오차를
표적 실측과 비교한다. 실측 비교 시각은 기준 추정기의 정상 구간 편향이 최소가 되는 지연으로 맞춘다
(측정 bbox 가 표적보다 늦는 효과를 추정기와 무관한 상수로 보고 모든 변형에 똑같이 적용).

전환부 판정은 실측 속도 부호 변화 시각 기준 경과시간 구간별로 |횡오차| <= 명중 반경 비율을 비교한다.

사용: python replay_reversal.py RESULT.json [RESULT.json ...]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from aiming_engine.target_state import ConstantVelocityEstimator  # noqa: E402
from aiming_engine.types import Vector3  # noqa: E402

HIT_RADIUS_M = 0.1975
LAG_S = 0.17  # FCS 의 motion_measurement_lag_s
BINS = [(-9, 0.0), (0.0, 0.15), (0.15, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 1.0), (1.0, 2.0), (2.0, 99)]


def load(path):
    d = json.load(open(path))
    gt_t, gt_p = [], []
    frames = []
    for s in d["track_samples"]:
        if s.get("ground_truth_world_m") is not None and (not gt_t or s["ground_truth_stamp"] > gt_t[-1]):
            gt_t.append(s["ground_truth_stamp"])
            gt_p.append(s["ground_truth_world_m"])
        if s.get("measurement_admitted") and s.get("meas_pos_world_m") and s.get("observer_world_m"):
            frames.append((s["stamp"], np.array(s["meas_pos_world_m"]), np.array(s["observer_world_m"])))
    gt_t, gt_p = np.array(gt_t), np.array(gt_p)
    vy = np.gradient(gt_p[:, 1], gt_t)
    sg = np.sign(np.convolve(vy, np.ones(5) / 5, "same"))
    reversals = gt_t[1:][sg[1:] * sg[:-1] < 0]
    return frames, gt_t, gt_p, reversals


def truth_at(gt_t, gt_p, t):
    return np.array([np.interp(t, gt_t, gt_p[:, k]) for k in range(3)])


def run_variant(factory, frames, gt_t, gt_p, reversals, delta, reseed_gap=0.5):
    est = None
    last_t = None
    rows = []
    for stamp, pos, obs in frames:
        if est is None or (last_t is not None and stamp - last_t > reseed_gap):
            est = factory()
        est.update(Vector3.from_array(pos), stamp, observer=obs)
        last_t = stamp
        if not (gt_t[0] <= stamp + delta <= gt_t[-1]):
            continue
        lead = est.predict(stamp + LAG_S).position.to_array()
        truth = truth_at(gt_t, gt_p, stamp + delta)
        los = truth - obs
        u = los / np.linalg.norm(los)
        err = lead - truth
        perp = err - (err @ u) * u
        # 진행방향 기준 부호: 월드 y 의 실측 속도 부호
        k = min(max(np.searchsorted(gt_t, stamp + delta), 1), len(gt_t) - 1)
        direction = np.sign(gt_p[k, 1] - gt_p[k - 1, 1]) or 1.0
        prev = reversals[reversals <= stamp + delta]
        since = stamp + delta - prev[-1] if len(prev) else np.nan
        k_used = getattr(est, 'chosen_k', None) or len(est._times)
        rows.append((since, float(np.linalg.norm(perp)), float(perp[1] * direction), k_used))
    return np.array(rows)


def calibrate_delta(factory, frames, gt_t, gt_p, reversals):
    best = None
    for delta in np.arange(-0.1, 0.31, 0.02):
        r = run_variant(factory, frames, gt_t, gt_p, reversals, delta)
        steady = r[(r[:, 0] > 1.0)] if len(r) else r
        if len(steady) < 20:
            continue
        score = abs(float(np.median(steady[:, 2])))
        if best is None or score < best[0]:
            best = (score, float(delta))
    return best[1] if best else 0.0


def gate_report(name, rows, ks=(8, 10, 12)):
    ok = rows[:, 1] <= HIT_RADIUS_M
    out = []
    for kmin in ks:
        m = rows[:, 3] >= kmin
        tr = m & (rows[:, 0] > 0.15) & (rows[:, 0] <= 0.6)
        out.append(f"k>={kmin}: pass {m.mean():.2f} inrad {ok[m].mean():.2f} | 0.15-0.6s pass {tr.sum()}/{((rows[:,0]>0.15)&(rows[:,0]<=0.6)).sum()} inrad {ok[tr].mean() if tr.any() else float('nan'):.2f}")
    return f"{'':34s} " + "  ".join(out)


def report(name, rows):
    ok = rows[:, 1] <= HIT_RADIUS_M
    out = [f"{name:34s} all {ok.mean():.2f}"]
    for lo, hi in BINS[1:]:
        m = (rows[:, 0] > lo) & (rows[:, 0] <= hi)
        if m.sum():
            out.append(f"{lo:.2f}-{hi:.2f}: {ok[m].mean():.2f}/{m.sum()}")
    return "  ".join(out)


def main(paths, variants):
    data = [load(p) for p in paths]
    base_factory = next(iter(variants.values()))
    delta = np.median([calibrate_delta(base_factory, f, gt, gp, rv) for f, gt, gp, rv in data])
    print(f"실측 비교 지연 delta = {delta:+.2f} s (기준 추정기의 정상 구간 편향 최소)")
    for name, factory in variants.items():
        rows = np.vstack([run_variant(factory, f, gt, gp, rv, delta) for f, gt, gp, rv in data])
        print(report(name, rows))
        print(gate_report(name, rows))


class AdaptiveWindowCV(ConstantVelocityEstimator):
    """[시험용] 횡방향 직선성을 만족하는 가장 긴 창으로 적합한다.

    횡성분(시선에 수직인 수평 방향) 잔차 RMSE 가 tau 이하인 가장 긴 최근 k 개(k_min..history) 샘플로 3축을
    OLS 적합한다. 전환이 창에 걸리면 긴 창의 횡잔차가 커져 전환 이후 구간만 쓰게 된다.
    """

    def __init__(self, *args, tau_m=0.06, k_min=6, **kw):
        super().__init__(*args, **kw)
        self.tau, self.k_min = tau_m, k_min
        self.chosen_k = None

    def predict(self, t_future):
        n = len(self._times)
        if n < 2 or self._lateral_axis is None or n <= self.k_min:
            return super().predict(t_future)
        last_t = self._times[-1]
        ts_all = np.array(self._times) - last_t
        ps_all = np.array(self._positions)
        lat_all = ps_all @ self._lateral_axis
        chosen = self.k_min
        for k in range(n, self.k_min - 1, -1):
            a = np.column_stack([ts_all[-k:], np.ones(k)])
            c, *_ = np.linalg.lstsq(a, lat_all[-k:], rcond=None)
            rmse = float(np.sqrt(np.mean((lat_all[-k:] - a @ c) ** 2)))
            if rmse <= self.tau:
                chosen = k
                break
        self.chosen_k = chosen
        a = np.column_stack([ts_all[-chosen:], np.ones(chosen)])
        coeffs, *_ = np.linalg.lstsq(a, ps_all[-chosen:], rcond=None)
        from aiming_engine.types import TargetVector
        return TargetVector(
            position=Vector3.from_array(coeffs[1] + coeffs[0] * (t_future - last_t)),
            velocity=Vector3.from_array(coeffs[0]),
            acceleration=Vector3(0.0, 0.0, 0.0),
            timestamp=t_future,
        )


def default_variants():
    def cv(**kw):
        base = dict(history_length=20, min_dt=1e-3, velocity_smoothing_alpha=1.0, maneuver_reset=True,
                    maneuver_gate_mrad=10.0, maneuver_gate_min_m=0.1, maneuver_hits=2, maneuver_keep=3)
        base.update(kw)
        return lambda: ConstantVelocityEstimator(**base)

    return {
        "baseline(현행)": cv(),
        "hits=1": cv(maneuver_hits=1),
        "history=12": cv(history_length=12),
        "history=10": cv(history_length=10),
        "no maneuver reset": cv(maneuver_reset=False),
        **{
            f"adaptive tau={tau} reset={'on' if rs else 'off'}": (
                lambda tau=tau, rs=rs: AdaptiveWindowCV(
                    history_length=20, min_dt=1e-3, velocity_smoothing_alpha=1.0, maneuver_reset=rs,
                    maneuver_gate_mrad=10.0, maneuver_gate_min_m=0.1, maneuver_hits=2, maneuver_keep=3,
                    tau_m=tau)
            )
            for tau in (0.03, 0.05, 0.08, 0.12)
            for rs in (True, False)
        },
    }


if __name__ == "__main__":
    main(sys.argv[1:], default_variants())
