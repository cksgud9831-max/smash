"""왕복 표적 실행 기록에서 측정 위치(meas_pos_world_m)의 횡방향 지연·잡음을 실측(ground truth)과 비교한다.

사용: python analyze_measurement_lag.py RESULT.json [RESULT.json ...]
등속 구간(실측 횡속도 0.6 m/s 이상)만 보며, 오차는 진행방향 기준 부호(음수 = 측정이 표적보다 뒤처짐)다.
지연은 -오차/속도, 고주파 잡음은 1 s 이동 중앙값을 뺀 잔차의 표준편차, 느린 성분은 그 이동 중앙값의 표준편차다.
검출 갱신(detector_refresh) 후 경과시간별 오차도 출력한다.
"""
import json
import sys

import numpy as np
import pandas as pd


def collect(path):
    ts = json.load(open(path))["track_samples"]
    gt_t, gt_p = [], []
    for s in ts:
        if s.get("ground_truth_world_m") and (not gt_t or s["ground_truth_stamp"] > gt_t[-1]):
            gt_t.append(s["ground_truth_stamp"])
            gt_p.append(s["ground_truth_world_m"])
    gt_t, gt_p = np.array(gt_t), np.array(gt_p)
    vy = np.gradient(gt_p[:, 1], gt_t)
    rows, last_ref = [], None
    for s in ts:
        t = s["stamp"]
        if s.get("detector_refresh"):
            last_ref = t
        if not (s.get("measurement_admitted") and s.get("meas_pos_world_m")) or last_ref is None:
            continue
        if not gt_t[0] <= t <= gt_t[-1]:
            continue
        v = np.interp(t, gt_t, vy)
        if abs(v) < 0.6:
            continue
        gy = np.interp(t, gt_t, gt_p[:, 1])
        rows.append((t, (s["meas_pos_world_m"][1] - gy) * np.sign(v), abs(v), t - last_ref, bool(s["detector_refresh"])))
    return pd.DataFrame(rows, columns=["t", "e", "speed", "since_ref", "refresh"])


def main(paths):
    parts = []
    for p in paths:
        df = collect(p).sort_values("t")
        df["slow"] = df.e.rolling(21, center=True, min_periods=5).median()
        df["hf"] = df.e - df.slow
        parts.append(df)
    df = pd.concat(parts)
    print(f"등속 프레임 {len(df)}: 오차 평균 {df.e.mean():+.3f} m, 지연 중앙값 {(-df.e / df.speed).median():+.3f} s")
    print(f"횡 잡음 표준편차 {df.e.std():.3f} m = 고주파 {df.hf.std():.3f} + 느린 성분 {df.slow.std():.3f}")
    print(df.groupby("refresh").e.agg(["mean", "std", "count"]).round(3).rename(index={True: "검출 갱신 프레임", False: "LK 전용 프레임"}))
    df["bin"] = pd.cut(df.since_ref, [-0.01, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0])
    print(df.groupby("bin", observed=True).e.agg(["mean", "std", "count"]).round(3))


if __name__ == "__main__":
    main(sys.argv[1:])
