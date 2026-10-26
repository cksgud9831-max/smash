#!/usr/bin/env python3
"""FCS 표적 위치·속도·리드 추정을 실측과 비교한다. <label>.track_diag.jsonl (fcs_timed.py) 을 읽는다. ROS 불필요.

프레임마다 오차를 시선(거리) / 가로(시선에 수직, 수평) / 세로 성분으로 나눈다.
    meas : 영상 한 장에서 역산한 표적 위치 - 실측        (측정 자체의 품질)
    est  : 운동 추정기가 촬영 시각에 대해 낸 위치 - 실측 (추정기의 품질)
    lead : 리드 점 - 촬영 시각 + 비행시간의 실측 위치     (조준해의 품질)

가로 오차를 표적의 가로 속도로 나누면 "추정이 몇 ms 늦는가"가 된다(양수 = 늦음, 음수 = 앞섬).
표적이 방향을 바꾼 뒤 경과 시간별로 묶어, 오차가 방향 전환 직후에만 생기는지 늘 있는지 가린다.

사용 예:
    python est_diag.py results/sim_runs/h6_amb5_r2 [...]
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np

REVERSAL_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.2), (1.2, math.inf)]
MOVING_MPS = 1.0  # 이보다 느린 프레임은 늦음(ms) 계산에서 뺀다


def load(prefix: str) -> list[dict]:
    with open(prefix + ".track_diag.jsonl", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    return [r for r in recs if r.get("gt_pos") is not None and r.get("cam_pos") is not None]


def _axes(gt: np.ndarray, cam: np.ndarray):
    u = gt - cam
    u /= np.linalg.norm(u)
    lat = np.array([-u[1], u[0], 0.0])
    n = np.linalg.norm(lat)
    lat = lat / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
    ver = np.cross(u, lat)
    return u, lat, ver


def _interp(ts: np.ndarray, ps: np.ndarray, t: float):
    if t < ts[0] or t > ts[-1]:
        return None
    return np.array([np.interp(t, ts, ps[:, k]) for k in range(3)])


def analyze(recs: list[dict]) -> dict:
    recs = sorted(recs, key=lambda r: r["t"])
    ts = np.array([r["t"] for r in recs])
    gts = np.array([r["gt_pos"] for r in recs], dtype=float)
    # 실측 속도: 앞뒤 프레임 중심차분 (같은 시각 중복은 건너뛴다)
    vel = np.zeros_like(gts)
    for i in range(len(recs)):
        j0, j1 = max(i - 1, 0), min(i + 1, len(recs) - 1)
        while j0 > 0 and ts[i] - ts[j0] < 0.02:
            j0 -= 1
        while j1 < len(recs) - 1 and ts[j1] - ts[i] < 0.02:
            j1 += 1
        if ts[j1] > ts[j0]:
            vel[i] = (gts[j1] - gts[j0]) / (ts[j1] - ts[j0])

    rows = []
    last_rev_t = None
    prev_sign = 0
    for i, r in enumerate(recs):
        gt = gts[i]
        u, lat, ver = _axes(gt, np.array(r["cam_pos"], dtype=float))
        v_lat = float(vel[i] @ lat)
        sign = 0 if abs(v_lat) < MOVING_MPS else (1 if v_lat > 0 else -1)
        if sign != 0 and prev_sign != 0 and sign != prev_sign:
            last_rev_t = r["t"]
        if sign != 0:
            prev_sign = sign
        row = {"t": r["t"], "v_lat": v_lat, "since_rev": None if last_rev_t is None else r["t"] - last_rev_t,
               "rel": r.get("rel")}
        for key in ("meas_pos", "est_pos"):
            if r.get(key) is not None:
                e = np.array(r[key], dtype=float) - gt
                row[key[:-4]] = (float(e @ u), float(e @ lat), float(e @ ver))
        if r.get("est_vel") is not None:
            row["est_v_lat"] = float(np.array(r["est_vel"], dtype=float) @ lat)
        if r.get("lead_pos") is not None and r.get("tof") is not None:
            g_imp = _interp(ts, gts, r["t"] + r["tof"])
            if g_imp is not None:
                e = np.array(r["lead_pos"], dtype=float) - g_imp
                row["lead"] = (float(e @ u), float(e @ lat), float(e @ ver))
        rows.append(row)
    return {"rows": rows}


def _lag_ms(row: dict, key: str) -> float | None:
    """가로 오차를 운동 방향 기준 시간으로. 양수 = 실측보다 뒤처짐."""
    if key not in row or abs(row["v_lat"]) < MOVING_MPS:
        return None
    return -row[key][1] / row["v_lat"] * 1000.0


def _med(xs):
    xs = sorted(x for x in xs if x is not None)
    return None if not xs else xs[len(xs) // 2]


def summarize(result: dict) -> dict:
    rows = result["rows"]
    moving = [r for r in rows if abs(r["v_lat"]) >= MOVING_MPS]
    static = [r for r in rows if abs(r["v_lat"]) < 0.2]

    def comp(rs, key, k):
        return _med([r[key][k] for r in rs if key in r])

    out = {
        "frames": len(rows), "moving_frames": len(moving), "static_frames": len(static),
        "static": {key: None if not static else [comp(static, key, k) for k in range(3)]
                   for key in ("meas", "est")},
        "moving_lag_ms": {key: _med([_lag_ms(r, key) for r in moving]) for key in ("meas", "est", "lead")},
        "moving_err_m": {key: None if not moving else [comp(moving, key, k) for k in range(3)]
                         for key in ("meas", "est", "lead")},
        "vel_ratio": _med([r["est_v_lat"] / r["v_lat"] for r in moving if "est_v_lat" in r]),
        "bins": [],
    }
    for lo, hi in REVERSAL_BINS:
        rs = [r for r in moving if r["since_rev"] is not None and lo <= r["since_rev"] < hi]
        abs_lat = [abs(r["lead"][1]) for r in rs if "lead" in r]
        out["bins"].append({
            "since_rev_s": [lo, hi], "n": len(rs),
            "est_lag_ms": _med([_lag_ms(r, "est") for r in rs]),
            "lead_lag_ms": _med([_lag_ms(r, "lead") for r in rs]),
            "vel_ratio": _med([r["est_v_lat"] / r["v_lat"] for r in rs if "est_v_lat" in r]),
            "lead_abs_lat_m": _med(abs_lat),
            "lead_within_hit_radius": (round(sum(1 for a in abs_lat if a <= 0.1975) / len(abs_lat), 2)
                                       if abs_lat else None),
        })
    return out


def _f(x, fmt="{:+.0f}"):
    return "-" if x is None else fmt.format(x)


def _v3(v):
    return "-" if v is None else "(" + ", ".join(_f(x, "{:+.3f}") for x in v) + ")"


def main() -> None:
    for prefix in sys.argv[1:]:
        s = summarize(analyze(load(prefix)))
        name = prefix.replace(chr(92), "/").split("/")[-1]
        print(f"== {name}: {s['frames']} frames ({s['moving_frames']} moving, {s['static_frames']} static)")
        print(f"   error (range, lateral, vertical) m  static meas {_v3(s['static']['meas'])}  est {_v3(s['static']['est'])}")
        print(f"   moving: meas {_v3(s['moving_err_m']['meas'])}  est {_v3(s['moving_err_m']['est'])}  "
              f"lead {_v3(s['moving_err_m']['lead'])}")
        lag = s["moving_lag_ms"]
        print(f"   moving lag ms (+ = behind): meas {_f(lag['meas'])}  est {_f(lag['est'])}  lead {_f(lag['lead'])}  "
              f"| est/true lateral speed {_f(s['vel_ratio'], '{:.2f}')}")
        print("   since reversal     n  est lag  lead lag  v ratio  |lead lat|  lead within 0.1975 m")
        for b in s["bins"]:
            lo, hi = b["since_rev_s"]
            label = f"{lo:.1f}-{hi:.1f} s" if hi != math.inf else f"{lo:.1f}+ s"
            print(f"   {label:12s} {b['n']:6d}  {_f(b['est_lag_ms']):>7}  {_f(b['lead_lag_ms']):>8}  "
                  f"{_f(b['vel_ratio'], '{:.2f}'):>7}  {_f(b['lead_abs_lat_m'], '{:.3f}'):>9}  "
                  f"{_f(b['lead_within_hit_radius'], '{:.2f}'):>6}")


if __name__ == "__main__":
    main()
