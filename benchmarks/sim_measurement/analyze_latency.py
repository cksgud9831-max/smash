#!/usr/bin/env python3
"""종단 지연 분석. <prefix>.lat_clock.json (clock_logger.py) 과 <prefix>.lat_fcs.json (fcs_timed.py) 을 읽는다.

모든 지연은 영상 헤더 시각(Gazebo 가 프레임을 렌더한 모의 시각)부터 잰다.
- wall ms: 이 PC 에서 실제로 걸린 시간
- sim ms : 그동안 모의 세계가 진행한 시간 — 이동 표적이 실제로 움직이는 시간

ROS 없이 numpy 만으로 동작한다.

사용 예:
    python analyze_latency.py results/sim_runs/f0.12_35m_r1 [...]
"""

from __future__ import annotations

import json
import sys

import numpy as np


def stats(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return {"n": int(a.size), "med": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def analyze_data(clk: dict, fcs: dict) -> dict:
    c = np.asarray(clk["clock"], dtype=float)
    wall, sim = c[:, 0], c[:, 1]
    keep = np.concatenate([[True], np.diff(sim) > 0])  # 단조 증가하는 모의 시각만
    wall, sim = wall[keep], sim[keep]

    def w2s(w):
        return np.interp(w, wall, sim, left=np.nan, right=np.nan)

    def s2w(s):
        return np.interp(s, sim, wall, left=np.nan, right=np.nan)

    out: dict = {"rtf": float(np.polyfit(wall, sim, 1)[0])}
    fr = np.array([f for f in fcs.get("frames", []) if f[3]], dtype=float)  # 조준해를 낸 프레임만
    if fr.size:
        ws, we, st = fr[:, 0], fr[:, 1], fr[:, 2]
        out["capture_to_fcs_start_wall_ms"] = stats((ws - s2w(st)) * 1e3)
        out["fcs_processing_wall_ms"] = stats((we - ws) * 1e3)
        out["capture_to_aim_wall_ms"] = stats((we - s2w(st)) * 1e3)
        out["capture_to_aim_sim_ms"] = stats((w2s(we) - st) * 1e3)
        out["frame_interval_wall_ms"] = stats(np.diff(ws) * 1e3)
    if clk.get("ann"):
        an = np.asarray(clk["ann"], dtype=float)
        out["capture_to_hud_topic_wall_ms"] = stats((an[:, 0] - s2w(an[:, 1])) * 1e3)
    if fcs.get("shots"):
        sh = np.asarray(fcs["shots"], dtype=float)  # wall, node_now, frame_ts
        out["capture_to_fire_wall_ms"] = stats((sh[:, 0] - s2w(sh[:, 2])) * 1e3)
        out["capture_to_fire_sim_ms"] = stats((w2s(sh[:, 0]) - sh[:, 2]) * 1e3)
        out["node_clock_lag_sim_ms"] = stats((w2s(sh[:, 0]) - sh[:, 1]) * 1e3)
    return out


def analyze(prefix: str) -> dict:
    with open(prefix + ".lat_clock.json") as f:
        clk = json.load(f)
    with open(prefix + ".lat_fcs.json") as f:
        fcs = json.load(f)
    return analyze_data(clk, fcs)


def main() -> None:
    for p in sys.argv[1:]:
        r = analyze(p)
        print(f"== {p.replace(chr(92), '/').split('/')[-1]}  RTF {r['rtf']:.2f}")
        for k, v in r.items():
            if k != "rtf" and v:
                print(f"  {k:32s} n={v['n']:5d} med={v['med']:7.1f} p95={v['p95']:7.1f} max={v['max']:7.1f}")


if __name__ == "__main__":
    main()
