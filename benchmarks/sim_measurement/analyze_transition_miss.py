"""왕복 표적 자동 격발 결과(09_auto_cross3_*.json)의 miss를 방향 전환 후 경과시간별로 분해한다.

사용: python analyze_transition_miss.py RESULT.json [RESULT.json ...]
표적 실측(ground_truth)에서 전환 시각을 구해 격발별 진행방향 기준 횡오차와 추정 속도를 모으고,
경과시간 구간별 명중률·후행량과 '추정 속도 급락' 후보 게이트(속도 < ratio × 직전 window 75퍼센타일)를 평가한다.
후보 게이트 평가는 같은 데이터에서 문턱을 본 표본 내 평가이므로 채택 전에 새 실행으로 검증해야 한다.
"""
import json
import sys

import numpy as np

HALF, TRANS_M = 1.5, 0.417


def load(path):
    d = json.load(open(path))
    ts = [s for s in d["track_samples"] if s.get("ground_truth_world_m")]
    t, i = np.unique([s["ground_truth_stamp"] for s in ts], return_index=True)
    y = np.array([s["ground_truth_world_m"][1] for s in ts])[i]
    vy = np.gradient(y, t)
    sg = np.sign(np.convolve(vy, np.ones(5) / 5, "same"))
    rev = t[1:][sg[1:] * sg[:-1] < 0]
    est = [s for s in d["track_samples"] if s.get("target_velocity_world_mps")]
    est_t = np.array([s["stamp"] for s in est])
    est_sp = np.array([abs(s["target_velocity_world_mps"][1]) for s in est])
    res = {h["shot_id"]: h for h in d["hits"]}
    rows = []
    for f in d["fires"]:
        h = res.get(f["shot_id"])
        if not h or "ground_truth_point_world_m" not in h:
            continue
        gy, it = h["ground_truth_point_world_m"][1], h["impact_time_s"]
        k = min(max(np.searchsorted(t, it), 1), len(t) - 1)
        direction = np.sign(vy[k]) or 1.0
        prev = rev[rev <= it]
        j = np.searchsorted(est_t, f["fire_time_s"]) - 1
        win = est_sp[(est_t > est_t[j] - 2.0) & (est_t <= est_t[j])] if j >= 0 else np.array([np.nan])
        rows.append(dict(
            hit=h["verdict"] == "HIT", trans=abs(gy) >= HALF - TRANS_M,
            since=it - prev[-1] if len(prev) else np.nan,
            lat=(h["projectile_point_world_m"][1] - gy) * direction,
            speed=est_sp[j] if j >= 0 else np.nan, p75=np.nanpercentile(win, 75)))
    return rows


def main(paths):
    rows = [r for p in paths for r in load(p)]
    n, hits = len(rows), sum(r["hit"] for r in rows)
    print(f"전체 {hits}/{n} ({hits / n:.1%}), 전환부 {sum(r['hit'] for r in rows if r['trans'])}/{sum(r['trans'] for r in rows)}")
    print("전환 후 경과시간 구간별 (횡오차 lat<0 = 진행방향 뒤처짐)")
    for lo, hi in [(0, .15), (.15, .3), (.3, .45), (.45, .6), (.6, 1), (1, 2), (2, 3), (3, 9)]:
        s = [r for r in rows if lo < r["since"] <= hi]
        if s:
            print(f"  {lo:.2f}-{hi:.2f}s n={len(s):3d} hit={np.mean([r['hit'] for r in s]):.2f} lat={np.mean([r['lat'] for r in s]):+.3f} m")
    print("후보 게이트: 추정 속도 < ratio × 직전 2 s 75퍼센타일 이면 차단")
    for ratio in (0.4, 0.5, 0.6):
        keep = [r for r in rows if not r["speed"] < ratio * r["p75"]]
        blocked = n - len(keep)
        kt = [r for r in keep if r["trans"]]
        print(f"  ratio {ratio}: 차단 {blocked}발, 유지 {len(keep)}발 명중 {np.mean([r['hit'] for r in keep]):.1%}, 전환부 {sum(r['hit'] for r in kt)}/{len(kt)}")


if __name__ == "__main__":
    main(sys.argv[1:])
