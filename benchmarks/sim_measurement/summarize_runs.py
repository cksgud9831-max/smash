#!/usr/bin/env python3
"""run_case.sh 결과 폴더를 한 표로 요약한다. ROS 없이 동작한다 (Windows 에서도 가능).

라벨이 run_matrix.sh 형식(f<fov>_<dist>m_r<rep>)이면 화각·반복 번호를 뽑고,
아니면 라벨 그대로 한 줄로 보여 준다. 요약 JSON 을 결과 폴더에 함께 저장한다.

사용 예:
    python summarize_runs.py                              # results/sim_runs 전체
    python summarize_runs.py results/sim_runs --glob "ms_*"
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_latency import analyze  # noqa: E402

LABEL_RE = re.compile(r"f([\d.]+)_(\d+)m_r(\d+)")
# 옛 결과 JSON 에 경사거리가 없을 때 쓰는 카메라 높이 근사값 (auto_aim_probe.py 기본값과 같다)
CAM_HEIGHT_M = 0.7
_SUFFIXES = (".lat_clock.json", ".lat_fcs.json", "_summary.json")


def summarize_case(probe_path: str) -> dict:
    label = os.path.basename(probe_path)[:-5]
    prefix = probe_path[:-5]
    with open(probe_path, encoding="utf-8") as f:
        pj = json.load(f)

    bbox, fcs_hz = [], []
    try:
        with open(prefix + ".prof.log", encoding="utf-8", errors="ignore") as f:
            for line in f:
                mb = re.search(r"bboxW med=([\d.]+)px", line)
                if mb:
                    bbox.append(float(mb.group(1)))
                mr = re.match(r"_on_image\s+n=\s*\d+ rate=\s*([\d.]+)/s", line)
                if mr:
                    fcs_hz.append(float(mr.group(1)))
    except FileNotFoundError:
        pass
    try:
        lat = analyze(prefix)
    except (FileNotFoundError, ValueError, KeyError, IndexError, TypeError):
        lat = {}

    m = LABEL_RE.search(label)
    fx, px = pj.get("fx"), pj.get("camera_px")
    fov_deg = (math.degrees(float(m.group(1))) if m
               else (math.degrees(2 * math.atan(px / 2 / fx)) if fx and px else None))
    rng = pj.get("slant_range_m")
    if rng is None and pj.get("target"):
        rng = round(math.dist((0.0, 0.0, CAM_HEIGHT_M), pj["target"]), 1)
    v = pj.get("verdicts") or {}
    state = pj.get("state_frac") or {}
    return {
        "label": label,
        "valid": pj.get("valid"),
        "invalid_reasons": pj.get("invalid_reasons"),
        "fov_deg": None if fov_deg is None else round(fov_deg, 1),
        "range_m": rng,
        "rep": int(m.group(3)) if m else None,
        "acquired": pj.get("acquire_s") is not None,
        "acquire_s": pj.get("acquire_s"),
        "align_px": pj.get("align_final_px"),
        "bbox_px": median(bbox) if bbox else None,
        "fcs_hz": median(fcs_hz[1:] or fcs_hz) if fcs_hz else pj.get("fcs_hz"),
        "ready": state.get("READY", 0.0),
        "search": state.get("SEARCH", 0.0),
        "shots": pj.get("shots", 0),
        "hits": v.get("HIT", 0),
        "judged": v.get("HIT", 0) + v.get("MISS", 0),
        "miss_mean": (pj.get("miss_m") or {}).get("mean"),
        "miss_max": (pj.get("miss_m") or {}).get("max"),
        "rtf": lat.get("rtf"),
        "aim_lat_med": (lat.get("capture_to_aim_wall_ms") or {}).get("med"),
        "aim_lat_p95": (lat.get("capture_to_aim_wall_ms") or {}).get("p95"),
        "fire_lat_med": (lat.get("capture_to_fire_wall_ms") or {}).get("med"),
        "fire_lat_sim_med": (lat.get("capture_to_fire_sim_ms") or {}).get("med"),
    }


def median(xs):
    return sorted(xs)[len(xs) // 2]


def _fmt(x, f="{:.0f}"):
    return "-" if x is None else f.format(x)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir", nargs="?", default=os.path.join(here, "..", "..", "results", "sim_runs"))
    ap.add_argument("--glob", default="*", help="라벨 필터 (예: 'ms_f0.12_*')")
    args = ap.parse_args()

    paths = [p for p in sorted(glob.glob(os.path.join(args.runs_dir, args.glob + ".json")))
             if not p.endswith(_SUFFIXES)]
    rows = []
    for p in paths:
        try:
            rows.append(summarize_case(p))
        except (json.JSONDecodeError, OSError) as e:
            print(f"skip {p}: {e}", file=sys.stderr)
    if not rows:
        print(f"no runs in {args.runs_dir} matching {args.glob!r}")
        return

    rows.sort(key=lambda r: (-(r["fov_deg"] or 0), r["range_m"] or 0, r["rep"] or 0, r["label"]))
    print(f"{'label':28s} {'ok':>2} {'FOV':>5} {'range':>6} | {'bbox':>4} {'FCS':>5} {'READY':>6} {'SEARCH':>6} | "
          f"{'shots':>5} {'hit/judged':>10} {'miss':>6} {'max':>6} | {'RTF':>4} {'aim_med':>7} {'aim_p95':>7} "
          f"{'fire_med':>8}")
    for r in rows:
        ok = {True: "y", False: "N", None: "?"}[r["valid"]]
        print(f"{r['label'][:28]:28s} {ok:>2} {_fmt(r['fov_deg'], '{:.1f}'):>5} {_fmt(r['range_m'], '{:.1f}'):>6} | "
              f"{_fmt(r['bbox_px']):>4} {_fmt(r['fcs_hz'], '{:.1f}'):>5} {r['ready'] * 100:5.0f}% "
              f"{r['search'] * 100:5.0f}% | {r['shots']:5d} {r['hits']:>4d}/{r['judged']:<5d} "
              f"{_fmt(r['miss_mean'], '{:.3f}'):>6} {_fmt(r['miss_max'], '{:.3f}'):>6} | "
              f"{_fmt(r['rtf'], '{:.2f}'):>4} {_fmt(r['aim_lat_med']):>7} {_fmt(r['aim_lat_p95']):>7} "
              f"{_fmt(r['fire_lat_med']):>8}")
    for r in rows:
        if r["valid"] is False:
            print(f"  {r['label']} invalid: {'; '.join(r['invalid_reasons'] or [])}")
    hits = sum(r["hits"] for r in rows)
    judged = sum(r["judged"] for r in rows)
    if judged:
        print(f"\ntotal hits {hits}/{judged} ({100 * hits / judged:.1f} %), {len(rows)} runs")
    out = os.path.join(args.runs_dir, "runs_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
