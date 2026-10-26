#!/usr/bin/env python3
"""Render one moving-target live-fire run into the mandatory branch report."""

from __future__ import annotations

import argparse
import json
import math
import os


def rate(items: list[dict]) -> tuple[int, int, float | None]:
    hits = sum(x.get("verdict") == "HIT" for x in items)
    judged = sum(x.get("verdict") in ("HIT", "MISS") for x in items)
    return hits, judged, (hits / judged if judged else None)


def fmt_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.2f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("report")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--command", default="")
    ap.add_argument("--sim-log", required=True)
    ap.add_argument("--probe-log", required=True)
    ap.add_argument("--cleanup", default="UNKNOWN")
    ap.add_argument("--speed-kmh", type=float, default=3.0)
    ap.add_argument("--half-length", type=float, default=1.5)
    ap.add_argument("--turn-window", type=float, default=0.5)
    args = ap.parse_args()

    with open(args.result, encoding="utf-8") as f:
        data = json.load(f)
    judged_events = [x for x in data.get("hits", []) if x.get("verdict") in ("HIT", "MISS")]
    speed = args.speed_kmh / 3.6
    edge = max(0.0, args.half_length - speed * args.turn_window)

    def lateral(event: dict) -> float | None:
        p = event.get("ground_truth_point_world_m")
        return float(p[1]) if isinstance(p, list) and len(p) >= 2 else None

    near = [x for x in judged_events if lateral(x) is not None and abs(lateral(x)) >= edge]
    cruise = [x for x in judged_events if x not in near]
    h_all, n_all, r_all = rate(judged_events)
    h_turn, n_turn, r_turn = rate(near)
    h_cruise, n_cruise, r_cruise = rate(cruise)
    valid = bool(data.get("valid")) and n_all > 0
    passed = valid and r_all is not None and r_all >= 0.90 and r_turn is not None and r_turn >= 0.90
    verdict = "PASS" if passed else ("FAIL" if valid else "INVALID")

    align = data.get("track_align_px") or {}
    clock = data.get("clock") or {}
    miss = data.get("miss_m") or {}
    state = data.get("state_frac") or {}
    files = [
        f"- 결과 JSON: [{os.path.basename(args.result)}]({args.result.replace(os.sep, '/')})",
        f"- 시뮬레이터 로그: [{os.path.basename(args.sim_log)}]({args.sim_log.replace(os.sep, '/')})",
        f"- 프로브 로그: [{os.path.basename(args.probe_log)}]({args.probe_log.replace(os.sep, '/')})",
    ]

    lines = [
        f"# 분기 보고서 — {args.run_id}", "",
        f"- 판정: **{verdict}**", f"- 종료 정리: `{args.cleanup}`", "",
        "## 1. 목적", "",
        "35 m, 3 km/h 좌우 왕복 UAV를 실제 추종하고 READY 게이트를 통과해 격발했을 때 전체와 방향 전환 구간 명중률이 각각 90% 이상인지 검증한다.", "",
        "## 2. 조건", "",
        "- 640×640, HFOV 0.2 rad, 왕복 반폭 1.5 m", "- CUDA YOLO + D3D12 Gazebo, GPU LiDAR OFF", "- viewer/HUD ON",
        f"- 전환 구간 정의: 궤도 끝점으로부터 {speed * args.turn_window:.3f} m 이내(방향 전환 전후 각각 약 {args.turn_window:.1f}초)",
        "", "## 3. 실행 명령", "", f"```bash\n{args.command}\n```", "", "## 4. 원본 파일", "",
        *files, "", "## 5. 결과", "",
        "| 지표 | 결과 |", "|---|---:|",
        f"| 유효성 | {data.get('valid')} {data.get('invalid_reasons', [])} |",
        f"| 관측 처리율 | {data.get('camera_hz', 'N/A')} Hz |",
        f"| READY 비율 | {100 * data.get('ready_frac', 0):.1f}% |",
        f"| 조준오차 중앙값 / P90 | {align.get('median', 'N/A')} / {align.get('p90', 'N/A')} px |",
        f"| 격발 / 판정 완료 | {data.get('shots', 0)} / {n_all} |",
        f"| 전체 명중 | {h_all}/{n_all} ({fmt_rate(r_all)}) |",
        f"| 방향 전환 구간 명중 | {h_turn}/{n_turn} ({fmt_rate(r_turn)}) |",
        f"| 등속 구간 명중 | {h_cruise}/{n_cruise} ({fmt_rate(r_cruise)}) |",
        f"| miss distance 중앙값 / 최대 | {miss.get('median', 'N/A')} / {miss.get('max', 'N/A')} m |",
        f"| clock 최대 공백 / stall | {clock.get('max_gap_s', 'N/A')} s / {clock.get('stalls', 'N/A')} |",
        f"| 상태 분포 | `{json.dumps(state, ensure_ascii=False)}` |", "",
        "## 6. 판정", "",
        ("전체와 전환 구간이 모두 90% 이상이며 시험이 유효하다." if passed else
         "시험은 유효하지만 전체 또는 전환 구간 90% 기준을 충족하지 못했다." if valid else
         "계측 유효성 조건을 충족하지 못해 성능 판정에 사용하지 않는다."), "",
        "## 7. 다음 분기", "",
        ("동일 설정 반복시험으로 재현성을 확인한다." if passed else
         "miss 이벤트를 방향·프레임 나이·조준오차 기준으로 분해한 뒤 feedforward/READY 게이트를 수정한다."), "",
        "## 8. 종료 확인", "", f"- 잔류 프로세스: `{args.cleanup}`", "",
    ]
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(json.dumps({"verdict": verdict, "hit_rate": r_all, "turn_hit_rate": r_turn,
                      "cruise_hit_rate": r_cruise, "judged": n_all}))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())

