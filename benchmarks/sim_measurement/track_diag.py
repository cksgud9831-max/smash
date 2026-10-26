#!/usr/bin/env python3
"""추적기가 실제 표적을 물고 있는지 요약한다. <label>.track_diag.jsonl (fcs_timed.py) 을 읽는다. ROS 불필요.

프레임마다 추적기 박스 중심과 실측 표적 위치(촬영 시각으로 보간해 같은 영상의 카메라 자세로 투영한
화소)의 거리를 잰다. 박스가 있는데 그 거리가 박스 크기보다 크면 "이탈"(표적이 아닌 것을 추적)로 본다.
이탈 중에도 신뢰도가 STABLE 이면 조준·격발이 엉뚱한 곳을 향한다.

사용 예:
    python track_diag.py results/sim_runs/amb5_comp_true_r4 [...]
"""

from __future__ import annotations

import json
import math
import sys


def load(prefix: str) -> list[dict]:
    with open(prefix + ".track_diag.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def frame_error(rec: dict) -> float | None:
    """박스 중심과 실측 투영 화소의 거리(px). 둘 중 하나라도 없으면 None."""
    b, g = rec.get("bbox"), rec.get("gt_px")
    if b is None or g is None:
        return None
    cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return math.hypot(cx - g[0], cy - g[1])


def is_off_target(rec: dict, err: float | None, min_px: float = 20.0) -> bool:
    if err is None:
        return False
    b = rec["bbox"]
    size = max(b[2] - b[0], b[3] - b[1])
    return err > max(min_px, size)


def summarize(recs: list[dict]) -> dict:
    tracked = [r for r in recs if r.get("bbox") is not None and r.get("gt_px") is not None]
    errs = [frame_error(r) for r in tracked]
    off = [is_off_target(r, e) for r, e in zip(tracked, errs)]
    off_stable = [o and r.get("rel") == "STABLE" for r, o in zip(tracked, off)]

    # 연속 이탈 구간
    episodes, start = [], None
    for i, o in enumerate(off):
        if o and start is None:
            start = i
        if (not o or i == len(off) - 1) and start is not None:
            end = i if o else i - 1
            seg = tracked[start:end + 1]
            episodes.append({
                "t0": seg[0]["t"], "t1": seg[-1]["t"], "frames": len(seg),
                "max_err_px": round(max(errs[start:end + 1]), 1),
                "stable_frames": sum(1 for r in seg if r.get("rel") == "STABLE"),
                "gt_y_at_start": seg[0].get("gt_y"),
                "yolo_frames": sum(1 for r in seg if r.get("yolo")),
                "recovery_frames": sum(1 for r in seg if r.get("recovery")),
            })
            start = None
    se = sorted(e for e in errs if e is not None)
    return {
        "frames": len(recs),
        "tracked_frames": len(tracked),
        "err_px_median": round(se[len(se) // 2], 1) if se else None,
        "err_px_p90": round(se[int(0.9 * (len(se) - 1))], 1) if se else None,
        "off_target_frac": round(sum(off) / len(tracked), 3) if tracked else None,
        "off_target_while_stable_frac": round(sum(off_stable) / len(tracked), 3) if tracked else None,
        "episodes": episodes,
    }


def main() -> None:
    for prefix in sys.argv[1:]:
        s = summarize(load(prefix))
        print(f"== {prefix.replace(chr(92), '/').split('/')[-1]}: {s['tracked_frames']}/{s['frames']} frames with box+GT, "
              f"err median {s['err_px_median']} px p90 {s['err_px_p90']} px, off-target {s['off_target_frac']}, "
              f"off-target while STABLE {s['off_target_while_stable_frac']}")
        for e in s["episodes"][:15]:
            print(f"   t {e['t0']:.2f}-{e['t1']:.2f} s  {e['frames']:3d} fr  max {e['max_err_px']:6.1f} px  "
                  f"STABLE {e['stable_frames']:3d}  yolo {e['yolo_frames']:2d}  recov {e['recovery_frames']:2d}  "
                  f"gt_y {e['gt_y_at_start']}")
        if len(s["episodes"]) > 15:
            print(f"   ... {len(s['episodes']) - 15} more episodes")


if __name__ == "__main__":
    main()
