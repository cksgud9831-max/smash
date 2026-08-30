#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yolo_follower_v20_final as m

MODEL = ROOT / "models/ga_yolo11s_960_best.engine"
OUT = ROOT / "results/final_reproduced"
OUT.mkdir(parents=True, exist_ok=True)

SEQUENCES = {
    "seq1": ROOT / "datasets/Anti-UAV300/test/20190925_111757_1_1/visible.mp4",
    "seq2": ROOT / "datasets/Anti-UAV300/test/20190926_134054_1_1/visible.mp4",
}

m.DEVICE = 0

if not MODEL.exists():
    raise FileNotFoundError(f"Model not found: {MODEL}")

for name, video in SEQUENCES.items():
    if not video.exists():
        raise FileNotFoundError(
            f"Dataset not found: {video}\nSee docs/DATASET.md"
        )

    print("\n" + "=" * 70)
    print("FINAL BENCHMARK:", name)
    print("=" * 70)

    report = m.run_benchmark(
        video_path=str(video),
        model_path=str(MODEL),
        max_frames=1000,
        sample_interval=1.0,
        use_tegrastats=True,
    )

    (OUT / f"{name}.json").write_text(
        json.dumps(report, indent=4)
    )

    t = report["tracker_metrics"]
    print(
        f"acq={t.get('initial_acquisition_frame')} | "
        f"P={t.get('precision_percent',0):.2f}% | "
        f"AUC={t.get('success_auc_percent',0):.2f}% | "
        f"IoU={t.get('mean_iou_percent',0):.2f}% | "
        f"FPS={t.get('overall_fps',0):.2f}"
    )
