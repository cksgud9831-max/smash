#!/usr/bin/env python3
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import yolo_follower_v20_visual as m

VIDEO = ROOT / "datasets/Anti-UAV300/test/20190926_134054_1_1/visible.mp4"
MODEL = ROOT / "models/ga_yolo11s_960_best.engine"

if not VIDEO.exists():
    raise FileNotFoundError(
        f"Dataset not found: {VIDEO}\nSee docs/DATASET.md"
    )

if not MODEL.exists():
    raise FileNotFoundError(f"Model not found: {MODEL}")

m.DEVICE = 0

r = m.run_benchmark(
    video_path=str(VIDEO),
    model_path=str(MODEL),
    max_frames=1000,
    sample_interval=1.0,
    use_tegrastats=False,
)

print("Acquisition frame:",
      r["tracker_metrics"].get("initial_acquisition_frame"))
print("Demo video:", ROOT / "output_video/v20_FINAL_demo.mp4")
