# Reproduction Guide

## Requirements

- NVIDIA Jetson
- CUDA
- TensorRT
- Python 3
- PyTorch
- Ultralytics
- OpenCV
- NumPy

Exact installed versions are stored under environment/.

## Model

Preferred Jetson deployment model:
models/ga_yolo11s_960_best.engine

Portable trained model:
models/ga_yolo11s_960_best.pt

The TensorRT engine may not work on another Jetson or TensorRT version.
Regenerate the engine from the PT file when necessary.

## Final Benchmark

Run:
python3 scripts/run_final_benchmarks.py

## Visualization

Run:
python3 scripts/run_demo.py

Expected demo output:
output_video/v20_FINAL_demo.mp4

## FPS Interpretation

- Approximately 20 FPS means the system successfully follows a paced 20 FPS input.
- It is not the maximum processing speed.
- Maximum measured fixed-20 throughput was approximately 55.3 FPS.

## Integrity

Run:
bash scripts/verify_integrity.sh

to verify files against MANIFEST.sha256.
