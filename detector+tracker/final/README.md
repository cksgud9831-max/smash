# K-MOSA Anti-UAV Edge Tracking

Jetson-based real-time Anti-UAV detection and tracking system.

## Core Pipeline

GA-optimized YOLO11s -> TensorRT FP16

SEARCH:
- Full-frame TensorRT YOLO every 5 frames
- 2-hit confirmation

TRACK:
- CPU Lucas-Kanade Optical Flow
- Async GPU TensorRT YOLO
- YOLO anchor correction every 20 frames

## Main Components

- GA-optimized YOLO11s UAV detector
- TensorRT FP16 deployment on NVIDIA Jetson
- CPU Optical Flow tracking
- Asynchronous CPU-GPU execution
- SEARCH / TRACK state separation
- 2-hit confirmation against single-frame false acquisition

## Final Files

- src/yolo_follower_FINAL_FROZEN.py
- src/yolo_follower_v20_final.py
- src/yolo_follower_v20_visual.py
- models/ga_yolo11s_960_best.pt
- models/ga_yolo11s_960_best.engine
- media/v20_FINAL_demo.mp4
- results/final/
- docs/RESULTS.md

## Performance Summary

- Real-time 20 FPS input: approximately 20 FPS processing
- Maximum measured throughput: approximately 55.3 FPS
- TRACK correction interval: 20 frames
- SEARCH interval: 5 frames
- SEARCH confirmation: 2 hits

See docs/RESULTS.md for exact measurements.

## Important Notes

Tracker Precision means the percentage of evaluated tracking frames with IoU >= 0.5.
It is NOT YOLO detector Precision.

The Anti-UAV dataset is not included in this package.

The TensorRT engine is hardware and TensorRT-version dependent.
Regenerate the engine from the PT model when using another target device.

Before public GitHub release, review LICENSE_NOTICE.md.
