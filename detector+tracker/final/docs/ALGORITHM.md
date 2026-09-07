# Final Algorithm

## 1. SEARCH State

- Full-frame TensorRT YOLO is executed every 5 frames.
- A single detection is treated only as a candidate.
- Two confirmed detection hits are required before TRACK begins.
- This prevents a single-frame false acquisition.

Observed during development:
- Sequence 2 contained a false detection at frame 7 while GT was ABSENT.
- The false detection occurred only for one frame.
- The real UAV was detected continuously from approximately frame 133.
- Therefore, 2-hit confirmation was selected.

## 2. TRACK State

- CPU Lucas-Kanade Optical Flow tracks the UAV between detector calls.
- TensorRT YOLO runs asynchronously on the GPU.
- YOLO performs anchor correction every 20 frames.
- CPU tracking continues while GPU inference is running.
- Returned YOLO boxes are applied after motion compensation.
- Optical Flow is reinitialized after YOLO correction.

## 3. Final Parameters

- Input shape: 384 x 640
- SEARCH interval: 5 frames
- SEARCH confirmation: 2 hits
- TRACK YOLO interval: 20 frames
- Detector confidence threshold: 0.25
- GPU: TensorRT YOLO
- CPU: Optical Flow
- Execution: asynchronous CPU-GPU

## 4. Why TRACK Interval 20

Interval 15 gave slightly higher IoU but required substantially more YOLO calls.
Interval 20 retained nearly the same tracking accuracy with lower GPU usage.
Intervals 25 and 30 showed increasing drift.

## 5. Adaptive Scheduling

Several adaptive scheduling variants were evaluated.
They were not selected because none outperformed the fixed-20 policy.
The corresponding JSON files are preserved under results/ablation/adaptive/.
