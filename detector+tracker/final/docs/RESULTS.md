# Final Experimental Results

## Final Configuration

| Item | Setting |
|---|---|
| Detector | GA-optimized YOLO11s |
| Deployment | TensorRT FP16 |
| TensorRT input | 384 x 640 |
| SEARCH interval | 5 frames |
| SEARCH confirmation | 2 hits |
| TRACK correction interval | 20 frames |
| CPU tracker | Lucas-Kanade Optical Flow |
| GPU role | YOLO detection / re-detection |
| CPU-GPU execution | Asynchronous |

## Final Sequence Results

| Sequence | Acquisition | Delay | Tracker P* | AUC | Mean IoU | Overall FPS | YOLO calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| Seq1 | 5 | 0.25 s | 98.59% | 77.13% | 78.43% | 19.82 | 51 |
| Seq2 | 140 | 0.50 s | 91.50% | 69.19% | 70.16% | 19.99 | 71 |

\* Tracker P is the percentage of evaluated tracking frames with IoU >= 0.5.
It is not detector classification Precision.

## Maximum Throughput

- Fixed-20 TensorRT asynchronous maximum throughput: **55.30 FPS**
- Mean IoU: **78.40%**
- Success AUC: **77.02%**
- YOLO calls: **50**

The ~20 FPS final-sequence values are paced evaluations using a 20 FPS video source.
They do not represent the maximum processing capability.

## TRACK Interval Ablation

| Interval | Tracker P | AUC | Mean IoU | YOLO calls |
|---:|---:|---:|---:|---:|
| 15 | 99.90% | 78.22% | 79.67% | 67 |
| 20 | 100.00% | 77.90% | 79.28% | 50 |
| 25 | 99.20% | 76.23% | 77.47% | 40 |
| 30 | 98.90% | 74.10% | 75.37% | 34 |

## SEARCH Interval Ablation

| SEARCH interval | Acquisition frame | GT present | SEARCH YOLO calls | Mean IoU |
|---:|---:|---|---:|---:|
| 1 | 134 | True | 135 | 68.21% |
| 2 | 136 | True | 69 | 68.61% |
| 5 | 140 | True | 29 | 70.25% |
| 10 | 150 | True | 16 | 69.39% |

## Final Selection

- SEARCH interval: **5 frames**
- SEARCH confirmation: **2 hits**
- TRACK YOLO correction interval: **20 frames**

Adaptive scheduling variants were evaluated but were not selected because
they did not outperform the fixed-20 tracking policy.
