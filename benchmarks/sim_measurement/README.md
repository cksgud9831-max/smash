# sim_measurement — Gazebo 시뮬레이터 측정 도구

SMASH FCS 시뮬레이터(`simulation/`)의 성능을 **같은 방법으로 반복 측정**하기 위한 도구다.
2026-09-18~19 세션에서 임시 폴더에 두고 쓰던 것을 옮겼다(측정 결과는 `docs/sessions/session_work_summary_2026-09-18.md` 10·11절).
측정 전용이며 FCS 동작은 바꾸지 않는다.

## 구성

| 파일 | 실행 위치 | 역할 |
|---|---|---|
| `run_case.sh` | WSL | 측정 1회: 헤드리스 시뮬레이터 기동 → 계측 FCS 교체 → 지연 기록 → 자동 조준 측정 → 정리 |
| `run_matrix.sh` | WSL | 화각 × 거리 × 반복 매트릭스. 조건마다 `run_case.sh` 호출 |
| `auto_aim_probe.py` | WSL (ROS) | 표적으로 선회 → 포착 → 정렬 → 측정 구간 동안 상태 분포·격발·명중·빗나감 기록 |
| `fcs_timed.py` | WSL (ROS) | FCS 노드를 단계별 시간 계측과 함께 실행. 프레임·격발 시각 기록 |
| `clock_logger.py` | WSL (ROS) | 모의 시각 ↔ 실제 시각 대응, HUD 영상·격발 도착 시각 기록 |
| `analyze_latency.py` | 어디서나 | 촬영 → FCS 수신 / 처리 / 조준점 / 격발 / HUD 지연 (실제 ms, 모의 ms) |
| `summarize_runs.py` | 어디서나 | 결과 폴더를 한 표로 요약하고 `runs_summary.json` 저장 |
| `alignment.py` | 어디서나 | 프로브의 정렬 계산 (`tests/test_sim_measurement.py` 에서 시험) |

## 사용법

먼저 런처로 빌드가 끝나 있어야 한다(`simulation\run_smash_fcs.bat` 3번).
측정 중에는 **실행 중인 SMASH 시뮬레이터가 종료된다**(런처의 `cleanup` 과 같은 패턴만 정리하므로 다른 ROS/Gazebo 작업은 건드리지 않는다).

```bash
# 1회 측정: 35 m 정지 표적, 화각 6.9°, 측정 60초
CAMERA_HFOV=0.12 bash benchmarks/sim_measurement/run_case.sh f0.12_35m_r1 33.0 0.0 12.5 60

# 매트릭스: 화각 6.9° 에서 10~100 m, 2회 반복 (조건당 약 3분)
FOVS="0.12" DISTS="10 20 35 60 100" REPS=2 bash benchmarks/sim_measurement/run_matrix.sh ms_

# 요약 (Windows 에서도 가능)
python benchmarks/sim_measurement/summarize_runs.py --glob "ms_*"
python benchmarks/sim_measurement/analyze_latency.py results/sim_runs/ms_f0.12_35m_r1
```

Windows 에서 바로 부를 때:

```bash
wsl -d Ubuntu-24.04 -e bash -lc "cd /mnt/d/Aiming && CAMERA_HFOV=0.12 bash benchmarks/sim_measurement/run_case.sh f0.12_35m_r1 33.0 0.0 12.5 60"
```

`run_case.sh` 의 환경변수(`CAMERA_PX`, `DEMO_KILL`, `PROFILE`, `NOCLOCK`, `NOIMG`, `EXTRA_LAUNCH`, `SIM_RUNS_DIR`)는 스크립트 머리말에 정리돼 있다.
결과는 기본으로 `results/sim_runs/` 에 쌓이며 git 에는 올리지 않는다.

## 거리별 표적 위치 (앙각 약 20°)

| 라벨 거리 | x y z (m) | 카메라 기준 경사거리 |
|---|---|---|
| 10 | 10.0 0.0 4.2 | 10.6 m |
| 20 | 18.8 0.0 7.5 | 20.0 m |
| 35 | 33.0 0.0 12.5 | 35.0 m |
| 60 | 56.4 0.0 21.2 | 60.0 m |
| 100 | 94.0 0.0 34.9 | 100.0 m |

세션 문서의 거리(10.8 m, 35.3 m 등)는 지면 원점 기준이다.

## 측정 주의 (세션 문서 6절)

- **성능은 RTF 가 아니라 카메라 토픽 주기와 FCS 처리율로 본다.** Gazebo 가 보고하는 `real_time_factor` 는 늘 약 1.0 이다. 실제 RTF 는 `analyze_latency.py` 가 `/clock` 과 실제 시각으로 계산한다.
- 측정용으로 640 영상을 추가 구독하면 그 자체가 전송 부하가 된다. 그래서 `NOIMG=1` 이 기본이며, 이때 FCS 처리율은 `fcs_timed.py` 의 `_on_image` 호출률로 본다.
- `top -b -n1` 의 첫 샘플은 기동 이후 평균이다. 프로브는 두 번째 샘플을 `.top.txt` 에 남긴다.
- 측정 전 Windows 쪽에서 CPU 를 쓰는 작업이 없는지 확인한다(4코어를 Gazebo 와 FCS 가 나눠 쓴다). 다른 작업이 돌면 처리율·지연은 비교할 수 없고, 탐지·READY·명중 결과만 유효하다.
- `fcs_timed.py` 는 노드 내부 이름(`_on_image`, `_tracker`, `_AimingManager` 등)에 기대므로 `smash_fcs_node.py` 를 크게 바꾸면 함께 고쳐야 한다.
