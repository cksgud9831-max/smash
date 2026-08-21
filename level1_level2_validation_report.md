# 조준 파이프라인 2단계 검증 보고서

**검증 프로그램 종합 · Level 1 + Level 2** · 2026-08-19 작성

실제 하드웨어(IMU·레이저) 없이 `aiming_engine`과 `bridge` 파이프라인을 검증하기 위해 두 단계로 나눠 진행했습니다.

- **1차 (Monte Carlo Oracle)**: 순수 합성 궤적으로 `aiming_engine` 수학 코어 자체의 수렴성·강건성·런타임 비용을 검증
- **2차 (Gazebo 통합)**: 실제 영상의 2D 트래킹에 Gazebo가 생성한 비-Mock IMU/거리 신호를 결합해 `bridge` 계층까지 포함한 파이프라인 전체가 실제(Mock 아닌) 센서 입력을 문제없이 소화하는지 확인

> ⚠️ **두 단계 모두 실제 하드웨어(BNO08x IMU, TF02-Pro 레이저, 캘리브레이션된 카메라)는 사용하지 않았습니다.** 1차는 완전히 합성된 궤적, 2차는 실제 영상 + 시뮬레이션된 IMU/거리의 조합입니다. 실전 성능에 대한 주장은 아직 할 수 없습니다 — 남은 Level 3(실 하드웨어)는 문서 하단 참고.

## 핵심 지표 요약

| 지표 | 값 | 비고 |
|---|---:|---|
| L1 · 오라클 수렴률 | **100.0%** | 1800/1800, 잔차 <1mm |
| L1 · 예측오차 1m 미만 비율 | **99.8%** | 가정된 노이즈 조건 |
| L2 · 트래킹 IoU 평균 | **0.757** | test/visible.json GT 대비 |
| L2 · 거리 정합성 오차 (평균) | **0.084 m** | 재구성 distance_m vs Gazebo 실제 range_m |

전체 수치는 아래 각 레벨 섹션 및 원본 산출물(`aim_oracle_validation.html`, `gazebo/performance.json`) 참고.

---

## Level 1 — Monte Carlo Oracle 검증 (aiming_engine 수학 코어)

실영상 기반 테스트는 "영상 속 실제 표적이 어디 있었는지" GT가 없어 조준 정확도를 증명하기 어렵다는 문제(순환 논리)를 풀기 위해, 실제 3D 궤적을 직접 정의하고 완전한 정보로 푼 오라클과 비교하는 방식으로 전환했습니다. 4단계 구조(①솔버 검증 → ②강건성 → ③런타임 → ④하드웨어) 중 ①②③을 다뤘습니다. 전체 상세 내용과 방법론은 `aim_oracle_validation.html` 참고 — 아래는 그 요약입니다.

### ① 솔버 검증 — 1800개 무작위 궤적 전부 수렴

거리 50–100m, 방위각 ±20°, 고도각 ±8° 범위 무작위 궤적에서 오라클(정답 궤적을 `HitEquation`+`NewtonSolver`+`ProjectileModel`에 직접 대입한 해)이 100% 수렴, 잔차 항상 1mm 미만.

이는 솔버 수치해석의 건전성과 `CoordinateTransform.image_to_world` 복원 경로에 계통적 버그가 없음을 보여주지만, **물리식 자체(HitEquation/ProjectileModel)의 실세계 타당성은 증명하지 않습니다** — 오라클과 실제 해가 같은 코드를 공유하기 때문입니다.

| 지표 | 값 |
|---|---:|
| 오라클 수렴률 | 1800 / 1800 (100.0%) |
| 오라클 잔차 최댓값 | 9.86 × 10⁻⁴ m |

### ② 강건성 — 노이즈 · 급기동 · 손떨림 12개 조합

정지/등속/급기동 궤적 × 고정/손떨림 플랫폼 × clean/노이즈 관측, 각 150회씩 총 1800개 시나리오. 관측이 완벽하면(clean) 정지·등속은 오차가 사실상 0이고, 급기동만 등속도 가정 위반으로 0.026~0.032° 오차가 남습니다. 가정된 노이즈(픽셀 σ=2px, 거리 σ=0.3m — **실측치 아님**) 주입 시 최악 조합에서도 예측오차 1m 미만 비율은 99.3% 유지.

| 조건 | 각도오차 평균 | 예측오차 평균 | 예측오차 P95 | 1m 미만 비율 |
|---|---:|---:|---:|---:|
| 노이즈 없음 (clean) | 0.010° | 0.020 m | 0.165 m | 100.0% |
| 가정된 노이즈 조건 (전체 평균) | 0.144° | 0.356 m | 0.690 m | 99.7% |
| 급기동·손떨림·노이즈 (최악 조합) | 0.147° | 0.343 m | 0.666 m | 99.3% |

### ③ 런타임 — 지연시간 · 조준 흔들림

81,000회 `AimingManager.update()` 호출 계측 (aiming_engine 단독, OpticalFlowTracker 미포함).

| 지표 | 평균 | 중앙값 | P95 |
|---|---:|---:|---:|
| 지연시간 | 1.13 ms | 0.97 ms | 2.31 ms |
| 조준 흔들림 (프레임간 각도 변화) | 0.70° | 0.51° | 1.89° |

---

## Level 2 — Gazebo 통합 테스트 (bridge 계층 포함 end-to-end)

1차 검증은 이미지·트래커·실제 카메라를 전혀 쓰지 않았습니다. 2차는 `bridge.OpticalFlowTracker`가 실제 영상(`test/visible.mp4`)에서 뽑은 2D 트래킹 결과에, Gazebo Sim(Jetty, v10.5.0)이 생성한 **Mock이 아닌** IMU 자세·거리 신호를 결합해 `TrackerFrame`을 구성하고 `AimingManager`까지 실제로 통과시켰습니다.

### 데이터 흐름

```
SOURCE 1 (실제)                SOURCE 2 (시뮬레이션)             MERGE
test/visible.mp4         →     Gazebo (aiming_test.sdf)    →    TrackerFrame
OpticalFlowTracker              platform IMU 센서 +               프레임 인덱스로 결합
→ bbox, confidence               pose_publisher                  → AimingManager.update()
(20fps, 1920×1080)               → 자세, 거리
```

- **`gazebo/worlds/aiming_test.sdf`** — platform(IMU 부착)과 target 모델, 둘 다 물리 대신 `/world/aiming_test/set_pose` 서비스로 매 프레임 외부에서 자세/위치를 직접 명령. 카메라/렌더링 센서는 없음(2D는 영상이 담당하므로 의도적으로 제외).
- **`01_gazebo_bridge_node.py`** — 고정 프로파일(플랫폼 흔들림 진폭 2°·2Hz, 표적 75m 지점에서 8m/s 횡이동)로 300프레임(15초, 20fps) 구동, `/platform/imu` 구독으로 실제 시뮬레이션 IMU 값 기록 → `gazebo_result.json`.
- **`02_e2e_simulation_runner.py`** — 두 JSON을 프레임 인덱스로 병합, IMU 쿼터니언 → 4×4 extrinsic, Gazebo 거리 → `laser_range_m`로 매핑해 `TrackerFrame` 구성 후 `AimingManager` 실행 → `performance.json`.
- **`03_render_result_video.py`** — 트래킹 박스·GT 박스·조준 레티클·상태를 원본 영상에 오버레이해 `gazebo/level2_result.mp4`로 렌더링 (육안 확인용).

### 결과 — 300프레임 (15초) 실행 결과

| 지표 | 값 |
|---|---:|
| 프레임 처리 / 트래킹 성공 | 300 / 300 (100.0%) |
| Recovery event (추적 실패 후 재탐지) | 0회 |
| 트래킹 IoU vs 사람 라벨링 GT (평균 / 중앙값) | 0.757 / 0.774 |
| 솔버 수렴률 | 100.0% |
| Aim Ready 비율 (트래킹 성공 프레임 중) | 97.3% |
| **거리 정합성** \|distance_m − gazebo_range_m\| (평균 / 최댓값) | **0.084 m / 0.557 m** |

마지막 행이 이 테스트의 핵심 지표입니다 — `AimingManager`가 IMU 자세(extrinsic)와 레이저 거리만으로 역산한 `distance_m`이, Gazebo가 애초에 명령한 실제 `range_m`과 평균 8cm, 최대 56cm 차이로 일치합니다. 즉 **Mock이 아닌 실제(시뮬레이션된) IMU/거리 신호를 넣어도 좌표 복원 체인이 정상 동작함**을 확인했습니다. IoU는 `test/visible.json`이라는 독립적인 사람 라벨링 GT와 비교한 유일한 지표로, 기존 `known_issues.md` 기록치(0.760)와 일치합니다.

### 이 테스트가 증명하는 것과 증명하지 않는 것

**✅ 확인된 것**
bridge 계층이 Mock이 아닌 IMU 쿼터니언·레이저 거리를 받아도 죽지 않고 정상 동작. `image_to_world`가 시뮬레이션이 명령한 3D 시나리오를 8cm 이내로 정확히 재구성. 실제 트래커가 실제 영상에서 GT와 0.757 IoU로 계속 추적.

**❌ 아직 확인 안 된 것**
**영상 속 실제 드론과 Gazebo의 표적은 서로 무관한 별개 개체입니다** — 화면 속 드론이 정말 75m·8m/s로 움직였다는 뜻이 아니라, 두 독립된 데이터 소스를 이어붙였을 때 파이프라인이 정상 동작하는지 본 것입니다. IMU도 실제 BNO08x가 아닌 Gazebo의 이상적인 시뮬레이션 센서라 실측 bias/drift/latency가 없고, 레이저도 실제 TF02-Pro가 아닌 Gazebo가 계산한 정확한 유클리드 거리입니다.

---

## Level 3 — 다음 단계: 실 하드웨어 검증 (미실시)

L1·L2 모두 실 하드웨어 없이 얻은 결과입니다. 아래가 없으면 "실전 성능"을 주장할 수 없습니다.

**남은 일:**
- `bridge/pose_source.py`의 `Bno08xPoseSource`를 실제 BNO08x IMU에 연결, bias/drift/latency/축 정렬 오차 실측
- `bridge/range_sensor.py`의 `Tf02ProRangeSensor`를 실제 TF02-Pro 레이저에 연결, 무효판정 기준·노이즈 분포 실측
- 실측 노이즈 분포로 L1 Monte Carlo 재실행 — "가정된 노이즈 조건"을 "실측 기반 조건"으로 교체
- 실제 카메라 캘리브레이션 (현재 intrinsics는 플레이스홀더)
- 가능하면 실사격/계측 사격으로 조준점 대 실제 탄착점 비교

---

*Level 1: `aiming_engine` 단독, 순수 파이썬 합성 궤적, 이미지·트래커·실제 카메라 미사용 — 상세는 `aim_oracle_validation.html`.*
*Level 2: `test/visible.mp4` (실제 영상) + Gazebo Sim 10.5.0 (WSL2 Ubuntu-24.04) 시뮬레이션 IMU/거리 — 산출물은 `gazebo/gazebo_result.json`, `gazebo/performance.json`, `gazebo/level2_result.mp4`.*
*둘 다 설정 변경 없이 `config/aiming_engine.yaml` / `config/bridge.test_video.yaml` 기준으로 실행.*
