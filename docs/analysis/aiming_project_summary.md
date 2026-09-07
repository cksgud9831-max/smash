# Aiming 프로젝트 코드 학습 정리

## 1. 전체 개요

이 프로젝트는 **AI 스마트 조준경(Smart Optical Scope) 조준 보조 시스템**이다. 처리 파이프라인은 다음과 같다.

```
Camera -> YOLO11s-GA (탐지) -> SmartTracker (추적) -> Aiming Engine (조준 계산) -> HUD -> 사람 운용자
```

가장 중요한 설계 원칙은 코드 전반에 반복적으로 명시되어 있다: **이 시스템에는 발사(fire) 명령이 어디에도 없다.** `AimAssistOutput`의 모든 값은 HUD에 표시되는 참고 정보(advisory)이며, 실제 발사 여부와 시점은 항상 사람 운용자가 결정한다. `aiming_engine/README.md`에는 원래 자율 화기 통제(fire-control) 시스템으로 설계됐다가, 수학 로직은 그대로 두고 "자동 발사"에 해당하는 의사결정 레이어를 제거하고 이름을 바꿔 지금의 조준 보조(aim-assist) 시스템으로 재구성했다는 이력이 적혀 있다.

리포지토리는 크게 세 부분으로 나뉜다.

- **`aiming_engine/`**: 탄도/조준 계산 핵심 로직 (순수 수학, 하드웨어 독립적)
- **`bridge/`**: 실제 카메라·탐지기·추적기·거리계·자세추정 장비를 `aiming_engine`이 이해하는 입력(`TrackerFrame`)으로 변환하는 접착 계층
- **`detector+tracker/`**: YOLO 탐지 모델 학습(GA 하이퍼파라미터 탐색)과, `bridge`와는 별개로 존재하는 실험용 옵티컬플로우 기반 추적기(`yolo_follower_v3_reactivate.py`) 및 벤치마크/GA 튜닝 스크립트

그 외 `config/`(YAML 설정), `tests/`(pytest 단위 테스트), `examples/`(데모 스크립트), `paper_text.txt`(참고 논문 발췌), `aiming_engine_report.html`(기존 보고서로 추정)가 있다.

## 2. `aiming_engine/` — 조준 계산 코어

### 2.1 좌표계 규약

- **World 프레임**: 관성계, Z-up. 표적 운동 물리(`TargetState`, `LeadPrediction`, `HitEquation`, `ProjectileModel`)는 전부 이 프레임에서 계산된다.
- **Camera 프레임**: 핀홀 카메라 표준 (X right, Y down, Z forward)
- **Scope 프레임**: FLU (X forward/보어사이트, Y left, Z up). azimuth=0, elevation=0이 "보어사이트 정면"을 의미. HUD에 표시할 최종 azimuth/elevation만 이 프레임으로 변환한다.
- 변환 순서는 `Image -> Camera -> Scope -> World`로 고정. `coordinate_transform.py`가 이 전체 체인(`CoordinateTransform`)을 담당하며, 카메라→스코프는 "고정된 축 재배열 행렬"과 `ScopeConfig`의 미세한 마운트 오차 보정 회전을 합성한 것.
- `laser_range_m`은 픽셀 시선(ray) 방향의 슬랜트 거리로 취급(z-depth 아님).

### 2.2 핵심 수학 파이프라인

1. **`TargetState` (target_state.py)**: 표적의 world 좌표 위치 이력을 버퍼에 쌓아 위치/속도/가속도를 추정. `MotionEstimator`라는 추상 전략 인터페이스를 두고, 현재는 `ConstantVelocityEstimator`(최소자승 선형 피팅, numpy `lstsq`로 x/y/z 동시 계산, 가속도는 항상 0)만 구현되어 있다. 나중에 칼만필터/등가속도 추정기로 교체 가능하도록 설계.

2. **`LeadPrediction` (lead_prediction.py)**: Newton solver의 초기값(warm start)만 만들어주는 "빠른 1차 근사". 현재 표적 위치까지의 직선거리/탄속으로 비행시간 t0을 추정하고, 그 시점의 표적 예측 위치를 향한 직선 방향으로 azimuth0/elevation0을 계산 (중력 무시).

3. **`ProjectileModel` (projectile_model.py)**: 발사체 궤적 적분. `ForceModel` 플러그인 리스트(`forces.py`)의 가속도를 합산해 RK4로 적분. v1은 중력만 있는데, 이 경우 RK4 대신 닫힌해(`p = p0 + v0*t + 0.5*g*t²`)를 써서 O(1)로 계산(중력만 있으면 RK4와 결과가 완전히 같음). 항력/바람 같은 비선형 힘이 추가되면 자동으로 RK4로 폴백.

4. **`HitEquation` (hit_equation.py)**: `잔차(residual) = ProjectilePosition(t, az, el) - TargetPosition(t)`를 정의만 하고 풀지는 않는다. `TargetState.predict()`와 `ProjectileModel.state_at()`을 합성해서, "미리 계산한 고정 리드포인트"가 아니라 **비행시간과 표적 운동을 동시에(연립방정식으로) 푸는** 구조가 핵심.

5. **`NewtonSolver` (newton_solver.py)**: 범용 다변수 뉴턴-랩슨 solver. 탄도학을 전혀 모르고 `equation.residual(x)`만 호출한다. 야코비안은 전진 유한차분으로 근사(해석적 미분 불필요). `x = [time_of_flight, azimuth, elevation]` 3원 연립방정식을 3~5회 반복 이내로 수렴시키는 것이 목표.

6. **`AimSolver` (aim_solver.py)**: 위 요소들을 조합해 프레임별 조준해를 계산하는 오케스트레이터. `LeadPrediction`의 근사값을 초기값으로 쓰되, `warm_start` 옵션이 켜져 있으면 **이전 프레임의 수렴 해**를 다음 프레임 초기값으로 재사용 — 표적이 부드럽게 움직일 때는 1~2회 반복만으로 수렴해 실시간(45fps) 예산을 지킨다. 최종적으로 world 방향을 scope 방향(azimuth/elevation)으로 변환하고, "탄도 보정값"(순수 조준선 대비 얼마나 리드를 줬는지, `ballistic_offset`)도 함께 계산해 HUD가 시각화할 수 있게 한다.

### 2.3 신뢰도/상태 판단 계층

- **`HitProbability` (hit_probability.py)**: 추적신뢰도, 팔로워 안정성, 조준오차, 거리, 표적속도 5개 요소를 각각 [0,1]로 정규화(지수감쇠 함수 등)한 뒤 YAML의 가중치로 가중평균 — "명중 확률" 추정치를 산출. 의도적으로 단순한 v1 모델(주석에 "구현 정책상 단순하게 유지"라 명시)이며, 추후 탄도 분산 공분산 기반 모델로 교체 가능하도록 이 파일만 건드리면 되게 설계.

- **`AimReadinessLogic` (aim_readiness_logic.py)**: "지금 이 조준을 믿어도 되는가"를 판단하는 순수 불리언 AND 게이트(추적 안정, 팔로워 안정, 명중확률 임계값 초과, 유효 사거리 범위, 조준오차 임계값 미만). 가중치 없이 단순 AND로만 구성해 감사(audit)하기 쉽게 설계. 이 결과가 발사를 허가하는 게 아니라 HUD의 "녹색 레티클" 신호일 뿐임을 재차 강조.

- **`AimStateMachine` (aim_state_machine.py)**: `SEARCH -> TRACK -> AIM -> READY` 4단계 상태기계. READY는 "조준해가 안정적으로 신뢰할 만하다"는 의미일 뿐 발사 상태가 아니다(발사 상태 자체가 존재하지 않음). 프레임마다 추적신뢰도·팔로워안정성·조준적합여부(`aim_ok`)를 받아 카운터 기반으로 전이한다.

- **`AimingManager` (aiming_manager.py)**: 위 전체를 하나로 묶는 메인 컨트롤러. `TrackerFrame` 1개를 입력받아 `AimAssistOutput` 1개를 출력하는 `update()` 메서드가 유일한 공개 API. 좌표변환 → 표적상태 갱신 → 조준해 계산 → 거리/조준오차/표적속도 산출 → 명중확률 → 상태기계 갱신 → `aim_ready` 최종 판정(상태기계가 READY에 도달**하고** 동시에 `AimReadinessLogic`도 통과해야 함) 순서로 진행.

### 2.4 확장성 설계

- 새 힘(항력, 바람, G1/G7 탄도계수)을 추가하려면 `forces.py`에 `ForceModel` 서브클래스를 만들고 `FORCE_REGISTRY`에 등록 + YAML의 `projectile.forces`에 이름 추가만 하면 됨. `ProjectileModel`/`HitEquation`/`NewtonSolver`/`AimSolver`는 수정 불필요.
- 새 표적 운동 추정기(칼만필터, 등가속도)를 추가하려면 `target_state.py`에 `MotionEstimator` 서브클래스를 만들고 `build_estimator()`에 등록 + YAML `target_state.estimator` 값만 바꾸면 됨.
- 모든 튜닝 파라미터는 `config/aiming_engine.yaml`에 있고, `config.py`가 이를 타입이 명시된 frozen dataclass로 한 번만 로드한다. YAML을 직접 읽는 곳은 `config.py`뿐.

## 3. `bridge/` — 하드웨어/추적기 접착 계층

`bridge/config.py`는 `aiming_engine/config.py`와 동일한 패턴(YAML → frozen dataclass, 단일 로더)을 쓰지만, 다루는 대상은 아이밍 수학이 아니라 **하드웨어 캘리브레이션**(카메라 내부파라미터, 레이저, 자세추정 소스, 추적기 튜닝값)이다.

- **`Detector` (detector.py)**: `detect(frame_bgr) -> [[x1,y1,w,h,conf], ...]` 인터페이스. `OnnxDetector`는 onnxruntime CPU 추론(데스크톱 개발용), `TRTDetector`는 Jetson 전용 TensorRT 래퍼(`trt_infer.py`)를 지연 임포트해서 감싼다 — TensorRT가 없는 머신에서도 `import`만으로는 에러가 나지 않게 설계.
- **`trt_infer.py`**: `jetson_deployment.zip`에서 **그대로 벤더링(vendored)**된 파일(주석에 명시). TensorRT 10 API + pycuda로 순수 YOLO 추론
을 수행하는 최적화 클래스(`TRTYOLO`). Ultralytics/torchvision 의존성 없이 동작.
- **`bridge/smart_tracking/`**: 이것도 `jetson_deployment.zip`에서 그대로 벤더링된, 이 프로젝트 범위 밖의 별도 컨트리뷰션(주석에 "Out of scope for this project to modify" 명시).
  - **`custom_kalman.py`**: `ConstantAccelerationKalmanFilter`(CA-KF, 10차원 상태: 중심좌표/종횡비/높이 + 속도 4개 + 가속도 2개)와 `ConstantVelocityKalmanFilter`(CV-KF, 8차원 표준 칼만필터) 두 종류. CA-KF는 드론의 급격한 비선형 선회를 추적하기 위해 만들어졌고, 가속도 항이 위치예측에 과도하게 영향을 주지 않도록(발산 방지) 계수를 0.01배로 완만하게 낮춰놓았다.
  - **`improvement_tracker.py` (`SmartTracker`)**: 단일 UAV 특화 추적기. 특징: (1) CA-KF 내장, (2) 예측오차 공분산 기반 "위기 감지"(crisis_mode)와 동적 GMC(Global Motion Compensation, ORB 특징점+호모그래피로 카메라 흔들림 보정)/Re-ID 제어, (3) 급선회 포착 시 게이팅 영역을 선제적으로 확장. Re-ID는 단일 표적 도메인이라 완전히 꺼져 있고 순수 IoU 기반 헝가리안 매칭(`linear_sum_assignment`)만 사용. **알려진 특이사항**: `crisis_mode` 반환값은 `tracked_stracks[0]`에서만 파생되므로 트랙이 0개일 때 항상 `False`를 반환 — 호출자가 "트랙 없음"을 "안정"으로 오해하면 안 된다는 경고가 `frame_builder.py`와 파일 헤더 양쪽에 명시되어 있다.
- **`confidence.py`**: SmartTracker의 원시 신호(score, uncertainty, crisis_mode)를 `aiming_engine.TrackerFrame`이 기대하는 `tracking_confidence`(score × 안정성계수, [0,1] clip)와 `follower_state`("stable"/"unstable") 필드로 매핑하는 순수 함수들. 독립적으로 유닛테스트 가능하도록 분리.
- **`laser_alignment.py` (`LaserAligner`)**: 레이저 거리계의 보어사이트와 카메라 광학중심 사이의 물리적 오프셋을 보정. 레이저가 측정한 슬랜트거리를 카메라 픽셀 시선벡터에 투영해 최적 스칼라 거리값을 계산 — 보어사이트/픽셀시선 정렬오차는 완전히 제거하지만, 오프셋의 횡방향(광축에 수직) 성분으로 인한 시차(parallax)까지는 없애지 못한다고 문서화되어 있음(그래도 원시값을 그대로 쓰는 것보단 낫다).
- **`pose_source.py`, `range_sensor.py`**: 각각 짐벌/IMU 자세추정과 레이저 거리계의 추상 인터페이스(`PoseSource`, `RangeSensor`). 실제 하드웨어/프로토콜이 아직 정해지지 않아 현재는 고정값을 반환하는 Mock 구현체만 존재(`MockPoseSource`, `MockRangeSensor`).
- **`frame_builder.py` (`TrackerFrameBuilder`)**: 위 모든 조각(SmartTracker, RangeSensor, PoseSource, LaserAligner)을 소유하고 `build(detections, frame, timestamp) -> Optional[TrackerFrame]`로 한 프레임을 조립한다. 단일 표적 락온 로직(`_select_track`)이 있어, 이전에 고정한 track_id가 이번 프레임에도 confirmed 상태면 그걸 유지하고(조준해가 트랙 간에 튀는 것 방지), 아니면 가장 신뢰도 높은 트랙을 새로 선택한다. 추적 대상이 없거나 유효한 거리 샘플이 없으면 `None`을 반환하고, 호출자는 그 프레임에 대해 그냥 `AimingManager.update()`를 건너뛰면 된다(발사 게이팅이 아니므로 에러 취급 안 함).

## 4. `detector+tracker/` — 탐지 모델 학습 & 실험용 추적기

- **`ga_code/`**: Ultralytics YOLO(`yolo11s.pt`)를 Anti-UAV300 데이터셋으로 학습/튜닝하는 스크립트. `tune_yolo11s_ga.py`/`_v2.py`는 `model.tune()`으로 유전 알고리즘 기반 하이퍼파라미터 탐색(반복 10~15회)을 수행하고, `train_yolo11s_ga_final.py`는 그렇게 찾은 최종 하이퍼파라미터(lr0, momentum, box/cls/dfl loss 가중치, 증강 파라미터 등)로 30 epoch 본학습을 실행한다.
- **`ga_results/yolo11s_ga_final-3/`**: 위 학습의 결과물 — 학습곡선/PR곡선/혼동행렬 이미지, `results.csv`, 학습 배치 시각화, 그리고 최종 가중치 `weights/best.pt`, `last.pt`(각 약 19MB). `bridge/weights/base_416.onnx`(약 10MB, 416×416 입력)는 이 학습 결과를 ONNX로 변환한 배포용 모델로 보인다.
- **`code/yolo_follower_v3_reactivate.py`** (약 1,500줄): `bridge/smart_tracking`과는 **별개의, 더 이전 세대로 보이는** 단일 표적 추적 파이프라인. YOLO 탐지 + Lucas-Kanade 옵티컬플로우 팔로워를 결합한 하이브리드 추적기로, 다음을 포함한다.
  - ROI 기반 YOLO 재탐지, 모션-일관성 있는 bbox 선택(`select_motion_consistent_bbox_from_yolo`), bbox 블렌딩/보정 로직
  - 다수의 GA로 튜닝된 임계값 상수(ROI_MARGIN, CENTER_SHIFT_THRES 등, 주석에 "GA-efficient candidate" 표시)
  - 벤치마크용 리소스 샘플러(`BenchmarkResourceSampler`): CPU/메모리/온도(Jetson `tegrastats` 파싱 포함) 모니터링
  - 자체 유전 알고리즘 최적화 루프(`ga_optimize`, `evaluate_tracker`, `compute_fitness`, `mutate_ga_candidate`, `crossover_ga_candidates`) — 추적 파라미터 자체를 GT(ground truth) 주석과 비교한 IoU/재탐색시간 등으로 피트니스를 매겨 탐색
  - `main()`: 비디오 파일을 읽어 초기 YOLO 탐지 → LK 옵티컬플로우 추적 → 주기적/조건부 YOLO 재보정을 반복하며 결과 영상과 디버그 로그를 저장하는 실행 스크립트
  
  즉 이 파일은 `bridge/smart_tracking`(칼만필터+IoU 헝가리안 매칭 기반)과는 다른 접근(옵티컬플로우+YOLO 하이브리드, 자체 GA 튜닝)의 추적기 실험/벤치마크 코드로 보인다.

## 5. 설정 파일 (`config/`)

- **`aiming_engine.yaml`**: 탄속 850m/s, 중력만 적용, RK4 스텝 0.01s, 뉴턴솔버 최대 5회 반복(허용오차 1mm), warm_start 활성화, 표적추정기는 constant_velocity, 명중확률 가중치(추적신뢰도 0.3/팔로워안정성 0.2/조준오차 0.3/거리 0.1/속도 0.1), 조준준비 임계값(명중확률 0.75 초과, 조준오차 5mrad 미만, 유효사거리 5~800m) 등.
- **`bridge.yaml`**: 카메라 내부파라미터(현재 fx=fy=500, cx=320, cy=240 — 플레이스홀더, 실장비 캘리브레이션 필요라고 주석에 명시), 레이저/자세추정 소스는 둘 다 `mock`(고정거리 250m / identity 자세), 추적기 튜닝값(uncertainty_threshold=3.0, gating_factor=1.6 — SmartTracker 클래스 기본값과 다른, 실측으로 튜닝된 값이라고 주석에 명시), 탐지기 백엔드는 onnx(desktop 개발용).

## 6. 테스트 & 예제

`tests/`에는 `aiming_engine`과 `bridge`의 거의 모든 모듈(아이밍 솔버, 상태기계, 명중확률, 좌표변환, 뉴턴솔버, 발사체모델, 리드예측, 신뢰도매핑, 레이저정렬, 프레임빌더 등)에 대응하는 단위 테스트 파일이 1:1로 존재한다. `pytest.ini`는 `testpaths = tests`, `pythonpath = .`로 설정.

`examples/`에는 두 개의 데모가 있다.
- `run_aiming_engine_demo.py`: SmartTracker/탐지기 없이 합성 `TrackerFrame` 시퀀스(직선 접근하는 표적)만으로 `AimingManager`를 실시간 스모크 테스트하고 프레임별 상태/지연시간을 출력.
- `run_bridge_pipeline_demo.py`: 비디오 파일/웹캠 입력을 받아 Detector → TrackerFrameBuilder → AimingManager 전체 파이프라인을 데스크톱에서 end-to-end로 돌리는 CLI 데모(Mock 센서 사용).

## 7. 참고 자료

- **`paper_text.txt`**: "Algorithm and Simulation of Airborne Fire Control Aiming Angle Correction"(Boyu Gao 외, Shenyang University of Technology)이라는 학술 논문의 텍스트 추출본(5페이지, 수식 부분은 OCR/PDF 추출 특성상 다소 깨져 있음). 뉴턴 하강법(Newton descending method)과 변수각 적분법(variable angle integral)으로 사격제원(발사각)을 반복 보정하는 기법을 다루며, `aiming_engine`의 `HitEquation` + `NewtonSolver` 조합(비행시간·표적운동을 동시에 뉴턴법으로 푸는 방식)이 이 논문의 접근을 참고해 설계된 것으로 보인다.
- **`aiming_engine_report.html`**: 별도의 기존 산출물(HTML 리포트)로, 이번 학습에서는 내용을 열람하지 않았다.

## 8. 종합 인상

- 코드 전반의 설계 품질이 상당히 높다: 모든 모듈이 얇은 인터페이스(Protocol/ABC)로 분리되어 있고, "이 파일만 고치면 확장된다"는 문서화가 일관되게 되어 있다.
- 물리 계산(월드 프레임)과 표시용 변환(스코프 프레임)이 명확히 분리되어 있고, 실시간 예산(45fps)을 지키기 위한 최적화(닫힌해 단축경로, 워밍스타트)가 곳곳에 있다.
- **안전/윤리적 프레이밍이 코드베이스 전체에 명시적으로 새겨져 있다** — 거의 모든 파일 docstring에 "이것은 발사하지 않는 조준 보조 시스템"이라는 문구가 반복된다. 이는 실수로 시스템의 성격이 오해되는 것을 막기 위한 의도적 장치로 보인다.
- `bridge/smart_tracking`(칼만필터 기반)과 `detector+tracker/code/yolo_follower_v3_reactivate.py`(옵티컬플로우+YOLO 기반)는 서로 다른 두 세대/계열의 추적기이며, 전자가 현재 `bridge` 파이프라인에서 실제로 사용되는 쪽이다.
- 하드웨어 관련 부분(레이저, 짐벌/IMU, TensorRT 추론)은 대부분 Mock/미구현 상태이며, 실장비 연결 전 캘리브레이션이 필요하다는 주석이 반복적으로 남아 있다.
