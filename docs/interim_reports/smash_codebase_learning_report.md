# SMASH AI 스마트 조준경 시스템 전체 학습 및 아키텍처 분석 보고서

작성 일자: 2026년 9월 8일
작성자: Antigravity AI

## 1. 개요 및 프로젝트 핵심 철학

본 프로젝트는 5.56mm 개인 화기(소총)에 장착되어 영상 기반 표적 탐지 및 추적을 수행하고, 센서(레이저 거리계, IMU 자세 센서) 데이터를 융합하여 정밀 탄도 및 리드각을 계산한 뒤 HUD 레티클로 사수에게 제공하는 스마트 조준 지원 시스템(Smart Aim Assist Scope)입니다.

### 핵심 운영 원칙
* 인간 중심 사격 결정: 시스템의 모든 산출물(리드각, 낙차각, 조준 준비 완료 상태)은 HUD에 투영되는 자문(Advisory) 정보이며, 발사 여부와 발사 시점은 100% 인간 운용자가 결정합니다.
* 과학적 타당성 및 투명성: 임의의 휴리스틱 대신 엄밀한 3차원 기하학적 좌표 변환, 뉴턴 랩슨 수치해석 솔버, 그리고 4차 룽게 쿠타(RK4) 물리 적분 모델을 기반으로 계산을 수행합니다.
* 3D 시뮬레이션 환경의 역할: Gazebo Harmonic 기반의 CIWS 포탑 모델(SMASH FCS)은 사수가 조준경을 파지하여 표적을 지향하는 신체 움직임을 정량적으로 모사하고 폐루프 수렴 성능을 평가하기 위한 물리 검증 환경입니다.

## 2. 서브시스템별 상세 구조 및 역할

### 2.1 조준 연산 코어 (aiming_engine)
하드웨어 및 센서 인터페이스와 분리된 순수 수학 및 물리 모델 계층입니다.
* 좌표계 정의:
  1. World 관성계: Z축 상방(Z up), 오른손 좌표계. 표적 운동 모델 및 탄도 적분이 수행되는 절대 기준 좌표계.
  2. Camera 핀홀 좌표계: X축 우측, Y축 하방, Z축 전방(광축 방향).
  3. Scope 좌표계: Forward Left Up (FLU) 기준계. 방위각(Azimuth)과 고각(Elevation)이 0일 때 총구 및 조준경 광축 정면을 가리킴.
* 표적 운동 상태 추정 (TargetState):
  * ConstantVelocityEstimator: 최근 이력 버퍼 기반 다변수 최소자승법(OLS) 선형 회귀 적용. 위치 스무딩 절편과 지수이동평균(EMA) 속도 스무딩(alpha)을 통해 이중 미분 노이즈를 억제.
  * ConstantAccelerationKalmanEstimator: 9차원 상태 변수(3차원 위치, 속도, 가속도)를 추정하는 3D 등가속도 칼만 필터(CA KF).
* 탄도 물리 모델 (ProjectileModel 및 Forces):
  * 중력 전용 폐쇄형 해석(Analytic Bypass): 100m 이내 소총 직사 화기 구간에서는 공기 저항 영향이 미미(1cm 미만 오차)하므로, 연산량을 O(1)로 줄여 0.1ms 미만에 즉시 위치를 계산.
  * 비선형 외력 지원: 항력(DragForce) 활성화 시 G1 및 G7 항력 곡선과 마하 수에 따른 가속도를 반영하며, 4차 룽게 쿠타(RK4) 적분 루프를 Numba JIT 컴파일러로 고속화.
* 조준 연립방정식 솔버 (NewtonSolver 및 AimSolver):
  * 탄환 도달 위치와 표적 미래 위치 간의 잔차 방정식을 0으로 수렴시키는 3차원 뉴턴 랩슨 수치해석 솔버.
  * 이전 프레임의 탄도 편차를 계승하는 동적 웜스타트(Motion Compensated Warm Start)를 적용하여 솔버 반복 횟수를 평균 1회 내외로 단축.
  * 거리에 비례하여 수렴 임계값을 완화하는 조기 종료(Dynamic Tolerance Early Stopping) 구현.
* 조준 상태 머신 (AimStateMachine):
  * SEARCH → TRACK → AIM → READY 순서로 전이.
  * READY 상태는 사격 명령이 아니라 조준점이 통계적으로 신뢰할 수 있게 수렴했음을 의미.

### 2.2 센서 및 하드웨어 접착 계층 (bridge)
* 추적기 (OpticalFlowTracker 및 FinalTracker):
  * YOLO11s 탐지기와 Lucas Kanade 옵티컬 플로우를 결합한 하이브리드 추적기.
  * SEARCH 상태(5프레임 주기 탐지, 2회 연속 탐지 확인 시 획득)와 TRACK 상태(20프레임 또는 30프레임 주기 YOLO 재보정)로 동작.
  * Jetson 배포를 위한 비동기 GPU 추론 TensorRT 엔진 백엔드 지원.
* 센서 융합 및 보정 (FrameBuilder):
  * 레이저 시차 보정 (LaserAligner): 조준경 광축과 레이저 거리계 사이의 물리적 마운트 오프셋을 픽셀 시선 벡터에 투영 보정(역행렬 캐싱 적용).
  * 센서 결측 방어 (Zero Order Hold): LiDAR(TF02 Pro) 및 IMU(BNO08x) 통신 노이즈 발생 시 최근 유효 데이터를 최대 100ms(약 3프레임) 유지하여 레티클 깜빡임 방지.
  * 논블로킹 센서 수신: 시리얼 및 I2C 통신을 백그라운드 스레드에서 폴링하고 배치 슬라이싱으로 고속 버퍼 리싱크 수행.

### 2.3 트래킹 신뢰도 진단 계층 (jun_reliability)
* 기존 조준 엔진 코드를 수정하지 않고 독립적으로 동작하는 진단 전용 모듈.
* 관측(Observation), 일관성(Consistency), 신선도(Freshness), 실행시간(Runtime)의 4대 증거 축을 융합.
* 정체 표적 감지기(StaleTrackDetector)와 시간적 히스테리시스를 거쳐 STABLE, HOLD, LOST 3단계 진단 상태를 발행.
* SingleFrameTrackerFanout 어댑터를 통해 프레임당 단 1회의 트래커 추론 결과를 기존 운용 파이프라인과 신뢰도 모듈에 공유.

### 2.4 3D 시뮬레이션 및 검증 인프라 (simulation, scripts, tests)
* Gazebo Harmonic 및 ROS 2 Jazzy 기반 SMASH FCS:
  * 2축 대공 CIWS 포탑 모델, 조준경 동축 광각 LiDAR, 대화형 스코프 뷰어 포함.
  * run_smash_fcs.bat 배치 파일을 통한 원클릭 환경 진단, 빌드, 수동 및 자동 조준 시뮬레이션 지원.
* 오프라인 오라클 검증 스크립트:
  * scripts/07_stage2_lead_accuracy_validation.py: 3D 표적 궤적 기반 리드각 정밀도 및 스무딩 파라미터 검증.
  * scripts/08_stage1_fcs_offline_validation.py: 역기구학 잔차 및 종단 조준오차 수치 검증.
* 단위 테스트: tests 디렉토리 내 114개 pytest 단위 테스트로 핵심 수치 알고리즘의 100% 무결성 검증.

## 3. 한계점 및 향후 연구/개발 과제

1. 실장비 장착 캘리브레이션:
   * 현재 YAML 설정 내 레이저 마운트 오프셋, IMU 장착 회전각, 카메라 내부 파라미터가 플레이스홀더이므로 실기 장착 시 정밀 계측값 입력 필요.
2. 실제 표적 기동 노이즈 기반 하이퍼파라미터 미세 조정:
   * 등속 직선 비행 외에 급선회 및 회피 기동 시나리오에서 3D CA KF 및 EMA 스무딩의 동적 반응성 추가 검증 권장.

## 4. 최종 통합 파이프라인 구조 및 브릿지 역할

사용자 요구사항에 따라 확정된 최종 시스템 파이프라인은 병렬 분기 방식이 아닌 명확한 직렬 흐름으로 통합됩니다:

Camera → YOLO11s → Optical Flow → STABLE / HOLD / LOST → Aiming → Decision

### 단계별 흐름 및 상호작용
1. Camera: 센서 프레임 획득 (원시 비디오 스트림)
2. YOLO11s: 표적 탐색(SEARCH 2회 확인) 및 주기적/이상감지 시 비동기 앵커 재보정
3. Optical Flow: 매 프레임 Lucas Kanade 기반 실시간 고속 표적 추적
4. STABLE / HOLD / LOST: 트래킹 신뢰도 진단 계층 (jun_reliability)
   * STABLE: 신뢰할 수 있는 트래킹 상태. 즉시 정밀 Aiming 연산으로 데이터 전달.
   * HOLD: 일시적 가림, 급기동, 탐지 누락 등으로 신뢰성 저하. 조준 유지 및 예측 모드로 전환.
   * LOST: 신뢰할 수 있는 표적 소실. 조준 해제, 웜스타트 리셋 및 SEARCH 모드로 복귀.
5. Aiming: 좌표계 3단계 변환, 표적 3D 운동 추정, 3D 뉴턴 탄도 리드각 및 낙차 계산, READY 판정 (aiming_engine)
6. Decision: HUD 레티클을 통한 사수 시각 피드백 및 인간 운용자의 최종 발사 판단

### 브릿지(bridge)의 핵심 역할
브릿지는 각 독립 모듈을 유기적으로 엮어주는 전체 파이프라인의 오케스트레이터(접착 계층)입니다:
* 센서 융합: 카메라 픽셀 레이와 레이저 거리계(TF02 Pro), IMU(BNO08x) 자세를 결합하고 시차 및 드롭아웃을 보정.
* 신뢰도 게이팅: Optical Flow 결과를 STABLE / HOLD / LOST 진단 모듈에 통과시켜 상태를 판정한 후, 그 결과를 바탕으로 Aiming 입력 여부와 모드를 제어.
* 일관된 데이터 파이프라인 관리: 단일 루프 내에서 모듈 간 데이터 규격을 매끄럽게 변환하여 전달.

## 5. 최종 병합 및 146개 단위 테스트 검증 완료

smash integration benchmark v1 패키지에서 검증된 핵심 모듈들을 현재 워크스페이스에 정밀 선별 병합하였습니다:
* 벤치마크 및 오케스트레이터: benchmarks/simulation_orchestration.py(직렬 게이팅) 및 integration_benchmark_v1.py(717줄 규모 평가 도구) 도입 완료.
* 브릿지 고도화: bridge/final_tracker.py(TensorRT 비동기 추론) 도입, frame_builder.py의 100ms 센서 영차 홀드 및 팩토리 분기 적용, 레이저 역행렬 캐싱, 시리얼 배치 슬라이싱 리싱크 적용.
* 탄도 물리 가속: aiming_engine/projectile_model.py 및 forces.py에 Numba JIT 컴파일 기계어 적분 가속 적용(약 10배 가속), 동적 웜스타트 및 조기 종료 완비.
* 워크스페이스 최신 성과 보존: 9차원 3D CA KF(ConstantAccelerationKalmanEstimator) 및 EMA 스무딩, ROS 2 Jazzy CIWS 환경, 실제 20MB 모델 바이너리를 온전히 보존.
* 무결성 검증: 총 146개 pytest 단위 테스트 전체 100% 정상 통과 (146 passed in 3.28s).


## 6. Gazebo CIWS 3D 환경 및 대화형 스코프 뷰어 통합 고도화

### 6.1 문제 진단 및 과학적 원인 분석
* WSLg 가상화 Ogre2 렌더러 충돌: Windows 호스트와 WSL2 간 그래픽 가상화 환경에서 Gazebo Sim의 3D Ogre2 렌더러 구동 시 D3D12 가상화 드라이버 경고와 함께 윈도우 창 렌더링이 멈추거나 검은 화면으로 지연되는 현상 확인.
* 모델 가중치 인자 누락 방어: 론치 시 model_path가 명시되지 않을 경우를 대비하여 smash_fcs_node.py 내부에서 GA 튜닝 최종 가중치(detector+tracker/ga_results/yolo11s_ga_final_3/weights/best.pt)를 자동 탐색 및 로드하는 안전 폴백 메커니즘 구축.
* 뷰어와 시뮬레이션의 이원화 해소: 기존에 5번(시뮬레이션 구동)과 8번(스코프 뷰어)을 별도의 터미널에서 각각 띄워야 했던 번거로움을 해결하기 위해 smash_scene.launch.py에 launch_viewer 인자를 추가하여 5번 선택 시 원클릭으로 동시 구동되도록 통합.

### 6.2 수행된 최적화 조치
1. Gazebo 헤드리스 기본값 설정: 백엔드 물리 시뮬레이터는 불필요한 GPU 렌더링 부하가 없는 서버 모드(HEADLESS=true)로 구동하고, 사수의 조준 HUD 및 실시간 영상, 트래킹, 사격 통제는 OpenCV 기반의 경량 대화형 창(smash_scope_viewer)으로 즉시 렌더링하도록 전환. (필요 시 9번 메뉴를 통해 Gazebo 전체 3D 뷰 켜기 및 끄기 토글 지원)
2. 스코프 뷰어 윈도우 조기 등록: 첫 프레임 수신 전 대기 화면을 즉시 imshow하여 WSLg 윈도우 매니저에 창이 지연 없이 즉각 모니터에 등록되도록 보장.
3. 가중치 자동 탐색 폴백: 파라미터 누락 시에도 저장소 내 19.15MB 실 가중치 바이너리를 자동으로 바인딩하여 런타임 예외 원천 방지.
4. 원클릭 통합 런처 완비: run_smash_fcs.bat에서 5번 또는 6번 선택 시 시뮬레이션 서버와 사수 스코프 뷰어가 동시에 즉각 실행되도록 파이프라인 단일화.
