# 탐지 추적기 브릿지 조준엔진 연동 구조 정밀 코드 분석 보고서

## 1. 분석 개요

본 보고서는 detector+tracker, bridge, aiming_engine 세 핵심 모듈 간의 코드 결합 방식, 데이터 전송 구조, 인터페이스 연결성을 딥다이브 코드 분석을 통해 입증한 내역을 정리한 문서입니다.

## 2. 모듈 간 연결 구조 및 데이터 흐름

전체 파이프라인의 데이터 흐름은 다음과 같은 3단계 연동 구조로 연결되어 있습니다:

### 1단계: detector+tracker => bridge (OpticalFlowTracker)
* detector+tracker 모듈에는 YOLO11s GA 학습 모델 가중치와 유전 알고리즘 기반 옵티컬 플로우 추적 알고리즘이 위치함.
* bridge/optical_flow_tracker.py 의 OpticalFlowTracker 클래스는 detector+tracker의 추적 알고리즘을 이식하여 로드함.
* 매 프레임 YOLO 추론을 수행하지 않고, 30프레임 주기 또는 추적 손실 시에만 YOLO 재탐지를 수행하며 프레임 사이에는 Lucas-Kanade 2D 옵티컬 플로우(cv2.calcOpticalFlowPyrLK)로 픽셀 이동을 고속 추적함.
* 추적 결과는 FollowerResult DTO(bbox, score, is_recovery_event) 객체로 래핑되어 다음 단계로 전달됨.

### 2단계: bridge 내부 센서 융합 => TrackerFrame 생성 (frame_builder.py)
* bridge/frame_builder.py 의 TrackerFrameBuilder 클래스가 바인딩 핵심 역할을 수행함.
* OpticalFlowTracker가 출력한 2D 바운딩 박스에서 표적 픽셀 중심 좌표 center_px (u, v)를 구함.
* bridge/confidence.py 모듈을 호출하여 추적 신뢰도 tracking_confidence 및 팔로워 상태 follower_state를 산출함.
* RangeSensor(TF02 Pro 거리계 모듈)로부터 관측한 거리를 획득하고, LaserAligner를 통해 레이저 보어사이트 오프셋 정렬 거리를 계산함.
* PoseSource(BNO08x IMU 모듈)로부터 카메라 플랫폼 4x4 변환 행렬 camera_extrinsics를 획득함.
* 위 모든 데이터를 하나로 묶어 aiming_engine이 수용하는 표준 데이터 객체인 TrackerFrame DTO로 바인딩하여 반환함.

### 3단계: bridge => aiming_engine (AimingManager.update)
* aiming_engine/aiming_manager.py 의 AimingManager.update(tracker_frame) 메서드가 바인딩 접점으로 구동함.
* CoordinateTransform 모듈이 TrackerFrame 내부의 2D 픽셀 center_px + 슬랜트 거리 + 4x4 자세 행렬을 3D 관성 월드 좌표계 Vector3(X, Y, Z meters)로 역투영함.
* TargetState 모듈이 3D 월드 좌표를 입력받아 3D 위치, 속도, 가속도를 추정함 (ConstantVelocityEstimator 또는 ConstantAccelerationKalmanEstimator 선택 연동).
* AimSolver 모듈이 탄도 적분과 뉴턴 솔버(NewtonSolver)를 거쳐 리드 포인트를 계산하고 스코프 기준 Azimuth, Elevation 조준해를 산출함.
* HitProbability 및 AimReadinessLogic, AimStateMachine을 거쳐 최종 HUD 레티클 오프셋 및 aim_ready 신호가 담긴 AimAssistOutput 객체를 출력함.

## 3. 코드 인터페이스 바인딩 요약

### 1. OpticalFlowTracker => TrackerFrameBuilder
* 데이터 규약: FollowerResult (bbox, score, is_recovery_event)
* 연결 지점: bridge/frame_builder.py 57줄 result = self._tracker.update(frame_bgr)

### 2. TrackerFrameBuilder => AimingManager
* 데이터 규약: aiming_engine.types.TrackerFrame
* 연결 지점: examples/run_bridge_pipeline_demo.py 72줄 tracker_frame = frame_builder.build(frame, timestamp)

### 3. AimingManager => HUD / 운용자
* 데이터 규약: aiming_engine.types.AimAssistOutput
* 연결 지점: examples/run_bridge_pipeline_demo.py 77줄 output = manager.update(tracker_frame)

## 4. 검증 결론

코드 분석 결과, detector+tracker, bridge, aiming_engine 모듈은 역할이 명확히 분리되어 있으며, DTO 객체(FollowerResult, TrackerFrame, AimAssistOutput)를 매개체로 하여 완벽하고 매끄럽게 end to end로 연결되어 동작함을 확인하였습니다.
