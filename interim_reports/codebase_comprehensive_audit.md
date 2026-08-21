# 코드베이스 종합 정밀 진단 및 개선 완료 보고서

## 1. 개요

본 보고서는 AI 스마트 조준경(Smart Optical Scope) 시스템 전체 코드베이스(Aiming Engine, Bridge 계층, OpticalFlowTracker, 설정 파일 및 테스트 모듈)를 처음부터 끝까지 정밀하게 검토하고, 코드 내 수학적/논리적 결함 및 정합성 이슈를 찾아내어 직접 수정 완료한 내용과 추가 개선 필요 사항을 정리한 문서입니다.

## 2. 발견된 문제점 및 즉시 수정 완료 항목

정밀 검토 과정에서 도출된 핵심 문제점을 보완하고, 코드 수정을 적용한 후 108개 전체 단위 테스트(pytest)를 100% 통과시켰습니다.

### 수정 1. TargetState의 위치 스무딩 수식 결함 보완 (aiming_engine/target_state.py)
* 현상: 기존 ConstantVelocityEstimator는 최소자승법(OLS) 선형 회귀를 수행하여 속도(coeffs[0])와 t=last_t 시점의 회귀 절편(coeffs[1])을 산출함에도 불구하고, 미래 위치 계산 시 회귀 절편 대신 노이즈가 포함된 가장 최근의 원시 측정 위치(last_p)를 기준점으로 사용하여 위치를 예측함.
* 영향: 카메라 픽셀 떨림이나 레이저 거리 오차가 발생하는 상황에서 원시 측정 노이즈가 미래 예측 좌표 전체에 그대로 덧셈 오차로 전파되어 HUD 조준점 떨림을 가중시킴.
* 조치: coeffs[1] (OLS 회귀로 노이즈가 제거된 위치)을 기준 위치로 설정하도록 수정하여 OLS 선형 회귀의 위치 스무딩 성능을 완전히 활용함.

### 수정 2. AimSolver의 Warm Start 수치 발산 및 오염 방지 (aiming_engine/aim_solver.py)
* 현상: NewtonSolver가 수렴하지 못한 경우에도 수렴 실패한 x 해를 그대로 이전 해(previous_solution)로 보관함.
* 영향: 급기동이나 초기 프레임 이상치 발생 시 이전 해가 NaN, Inf 또는 과도하게 큰 잔차 값으로 오염되어 이후 프레임에서도 연속적으로 뉴턴 솔버가 수렴 실패에 빠지는 위험이 존재함.
* 조치: 뉴턴 솔버 결과가 수렴하였거나 수치적으로 유효한 범위(잔차 50m 이하 및 정상 수치)인 경우에만 이전 해를 보관하고, 그렇지 않은 경우 warm start를 즉시 리셋(None)하여 안전한 naive_guess로 폴백하도록 수정함.

### 수정 3. NewtonSolver 수치적 업데이트 경계 조건 강화 (aiming_engine/newton_solver.py)
* 현상: 뉴턴 랩슨 반복 과정에서 delta 스텝 업데이트 시 비행시간(x[0])이 음수로 넘어갈 수 있는 가능성과 수치적 NaN, Inf에 대한 방어 로직 부족.
* 조치: 비행시간 x[0]가 물리적으로 0 초과(최소 0.0001초)를 항상 유지하도록 클램핑을 적용하고, 유효하지 않은 수치 발생 시 조기 이탈하도록 안정성을 강화함.

### 수정 4. 3D Constant Acceleration 칼만 필터 (CA KF) 추정기 추가 구현 (aiming_engine/target_state.py)
* 현상: 기존 TargetState는 3D 월드 관성계에서 등속 운동(Constant Velocity) 기반 최소자승 선형 피팅만 존재하여, 드론 표적이 급선회하거나 3D 가속도가 발생하는 기동 시 표적 위치 외삽 지연 오차가 유발되었음.
* 조치: 추적기(Tracker) 코드를 일절 변경하지 않은 채, 3D 월드 관성계 좌표계 상에서 9차원 상태 변수 (3D 위치, 3D 속도, 3D 가속도)를 추적하는 3D ConstantAccelerationKalmanEstimator (CA KF) 클래스를 구현함. config/aiming_engine.yaml 설정에서 estimator: constant_acceleration 지정을 지원하고 관련 하이퍼파라미터를 파싱하도록 확장함.

## 3. 추가 점검 및 향후 개선 추천 항목

### 1. OpticalFlowTracker 신뢰도(tracking_confidence) 민감도 보완
* 현상: 현재 OpticalFlowTracker는 YOLO 재탐지 성공 시 점수를 고정 유지하는 방식을 사용함.
* 개선안: 옵티컬플로우 특징점(Keypoints)의 가용 비율(Tracked keypoints / Initial keypoints) 및 궤적 분산을 신뢰도 산출 수식에 반영하여, 표적이 왜곡되거나 부분 가림 발생 시 추적 불안정 상태를 빠르게 감지하도록 보완 추천.

### 2. Gazebo 3D 시뮬레이션 연동 노드 구축 (진행 중인 작업)
* 현상: 현재 파이프라인 통합 및 우분투 Gazebo 시뮬레이터 검증 준비 단계임.
* 조치 계획: scripts 폴더 내에 01_gazebo_bridge_node.py 및 02_e2e_simulation_runner.py 스크립트를 작성하여 Gazebo 센서 토픽 수신 및 Ground Truth 정량 평가 파이프라인을 완성할 예정.

## 4. 검증 결과

수정 및 3D CA KF 모듈 추가 반영 후 전체 pytest 단위 테스트 109개를 수행한 결과, 109개 테스트가 모두 오류 없이 정상 통과하였습니다 (109 passed in 1.83s).
