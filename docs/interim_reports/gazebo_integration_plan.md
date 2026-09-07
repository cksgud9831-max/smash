# Gazebo 시뮬레이션 기반 스마트 조준경 통합 검증 계획서

## 1. 개요 및 연구 배경

본 보고서는 스마트 조준경(Smart Optical Scope) 시스템의 최종 검증을 위한 우분투 Gazebo 시뮬레이션 환경 구축 및 통합 테스트 계획을 다룹니다.

현재까지 진행된 주요 연구 성과:
* Anti UAV 데이터셋 기반 YOLO11s GA 탐지 모델 학습 완료
* Jetson Orin Nano 온디바이스 타겟에서 Detector 및 Tracker 알고리즘 실시간 동작 확인
* Aiming Engine과 Detector, Tracker 사이의 하드웨어 독립적 접착 계층(Bridge) 완성
* 파이썬 기반 몬테카를로 시뮬레이션을 통한 Aiming Engine 수학적 조준해 검증 완료

## 2. Gazebo 시뮬레이션 환경 구축 목적

몬테카를로 시뮬레이션은 수학적 탄도학 계산 및 표적 궤적 추정 알고리즘의 이론적 타당성을 검증하였으나, 실제 환경에서의 다음 요소들을 모사하는 데 한계가 있습니다:
* 카메라 렌즈의 영상 획득 및 픽셀 시선 변환 과정에서의 렌즈 왜곡 및 모션 블러
* 표적 기동에 따른 카메라 화상 내 픽셀 크기 변화 및 탐지, 추적 신뢰도 변동
* 레이저 거리계와 IMU 센서의 측정 주기 불일치 및 통신 지연
* 조준경 마운트 짐벌의 진동 및 이동 환경에서의 자세 측정 오차

따라서 우분투 환경의 Gazebo를 통해 가상의 3D 세계를 구성하고 표적 UAV, 카메라, 레이저 거리계, IMU 센서를 동적으로 운용함으로써 전체 파이프라인의 End to End 정합성을 검증합니다.

## 3. 검증 환경 아키텍처 및 데이터 흐름

Gazebo 환경과 스마트 조준경 파이프라인 간 데이터 연동 구조는 다음과 같습니다:

1. Gazebo 3D World (우분투)
   * 표적 UAV 동적 비행 경로 생성 (정지 비행, 직선 등속 비행, 급선회 비행)
   * 센서 모듈 모사: RGB 카메라 센서, Laser Range Finder 센서, IMU 자세 센서

2. ROS2 또는 Direct Socket/UDP Interface Bridge
   * Gazebo 센서 데이터를 수신하여 조준경 표준 프로토콜로 변환
   * Camera Frame (BGR Image) => Bridge Detector / TRTDetector / OnnxDetector
   * Range Data => Bridge LaserAligner / MockRangeSensor 대체
   * Orientation Data => Bridge PoseSource

3. 스마트 조준경 파이프라인 (Bridge + Aiming Engine)
   * TrackerFrameBuilder를 통한 센서 융합 및 TrackerFrame 생성
   * SmartTracker (CA KF 기반) 표적 위치 및 가속도 추적
   * AimingManager 조준해 풀이 (Newton Solver 및 RK4 궤적 적분)
   * AimReadinessLogic 및 HitProbability를 통한 조준 상태(READY) 판정

4. 평가 및 시각화 (Validation & Visualization)
   * Ground Truth 표적 위치 대비 조준 예측 위치 비교 오차 산출
   * 수렴 시간, 지연 시간, 표적 추적 유지율 측정
   * intermediate_results 및 visualizations 폴더 내 결과 자동 저장

## 4. 검증 시나리오 및 주요 평가 지표

시나리오 1: 등속 직선 비행 표적
* 거리 100m ~ 500m 구간에서 일정 속도로 이동하는 표적 조준
* 평가 지표: 조준 오차 수렴 속도 및 뉴턴 솔버 반복 횟수

시나리오 2: 급선회 기동 표적
* 선회 가속도가 발생하는 표적 조준
* 평가 지표: CA KF 가속도 추정 정확도 및 AimReady 상태 유지 비율

시나리오 3: 센서 노이즈 및 카메라 흔들림 환경
* IMU 측정 오차 및 카메라 짐벌 진동 부가
* 평가 지표: LaserAligner 보정 효과성 및 추적 신뢰도 저하 시 가중치 반응

## 5. 파이프라인 연동 스크립트 구성 방안

주요 실행 스크립트는 scripts 폴더 내 두 자리 숫자 접두사를 부여하여 체계적으로 관리합니다:
* scripts/01_gazebo_bridge_node.py: Gazebo 센서 토픽 수신 및 파이프라인 입력 생성
* scripts/02_e2e_simulation_runner.py: 전체 시스템 통합 실행 및 데이터 로깅
* scripts/03_evaluate_simulation_results.py: Ground Truth 대비 성능 분석 및 시각화

## 6. 결론 및 향후 추진 과제

Gazebo 시뮬레이션을 완료한 후 결과를 객관적으로 분석하여 파라미터를 미세 조율할 예정입니다. 검증된 데이터는 intermediate_results 폴더에 저장하며 시각화 자료는 visualizations 폴더에 보관하여 과학적 타당성을 확보하겠습니다.
