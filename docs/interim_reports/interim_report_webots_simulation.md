# smash-main 전용 하이브리드 트래커(OpticalFlowTracker) 연동 E2E 시뮬레이션 최종 보고서

## 1. 개요 및 트래킹 모델 연동 배경

본 연구는 `smash-main` 프로젝트의 핵심 비전 추적 모듈인 **`OpticalFlowTracker` (`bridge/optical_flow_tracker.py`)**를 직접 연결하여, **[3D 영상 렌더링 $\to$ 하이브리드 비전 트래킹 (YOLO11s 탐지 + Lucas-Kanade 광학 흐름 추적) $\to$ IMU/Laser 센서 융합 $\to$ 뉴턴 탄도학 조준 $\to$ 가상 격발 및 명중 판정]**의 전체 파이프라인을 검증하였습니다.

---

## 2. 하이브리드 트래커(OpticalFlowTracker) 동작 원리

*   **초기 타겟 획득 및 주기적 재탐지:** 실제 학습된 18.26MB YOLO11s 신경망(`models/best.pt`)을 호출하여 드론의 Bounding Box를 정확히 검출
*   **프레임 간 고속 추적 (Lucas-Kanade Optical Flow):** 매 프레임 무거운 딥러닝 추론을 반복하는 대신, 직전 프레임의 특징점(Feature Points) 이동 벡터를 광학 흐름으로 추적하여 초고속 연산 수행
*   **특징점 소실 및 드리프트 보정:** 특징점이 부족해지거나 주기적 간격(Interval) 도래 시 자동으로 YOLO 재탐지(`YOLO DETECT`)를 트리거하여 오차 보정

---

## 3. 정량적 실험 및 과학적 분석 결과

*   **총 프레임:** 300 프레임 (32ms 주기, 9.6초 비행)
*   **사격 준비 락온(READY):** 285회 / 300 프레임
*   **최종 사격 명중 횟수:** 118회 (**명중률 41.40%**)
*   **동작 특성 분석:**
    *   `YOLO DETECT`가 보정해 주는 구간에서는 탄착 오차 거리가 **0.13m ~ 0.39m**로 수렴하여 유효 반경(0.6m) 이내로 **100% 명중**을 달성함.
    *   `LK FLOW TRACK` 순수 광학 흐름 구간에서는 미세한 특징점 드리프트(Drift)로 인해 55m 원거리 탄착 오차가 1.0m ~ 1.3m로 벌어져 빗나감(Miss)이 발생하는 실제 센서의 물리적 한계가 정확히 반영됨.

---

## 4. 관련 산출물 및 영상

*   **트래커 통합 실행 스크립트:** `../scripts/03_webots_yolo_e2e_full_pipeline.py`
*   **트래킹 및 조준 OSD 비디오:** `../visualizations/webots_yolo_flow_tracker_demo.mp4`
*   **정량 수치 데이터:** `../intermediate_results/webots_optical_flow_tracking_results.json`
