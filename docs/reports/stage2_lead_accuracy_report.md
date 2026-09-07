# 로드맵 2단계 리드각 정밀도 검증 보고서 (등속 직선 비행)

작성일 2026년 9월 4일. `scripts/07_stage2_lead_accuracy_validation.py` 실행 결과.

## 검증 방법

Level 1 오라클 검증(`aim_oracle_validation.html`)과 같은 방법론이다. 실제 3D 궤적을 직접
정의해 정답을 아는 상태에서(`drone_motion_models.compute_drone_linear_position`, 사거리
35m 고정, y축 왕복 등속 1.5 m/s, 왕복 반경 14m), 카메라와 YOLO와 OpticalFlowTracker 전체
비전 파이프라인은 거치지 않고 `aiming_engine.TargetState`(`ConstantVelocityEstimator`)만
단독으로 떼어내 검증했다. 이렇게 하면 비전 파이프라인 자체의 노이즈와, known_issues.md
1.5절이 지적한 이중 미분 구조가 만드는 노이즈를 분리해서 볼 수 있다.

측정한 두 파라미터는 `target_state.history_length`(기존 10/20/30 재스윕, 로드맵 B4)와
새로 추가한 `target_state.velocity_smoothing_alpha`(로드맵 A2, 1.0은 기존과 동일한
무보정 동작)다. 위치 측정 노이즈는 실측치가 아니라 Level 1 검증과 같은 가정치다. 06번
인터랙티브 시뮬레이터의 카메라 초점거리와 35m 사거리로 역산해 횡방향(y, z) 위치오차
시그마를 0.1m로, TF02 Pro 레이저 데이터시트 스펙에 맞춰 사거리 방향(x) 시그마를 0.05m로
두었다. 45fps로 40초간(등속 왕복 주기 약 18.7초를 두 번 이상 포함) 시뮬레이션했다.

## 핵심 결과

정상 비행 구간(방향 반전 지점에서 0.5초 이상 떨어진 구간) 기준, 각도오차 평균값은 다음과
같다.

| history_length | alpha 1.0 | alpha 0.9 | alpha 0.7 | alpha 0.5 | alpha 0.3 |
|---|---:|---:|---:|---:|---:|
| 10 | 0.1544 deg | 0.1535 deg | 0.1509 deg | 0.1466 deg | 0.1389 deg |
| 20 (기존 기본값) | 0.0983 deg | 0.0982 deg | 0.0977 deg | 0.0969 deg | 0.0951 deg |
| 30 | 0.0764 deg | 0.0764 deg | 0.0762 deg | 0.0760 deg | 0.0754 deg |

전체 수치(위치오차 포함, 선회 근접 구간 별도 집계)는 `stage2_lead_accuracy_results.json`
참고.

## 해석

두 파라미터 모두 오차를 줄이는 방향으로 일관되게 작동했지만 효과 크기는 크게 다르다.
history_length를 10에서 20으로 늘리면 각도오차 평균이 0.1544도에서 0.0983도로 약 36%
줄었고, 20에서 30으로 늘리면 다시 0.0764도로 약 22% 더 줄었다. 반면 velocity_smoothing_alpha를
같은 history_length 안에서 1.0에서 0.3까지 낮췄을 때의 개선폭은 훨씬 작다(history_length=10
기준 0.1544도에서 0.1389도로 약 10% 감소).

이 결과를 velocity_smoothing_alpha가 별 효과가 없다는 뜻으로 해석하면 안 된다. 이번
검증에서 주입한 노이즈는 프레임마다 독립인(i.i.d.) 가우시안 노이즈다. 이런 노이즈에는
표본 수를 늘리는 효과를 가진 history_length 확장이 원래 유리하다. known_issues.md 1.5절이
실제로 우려한 문제는 이미 한 번 스무딩되어 프레임 간에 서로 연관된(자기상관이 있는) 잔여
노이즈였는데, 이런 종류의 노이즈에서는 단순히 창을 늘리는 것보다 velocity_smoothing_alpha
같은 지수적 감쇠가 상대적으로 더 유리할 가능성이 있다. 이번 검증은 그 실제 상관관계 있는
노이즈까지는 재현하지 않았으므로, velocity_smoothing_alpha의 진짜 효과를 확인하려면 실제
OpticalFlowTracker와 06번 인터랙티브 시뮬레이터를 통과한 진짜 트래킹 잔여노이즈로 같은
스윕을 다시 해봐야 한다.

## 결론 및 권장 설정

이번 결과만 놓고 보면 history_length를 20에서 30으로 올리는 쪽이 velocity_smoothing_alpha를
낮추는 쪽보다 이득이 크다. 다만 known_issues.md 1.5절은 history_length를 더 키우면 급기동
구간에서 반응 지연이 커지는 트레이드오프를 이미 경고했다(history_length=40이면 창이 2초로
늘어나 반응이 느려짐). 3단계(선회 기동)를 앞두고 있으므로, 지금 당장 기본값을 30으로
올리기보다는 velocity_smoothing_alpha를 우선 실제 트래킹 노이즈로 재검증한 뒤, 두 파라미터를
함께 놓고 급기동 반응성과 정상비행 정밀도 사이의 트레이드오프를 다시 판단하는 것을 권장한다.
config/aiming_engine.yaml의 기본값(history_length=20, velocity_smoothing_alpha=1.0)은
이번 검증으로 바꾸지 않았다.

## 남은 일 (후속 검증)

1. 06번 인터랙티브 시뮬레이터를 통과한 실제 트래킹 잔여노이즈로 같은 alpha 스윕 재실행.
2. 3단계(선회 기동) 착수 시 history_length와 velocity_smoothing_alpha를 급기동 반응성
   지표와 함께 재평가.
3. Level 3(실 하드웨어) 연결 후에는 이번 검증의 가정 노이즈(횡방향 0.1m, 사거리방향 0.05m)를
   실측 노이즈로 교체해 재실행.
