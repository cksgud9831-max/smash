# SMASH 세션 작업 정리 (2026년 9월 4일)

이 문서는 이번 대화에서 진행한 작업 전체를 정리한 것이다. 대상 코드베이스는 D 드라이브
Aiming 폴더이며, 사용자가 승인한 로드맵(claude/smash_roadmap.md, Claude 프로젝트 문서로
저장됨) 순서대로 실제 코드 작업까지 진행했다.

## 1. 현재 상황 조사

컴퓨터에 연결해 D 드라이브 Aiming 폴더의 실제 코드를 직접 열어 확인했다. 검토한 문서와
코드는 known_issues.md, aiming_project_summary.md, level1_level2_validation_report.md,
sensor_integration_gap_analysis.md, run_pipeline.py, scripts/06_gazebo_interactive_shooter.py,
gazebo/worlds/aiming_test.sdf다. 이를 근거로 다섯 가지 핵심 문제를 확인했다.

1. scripts/06_gazebo_interactive_shooter.py는 이름과 달리 실제 Gazebo와 연동되어 있지
   않다. run_pipeline.py의 4단계 메뉴가 Gazebo GUI와 00_live_demo_drive.py를 함께 띄우지만,
   06번 스크립트의 드론 위치는 전부 파이썬 코드 안의 사인함수로 직접 계산되며 Gazebo 토픽을
   구독하는 코드는 없다.
2. 지금 드론 운동은 등속 직선 비행이 아니라 사거리를 고정한 채 y축과 z축만 사인파로 흔드는
   제자리 진동이다. 사용자가 정의한 2단계 목표(1~2 m/s 등속 직선 비행)와 다르다.
3. 탄환 비행 로직에 실제 결함이 있었다. bullet_start_point와 bullet_end_point가 항상 같은
   화면 중앙 좌표로 고정되어 있어, 탄환이 화면상에서 전혀 이동하지 않고 명중 판정도 발사
   시점의 조준 품질과 무관한 근사치였다.
4. known_issues.md 1.5절에 이미 진단되어 있는 문제로, aiming_engine의 TargetState가
   트래커가 계산한 속도 신호를 무시하고 원시 위치 이력을 다시 최소자승으로 미분해, 이중
   미분 노이즈가 리드각 오차로 이어질 위험이 있었다.
5. Level 1 오라클 검증에서 등속 궤적은 이미 각도오차 평균 0.010도 수준으로 검증되어
   있었지만, 이는 별도의 오프라인 러너 결과일 뿐 06번 인터랙티브 경로에서 검증된 적은
   없었다.

## 2. 로드맵 작성 및 승인

위 진단을 근거로 Phase A(선결 문제 해결), Phase B(2단계 등속 직선 비행 본구현),
Phase C(3단계 선회 기동 준비), Phase D(4~5단계 장기 과제)로 구성된 로드맵을 작성해
Claude 프로젝트 문서(claude/smash_roadmap.md)에 저장했다. 착수 순서를 물었고, 로드맵
순서대로 전부 진행하는 쪽으로 결정되었다.

## 3. 실제 코드 작업

### A1. 탄환 비행 로직 수정

scripts/06_gazebo_interactive_shooter.py의 격발 처리 블록에서 bullet_end_point를 화면
중앙 고정값 대신, 격발 시점에 뉴턴솔버가 계산해 둔 미래 조준점(aim_point_px)으로
바꿨다. 표적이 포착되지 않은 상태(허공 사격)에서는 기존처럼 화면 중앙을 종점으로 쓴다.
이제 탄환이 실제로 화면에서 이동하고, 명중 판정도 격발 시점의 조준 정확도와 연결된다.

### B1. 등속 직선 왕복 비행 드론 모션

새 모듈 scripts/drone_motion_models.py에 compute_drone_linear_position 함수를 추가했다.
사거리(x=35m)와 고도(z=3.5m)는 고정하고 y축만 속력 1.5 m/s, 왕복 반경 14m로 순수 등속
직선 왕복시킨다. 4만 프레임 시뮬레이션으로 방향이 바뀌는 두 끝점의 순간을 제외하면 속도
편차가 없음을 확인했다. scripts/06_gazebo_interactive_shooter.py는 이 함수를 임포트해서
쓰도록 교체했다.

### A2. 계획 수정. velocity_px 대신 EMA 속도 스무딩

원래 로드맵에 적었던 "velocity_px 반영" 방안은 실제 코드를 확인한 결과 그대로 적용할 수
없었다. bridge/frame_builder.py는 production 경로에서 velocity_px를 항상 (0.0, 0.0)
플레이스홀더로만 채우고 있고, 06번 인터랙티브 스크립트 쪽은 반대로 단일 프레임 차분으로
직접 계산해 넘기므로 오히려 기존 OLS 추정보다 노이즈가 크다. 두 경로 모두 "이미 스무딩된
외부 속도 신호"라는 원래 전제가 성립하지 않았다.

대신 aiming_engine/target_state.py의 ConstantVelocityEstimator에
velocity_smoothing_alpha(지수이동평균 계수)를 새로 추가했다. 기본값 1.0은 스무딩 없음,
기존 Level 1/Level 2 검증 수치와 완전히 동일한 동작이라 회귀가 없다.
aiming_engine/config.py와 config/aiming_engine.yaml에도 이 값을 노출했다. 단위 테스트로
alpha=1.0이 기존과 동일함과, alpha<1.0이 실제로 속도 변화에 지연을 만든다는 것을
확인했다.

### B2, B4. 정밀도 정량 검증 스크립트 작성 및 실행

새 스크립트 scripts/07_stage2_lead_accuracy_validation.py를 작성해 실제로 실행했다.
Level 1과 같은 오라클 방법론으로, TargetState만 단독으로 떼어내 알려진 등속 궤적(GT)과
예측을 비교했다. history_length(10/20/30, B4 재스윕 포함)와 velocity_smoothing_alpha
(1.0/0.9/0.7/0.5/0.3)를 조합한 15개 설정으로 45fps 40초 시뮬레이션을 돌렸다.

핵심 결과(정상 비행 구간 각도오차 평균)는 history_length를 10에서 20으로 늘리면
0.1544도에서 0.0983도로 약 36퍼센트 감소, 20에서 30으로 늘리면 0.0764도로 약 22퍼센트
추가 감소했다. 같은 history_length 안에서 velocity_smoothing_alpha를 1.0에서 0.3으로
낮추는 효과는 이보다 훨씬 작았다(history_length=10 기준 약 10퍼센트 감소).

솔직한 해석을 보고서에 남겼다. 이번에 주입한 노이즈는 프레임마다 독립인 가우시안
노이즈라 표본을 늘리는 효과를 가진 history_length가 원래 유리한 조건이었다.
known_issues.md 1.5절이 실제로 우려한 노이즈는 이미 한 번 스무딩되어 프레임 간에
서로 연관된 잔여 노이즈였는데, 이런 노이즈에서는 velocity_smoothing_alpha가 상대적으로
더 유리할 가능성이 있다. 그래서 config/aiming_engine.yaml의 기본값(history_length=20,
velocity_smoothing_alpha=1.0)은 이번 결과만으로 바꾸지 않았다. 전체 수치는
stage2_lead_accuracy_results.json에, 해석과 권장사항은 stage2_lead_accuracy_report.md에
저장했다.

### A3. Gazebo 실연동 보류 결정

06번 스크립트를 자체 완결형으로 유지하기로 정했다. 실제 센서 신호가 bridge 계층을
문제없이 통과하는지는 이미 Level 2 검증(gazebo/02_e2e_simulation_runner.py 경로)에서
별도로 확인되어 있고, 지금 gz 토픽 실시간 구독을 새로 붙이는 것은 검증 없이 반영하기엔
부담이 큰 아키텍처 변경이라 판단했다. 로드맵 2~3단계의 목적은 06번 스크립트가 지금 형태로도
충분히 달성할 수 있다. 실시간 Gazebo 동기화는 Level 3(실 하드웨어 검증) 준비 단계에서
다시 검토하는 쪽을 권장했으며, 사용자가 원하면 언제든 이 결정을 뒤집을 수 있다.

## 4. 검증 및 반영

수정한 모든 파이썬 파일에 문법 검사(py_compile, ast.parse)를 돌렸다. TargetState의 새
동작(alpha=1.0이 기존과 동일, alpha<1.0이 실제로 지연을 만드는지)에 대해 단위 테스트를
작성해 통과를 확인했고, drone_motion_models.py의 등속 성질도 4만 프레임 시뮬레이션으로
검증했다. 이후 변경된 파일 여덟 개를 D 드라이브 Aiming 폴더의 원래 경로에 정상적으로
반영했다(모두 rejected 없이 기록됨).

## 5. 변경 및 생성된 파일 목록

| 파일 | 종류 | 내용 |
|---|---|---|
| scripts/06_gazebo_interactive_shooter.py | 수정 | A1(탄환 로직), B1(모션 임포트) 반영 |
| scripts/drone_motion_models.py | 신규 | B1 등속 직선 왕복 비행 모델 |
| scripts/07_stage2_lead_accuracy_validation.py | 신규 | B2/B4 정밀도 검증 스크립트 |
| aiming_engine/target_state.py | 수정 | A2 velocity_smoothing_alpha 추가 |
| aiming_engine/config.py | 수정 | A2 관련 설정 필드 추가 |
| config/aiming_engine.yaml | 수정 | A2 관련 설정 노출 |
| stage2_lead_accuracy_results.json | 신규 | B2/B4 실행 결과 원본 수치 |
| stage2_lead_accuracy_report.md | 신규 | B2/B4 결과 해석 및 권장사항 |

## 6. 결과물 위치

코드 변경은 컴퓨터의 D:\Aiming 폴더(scripts, aiming_engine, config 하위)에 반영되어
있다. 로드맵과 진행 현황 전체 기록은 Claude 프로젝트 문서 claude/smash_roadmap.md에,
이번 검증의 상세 결과는 D:\Aiming\stage2_lead_accuracy_report.md와
stage2_lead_accuracy_results.json에 저장되어 있다.

## 7. 다음 단계 (남은 일)

1. 06번 인터랙티브 시뮬레이터를 통과한 실제 트래킹 잔여노이즈로 velocity_smoothing_alpha
   스윕 재실행.
2. 3단계(선회 기동) 착수. 원호 또는 나선 모션 모델 추가와, 이미 구현되어 있지만 현재
   미사용 상태인 ConstantAccelerationKalmanEstimator로 전환 검토.
3. Phase D(고속 조건 재검증, 100회 벤치마크 재활용)는 Phase C 이후 진행.
