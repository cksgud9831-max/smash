# 현재 코드베이스 문제점 정리

작성일: 2026-08-18. `bridge/` 하드웨어 센서 연동(TF02-Pro 레이저, BNO08x IMU) 코드를 중심으로 검토한 결과.
각 항목은 상태(`[ ]` 미해결 / `[x]` 해결)로 추적한다.

## 0. 트래커 교체: SmartTracker → OpticalFlowTracker (2026-08-18)

`bridge/smart_tracking/`(Kalman필터+Hungarian 매칭 기반)이 삭제되고, 대신 `detector+tracker/code/
yolo_follower_v3_reactivate.py`에서 이미 검증·GA 튜닝된 YOLO 초기/주기 탐지 + Lucas-Kanade 옵티컬플로우
팔로워 알고리즘을 `bridge/optical_flow_tracker.py`(`OpticalFlowTracker`)로 포팅해 프로덕션 트래커로
교체했다. 아래 **1.4, 1.5 절은 전부 SmartTracker 시절 이슈이며 지금은 해당 없음(historical)** — 남겨두는
이유는 "왜 예전엔 이런 문제가 있었는지"와 "새 트래커로 바뀌면서 무엇이 좋아졌는지"의 기록 때문이다.

**바뀐 것**: `bridge/frame_builder.py`가 더 이상 외부 `Detector.detect()`를 매 프레임 호출하지 않는다
(`OpticalFlowTracker`가 필요할 때만—초기화/30프레임마다/실패 복구 시—내부적으로 자체 YOLO 호출을
수행하므로, 매 프레임 외부에서 탐지를 강제하면 이 효율성 이점이 사라진다). `TrackerFrameBuilder.build()`
시그니처가 `build(detections, frame, timestamp)` → `build(frame, timestamp)`로 변경됐다.
`bridge/confidence.py`의 `compute_tracking_confidence`/`compute_follower_state`도 새 트래커의 신호
(마지막 YOLO 점수, "이번 프레임에 non-periodic 재탐지가 발동했는가")에 맞게 재작성했다. `bridge/detector.py`
(ONNX/TRT)와 `config/bridge.yaml`의 `detector:` 섹션은 삭제하지 않고 남겨뒀지만 현재 기본 파이프라인에서는
쓰이지 않는다(`onnx.weights_path`가 가리키는 `bridge/weights/`도 이번에 함께 삭제됨).

**검증 결과** (`test/visible.mp4`, 1000프레임, `config/bridge.test_video.yaml`):

| 지표 | SmartTracker (이전) | OpticalFlowTracker (현재) |
|---|---|---|
| GT 대비 평균 IoU | 0.761 | 0.760 (동일) |
| 트래킹 끊김/재획득 | track ID 7회 전환 | **0회**(한 번도 안 놓침) |
| "표적 없음" 프레임 | 종종 발생 | **0/1000** |
| YOLO 호출 비율 | 매 프레임(100%) | **3.4%**(34/999) |
| 상태분포(READY) | 702/948 | **992/1000** |
| aim_ready=True | 489프레임 | **992프레임** |
| hit_probability | 0.70~0.88 | **0.81~0.92** |
| 처리 속도 | — | 48.8 fps (GPU) |
| 조준점 프레임간 흔들림(1.5절 지표, mean) | 6.46px(사전) / history_length=20 조정 후 개선 | **2.80px** (튜닝 없이) |

탐지 정확도는 사실상 동일하지만, 트래킹 연속성·조준 안정성·효율성 모두 크게 개선됐다. 이는 옵티컬플로우가
프레임 간 변화량 자체를 직접 추종하고(YOLO는 30프레임마다 부드럽게 보정만) GA로 이 정확한 영상에 맞춰
튜닝된 결과로 보인다 — 다른 영상/실장비에서도 이 정도로 잘 맞을지는 별도 검증이 필요하다(아래 "남은 리스크"
참고).

**남은 리스크**:
- `OpticalFlowTracker`의 GA 튜닝 상수(`ROI_MARGIN`, `CENTER_SHIFT_THRES` 등)는 `test/visible.mp4` 한
  영상에 대해 GA 탐색된 값이다. 다른 해상도/표적 크기/프레임레이트의 실제 영상에서도 잘 맞을지는
  검증되지 않았다 — SmartTracker의 `uncertainty_threshold`가 해상도별로 재튜닝이 필요했던 것과 같은 종류의
  리스크가 잠재되어 있다.
- `tracking_confidence`가 이제 "마지막 YOLO 점수를 그대로 유지, 복구 이벤트 시에만 0"이라는 2단계에
  가까운 값이라, SmartTracker의 연속적인 불확실성 기반 값보다 훨씬 덜 민감하다. 실패를 더 빨리/세밀하게
  감지해야 하는 상황(예: 표적이 서서히 흐려지는 경우)을 놓칠 수 있다.
- `bridge/detector.py`(ONNX/TRT)가 죽은 코드가 됐다 — 완전히 삭제할지, 다른 백엔드 전환 옵션으로 남겨둘지
  결정이 필요하다.
- `ultralytics.YOLO(.pt)`를 그대로 쓰기로 했으므로(사용자 확인 완료), Jetson 배포 시 torch+ultralytics
  설치가 필요하다 — `bridge/trt_infer.py`가 원래 피하려던 무거운 의존성이 다시 들어온 것이므로, 실제
  Jetson 배포 전에 속도/용량 검증이 필요하다.

## 1. 실제 코드 결함 / 리스크

### 1.1 `[x]` `Tf02ProRangeSensor`/`Bno08xPoseSource`가 실행 경로에서 절대 `close()` 되지 않음 — 수정 완료

**수정 내용**: `RangeSensor`/`PoseSource` 베이스 클래스에 기본 no-op `close()`를 추가해 `MockRangeSensor`/
`MockPoseSource`도 동일하게 호출 가능하게 했고, `run_bridge_pipeline_demo.py`를 `try/finally`로 감싸
정상 종료·예외·`cap.isOpened()` 실패 등 모든 경로에서 `range_sensor.close()`/`pose_source.close()`가
호출되도록 했다.

`examples/run_bridge_pipeline_demo.py`는 `build_range_sensor`/`build_pose_source`로 두 센서를 생성만 하고,
루프 종료 시(`cap.release()` 이후) `close()`를 호출하지 않는다. `close()`는 `tests/test_range_sensor_tf02pro.py`,
`tests/test_pose_source_bno08x.py`에서만 쓰인다.

- 백그라운드 폴링 스레드가 daemon=True라 프로세스 종료 시 같이 죽긴 하지만, 시리얼 포트/I2C 핸들이
  정상적으로 닫히지 않는다.
- 루프 중 예외(예: `cap.read()` 실패, 사용자 Ctrl+C)가 나면 포트가 열린 채로 남는다.
- `MockRangeSensor`/`MockPoseSource`는 애초에 `close()` 메서드가 없어서, 데모 코드가 무조건 `close()`를
  호출하도록 고치려면 베이스 클래스에 기본(no-op) `close()`를 추가해야 한다.

**영향**: 데스크톱 데모 단발 실행에서는 눈에 띄는 문제가 없지만, 실제 Jetson 배포에서 반복 실행/재시작
시나리오나 시리얼 포트를 다른 프로세스와 공유하는 상황에서 포트 점유가 풀리지 않을 수 있다.

### 1.2 `[ ]` 비디오 재생 시 `timestamp`와 센서 staleness 판정 기준이 서로 다른 시계를 씀

`run_bridge_pipeline_demo.py`의 `timestamp = frame_idx / args.fps`는 영상 내 가상 시간이다. 반면
`Tf02ProRangeSensor.read()`/`Bno08xPoseSource.read()`는 인자로 받은 `timestamp`를 완전히 무시하고
내부적으로 `time.time()`(실제 벽시계)로만 staleness를 판정한다
(`bridge/range_sensor.py:169`, `bridge/pose_source.py:207`).

녹화 영상을 재생하면서 실장비 센서를 동시에 붙이는 벤치 테스트(스코프 정렬 검증 등)에서
`TrackerFrame.timestamp`는 영상 시간인데 `laser_range_m`/`camera_extrinsics`는 실제 "지금" 값이라 서로
대응이 안 맞는 조합이 나올 수 있다. `read(timestamp)` 시그니처가 실질적으로 죽은 파라미터라, 나중에
"센서값이 카메라 프레임과 안 맞는다" 류의 버그를 디버깅하기 어렵게 만든다.

**영향**: 라이브 웹캠 사용 시엔 `timestamp ≈ time.time()`이라 드러나지 않는다. 녹화 영상 + 실장비 조합
테스트에서만 문제가 된다. 최소한 문서화가 필요.

### 1.3 `[ ]` `Tf02ProRangeSensor._poll_loop`의 리싱크 경로 비효율

헤더 불일치 시 `del buffer[0:1]`로 한 바이트씩 지우며 루프를 도는데, 노이즈가 길게 낀 구간에서는 매
바이트마다 `bytearray` 앞부분을 통째로 시프트(O(n))하는 게 반복된다. 실사용 트래픽량(9바이트 프레임,
최대 1000Hz)에서는 무시할 수준이라 버그는 아니고, 우선순위 낮음.

### 1.4 `[historical: SmartTracker 시절 이슈, 0절 참고]` `tracker.uncertainty_threshold: 3.0`이 실영상에서 `tracking_confidence`를 영구적으로 0으로 고정 — 원인 진단 완료, 테스트용 설정으로 우회

`test/visible.mp4`(1920x1080, 20fps, Anti-UAV 포맷 GT 포함)로 전체 파이프라인을 처음 돌렸을 때
`AimStateMachine`이 1000프레임 내내 `SEARCH`에서 한 번도 벗어나지 못하고 `aim_ready`도 항상 `False`였다.
Detector+Tracker 자체 품질은 좋았는데(1.4.1 참고) 왜 상태가 안 올라가는지 추적한 결과:

- `bridge/smart_tracking/improvement_tracker.py:248`의 `crisis_mode = max_uncertainty > self.uncertainty_threshold`가
  이 영상에서는 `threshold=3.0`일 때 **960/960프레임 전부 `True`**로 고정됨(스윕 결과: 3.0/6.0→100%,
  10.0→24.7%, 15.0 이상→0%). CA-KF 위치 공분산이 표적 bbox 높이(이 영상 기준 73~109px)에 비례해서 커지는
  구조라, 실측 uncertainty가 정상적으로도 ~5~6px 수준으로 유지되는데 `bridge.yaml`의 `3.0`은 그보다 훨씬
  타이트하다.
- `bridge/confidence.py:compute_tracking_confidence`는 `uncertainty > threshold`면
  `stability_factor = clip(1 - uncertainty/threshold, 0, 1)`가 **정확히 0**이 되므로, `crisis_mode`가 항상
  `True`인 상황에서는 `track.score`가 아무리 좋아도(실측 0.68~0.78) `tracking_confidence`가 **항상 정확히
  0.000**으로 나왔다.
- `aiming_engine/aim_state_machine.py`의 SEARCH→TRACK 전이는 `tracking_confidence >=
  aim_state_machine.tracking_confidence_threshold(0.6)`가 3프레임 연속 필요한데, 0.000은 절대 이 문턱을
  넘을 수 없어 상태기계가 구조적으로 SEARCH에 갇혔다.

**이건 `aiming_engine`/`bridge` 코드 버그가 아니다** — `compute_tracking_confidence`, `crisis_mode`,
`AimStateMachine` 모두 설계된 그대로 동작하고 있다. 문제는 `bridge.yaml`의 `tracker.uncertainty_threshold: 3.0`
(주석에 "jetson_team_eval_package에서 튜닝된 값"이라고 명시됨)이 **이 특정 영상의 해상도/표적 크기 스케일과
전혀 안 맞는다**는 것 — 실측 없이 다른 문맥에서 튜닝된 값을 그대로 가져다 쓰면 겉보기엔 정상 동작(에러도
경고도 없음)하면서 조용히 "항상 위기 상태"로 고정되는, 발견하기 아주 어려운 종류의 설정 오류다.

**우회 조치**: `config/bridge.test_video.yaml`에서만 `tracker.uncertainty_threshold`/
`confidence.uncertainty_threshold`를 `30.0`으로 재설정(3/6/10/15/20/27/30/40 스윕 후, `tracking_confidence`가
과반 프레임에서 0.6을 넘는 최소값 채택). `config/bridge.yaml`(실장비용)은 건드리지 않았다 — 이 값은 실제
카메라/거리에서 다시 스윕해야 의미가 있다.

**결과**: 재실행 시 `SEARCH 198 / TRACK 8 / AIM 40 / READY 702`(948 트래킹 프레임 중), `aim_ready=True` 489
프레임, `hit_probability` 0.70~0.88 — 파이프라인이 실제로는 정상 동작함을 확인.

**남는 일반화된 리스크**: `tracker.uncertainty_threshold` (crisis_mode 게이트) ↔
`aim_state_machine.tracking_confidence_threshold` (상태 전이 게이트) 두 값은 서로 다른 파일
(`bridge.yaml` vs `aiming_engine.yaml`)에 있지만 사실상 한 쌍으로 튜닝되어야 하는 값이다. 둘 중 하나만
바꾸면(예: 카메라 해상도 변경, 표적 크기 변화) 위와 같이 시스템이 "항상 SEARCH"로 조용히 고착될 수 있다.
이 상호의존성을 두 YAML 어디에도 명시적으로 문서화해두지 않았다.

### 1.5 `[historical: SmartTracker 시절 이슈, 0절 참고]` 표적이 움직일 때 조준점(HUD 리드포인트)이 심하게 흔들림 — 원인 진단 + 완화 완료

`test/visible_aim_overlay.mp4`를 육안으로 확인한 결과 표적이 움직이는 구간에서 조준점(초록/주황 십자)이
표적 대비 눈에 띄게 떨렸다. 정량적으로 재현/분해한 결과:

- `bridge/smart_tracking/improvement_tracker.py:STrack.to_tlwh()`가 반환하는 `center_px`는 이미
  SmartTracker 자체 CA-Kalman필터로 한 번 스무딩된 값인데도, 프레임 간 픽셀 변동이 상당했다(원본 bbox
  중심의 프레임당 이동량: 평균 15.67px, p95 60.65px, 최대 252px — 이 중 대부분은 실제 빠른 표적 이동이지만
  일부는 잔여 노이즈).
- `aiming_engine/target_state.py:ConstantVelocityEstimator`는 이 값을 받아 **가중치·이상치 제거·스무딩 전혀
  없는 단순 최소자승(OLS) 직선 피팅**을 최근 `history_length`(기존 10, 20fps 기준 0.5초 창) 샘플에 대해
  다시 수행해 속도를 추정한다. 즉 이미 한 번 필터링된 신호를 아무 감쇠 없이 **두 번째로 미분**하는 구조라,
  잔여 노이즈가 고스란히(또는 그 이상) 속도 추정치로 넘어간다.
- 이 속도가 `lead_point_world = position + velocity * time_of_flight`로 그대로 외삽되어 HUD 조준점에
  반영되므로, 원본 트래킹 신호의 작은 흔들림이 조준점에서 크게 보이는 흔들림으로 나타났다.
- `history_length`를 3/6/10/15/20/30/40으로 스윕하며 "조준점-원시위치 오프셋의 프레임간 변화량"(리드
  예측 단계 자체가 만들어내는 흔들림만 분리 측정)을 측정: **10→평균 6.46px(최대 60px) / 20→평균 2.78px /
  30→평균 1.35px / 40→평균 0.88px** — history_length를 늘릴수록 뚜렷하게 감소.

**적용한 조치**: `config/aiming_engine.yaml`의 `target_state.history_length`를 **10 → 20**(0.5초 → 1.0초
창)으로 상향. 조준점의 "원시 위치 대비 프레임간 이동량 증폭률"이 1.35배 → 1.10배로 줄어(즉 리드 예측
단계가 만들어내던 추가 흔들림이 거의 제거됨) `test/visible_aim_overlay.mp4`를 재생성해 확인했다.

**남은 큰 조준점 점프는 버그가 아니라 실제 빠른 표적 이동**: 조정 후에도 남아있는 가장 큰 점프들(예:
frame 884→887, frame 511→515)을 GT(`visible.json`)와 대조한 결과, 실제로 표적이 3~5프레임 사이에
150~250px씩 진짜로 이동하는 구간이었다. 이런 구간에서는 조준점이 따라 움직이는 게 정상이며, 스무딩
창을 더 늘리면 노이즈는 더 줄지만 대신 이런 실제 급기동 구간에서 반응 지연/오버슈트가 커진다
(`history_length=40`이면 1초→2초 창이 되어 반응이 상당히 느려짐) — **30~40까지 더 낮추는 게 항상
좋은 것은 아니며, 실제 표적 기동성에 맞춰 값을 골라야 하는 트레이드오프**다.

**더 근본적인 개선 여지 (구현하지 않음, 다음 단계 후보)**: `frame_builder.py:74`가 이미
`TrackerFrame.velocity_px = (track.mean[4], track.mean[5])`로 SmartTracker의 Kalman필터가 추정한
(스무딩된) bbox 속도를 채워 넣고 있는데, **`aiming_engine`은 이 필드를 어디서도 읽지 않는다**
(`grep -r velocity_px aiming_engine` → `types.py` 정의 한 곳뿐). 지금은 이미 계산된 속도 추정치를 버리고
`TargetState`가 완전히 새로 미분하는 구조라 이중 미분 노이즈가 생긴다. 장기적으로는 `velocity_px`(+
range-rate)를 `TargetState`의 초기 속도 추정에 반영하거나, `MotionEstimator`를 실제 칼만필터 기반으로
교체(이미 `target_state.py` docstring에 "future: constant_acceleration, kalman"으로 예비된 확장 지점)하는
쪽이 창 크기만 늘리는 것보다 지연 없이 노이즈를 줄이는 더 나은 해법이다.

## 2. 설계상 아직 비어 있는 부분 (실장비 투입 전 확정 필요)

- `config/bridge.yaml`의 `laser.mount_offset_m`, `pose.bno08x.mount_rotation_rad`가 전부 `(0,0,0)`
  플레이스홀더, `Bno08xPoseSource._IMU_TO_PLATFORM_AXES`도 단위행렬 플레이스홀더 — 실기 장착 후 실측
  없이 그대로 돌리면 `LaserAligner`/`quaternion_to_platform_rotation`이 조용히 틀린 값을 낼 뿐 에러가
  나지 않는다.
- `min_signal_strength: 100`(YAML)도 실측 전 "확인 안 됨" 플레이스홀더.
- `PoseSample`은 `Optional`이 아니라 항상 반환되는 시그니처인데, `Bno08xPoseSource`는 첫 샘플이 오기
  전엔 `valid=False`로 identity를 반환한다 — `frame_builder.py`가 이미 방어하고 있어 안전하지만, 이
  계약을 인터페이스 docstring에 명시해두면 다음 `PoseSource` 구현체를 만들 때 실수를 줄일 수 있다.

## 3. `test/visible.mp4` 실전 검증 (2026-08-18)

`test/visible.mp4`(1920x1080, 20fps, 1000프레임/약 50초, 표적이 화면 중앙 부근에서 좌우로 크게 이동)와
동봉된 GT(`test/visible.json`, Anti-UAV 포맷 `exist`/`gt_rect`, 1000프레임 전부 표적 존재)를 이용해 실제
카메라/레이저/IMU 없이도 검증 가능한 범위까지 파이프라인을 단계적으로 실행했다. 레이저/IMU는 요청대로 Mock
고정값(거리 100m→이후 30.0 스윕 결과 반영, roll/pitch/yaw=0)을 사용했다. 이 영상 전용 설정은
`config/bridge.test_video.yaml`에 분리(1920x1080에 맞춘 카메라 intrinsics 스케일링 포함, `config/bridge.yaml`
원본은 변경하지 않음).

### Test 1 — Detector + SmartTracker (GT 비교)

- 탐지된 프레임: 987/1000 (98.7%)
- confirmed track 있는 프레임: 960/1000 (96.0%)
- 사용된 track ID: 7개(끊김 후 재획득마다 새 ID) — **연속 트래킹 중 ID가 튄 적은 0회**
- GT 대비 평균 IoU: **0.761** (960프레임 중 99.6%가 IoU>0.5, 최소 0.462)

→ Detector/Tracker 자체는 이 영상에서 매우 잘 동작한다.

### Test 2/4 — 전체 파이프라인 (`run_bridge_pipeline_demo.py`, 수정 없이 그대로 사용)

초기 실행(원래 `bridge.yaml` 값 그대로, threshold=3.0)에서는 1.4에서 진단한 문제로 1000프레임 내내 SEARCH에
고착되어 `aim_ready`가 한 번도 True가 되지 않았다. `bridge.test_video.yaml`에서 threshold를 30.0으로
재조정한 뒤 재실행하면:

- 상태 분포(948 트래킹 프레임): `SEARCH 198 / TRACK 8 / AIM 40 / READY 702`
- `aim_ready = True`: 489프레임
- `hit_probability`: 0.70 ~ 0.88

→ 좌표변환 → TargetState → AimSolver(Newton) → HitProbability → AimReadinessLogic → AimStateMachine 전체
체인이 실영상 입력으로 끝까지 정상 동작하고, 상당 시간 "조준 가능(READY)" 상태에 도달함을 확인.

### Test 3 — IMU(roll/pitch/yaw) 민감도 (합성 데이터, 영상 불필요)

`MockPoseSource`는 현재 identity 고정값만 지원해 영상 재생 중 IMU 값을 실시간으로 바꿀 수 없으므로, 월드좌표에
고정된 가상 표적(200m 전방, 10m 좌측, 3m 상방)에 대해 플랫폼 자세(=IMU가 리포트할 값)만 바꿔가며
`AimingManager`를 직접 호출했다.

| 케이스 | azimuth | elevation |
|---|---|---|
| identity | -2.862° | 0.936° |
| yaw +5° | -7.862° (Δ-5.000°) | 0.936° |
| yaw +10° | -12.862° (Δ-10.000°) | 0.936° |
| yaw -10° | 7.138° (Δ+10.000°) | 0.936° |
| pitch +5° | -2.877° | 5.930° (Δ+4.994°) |
| pitch -5° | -2.869° | -4.058° (Δ-4.994°) |

모든 케이스에서 역변환한 표적 월드좌표는 정확히 `(200.00, -10.00, 3.00)`으로 동일하게 복원됐고(왕복
일관성 확인), azimuth/elevation은 입력한 yaw/pitch 변화량과 거의 정확히(오차 <0.01°) 선형으로 반응했다.

→ `Bno08xPoseSource`가 실제로 연결되면 그 쿼터니언 → `camera_extrinsics`가 `CoordinateTransform`/
`AimSolver`를 거쳐 HUD azimuth/elevation에 물리적으로 올바르게 반영될 것으로 확신할 수 있다.

## 4. 다음 단계

1. **1.1 수정 완료** — `RangeSensor`/`PoseSource` 베이스 클래스에 기본 `close()`(no-op) 추가, 데모 스크립트를
   `try/finally`로 감싸 항상 `close()` 호출.
2. **1.4 진단 완료** — 원인은 `bridge.yaml`의 `tracker.uncertainty_threshold` 스케일 불일치. 실장비 연결 후
   실측 카메라/표적 스케일로 `bridge.yaml`(운영용)을 별도로 재튜닝할 것. `bridge.test_video.yaml`의 `30.0`은
   이 영상 전용이며 실장비에 그대로 쓰면 안 됨.
3. 1.2는 문서화(또는 필요 시 `TrackerFrame`에 "센서 샘플 실제 캡처 시각" 필드 추가) 여부 결정.
4. 실기 장착 후 2절의 플레이스홀더 값들을 실측치로 교체.
5. `tracker.uncertainty_threshold` ↔ `aim_state_machine.tracking_confidence_threshold` 상호의존성을 두
   YAML 파일에 상호 참조 주석으로 명시(1.4 참고).

## 5. `DragForce` 추가 (2026-08-19) — 기본 비활성, 데이터 미검증

`aiming_engine/forces.py`에 `DragForce(ForceModel)`를 추가해 `projectile.forces`에 `drag`를 넣으면
공기저항(마하수 기반 Cd 곡선, G1/G7 형태)이 탄도 계산(RK4)에 반영되도록 확장했다. `config/aiming_engine.yaml`
기본값은 그대로 `forces: [gravity]`로 두었다 — 아래 사유로 지금 당장 켜면 안 된다.

- **`G1_DRAG_TABLE`/`G7_DRAG_TABLE`(`forces.py`)은 여전히 미검증이다 (2026-08-19 갱신, 신뢰도는 올라갔지만
  아직 확정 아님).** 처음 버전은 G1/G7 곡선의 일반적으로 알려진 *형태*(아음속 평탄 → 마하 1 부근 급상승 →
  초음속 완만한 감소)만 재현한 것이었다. 지금 버전은 표준 Ingalls/McCoy G1/G7 항력함수의 실제 발표 수치를
  기억에 의존해 재현하려 시도한 것으로 한 단계 나아졌지만, **작성 시점에 실시간으로 공인 출처와 대조하며
  옮겨적은 게 아니라서 개별 지점이 몇 % 어긋나 있을 수 있다.** JBM Ballistics나 McCoy의 *Modern Exterior
  Ballistics*로 몇 개 지점이라도 직접 대조하기 전에는 HUD 참고 표시 이상의 용도로 신뢰하면 안 된다.
- **`mass_kg`/`diameter_m`가 플레이스홀더(0.0)다.** `drag`를 활성화하려면 반드시 실제 탄자재원 값으로
  바꿔야 하고(0이면 `DragForce.__init__`이 `ValueError`로 즉시 실패하도록 방어해둠), 지금 이 프로젝트가
  실제 어떤 탄약을 쓰는지 확인된 바 없다.
- **탄도계수(BC) 대신 질량+직경+Cd(마하)를 직접 쓰는 방식을 택했다** — 고전적 BC는 파운드/제곱인치 단위계에
  묶여 있어 이 프로젝트의 SI 단위 규약과 안 맞고, 단위환산 상수를 검증 없이 들여오는 것보다 물리량을 그대로
  쓰는 쪽이 투명하다고 판단.
- 드래그를 켜면 `ProjectileModel`의 중력-only 닫힌해 지름길이 자동으로 꺼지고 RK4로 전환된다(`forces`가
  2개 이상이면 `_gravity_vec is None`) — `tests/test_forces.py`, `tests/test_aim_solver.py`의
  `test_converges_with_drag_enabled_across_range`(100/300/600m)로 뉴턴솔버가 이 비선형 힘에서도 정상
  수렴하는 것까지는 확인했다. **물리량 자체(항력표, 질량/직경)의 정확도는 별개로 검증이 필요하다.**

- **RK4 고정 스텝(`integrator_step`)이 드래그 활성화 시에도 수렴하는지는 이번에 별도로 확인했다
  (2026-08-19).** `config/aiming_engine.yaml`의 기본값 `0.01`초가 실제로 충분히 촘촘한지 이전에는
  검증된 적이 없었다(주석에 "tighten once drag is enabled"라고만 적혀 있었음). 완전히 다른 수치적분
  방법(scipy `solve_ivp`의 적응형 RK45, `rtol=atol=1e-10`)으로 같은 `ForceModel`을 독립적으로 적분해
  대조한 결과(`tests/test_forces.py::test_production_integrator_step_converged_for_drag`), 초속
  800m/s·비행시간 1.0초(수백 m 사거리)까지 5mm 이내로 일치 — 지금 기본 스텝은 이미 충분히 촘촘하다.
  단, 이건 수치적분의 *수렴성*만 확인한 것이지 항력표/질량/직경 등 *물리량 자체의 정확도*와는 별개다.

**다음 단계**: 실제 탄약의 mass_kg/diameter_m 확정 + G1/G7 표를 JBM Ballistics 등 공인 출처와 직접 대조한
뒤에만 `forces: [gravity, drag]`로 전환할 것.
