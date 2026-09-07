# 레이저/IMU 센서 통합 — 구조 점검 및 갭 분석

## 1. 현재 파이프라인에서 통합 지점이 어디인지

```
Camera --> Detector.detect() --> SmartTracker.update() --> TrackerFrameBuilder.build()
                                                                    |
                                                    RangeSensor.read()   PoseSource.read()
                                                    (레이저 거리)        (IMU/짐벌 자세)
                                                                    |
                                                                    v
                                                          aiming_engine.TrackerFrame
                                                                    |
                                                                    v
                                                          AimingManager.update()  (조준 알고리즘, 완성됨)
```

다행히 이 작업을 위한 자리는 이미 코드에 파여 있다. `bridge/frame_builder.py`가 매 프레임마다 `RangeSensor.read(timestamp)`와 `PoseSource.read(timestamp)`를 호출해서 `TrackerFrame`을 조립하고, 그 결과를 그대로 `aiming_engine.AimingManager`에 넘긴다. 즉 지금 필요한 작업은 **`RangeSensor`와 `PoseSource` 두 추상 인터페이스의 실제 하드웨어 구현체를 만들어 끼워 넣는 것**이지, 파이프라인 구조 자체를 새로 설계할 필요는 없다.

- `RangeSensor` (`bridge/range_sensor.py`): `read(timestamp) -> Optional[RangeSample(distance_m, timestamp, valid)]`. 지금은 `MockRangeSensor`(고정 250m)만 존재.
- `PoseSource` (`bridge/pose_source.py`): `read(timestamp) -> PoseSample(extrinsic: 4x4, timestamp)`. 지금은 `MockPoseSource`(identity, 즉 위치 원점·자세 레벨)만 존재.
- 두 팩토리 함수 `build_range_sensor()` / `build_pose_source()`(각 파일 하단)가 `config/bridge.yaml`의 `laser.sensor` / `pose.source` 문자열로 분기하며, 이미 `serial` / `gimbal_imu`라는 이름 자리가 주석으로 예약되어 있다("not implemented yet — hardware TBD").

## 2. 이미 준비되어 있어 재사용 가능한 부분

- **`LaserAligner`** (`bridge/laser_alignment.py`): 레이저 보어사이트와 카메라 광학중심 간 물리적 오프셋을 보정해 슬랜트 거리를 카메라 픽셀 시선벡터에 투영하는 로직이 이미 구현·문서화되어 있다. 실제 레이저 드라이버가 원시 거리값(raw slant range, float)만 돌려주면 이 보정 로직에 바로 태울 수 있다.
- **`CoordinateTransform`** (`aiming_engine/coordinate_transform.py`): `camera_extrinsics`(World<-Platform 4x4)를 받아 Image→Camera→Scope→World 전체 체인을 처리하는 로직이 이미 있다. 즉 `PoseSource`가 매 프레임 4x4 행렬(회전+위치)만 정확히 채워주면, 그 뒤 좌표변환·조준 계산은 손댈 필요가 없다.
- **`config.py` 로딩 패턴**: YAML → 타입 명시된 frozen dataclass 로 로드하는 방식이 `aiming_engine`과 `bridge` 양쪽에 일관되게 있어서, 새 센서 설정 필드를 추가할 때 따라야 할 패턴이 명확하다.
- **`TrackerFrameBuilder`의 실패 처리**: 거리 샘플이 없거나 유효하지 않으면(`range_sample is None or not range_sample.valid`) 그냥 그 프레임을 스킵(`None` 반환)하도록 이미 되어 있다 — 발사 게이팅이 아니므로 하드웨어 순간 오류를 에러로 취급하지 않고 조용히 넘어가는 구조가 이미 마련돼 있다.

## 3. 갭(비어 있는 부분) — 실제 구현 시 결정/작업이 필요한 지점

### 3.1 `RangeSensor` 실제 구현체 부재
- 프로토콜(UART/시리얼, I2C, CAN 등), 명령/응답 포맷, 보율(baud rate) 등이 아직 코드에 반영되어 있지 않음 — `config/bridge.yaml`의 `laser.sensor: mock | serial`에서 `serial` 분기가 비어 있다.
- **`valid` 판정 기준**을 정의해야 한다. 레이저는 노 리턴(no return, 표적이 없거나 너무 멀 때), 다중 반사, 신호 약함 등으로 무효 판정이 나는 경우가 흔한데 이걸 무엇으로 판단할지(에러 코드? 특정 센티널 값? 신호 강도 임계치?) 하드웨어 스펙에 따라 결정 필요.
- **샘플링 주기 불일치**: 카메라는 30fps 고정이지만 레이저 측거 주기는 그보다 느리거나(예: 10Hz) 비동기일 수 있다. 지금 인터페이스는 `read(timestamp)`가 그 프레임의 최신값을 바로 준다고 가정하는데, 실제로는 "가장 최근 샘플을 캐싱해두고 그게 너무 오래됐으면(staleness) invalid 처리" 하는 로직이 드라이버 내부에 필요하다.

### 3.2 `PoseSource` 실제 구현체 부재
- **IMU만으로는 위치(translation)를 알 수 없다.** IMU/AHRS는 보통 자세(roll/pitch/yaw 또는 쿼터니언)만 제공한다. `camera_extrinsics`는 4x4 전체(회전+위치)가 필요하므로, 플랫폼이 고정되어 있다고 가정하고 위치는 `(0,0,0)` 등 고정값을 쓸지, 아니면 별도 위치추정(GPS 등)이 있는지 확인이 필요하다. (원래 `aiming_engine/README.md`의 설계도 스코프 플랫폼이 삼각대/고정 마운트라는 전제가 강해 보인다.)
- **짐벌이 있다면** 짐벌 인코더 각도 + IMU 자세를 합성해서 최종 extrinsic을 만들어야 한다 — 지금 코드에는 짐벌 인코더 인터페이스가 아예 없다.
- **좌표계/축 정의 불일치 가능성**: IMU가 내놓는 좌표계(NED, ENU, 또는 제조사 자체 정의)가 `aiming_engine`이 요구하는 "World: right-handed, Z-up"과 다를 수 있다. `PoseSource` 구현 내부에서 이 변환을 책임져야 한다(마치 `coordinate_transform.py`가 카메라축↔스코프축을 고정 회전행렬로 재배열하듯).
- **드리프트/바이어스**: IMU 자세 추정은 시간이 지나며 드리프트가 생기므로 필터링(상보필터/칼만필터/AHRS 라이브러리) 또는 주기적 보정이 필요할 수 있다 — 이 부분은 하드웨어/펌웨어가 자체 처리하는지, 우리 쪽에서 후처리해야 하는지 확인 필요.

### 3.3 타이밍 동기화
- `TrackerFrameBuilder.build()`는 카메라 프레임 타임스탬프 하나로 `range_sensor.read(ts)`와 `pose_source.read(ts)`를 순차 호출한다. 세 센서(카메라/레이저/IMU)가 서로 다른 클럭/주기를 가지므로, 프레임 타임스탬프에 가장 가까운 샘플을 찾아주는 보간/최근값 캐싱 로직이 각 드라이버 내부(혹은 공용 유틸)에 필요하다. 지금 Mock 구현들은 이 문제를 아예 회피(고정값 반환)하고 있어서 실제 구현 시 처음 마주치는 문제가 될 것이다.

### 3.4 설정 스키마 확장
- `bridge/config.py`의 `LaserConfig`/`PoseConfig` 데이터클래스에 시리얼 포트, 보율, IMU 축 매핑, 짐벌 오프셋 등 실제 연결 파라미터 필드를 추가해야 한다. 지금은 `mock_fixed_distance_m` / `mock_extrinsic` 같은 mock 전용 필드만 있다.

### 3.5 스레딩/실시간성
- 시리얼 통신은 보통 블로킹 I/O라서, 메인 프레임 루프(목표 30~45fps) 안에서 직접 `read()`를 블로킹으로 호출하면 프레임레이트가 흔들릴 수 있다. 별도 스레드가 백그라운드에서 계속 센서를 폴링해 최신값을 캐시에 써두고, `read()`는 그 캐시를 non-blocking으로 반환하는 producer/consumer 구조가 필요할 가능성이 높다 — 지금 인터페이스 시그니처(`read(timestamp) -> Sample`)는 이 구조를 자연스럽게 감당할 수 있게 이미 되어 있다(내부 구현만 바꾸면 됨).

### 3.6 실패/연결 끊김 처리
- 시리얼 연결이 끊기거나 IMU가 응답하지 않을 때 무엇을 반환할지 정책이 필요하다: 마지막 유효값 유지(staleness 태그와 함께)? `valid=False`? 예외 발생? 현재 `TrackerFrameBuilder`는 `range_sample`이 없거나 무효하면 그 프레임을 조용히 스킵하므로, 레이저 쪽은 이 정책과 잘 맞아떨어진다. 반면 `PoseSource.read()`는 `Optional`이 아니라 `PoseSample`을 항상 반환하는 시그니처라서, 자세 추정이 실패했을 때의 처리 방식은 지금 인터페이스에 정의돼 있지 않다 — 이 부분은 설계가 필요하다.

### 3.7 테스트 갭
- `tests/`에는 `MockRangeSensor`/`MockPoseSource`에 대한 테스트만 존재(간접적으로도 실제 하드웨어 구현체를 검증하는 테스트는 없음). 실제 드라이버가 생기면 시리얼 포트를 모사하는 fixture나 녹화된 로그 재생 방식의 테스트가 필요할 것이다.

## 4. 선정 센서 반영 — Benewake TF02-Pro (레이저), Jetson Orin 호환 IMU (미정)

### 4.1 Benewake TF02-Pro — 프로토콜은 이미 충분히 특정 가능

TF02-Pro는 Benewake TF 시리즈(TFmini/TF02/TF-Luna 등)가 공유하는 표준 UART 바이너리 프로토콜을 쓴다. 공식 데이터시트(Mouser) 기준 스펙은 다음과 같다.

- **통신**: UART, 기본 115200bps · 8N1(데이터 8비트, 정지비트 1, 패리티 없음). 보율은 9600~921600bps, 프레임레이트는 1~1000Hz(기본 100Hz)로 설정 명령을 통해 변경 가능.
- **측정범위**: 0.1~40m(90% 반사율 기준), 정확도 ±5cm(0.1~5m) / ±1%(5~40m), 분해능 1cm.
- **출력 프레임 구조**(TF 시리즈 공통 9바이트 프레임 — TF02-Pro 전용 세부 필드는 실물/정식 매뉴얼로 최종 검증 권장):
  - Byte 0-1: 헤더 `0x59 0x59`
  - Byte 2-3: 거리 저바이트/고바이트 (`DIST_L`, `DIST_H`)
  - Byte 4-5: 신호강도 저바이트/고바이트 (`AMP_L`, `AMP_H`)
  - Byte 6: 신뢰도/모드 바이트 (모델별로 의미가 조금씩 다를 수 있어 확인 필요)
  - Byte 7: 예약
  - Byte 8: 체크섬 (Byte0~7 합의 하위 8비트)
  - 무효/측정불가 시 특정 센티널 거리값(TF 시리즈에서 흔히 쓰이는 방식) 또는 신호강도가 임계치 미만일 때 무효로 처리 — 정확한 센티널 값·신뢰도 임계치는 실기/정식 매뉴얼로 확정 필요.

**설계 방향**: `RangeSensor`를 상속하는 `TF02ProRangeSensor`(가칭)를 만들어, 백그라운드 스레드가 시리얼 포트를 논블로킹으로 계속 읽어 9바이트 프레임을 파싱 → 최신 `RangeSample`을 캐시에 저장하고, `read(timestamp)`는 그 캐시값(및 staleness 체크)만 즉시 반환하는 producer/consumer 구조를 제안한다(3.5절 스레딩 이슈 해결). `config/bridge.yaml`의 `laser.sensor: serial`을 이 구현체에 매핑하고, `LaserConfig`에 `port`, `baud_rate`, `min_signal_strength`(무효 판정 임계치) 같은 필드를 추가하면 된다. 체크섬 검증 실패/헤더 불일치 프레임은 버리고 이전 캐시값을 유지(또는 staleness 초과 시 invalid)하는 방어 로직이 필요하다.

이 정도면 실제 `TF02ProRangeSensor` 구현으로 바로 들어갈 수 있는 수준의 스펙이 확보된 상태다. 다만 신뢰도 바이트(Byte 6)의 정확한 의미와 무효 판정 센티널 값은 모델별 편차가 있어 실기 테스트나 정식 매뉴얼 대조로 최종 확인하는 걸 권장한다.

### 4.2 IMU — "Orin과 연동 가능한 아무거나"에 대한 판단

IMU를 아직 안 정하셨다면, 이번 용도(스코프 자세 추정 → `PoseSource`의 회전 성분)에 맞춰 두 갈래를 제안한다.

- **내장 센서퓨전 IMU (추천)**: 칩 자체가 가속도계+자이로+지자기를 합성해 쿼터니언/오일러각을 바로 출력해주는 타입(예: BNO085 계열). 소프트웨어에서 별도 AHRS 필터(Madgwick/Mahony 등)를 구현할 필요가 없어 `PoseSource` 구현이 "시리얼/I2C로 쿼터니언 읽어서 4x4로 변환"만 하면 되는 수준으로 단순해진다. 다만 구형 BNO055는 일부 I2C 컨트롤러(특히 클록 스트레칭 처리)와 궁합 문제가 보고된 사례가 있어, 최근 보드에서는 UART 모드 사용이나 후속 모델(BNO085/BNO086 계열) 쪽이 커뮤니티에서 더 권장되는 편이다.
- **raw 9축 IMU + 자체 퓨전 (예: ICM-20948)**: 센서 자체는 저렴하고 검증된 칩이지만, 우리 쪽에서 AHRS 필터를 직접 구현/튜닝해야 해서 개발 범위가 늘어난다. ROS2 커뮤니티에서 Jetson Orin NX와 함께 자주 언급되는 후보이긴 하나, "바로 쓸 수 있는 자세값"을 얻으려면 추가 소프트웨어 스택이 필요하다.

지금 단계에서는 **내장 센서퓨전이 되는 IMU를 고르는 쪽을 권장**한다 — `PoseSource` 구현이 단순해지고, `aiming_engine`이 필요로 하는 건 어차피 최종 회전행렬(또는 쿼터니언)뿐이라 굳이 원시 IMU 데이터를 직접 필터링할 이유가 없다. 어떤 IMU든 최종적으로 `PoseSource.read()`가 "쿼터니언/오일러각 → World Z-up 좌표계로 축 재배열 → 4x4 extrinsic"으로 변환하는 얇은 어댑터 계층만 만들면 되므로, IMU 기종이 최종 확정되지 않아도 `PoseSource` 인터페이스와 축 변환 골격은 지금 먼저 설계해둘 수 있다.

**남은 결정 필요 사항 (IMU)**: 통신 방식(I2C vs UART), 출력 좌표계(NED/ENU 등, 제조사 스펙시트에 명시됨), 짐벌이 실제로 있는지(있다면 인코더 각도를 IMU 자세와 합성해야 함) — 이 세 가지만 정해지면 `PoseSource` 구현도 바로 진행 가능하다.

## 5. 다음 단계 제안

1. **TF02-Pro `RangeSensor` 구현**: 스펙이 충분히 확보됐으니 바로 착수 가능 — 원하시면 다음 턴에 시리얼 파서 + 백그라운드 폴링 스레드 + `LaserConfig` 확장까지 구현.
2. **IMU 기종 확정**: 위 두 갈래(내장 퓨전 vs raw+자체 필터) 중 방향을 정해주시면 `PoseSource` 구현으로 이어갈 수 있음. 방향만 정해지면 실제 모델은 나중에 바뀌어도 어댑터 계층만 갈아끼우면 되도록 설계 가능.
3. **플랫폼 이동성 확인**: 스코프 플랫폼이 고정(삼각대)인지 이동 가능한지 — 지금 코드베이스 전반(`config/aiming_engine.yaml`의 identity mock 등)이 고정 플랫폼(위치는 원점 고정, 자세만 실시간 갱신)을 전제로 하는 것처럼 보이는데, 이 가정이 맞는지 확인 필요.

Sources:
- [Benewake TF02-Pro Datasheet (Mouser)](https://www.mouser.com/datasheet/2/1099/Benewake_10152020_TF02_Pro-1954040.pdf)
- [Benewake TF-series LiDAR Datasheet (Seeed Studio, shared UART protocol reference)](https://statics3.seeedstudio.com/assets/file/bazaar/product/DE_LiDAR_TF02_Datasheet_V2.2.pdf)
- [Looking for an IMU for Jetson Orin NX that works in ROS2 (Bosch Sensortec Community)](https://community.bosch-sensortec.com/mems-sensors-forum-jrmujtaw/post/looking-for-an-imu-for-jetson-orin-nx-that-works-in-ros2-L8fPvn4U4YDKEpb)
