# SMASH Gazebo 및 ROS 2 시뮬레이션

이 디렉터리는 SMASH 대드론 조준 보조 알고리즘을 Gazebo Harmonic과 ROS 2 Jazzy에서 검증하는 환경이다. CIWS 형상의 Pan/Tilt 포탑은 사수가 총기를 움직이는 동작을 대신하는 시뮬레이션 장치이며, 실제 제품의 자동 포탑을 의미하지 않는다.

## 디렉터리 구성

```text
simulation/
├── ciws_turret_aerial_object_detection-main/
│   ├── ciws_turret/              포탑 URDF·메시·Gazebo 월드·카메라·레이저·제어기
│   ├── drone_sim/                표적 모델·통합 FCS·HUD·뷰어·격추 연출
│   └── README.md                 기반 시뮬레이터 요약
├── run_smash_fcs.bat             Windows/WSL2 통합 실행 메뉴
├── environment_setup_guide.md    WSL2·ROS 2·Gazebo 설치 및 환경 구성
└── README.md
```

실제 폴더 이름은 `ciws_turret_aerial_object_detection-main`이다. 언더스코어 형태의 `ciws_turret_aerial_object_detection_main`은 사용하지 않는다.

### `ciws_turret`

- `urdf/ciws_turret.urdf.xacro`: Pan/Tilt 포탑, 카메라 및 레이저 센서 모델
- `worlds/turret_world.sdf`: Gazebo 검증 월드
- `launch/view.launch.py`: 포탑·센서·ROS/Gazebo 브리지 실행
- `config/controllers.yaml`: ROS 2 제어기 설정
- `config/fastdds_large_messages.xml`: 640×640 영상 전송용 Fast DDS 공유메모리 설정

### `drone_sim`

- `launch/smash_scene.launch.py`: 포탑, 표적, FCS, 격발 판정, 격추 연출 및 뷰어 통합 실행
- `config/smash_fcs.yaml`: 추적·거리·탄도·포탑·격발 파라미터
- `drone_sim/smash_fcs_node.py`: 탐지, 추적, 거리 추정, 신뢰도 판단, 탄도 계산 및 격발 통합 노드
- `drone_sim/smash_fcs/hud.py`: 스코프 HUD, 잠금 괄호, 상태 표시 및 탄착 패널
- `drone_sim/smash_fcs/fire_control.py`: 자동 격발과 HIT/MISS 판정
- `drone_sim/smash_scope_viewer.py`: 스코프 영상 중계 서버(HTTP MJPEG, 포트 9999). Windows 창과 녹화는 `scripts/win_scope_viewer.py`
- `drone_sim/demo_director.py`: 명중한 드론의 추락 및 재출현 연출
- `models/`: 표적 드론 Gazebo 모델

## 실행 순서

1. `environment_setup_guide.md`에 따라 WSL2, ROS 2 Jazzy와 Gazebo Harmonic을 설치한다.
2. Windows에서 `simulation\run_smash_fcs.bat`을 실행한다.
3. 처음 설치한 환경에서는 메뉴 1 → 2 → 3 순서로 진단, 링크, 빌드를 수행한다.
4. 목적에 따라 메뉴 4(비사격) 또는 6(자동 격발)으로 시뮬레이션을 실행한다.

| 메뉴 | 기능 |
|---:|---|
| 1 | 환경 진단 (`check`) |
| 2 | WSL2 워크스페이스 심볼릭 링크 생성 (`link`) |
| 3 | `ciws_turret`, `drone_sim` colcon 빌드 (`build`) |
| 4 | 비사격 시뮬레이션과 스코프 뷰어(탐지·조준만) |
| 6 | 자동 격발 시뮬레이션—포가 리드점을 추종하고 READY 시 자동 격발 |
| 7 | 격발·명중 판정 오프라인 검증 |
| 8 | 실행 중인 시뮬레이터의 스코프 뷰어 다시 열기 |
| 9 | Gazebo GUI/헤드리스 모드 전환 |

스코프 뷰어에는 조준·격발 입력이 없다(수동 조작 방식은 폐기). `R`은 표적을 놓쳤을 때 대공 경계 자세로 복귀, `0~4`·`+/-`·`M`은 이동 표적 기동 방식·속도·출발/정지, `H`는 HUD 디버그 정보, `C`는 통계 초기화, `V`는 MP4 녹화를 켜거나 끈다.

## 현재 통합 상태

- 카메라 영상에서 YOLO 탐지와 다중 스케일 옵티컬플로우 추적을 수행한다.
- PointCloud 레이저와 바운딩박스 기하 추정을 조합해 거리를 계산한다.
- `aiming_engine`이 5.56 mm 탄도, 비행시간, 리드각과 명중확률을 계산한다.
- `jun_reliability`의 `STABLE/HOLD/LOST` 상태가 측정 수용과 READY 판정에 적용된다.
- `STABLE`에서는 현재 측정을 반영하고, `HOLD`에서는 마지막 신뢰 상태로 짧게 예측하며, `LOST`에서는 측정을 거부한다.
- HUD는 SEARCH/TRACK/ALIGN/READY 상태, LIVE/HELD 조준점, 탄착 분포와 격추 결과를 표시한다.
- 정지 호버링 표적은 약 10.8 m와 35.3 m 조건에서 명중 검증을 완료했다. 60 km/h 이상 이동 표적과 100 m 운용 조건은 후속 검증 범위다.

## 검증

저장소 루트에서 다음 명령으로 Python 테스트를 실행한다.

```bash
python -m pytest -q
```

전체 테스트 결과는 `323 passed`다(2026-10-15). ROS 2/Gazebo 실행 환경은 메뉴 1의 환경 진단과 메뉴 7의 격발 판정 검증을 함께 사용한다.


### 영상 뷰어 상태 패널

Windows 뷰어 오른쪽 패널에 영상 수신 FPS, 마지막 프레임 수신 후 경과 시간,
재연결 횟수, 녹화 시간과 파일 이름을 표시한다. 표시 FPS는 네트워크 수신률이며
시뮬레이션 처리율이나 종단 지연을 뜻하지 않는다.

2초 동안 새 영상이 오지 않으면 STALE 배너를 표시하고 진행 중인 녹화를 종료한다.
영상 연결은 자동으로 다시 시도하며, 복구 후 녹화는 V 키로 새로 시작한다.
녹화에는 기존 영상·자막만 저장하고 로컬 상태 패널은 포함하지 않는다.
F 키 전체화면과 F1 기존 도움말은 유지한다. 재연결 오류는 콘솔에 출력한다.


### 로컬 녹화 재생기

런처 **5번** 또는 `python scripts/recording_player.py [영상파일]`로 연다.
시뮬레이터가 꺼져 있어도 동작하며 네트워크 연결을 만들지 않는다.
파일 열기, 타임라인 탐색, ±5초 이동, 0.25~2배속, 전체화면을 지원한다.
Space는 재생·일시정지, 좌우 방향키는 5초 이동, F는 전체화면,
Esc는 전체화면 해제다. 끝까지 재생한 뒤 재생을 누르면 처음부터 시작한다.
OpenCV가 지원하는 MP4/AVI/MKV/MOV 파일을 열 수 있다. Pillow와 Tkinter가 필요하다.

런처 **O**는 녹화 폴더, **L**은 WSL ROS 로그 폴더를 연다.
기존 메뉴 번호는 유지했으며 작업 종료 코드와 마지막 작업 결과를 표시한다.

### 장면 조명·재질

`ciws_turret/worlds/turret_world.sdf`의 환경광은 차가운 색조, 햇빛은 따뜻한 색조로
조정했다. 지면에는 비금속·거칠기 0.95의 무광 재질을 적용했다.
그림자는 기존처럼 끈 상태다. 센서·형상·물리 설정은 수정하지 않았다.
조명 변경은 영상 픽셀에 영향을 주므로 과거 영상 측정 결과와 비교할 때 기록해야 한다.
변경한 월드를 설치본에 반영하려면 실행 중인 시뮬레이터를 종료한 뒤 런처 3번으로
다시 빌드하고 재시작한다. 2026-10-15 colcon 재빌드 후 비사격 설정으로 재시작하여 Gazebo 장면과 Windows
뷰어의 실제 영상 수신을 확인했다. 이전 장면과 동일 시점의 정량 전후 비교는 수행하지 않았다.
