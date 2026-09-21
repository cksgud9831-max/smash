# SMASH Gazebo 및 ROS 2 시뮬레이션

이 디렉터리는 SMASH 대드론 조준 보조 알고리즘을 Gazebo Harmonic과 ROS 2 Jazzy에서 검증하는 환경이다. CIWS 형상의 Pan/Tilt 포탑은 사수가 총기를 움직이는 동작을 대신하는 시뮬레이션 장치이며, 실제 제품의 자동 포탑을 의미하지 않는다.

## 디렉터리 구성

```text
simulation/
├── ciws_turret_aerial_object_detection-main/
│   ├── ciws_turret/              포탑 URDF·메시·Gazebo 월드·카메라·레이저·제어기
│   ├── drone_sim/                표적 모델·통합 FCS·HUD·뷰어·격추 연출
│   ├── assets/                   참고 이미지 및 시각화 자료
│   └── README.md                 기반 시뮬레이터 설명
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
- `drone_sim/smash_fcs/fire_control.py`: 수동·자동 격발과 HIT/MISS 판정
- `drone_sim/smash_scope_viewer.py`: 조준 조작, 격발 및 MP4 녹화 뷰어
- `drone_sim/demo_director.py`: 명중한 드론의 추락 및 재출현 연출
- `models/`: 드론·조류·헬기·항공기 Gazebo 모델

## 실행 순서

1. `environment_setup_guide.md`에 따라 WSL2, ROS 2 Jazzy와 Gazebo Harmonic을 설치한다.
2. Windows에서 `simulation\run_smash_fcs.bat`을 실행한다.
3. 처음 설치한 환경에서는 메뉴 1 → 2 → 3 순서로 진단, 링크, 빌드를 수행한다.
4. 목적에 따라 메뉴 4~6으로 시뮬레이션을 실행한다.

| 메뉴 | 기능 |
|---:|---|
| 1 | 환경 진단 (`check`) |
| 2 | WSL2 워크스페이스 심볼릭 링크 생성 (`link`) |
| 3 | `ciws_turret`, `drone_sim` colcon 빌드 (`build`) |
| 4 | 비사격 시뮬레이션과 스코프 뷰어 |
| 5 | 수동 격발 시뮬레이션—READY 상태에서 Space로 격발 |
| 6 | 자동 격발 시뮬레이션—조준은 수동, READY 시 격발만 자동 |
| 7 | 격발·명중 판정 오프라인 검증 |
| 8 | 실행 중인 시뮬레이터의 스코프 뷰어 다시 열기 |
| 9 | Gazebo GUI/헤드리스 모드 전환 |

스코프 뷰어는 방향키 또는 WASD로 조준하고, `T`로 대공 경계 자세, `R`로 수평 자세를 선택한다. `H`는 HUD 디버그 정보, `V`는 MP4 녹화를 켜거나 끈다.

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

현재 `main` 기준 전체 테스트 결과는 `171 passed`다. ROS 2/Gazebo 실행 환경은 메뉴 1의 환경 진단과 메뉴 7의 격발 판정 검증을 함께 사용한다.
