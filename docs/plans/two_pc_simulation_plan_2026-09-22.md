# PC 2대 분산 시뮬레이션 계획서

작성: 2026-09-22 · 대상: SMASH Gazebo 시뮬레이터(`simulation/`) · 상태: 계획(미착수)

## 1. 목적

- 시연 때 **Gazebo 3D 화면**과 **스코프 UI**를 함께 보여준다. 지금은 PC 한 대에서 3D 화면을 띄우면 PC가 멈춰서 headless로만 돌린다.
- 시뮬레이션 계산(물리, 카메라, FCS)은 지금 성능을 유지하거나 조금 개선한다.
- PC는 지금과 같은 사양 2대(i5-7400 4코어, GTX 1050 Ti, WSL2 Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic)라고 가정한다.

## 2. 근거: 2026-09-22 부하 측정

35 m 정지 표적, 카메라 640 px, 화각 6.9°에서 측정했다.

| 구성 | CPU (4코어) | 실제 RTF | 카메라 / FCS |
|---|---|---|---|
| 한 PC, 평소 구성 (뷰어 제외) | 91% | 0.84 | 약 25 Hz |
| Gazebo 단독 (FCS 종료) | 61% | 0.93 | 28.3 Hz |
| 한 PC, FCS 1~2스레드 제한 | 62~82% | 0.85~0.86 | 약 26 Hz |

측정에서 알게 된 것:
- **병목은 Gazebo의 단일 스레드 속도다.** CPU가 40% 남아도 RTF는 0.93에서 멈춘다. GPU는 평균 7~23%로 병목이 아니다.
- **FCS는 여유가 있다.** 처리에 쓰는 시간은 30% 정도이고, 스레드 제한 없이 약 1.3코어를 헛돌린다.
- **Gazebo와 FCS를 다른 PC로 나눠도 이득이 작다.** 카메라가 25에서 28 Hz가 되는 정도다. 대신 640 원본 영상(1.23 MB)을 1 GbE로 보내느라 프레임당 약 10 ms 지연이 붙는다.
- Windows 쪽에서 다른 앱이 GPU를 쓰면 RTF가 0.58~0.74까지 떨어진다. 이 영향이 스레드 설정보다 훨씬 크다.

## 3. 현재 구조

```
[PC 한 대]
 WSL2 ─ gz sim -s (서버, headless) ─ ros_gz_bridge ─ smash_fcs (YOLO·추적·조준·격발)
      └ smash_scope_viewer : /turret_camera/image_annotated 를 JPEG 로 바꿔 HTTP :9999 로 내보냄
 Windows ─ win_scope_viewer.py : http://localhost:9999 에서 영상을 받고, /fire·/slew·/cmd 로 조작
```

UI는 이미 HTTP로 떨어져 있다. 그래서 주소만 바꾸면 다른 PC에서도 붙을 수 있다.

## 4. 구성안 비교

### 안 1: 계산은 PC 1, 보여주기는 PC 2 (권장)

```
[PC 1: 계산]  WSL2 ─ gz sim -s + FCS + smash_scope_viewer(:9999)     ← 지금과 같음, headless
                    │ gz-transport (장면·자세, 작음)     │ HTTP MJPEG (HUD, 약 2~3 MB/s)
[PC 2: 표시]  WSL2 ─ gz sim -g (3D GUI 클라이언트)    Windows ─ win_scope_viewer.py
```

- FCS가 Gazebo와 같은 PC에 있어서 **제어 루프 지연이 늘지 않는다.** 원본 영상도 네트워크를 타지 않는다.
- 3D 화면을 그리는 부하와 뷰어·녹화 부하가 모두 PC 2로 간다. PC 1은 지금보다 가벼워진다.
- 네트워크에는 HUD MJPEG(약 20 Mbps)와 Gazebo 장면 갱신만 오간다.
- 월드에 `SceneBroadcaster` 플러그인이 이미 있다(`turret_world.sdf` 11행). 그래서 원격 GUI 클라이언트가 붙을 수 있다.

### 안 2: Gazebo는 PC 1, FCS와 UI는 PC 2 (권장하지 않음)

- ROS 2 DDS로 카메라 원본을 PC 2로 보낸다.
- 측정상 이득은 카메라 +12%뿐이다. 대신 지연이 약 10 ms 늘고, 공유메모리 대신 UDP 전송을 튜닝해야 한다.
- PC 1에서 3D GUI를 띄우면, 이미 한계인 Gazebo 서버와 같은 PC에서 경쟁한다.
- 이 안을 다시 볼 조건: 카메라 해상도나 개수가 늘거나 FCS 모델이 커져서, FCS가 CPU 병목이 되는 경우.

**이하 계획은 안 1 기준이다.**

## 5. 단계별 진행

### 0단계: 준비 (0.5~1일)

| 작업 | 내용 |
|---|---|
| 네트워크 | 두 PC를 유선 기가비트로 연결한다(같은 스위치 또는 직결). 고정 IP를 주고 서로 ping이 되는지 확인한다. |
| PC 2 환경 | `simulation/environment_setup_guide.md` 대로 WSL2, Jazzy, Gazebo를 설치하고 저장소를 받아 빌드한다(`run_smash_fcs.bat` 1~3번). 3D GUI가 월드의 모델과 메시를 불러와야 해서 `ciws_turret`, `drone_sim` 빌드가 필요하다. |
| WSL 네트워킹 | 두 PC 모두 `%UserProfile%\.wslconfig`에 `[wsl2]` `networkingMode=mirrored`를 넣고 `wsl --shutdown`한다. NAT 모드에서는 LAN에서 WSL 포트에 접근할 수 없고 멀티캐스트 탐색도 안 된다. |
| 방화벽 | PC 1에서 9999/TCP, gz-transport 탐색(10317~10318/UDP)과 동적 TCP를 **PC 2 IP에 대해서만** 연다. mirrored 모드에서는 Hyper-V 방화벽 규칙도 확인해야 한다. |
| 격리 | 두 PC 모두 `ROS_DOMAIN_ID`는 다르게 두고(예: PC 1 = 41, PC 2 = 42), `GZ_PARTITION=smash`, `GZ_IP=<각자 LAN IP>`로 맞춘다. 안 1은 DDS가 PC 사이를 오갈 필요가 없으므로, 섞이지 않게 막는다. |

완료 기준: PC 2에서 `curl http://<PC1>:9999/`가 응답하고, PC 2 WSL에서 `gz topic -l`을 치면 PC 1 서버의 토픽이 보인다.

### 1단계: 스코프 뷰어 원격 접속 (약 2시간)

| 파일 | 변경 |
|---|---|
| `scripts/win_scope_viewer.py` | `localhost:9999`로 고정된 주소를 환경변수 `SMASH_VIEWER_HOST`나 명령행 인자로 받게 한다. 기본값은 지금처럼 localhost다. |
| `simulation/run_smash_fcs.bat` | "원격 뷰어 접속(IP 입력)" 메뉴를 추가한다. 4~6번 실행에 "뷰어 없이 서버만" 선택지를 추가한다. |

검증:
- PC 2 화면의 초당 프레임을 측정한다. 목표는 로컬 대비 −10% 이내다.
- 키를 누른 뒤 화면에 반영될 때까지의 지연은 로컬 대비 +20 ms 이내를 목표로 한다.
- 수동 격발, 조준, 녹화가 모두 동작하는지 확인한다.
- PC 1의 RTF와 FCS 처리율이 뷰어를 떼어낸 뒤 같거나 오르는지 확인한다. 앞선 측정에는 뷰어가 빠져 있었으므로, 이번에 뷰어 부하도 함께 잰다.

주의: `/fire` 엔드포인트에는 인증이 없다. LAN 안에서만 쓰고, 방화벽을 PC 2 IP로만 연다.

### 2단계: Gazebo 3D 화면 원격 표시 (0.5일, 불확실성이 가장 큼)

| 파일 | 변경 |
|---|---|
| `ciws_turret/config/remote_view.gui.config` (신규) | 시점 조작만 되고 월드는 바꿀 수 없는 GUI 구성. 7절 참고. |
| `scripts/wsl_setup_smash_fcs.sh` | `gui_client` 하위 명령을 추가한다. 리소스 경로, `GZ_PARTITION`, `GZ_IP`를 설정하고 `gz sim -g --gui-config remote_view.gui.config`만 실행한다. 서버 실행(`run`)에도 같은 `GZ_PARTITION`과 `GZ_IP`를 내보낸다. |
| `simulation/run_smash_fcs.bat` | "원격 3D 화면(GUI 클라이언트)" 메뉴를 추가한다. |

검증:
- PC 2에 드론, 포탑, 탄도 표시가 실시간으로 움직이는지 확인한다.
- 마우스로 회전·이동·확대가 되고, 모델을 끌거나 지우는 도구는 보이지 않는지 확인한다.
- GUI를 붙이기 전후로 PC 1의 RTF와 FCS 처리율을 비교한다. 목표는 차이 −3% 이내다.
- PC 2의 3D 화면이 20 fps 이상인지 확인한다.

위험과 대안:
- WSLg에서 Gazebo GUI 성능이 모자랄 수 있다. 이 저장소가 WSLg 버그를 피하려고 Windows 네이티브 뷰어를 따로 만든 이력이 있다. 그러면 PC 2를 네이티브 Ubuntu(듀얼부팅)로 쓰는 방안을 검토한다.
- mirrored 모드에서 gz-transport 탐색이 안 되면 `GZ_RELAY=<PC1 IP>`로 고정 연결한다.

### 3단계: PC 1 여유 확보 (선택, 약 1시간)

- FCS 노드 기동 시 `torch.set_num_threads(2)`와 `cv2.setNumThreads(2)`를 설정한다. FCS 처리율은 그대로 두고 약 0.7코어를 돌려받는다.
- 시연용 PC 1에서는 GPU를 쓰는 다른 Windows 앱(브라우저 영상, ChatGPT 등)을 끄는 절차를 둔다.

### 4단계: 비교 시험 (0.5일)

같은 조건으로 단일 PC 구성과 2대 구성을 각 3회 이상 돌린다. `benchmarks/sim_measurement/` 도구를 쓴다.
- 정지 표적: 35 m, 100 m
- 이동 표적: 매복 A/B 시나리오(`smash-latency-compensation` 결론에 따른 시험)

| 지표 | 목표 |
|---|---|
| PC 1 실제 RTF, FCS 처리율 | 단일 PC 이상 |
| 촬영→격발 지연 (중앙값, p95) | 단일 PC 대비 +5 ms 이내 |
| READY 비율, 명중률 | 통계적으로 차이 없음 |
| PC 2 표시 (스코프, 3D) | 각각 20 fps 이상 |

주의: 측정 전에 다른 세션의 `run_case.sh`가 돌고 있지 않은지 반드시 확인한다(2026-09-22에 측정끼리 충돌한 사례가 있다). 결과에 TF 조회 실패가 섞였는지도 확인한다.

### 5단계: 운영 절차 정리 (약 2시간)

- 시연 순서를 정리한다: PC 1 서버 기동 → PC 2 3D 화면 → PC 2 스코프 뷰어 → 녹화.
- `simulation/README.md`와 `environment_setup_guide.md`에 2대 구성 절을 추가한다.
- 네트워크 없이 한 PC에서 돌리는 기존 경로는 그대로 유지한다. 기본값을 바꾸지 않는다.

## 6. 일정 요약

| 단계 | 기간 | 선행 조건 |
|---|---|---|
| 0 준비 | 0.5~1일 | PC 2, LAN 케이블 |
| 1 원격 뷰어 | 약 2시간 | 0 |
| 2 원격 3D 화면 | 약 0.5일 | 0 (1과 병행 가능) |
| 3 PC 1 여유 확보 | 약 1시간 | 선택 |
| 4 비교 시험 | 약 0.5일 | 1, 2 |
| 5 정리 | 약 2시간 | 4 |
| **합계** | **약 2~3일** | |

1단계만 해도 "PC 1은 계산, PC 2는 UI"라는 목표의 절반이 이뤄진다. 2단계는 WSLg 성능에 달려 있으므로 가장 먼저 짧게 시험해 볼 가치가 있다.

## 7. 결정 사항

| 항목 | 결정 (2026-09-22) | 계획에 주는 영향 |
|---|---|---|
| PC 2 환경 | 지금 로컬 데스크톱과 똑같이 세팅한다(Windows + WSL2 Ubuntu 24.04 + Jazzy + Gazebo Harmonic, 같은 저장소와 빌드). | 0단계는 설치 가이드를 그대로 따라 하면 된다. 2단계의 3D 화면은 WSLg 위에서 돈다. |
| 3D 화면 시점 | 사용자가 마우스로 조작한다(회전, 이동, 확대). | 2단계에 GUI 설정 파일 작업이 추가된다(아래 참고). |
| 녹화 위치 | 미정. 스코프는 PC 2 뷰어의 녹화 기능을 쓰고, 3D 화면은 PC 2의 화면 녹화 도구를 쓰는 안이 유력하다. | — |

### 3D 화면을 사용자가 조작할 때 주의할 점

- **시점 조작은 PC 1에 부하를 주지 않는다.** GUI 클라이언트의 시점 카메라는 PC 2에서만 그리는 화면이고, 센서가 아니다. 서버는 장면 갱신만 보낸다.
- **다만 Gazebo 기본 GUI는 월드를 바꿀 수 있다.** 이동·회전 도구나 모델 삭제 같은 조작이 `UserCommands`를 거쳐 PC 1 서버에 그대로 반영된다. 시연 중에 드론이나 포탑을 실수로 끌어 옮기면 시험이 망가진다.
- 대응: 전용 GUI 설정 파일(`ciws_turret/config/remote_view.gui.config`)을 둔다. 3D 장면, 시점 조작, 시계 표시만 남기고 변환 도구(Transform Control), 모델 삽입·삭제, 엔티티 속성 편집 플러그인은 뺀다. 이 파일은 `gz sim -g --gui-config <파일>`로 불러온다.
- 편의 기능(선택): 자주 쓰는 시점(포탑 뒤, 표적 옆, 전경)을 버튼이나 단축키로 불러오는 기능. GUI의 카메라 이동 기능(`/gui/move_to/pose`)을 쓰면 된다.

### 2단계 보강

- 1순위 작업: 지금 PC에서 먼저 짧게 시험한다. headless 서버를 띄운 상태에서 같은 PC의 WSLg로 `gz sim -g`를 붙여 본다. 화면이 뜨는지, 대략 몇 fps가 나오는지 보면 PC 2 준비 전에 WSLg 가능 여부를 가릴 수 있다. 같은 PC라 부하 수치 자체는 참고용이다.
- WSLg 성능이 모자라면 GUI 설정을 줄여 본다(그림자 끄기, 창 크기 줄이기). 그래도 부족하면 그때 PC 2 네이티브 Ubuntu를 다시 검토한다.
