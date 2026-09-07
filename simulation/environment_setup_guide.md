# SMASH FCS 대드론 시뮬레이션 환경 구축 가이드

본 문서는 SMASH 지능형 대드론 사격 통제 시스템(FCS) 3차원 물리 시뮬레이션을 재현하고 구동하기 위한 시스템 요구 사양, 의존성 패키지, 환경 변수 및 실행 절차를 상세히 정리한 가이드입니다.

============================================================

### 1. 시스템 기본 요구 사양

* **호스트 운영체제:** Windows 10 또는 Windows 11 (64bit)
* **게스트 서브시스템:** WSL2 기반 Ubuntu 24.04 LTS
* **그래픽 카드(GPU):** NVIDIA 지포스 계열 (CUDA 가속 및 Gazebo 하드웨어 렌더링 지원)
* **시스템 메모리:** 16GB RAM 이상 권장
* **저장 공간:** 30GB 이상의 여유 공간

============================================================

### 2. 핵심 프레임워크 및 소프트웨어 버전

* **로봇 운영체제:** ROS 2 Jazzy Jalisco
* **3차원 물리 시뮬레이터:** Gazebo Harmonic 8.11
* **프로그래밍 언어:** Python 3.12 (가상환경 smash_venv)
* **딥러닝 프레임워크:** PyTorch (CUDA 가속 활성화), Ultralytics (YOLO11)
* **비전 및 수치 연산 라이브러리:** OpenCV, NumPy, SciPy

============================================================

### 3. 필수 제어기 및 Gazebo 연동 패키지 구성

Gazebo 3D 월드 상에서 2축(Pan/Tilt) 대공 포탑을 모터 구동하고 센서 토픽을 브릿지하기 위해 아래의 ROS 2 패키지가 WSL2 Ubuntu 24.04 내에 구성되어야 합니다.

* **필수 패키지 목록:**
  1. ros_jazzy_ros2_control
  2. ros_jazzy_ros2_controllers
  3. ros_jazzy_gz_ros2_control
  4. ros_jazzy_controller_manager
  5. ros_jazzy_ros_gz
* **설치 및 관리 원칙:**
  * 환경 진단 런처를 실행하면 controller_manager 및 관련 패키지의 설치 여부를 자동 점검합니다.
  * 미설치 상태가 감지될 경우 출력되는 터미널 안내에 따라 사용자가 수동 설치를 수행합니다.

============================================================

### 4. 환경 변수 및 워크스페이스 심볼릭 링크 설정

1. **Gazebo 플러그인 환경 변수 주입:**
   * Gazebo 시뮬레이터가 ROS 2 제어기 플러그인을 정상 인식할 수 있도록 wsl_setup_smash_fcs.sh 실행 스크립트 내에 아래 환경 변수를 자동으로 주입합니다.
   * `export GZ_SIM_SYSTEM_PLUGIN_PATH="/opt/ros/jazzy/lib:$GZ_SIM_SYSTEM_PLUGIN_PATH"`

2. **파이썬 모듈 검색 경로 설정:**
   * 공통 라이브러리(bridge 패키지 등)를 원활하게 임포트할 수 있도록 프로젝트 루트 경로를 PYTHONPATH에 등록합니다.
   * `export PYTHONPATH="/mnt/d/Aiming:$PYTHONPATH"`

3. **ROS 2 워크스페이스 심볼릭 링크 연결:**
   * 윈도우 프로젝트 폴더와 WSL2 ROS 2 워크스페이스(~/ros2_ws/src/)를 심볼릭 링크로 연결하여 코드 수정 사항이 실시간으로 동기화되도록 구성합니다.
   * simulation/ciws_turret_aerial_object_detection_main/ciws_turret → ~/ros2_ws/src/ciws_turret
   * simulation/ciws_turret_aerial_object_detection_main/drone_sim → ~/ros2_ws/src/drone_sim

============================================================

### 5. 통합 마스터 런처를 통한 원클릭 실행 절차

Windows CMD 환경에서 `simulation/launchers/run_smash_fcs.bat` 를 실행하여 원하는 작업을 손쉽게 수행할 수 있습니다.

* **메뉴 번호별 주요 기능:**
  * **1번 (check):** WSL2 환경, ROS 2 Jazzy, Gazebo Harmonic, 필수 패키지 설치 여부 종합 진단
  * **2번 (link):** ROS 2 워크스페이스 src 디렉토리와 프로젝트 패키지 간 심볼릭 링크 자동 연결 (최초 1회 실행)
  * **3번 (build):** colcon build를 통한 패키지 컴파일 및 환경 소싱
  * **4번 (run):** 조준선 추종 전용 시뮬레이션 실행 (비사격 모드)
  * **5번 (manual):** 사수 수동 격발 모드 실행 (지정된 조건에서 격발 트리거 대기)
  * **6번 (auto):** 조준 정렬 일치(STATE: READY) 시 자동 격발 모드 실행
  * **8번 (viewer):** 대화형 스마트 스코프 뷰어 창 구동 (마우스 좌클릭 및 스페이스바 입력 시 인위적인 지연 없이 즉시 격발 연동)
