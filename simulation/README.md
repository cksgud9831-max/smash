# SMASH FCS 대드론 시뮬레이션 통합 모듈 (simulation)

본 폴더는 SMASH 지능형 대드론 사격 통제 시스템의 3차원 물리 시뮬레이션, 대화형 조준기, 비전 추적 및 격발/격추 판정 관련 모든 소스 코드와 실행 환경을 체계적으로 통합 관리하는 디렉토리입니다.

============================================================

### 1. 폴더 구조 및 구성 요소

1. **ciws_turret_aerial_object_detection_main/ (참고 오픈소스 및 Gazebo/ROS 2 3D 물리 환경 전체):**
   * **ciws_turret/:** 2축(Pan/Tilt) 대공 포탑 URDF/xacro 모델, Gazebo 센서 플러그인(카메라 및 레이저 LiDAR), 물리 월드(turret_world.sdf), 제어기 설정(controllers.yaml) 및 런치 파일.
   * **drone_sim/:** SMASH FCS 핵심 노드(smash_fcs_node.py), 대화형 스마트 스코프 뷰어(smash_scope_viewer.py), 3차원 비행 타깃(드론, 비행기, 헬기, 조류) 모델 및 런치 파이프라인(smash_scene.launch.py).
   * **assets/ 및 3D 모델 메쉬:** 비행체 및 포탑 3D 메쉬/텍스처 에셋 전체.

2. **실행 진입점 (저장소 루트 및 scripts/ 에 위치):**
   * **../run_smash_fcs.bat:** Windows CMD 환경에서 환경 진단, 빌드, 조준, 수동/자동 격발, 대화형 스코프 뷰어를 원클릭으로 구동하는 마스터 배치 파일.
   * **../scripts/wsl_setup_smash_fcs.sh:** WSL2 Ubuntu_24.04 환경 내부에서 ROS 2 Jazzy, Gazebo Harmonic, 심볼릭 링크 및 가상환경 경로를 자동 관리하는 스크립트. run_smash_fcs.bat 이 이 경로를 직접 호출한다.

============================================================

### 2. 주요 실행 방법

1. **Gazebo Harmonic + ROS 2 3D 포탑 시뮬레이터 실행:**
   * 저장소 루트의 run_smash_fcs.bat 실행 후 메뉴 선택:
     * 1번: 환경 진단 (check)
     * 2번: 워크스페이스 심볼릭 링크 연결 (link, 최초 1회)
     * 3번: 패키지 빌드 (build)
     * 4번: 조준 전용 시뮬레이션 실행 (run)
     * 5번: 사수 수동 격발 모드 실행 (manual)
     * 6번: READY 정렬 시 자동 격발 모드 실행 (auto)
     * 8번: 대화형 스마트 스코프 뷰어 창 실행 (마우스 좌클릭 / 스페이스바 즉시 격발)

2. **ROS 2 / Gazebo 없이 격발·판정 로직만 검증:**
   * run_smash_fcs.bat 7번(오라클 검증) 또는
     python scripts/08_stage1_fcs_offline_validation.py 실행.
