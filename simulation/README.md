# SMASH Gazebo 및 ROS 2 시뮬레이션 환경

본 디렉터리는 SMASH 대드론 사격 통제 알고리즘 검증을 위한 Gazebo 물리 시뮬레이션 및 ROS 2 패키지를 관리하는 공간입니다.

============================================================

### 디렉터리 구성

1. **ciws_turret_aerial_object_detection_main**: 시뮬레이터 핵심 ROS 2 패키지 모음
   * **ciws_turret**: 2축(Pan/Tilt) 대공 포탑 모델(URDF/Xacro), Gazebo 월드, RViz 설정 및 모터 제어기
   * **drone_sim**: 가상 드론 비행 시나리오 생성, Gazebo 센서 브릿지 및 시뮬레이션 검증 노드
   * **assets**: 3D 메쉬 및 시각화 리소스
2. **environment_setup_guide.md**: 팀원을 위한 WSL2, ROS 2 Jazzy, Gazebo Harmonic 설치 및 환경 구축 상세 가이드

============================================================

### 깃허브 협업 개발 워크플로우

1. **WSL2 환경 구성**: `environment_setup_guide.md` 문서를 참고하여 필수 패키지를 설치합니다.
2. **루트 런처 실행**: 프로젝트 루트의 `run_smash_fcs.bat`을 실행합니다.
   * 1번 (check): 환경 설치 상태 진단
   * 2번 (link): WSL2 워크스페이스와 `simulation` 내부 패키지 심볼릭 링크 자동 생성
   * 3번 (build): colcon build 수행
   * 4번~8번: 시뮬레이션 및 스마트 스코프 뷰어 실행
