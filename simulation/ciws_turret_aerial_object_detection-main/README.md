# ciws_turret_aerial_object_detection (SMASH 시뮬레이터용 축소판)

ROS 2 Jazzy + Gazebo Harmonic 위에서 CIWS 형상의 Pan/Tilt 포탑과 표적 드론을 띄우는 환경이다.
원본은 AJAXa239/ciws_turret_aerial_object_detection(MIT)이며, SMASH 자동 격발 시뮬레이터에
필요한 부분만 남겼다. 원본의 단순 YOLO 탐지기·비례제어 추종기·다중 표적(F-16, 헬기, 조류)·
조종 슬라이더는 제거했다(2026-10-13).

실행·구성·조작은 상위 폴더의 [`../README.md`](../README.md) 와
[`../environment_setup_guide.md`](../environment_setup_guide.md) 를 본다.

| 패키지 | 유형 | 내용 |
|---|---|---|
| `ciws_turret` | ament_cmake | 포탑 URDF/xacro, 메시, Gazebo 월드, ros2_control(pan + tilt), 카메라·레이저, RViz 카메라 점검 설정 |
| `drone_sim` | ament_python | 표적 드론 모델, SMASH FCS 노드, 포탑 구동 노드, 이동 표적(`target_mover`), 격추 연출(`demo_director`), 스코프 영상 중계 |
