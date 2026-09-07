"""smash_scene.launch.py — turret world + 단일 표적 + SMASH FCS 노드.

기존 scene.launch.py 를 대체한다. 차이는 다음 두 가지다.

    1. drone_detector (단순 YOLO 탐지기) 와 turret_tracker (단순 비례제어 P 추종기)를
       실행하지 않는다. 두 노드는 FCS 노드와 같은 토픽
       (/turret_controller/commands, /turret_camera/image_annotated)을 발행하므로
       동시에 띄우면 명령이 충돌한다. 파일 자체는 저장소에 남겨 두었으므로
       필요하면 scene.launch.py 로 언제든 되돌릴 수 있다.
    2. 다중 표적(F16, 헬기, 조류) 대신 단일 정지 호버링 드론만 스폰한다.
       로드맵 5단계 1단계(정지 호버링 요격률 검증)의 기준 조건이며, 1단계
       통합 검증에서 변수를 최소화하기 위함이다. 다중 표적 환경이 필요하면
       spawn_extra_targets:=true 로 켠다.

표적 기본 배치에 대한 해석 주의:
    로드맵의 "35m 상공 정지 호버링" 을 사거리 약 35 m 로 해석해 수평 33 m,
    고도 12.5 m (경사거리 35.3 m, 앙각 약 20.7도) 에 놓았다. 포탑 앙각 한계가
    0 ~ 85도 이므로 표적을 바로 머리 위에 두면 한계 근처에서 동작하게 되어
    1단계 검증 조건으로 부적절하다. 다른 배치를 원하면 target_x / target_y /
    target_z 인자로 지정할 것.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('drone_sim')
    turret_pkg = get_package_share_directory('ciws_turret')
    params_file = os.path.join(pkg, 'config', 'smash_fcs.yaml')

    args = [
        DeclareLaunchArgument(
            'core_path', default_value=os.environ.get('SMASH_CORE_PATH', ''),
            description='aiming_engine / bridge 를 담고 있는 SMASH 코어 저장소 루트'),
        DeclareLaunchArgument(
            'model_path', default_value='',
            description='YOLO11s 파인튜닝 가중치(.pt) 경로. 예: <core>/models/best_finetuned.pt'),
        DeclareLaunchArgument('device', default_value='cpu'),
        DeclareLaunchArgument('muzzle_velocity', default_value='920.0'),
        DeclareLaunchArgument('enable_drag', default_value='true'),
        DeclareLaunchArgument('target_x', default_value='33.0'),
        DeclareLaunchArgument('target_y', default_value='0.0'),
        DeclareLaunchArgument('target_z', default_value='12.5'),
        DeclareLaunchArgument(
            'target_sdf', default_value=os.path.join(pkg, 'models', 'drone', 'model.sdf'),
            description='스폰할 표적 모델 SDF'),
        DeclareLaunchArgument('spawn_extra_targets', default_value='false'),
        DeclareLaunchArgument(
            'enable_fire_control', default_value='false',
            description='로드맵 3단계: 가상 탄환 격발 및 Hit/Kill 판정 활성화'),
        DeclareLaunchArgument(
            'ground_truth_pose_topic', default_value='/model/drone/pose',
            description='drone/model.sdf 의 PosePublisher 가 발행하는(가정) gz 토픽. '
                        'WSL2 에서 `gz topic -l` 로 실제 이름을 확인할 것'),
        DeclareLaunchArgument(
            'fire_mode', default_value='manual',
            description='"manual"(트리거 상승 엣지 격발) 또는 "auto"(READY 자동 격발)'),
    ]

    share_parent = os.path.dirname(pkg)
    existing = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    paths = ':'.join(p for p in [existing, share_parent, os.path.join(pkg, 'models')] if p)
    set_res = SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', paths)

    turret = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turret_pkg, 'launch', 'view.launch.py')))

    drone = Node(
        package='ros_gz_sim', executable='create', output='screen',
        arguments=['-file', LaunchConfiguration('target_sdf'), '-name', 'drone',
                   '-x', LaunchConfiguration('target_x'),
                   '-y', LaunchConfiguration('target_y'),
                   '-z', LaunchConfiguration('target_z')])

    def spawn_extra(name, rel_sdf, x, y, z):
        return Node(
            package='ros_gz_sim', executable='create', output='screen',
            condition=IfCondition(LaunchConfiguration('spawn_extra_targets')),
            arguments=['-file', os.path.join(pkg, 'models', rel_sdf, 'model.sdf'),
                       '-name', name, '-x', str(x), '-y', str(y), '-z', str(z)])

    extras = [
        spawn_extra('heli', 'heli', -8, 5, 7),
        spawn_extra('plane', 'plane', 9, -6, 8),
        spawn_extra('bird_1', 'bird', -8, 6, 5),
        spawn_extra('bird_2', 'bird', 7, -5, 6),
    ]

    fcs = Node(
        package='drone_sim', executable='smash_fcs', name='smash_fcs', output='screen',
        parameters=[
            params_file,
            {
                'use_sim_time': True,
                'core_path': LaunchConfiguration('core_path'),
                'model_path': LaunchConfiguration('model_path'),
                'device': LaunchConfiguration('device'),
                'muzzle_velocity_mps': LaunchConfiguration('muzzle_velocity'),
                'enable_drag': LaunchConfiguration('enable_drag'),
                'enable_fire_control': LaunchConfiguration('enable_fire_control'),
                'ground_truth_pose_topic': LaunchConfiguration('ground_truth_pose_topic'),
                'fire_mode': LaunchConfiguration('fire_mode'),
            },
        ],
    )

    # 로드맵 3단계: Hit/Kill 판정용 Gazebo 실측(ground truth) 위치 브리지.
    # view.launch.py 의 기존 브리지(카메라/레이저)와는 별개로 여기에 둔다 —
    # 3단계 전용 추가라 1/2단계가 쓰는 공용 파일(view.launch.py)을 건드리지
    # 않는다. drone/model.sdf 의 PosePublisher 플러그인이 실제로 이 토픽
    # 이름으로 발행하는지는 WSL2 에서 `gz topic -l` 로 확인해야 한다(모듈
    # docstring 참고). enable_fire_control 이 꺼져 있으면 이 브리지가 떠 있어도
    # FCS 노드가 구독하지 않으므로 무해하다.
    ground_truth_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        name='smash_ground_truth_bridge', output='screen',
        condition=IfCondition(LaunchConfiguration('enable_fire_control')),
        arguments=[[
            LaunchConfiguration('ground_truth_pose_topic'),
            '@geometry_msgs/msg/PoseStamped[gz.msgs.Pose',
        ]],
        parameters=[{'use_sim_time': True}],
    )

    delayed = TimerAction(period=6.0, actions=[drone] + extras + [fcs, ground_truth_bridge])

    return LaunchDescription(args + [set_res, turret, delayed])
