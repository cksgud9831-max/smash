"""smash_fcs.launch.py — SMASH FCS 통합 노드만 단독 기동한다.

이미 turret world 가 떠 있는 상태(ros2 launch ciws_turret view.launch.py 등)에서
FCS 노드만 붙일 때 쓴다. 월드와 표적까지 한 번에 띄우려면 smash_scene.launch.py 를
사용할 것.

주의: 기존 drone_detector / turret_tracker 를 함께 띄우면 안 된다. 두 노드 모두
/turret_controller/commands 와 /turret_camera/image_annotated 를 발행하므로
FCS 노드와 명령이 충돌한다.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('drone_sim')
    params_file = os.path.join(pkg, 'config', 'smash_fcs.yaml')

    core_path = DeclareLaunchArgument(
        'core_path',
        default_value=os.environ.get('SMASH_CORE_PATH', ''),
        description='aiming_engine / bridge 를 담고 있는 SMASH 코어 저장소 루트')
    model_path = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description='OpticalFlowTracker 가 쓸 YOLO11s 파인튜닝 가중치(.pt) 경로')
    device = DeclareLaunchArgument(
        'device', default_value='cpu', description='"cpu" 또는 GPU 인덱스 문자열(예: "0")')
    muzzle_velocity = DeclareLaunchArgument(
        'muzzle_velocity', default_value='880.0',
        description='포구초속 [m/s]. 기본값은 5.56x45mm M855 14.5인치 총열 기준')
    enable_drag = DeclareLaunchArgument(
        'enable_drag', default_value='true', description='공기저항 모델 사용 여부')

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
            },
        ],
    )

    return LaunchDescription(
        [core_path, model_path, device, muzzle_velocity, enable_drag, fcs])
