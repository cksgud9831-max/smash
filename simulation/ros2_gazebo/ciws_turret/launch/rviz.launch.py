import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('ciws_turret')
    rviz_cfg = os.path.join(pkg, 'rviz', 'camera_check.rviz')

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', rviz_cfg],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )
    return LaunchDescription([rviz])
