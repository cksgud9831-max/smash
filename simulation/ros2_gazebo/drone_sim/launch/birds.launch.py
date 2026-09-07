import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('drone_sim')

    world = DeclareLaunchArgument('world', default_value='turret_world')
    count = DeclareLaunchArgument('count', default_value='4')

    share_parent = os.path.dirname(pkg)
    existing = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    paths = ':'.join(p for p in [existing, share_parent,
                                 os.path.join(pkg, 'models')] if p)
    set_res = SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', paths)

    model_sdf = os.path.join(pkg, 'models', 'bird', 'model.sdf')

    # spawn 4 birds at spread-out start positions
    starts = [(-8, 6, 5), (7, -5, 6), (-6, -7, 4), (9, 8, 7)]
    spawns = []
    for i, (x, y, z) in enumerate(starts, start=1):
        spawns.append(Node(
            package='ros_gz_sim', executable='create', output='screen',
            arguments=['-file', model_sdf, '-name', f'bird_{i}',
                       '-x', str(x), '-y', str(y), '-z', str(z)],
        ))

    wander = Node(
        package='drone_sim', executable='bird_wander', output='screen',
        parameters=[{
            'world': LaunchConfiguration('world'),
            'count': LaunchConfiguration('count'),
            'name_prefix': 'bird',
            'area': 12.0, 'z_min': 3.0, 'z_max': 8.0, 'speed': 2.0,
        }],
    )

    return LaunchDescription([world, count, set_res] + spawns + [wander])
