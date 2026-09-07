import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('drone_sim')
    turret_pkg = get_package_share_directory('ciws_turret')

    weights = DeclareLaunchArgument(
        'weights',
        default_value=os.path.expanduser('~/ros2_ws/src/ciws_project/drone_sim/weights/drone2_0.pt'))
    conf = DeclareLaunchArgument('conf', default_value='0.25')
    imgsz = DeclareLaunchArgument('imgsz', default_value='320')
    track = DeclareLaunchArgument('track', default_value='false')
    world_name = 'turret_world'

    # ---- resource paths: package models + built-in X3 fuel model ----
    x3_dir = os.path.expanduser(
        '~/ros2_ws/src/gz_fuel_models/fuel/fuel.ignitionrobotics.org/'
        'openrobotics/models/x3 uav/4')
    x3_sdf = os.path.join(x3_dir, 'model.sdf')
    share_parent = os.path.dirname(pkg)
    existing = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    paths = ':'.join(p for p in [existing, share_parent,
                                 os.path.join(pkg, 'models'), x3_dir] if p)
    set_res = SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', paths)

    # ---- 1) turret world (creates the sim + camera) ----
    turret = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turret_pkg, 'launch', 'view.launch.py')))

    # ---- helper builders ----
    def spawn(name, sdf, x, y, z):
        return Node(package='ros_gz_sim', executable='create', output='screen',
                    arguments=['-file', sdf, '-name', name,
                               '-x', str(x), '-y', str(y), '-z', str(z)])

    def wander(name, area, zmin, zmax, speed):
        return Node(package='drone_sim', executable='wander', output='screen',
                    name=f'wander_{name}',
                    parameters=[{'world': world_name, 'model': name,
                                 'area': area, 'z_min': zmin, 'z_max': zmax,
                                 'speed': speed}])

    drone_sdf = x3_sdf
    heli_sdf = os.path.join(pkg, 'models', 'heli', 'model.sdf')
    plane_sdf = os.path.join(pkg, 'models', 'plane', 'model.sdf')
    bird_sdf = os.path.join(pkg, 'models', 'bird', 'model.sdf')

    # ---- 2) spawn everything (delayed so the world is up first) ----
    spawns = [
        spawn('drone', drone_sdf, 6, 0, 6),
        spawn('heli', heli_sdf, -8, 5, 7),
        spawn('plane', plane_sdf, 9, -6, 8),
    ]
    bird_starts = [(-8, 6, 5), (7, -5, 6), (-6, -7, 4), (9, 8, 7)]
    for i, (x, y, z) in enumerate(bird_starts, start=1):
        spawns.append(spawn(f'bird_{i}', bird_sdf, x, y, z))

    # ---- 3) wander motion for each ----
    wanders = [
        wander('drone', 8.0, 4.0, 8.0, 3.0),
        wander('heli', 12.0, 5.0, 10.0, 4.0),
        wander('plane', 15.0, 6.0, 12.0, 6.0),
    ]
    bird_wander = Node(package='drone_sim', executable='bird_wander', output='screen',
                       parameters=[{'world': world_name, 'count': 4, 'name_prefix': 'bird',
                                    'area': 12.0, 'z_min': 3.0, 'z_max': 8.0, 'speed': 2.0}])

    # ---- 4) detector (+ optional tracker) ----
    detector = Node(package='drone_sim', executable='drone_detector', output='screen',
                    parameters=[{'weights': LaunchConfiguration('weights'),
                                 'conf': LaunchConfiguration('conf'),
                                 'imgsz': LaunchConfiguration('imgsz'), 'device': 'cpu'}])
    tracker = Node(package='drone_sim', executable='turret_tracker', output='screen',
                   condition=IfCondition(LaunchConfiguration('track')),
                   parameters=[{'image_w': 640, 'image_h': 640, 'kp': 0.6,
                                'deadband': 0.03, 'max_rate': 1.2, 'classes': ''}])

    # give the world ~6 s to come up before spawning + moving things
    delayed = TimerAction(period=6.0, actions=spawns + wanders + [bird_wander,
                                                                  detector, tracker])

    return LaunchDescription([weights, conf, imgsz, track, set_res, turret, delayed])
