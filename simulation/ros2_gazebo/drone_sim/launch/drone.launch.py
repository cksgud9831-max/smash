import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('drone_sim')

    weights = DeclareLaunchArgument(
        'weights',
        default_value=os.path.expanduser('~/ros2_ws/src/ciws_project/drone_sim/weights/drone2_0.pt'),
        description='path to YOLO .pt weights')
    conf = DeclareLaunchArgument('conf', default_value='0.25')
    imgsz = DeclareLaunchArgument('imgsz', default_value='640',
                                  description='YOLO inference size (320 = faster on CPU)')
    track = DeclareLaunchArgument('track', default_value='false',
                                  description='true = turret auto-follows the target')
    world = DeclareLaunchArgument('world', default_value='turret_world')
    # which model to spawn/fly: default = built-in Gazebo X3 quadcopter (textured)
    drone_model = DeclareLaunchArgument(
        'drone_model',
        default_value=os.path.join(pkg, 'models', 'drone', 'model.sdf'),
        description='SDF of the model to fly (default: your floating STL drone)')

    # let Gazebo find the drone model + mesh
    share_parent = os.path.dirname(pkg)
    existing = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    paths = ':'.join(p for p in [existing, share_parent,
                                 os.path.join(pkg, 'models')] if p)
    set_res = SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', paths)

    spawn = Node(
        package='ros_gz_sim', executable='create', output='screen',
        arguments=[
            '-file', LaunchConfiguration('drone_model'),
            '-name', 'drone',
            '-x', '6.0', '-y', '0.0', '-z', '5.0',
        ],
    )

    flight = Node(
        package='drone_sim', executable='drone_flight', output='screen',
        parameters=[{
            'world': LaunchConfiguration('world'),
            'model': 'drone',
            'radius': 6.0,
            'altitude': 5.0,
            'period': 20.0,
        }],
    )

    detector = Node(
        package='drone_sim', executable='drone_detector', output='screen',
        parameters=[{
            'weights': LaunchConfiguration('weights'),
            'conf': LaunchConfiguration('conf'),
            'imgsz': LaunchConfiguration('imgsz'),
            'device': 'cpu',
        }],
    )

    # optional: turret auto-tracking (enable with track:=true)
    tracker = Node(
        package='drone_sim', executable='turret_tracker', output='screen',
        condition=IfCondition(LaunchConfiguration('track')),
        parameters=[{
            'image_w': 640, 'image_h': 640,
            'kp': 0.6, 'deadband': 0.03, 'max_rate': 1.2,
            'classes': '',   # '' = follow any aerial target
        }],
    )

    return LaunchDescription([weights, conf, imgsz, track, world, drone_model, set_res,
                              spawn, flight, detector, tracker])
