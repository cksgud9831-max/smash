import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command


def generate_launch_description():
    pkg = get_package_share_directory('ciws_turret')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    xacro_file = os.path.join(pkg, 'urdf', 'ciws_turret.urdf.xacro')
    world_file = os.path.join(pkg, 'worlds', 'turret_world.sdf')

    # let Gazebo resolve package:// mesh URIs
    share_parent = os.path.dirname(pkg)
    # also let this Gazebo resolve package:// meshes from drone_sim (the drone
    # gets spawned into THIS world, so its meshes must be on THIS path)
    extra = []
    try:
        from ament_index_python.packages import get_package_share_directory as _gp
        extra.append(os.path.dirname(_gp('drone_sim')))
    except Exception:
        pass
    parts = [p for p in [os.environ.get('GZ_SIM_RESOURCE_PATH', ''),
                         share_parent] + extra if p]
    gz_path = ':'.join(parts)
    set_resource_path = SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', gz_path)

    # Gazebo Sim System Plugin Path for gz_ros2_control
    existing_plugin_path = os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', '')
    plugin_paths = ':'.join([p for p in ['/opt/ros/jazzy/lib', existing_plugin_path] if p])
    set_plugin_path = SetEnvironmentVariable('GZ_SIM_SYSTEM_PLUGIN_PATH', plugin_paths)

    robot_description = ParameterValue(Command(['xacro ', xacro_file]), value_type=str)

    # headless:=true 면 Gazebo GUI 없이 서버만 띄운다(gz sim -s).
    # WSLg 처럼 GUI 렌더링이 안 되는 환경에서도 카메라 센서는 정상 동작하므로,
    # 화면은 smash_scope_viewer(OpenCV 창)로 확인하면 된다.
    headless = LaunchConfiguration('headless')
    gz_launch = os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')

    gz_sim_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_launch),
        launch_arguments={'gz_args': f'-r -v 3 {world_file}'}.items(),
        condition=UnlessCondition(headless),
    )
    gz_sim_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_launch),
        launch_arguments={'gz_args': f'-s -r -v 3 {world_file}'}.items(),
        condition=IfCondition(headless),
    )

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
    )

    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', 'robot_description', '-name', 'ciws_turret', '-z', '0.05'],
    )

    jsb = Node(
        package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster', '-c', '/controller_manager'],
        output='screen',
    )
    turret = Node(
        package='controller_manager', executable='spawner',
        arguments=['turret_controller', '-c', '/controller_manager'],
        output='screen',
    )

    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                   '/turret_camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
                   '/turret_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
                   # SMASH FCS 동축 광각 레이저 거리측정기 (ciws_turret.urdf.xacro 참고).
                   # 2차원 격자가 필요하므로 LaserScan(가운데 행)이 아니라
                   # PointCloud2(전체 격자)를 브리지한다.
                   '/turret_camera/boresight_range/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked'],
        parameters=[{'use_sim_time': True}], output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'headless', default_value='false',
            description='true 면 Gazebo GUI 없이 서버만 실행 (gz sim -s)'),
        set_resource_path, set_plugin_path, gz_sim_gui, gz_sim_headless,
        rsp, bridge, spawn,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[turret])),
    ])
