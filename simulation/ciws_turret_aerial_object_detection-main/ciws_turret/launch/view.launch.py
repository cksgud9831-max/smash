import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
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

    # 컨트롤러 전환 제한시간을 기본값(5초)에서 크게 늘린다.
    #
    # 왜 필요한가 (2026-09-12 실측): 컨트롤러 활성화는 Gazebo 내부의 제어 갱신
    # 루프에서 처리되는데, 4코어 WSL2 처럼 CPU 가 포화된 환경에서는 이 루프가
    # 밀려 5초를 넘긴다. 그러면 spawner 가 다음과 같이 실패하고 죽는다.
    #     [controller_manager] Switch controller timed out after 5 seconds!
    #     [spawner_joint_state_broadcaster] Failed to activate controller
    # 특히 joint_state_broadcaster 가 실패하면 증상이 조용하고 치명적이다.
    #     /joint_states 중단 -> robot_state_publisher 가 동적 TF(turret_link,
    #     gun_link) 발행 중단 -> TF 트리가 두 조각으로 분리 -> FCS 의
    #     base_footprint <- camera_link_optical 조회 실패 -> 상태가
    #     TRACK(NO TF) 에서 벗어나지 못하고 조준점(리드 레티클)이 아예 생성되지
    #     않는다. 토픽과 컨트롤러 목록은 정상으로 보여 원인 추적이 어렵다.
    # 실행할 때마다 성공/실패가 갈리는 것도 이 경쟁 때문이다.
    #
    # --service-call-timeout 도 같은 이유로 늘린다 (2026-09-15 실측).
    # spawner 는 서비스 응답을 기본 10초만 기다리고, 받지 못하면 같은 요청을
    # 다시 보낸다(최대 3회). 그런데 첫 요청은 사라진 것이 아니라 밀려 있을 뿐이라
    # 뒤늦게 처리되고, 재전송된 load 요청은 중복으로 거부되어 spawner 가 죽는다.
    #     [spawner_joint_state_broadcaster] Failed getting a result from calling
    #         /controller_manager/load_controller in 10.0. (Attempt 1 of 3.)
    #     [controller_manager] A controller named 'joint_state_broadcaster' was
    #         already loaded inside the controller manager
    #     [spawner_joint_state_broadcaster] Failed loading controller joint_state_broadcaster
    # 이때 jsb 는 unconfigured 로 남고 turret_controller 는 아예 로드되지 않으므로
    # 포탑이 명령을 전혀 따르지 않는다. 응답 대기를 길게 잡아 재전송 자체를 막는다.
    SPAWNER_TIMEOUTS = [
        '--controller-manager-timeout', '60',
        '--switch-timeout', '60',
        '--service-call-timeout', '60',
    ]

    # 두 컨트롤러를 한 번의 spawner 호출로 함께 올린다.
    #
    # 왜 (2026-09-13 실측): 이전에는 jsb 를 올린 뒤 그 프로세스가 "종료"하면
    # turret 을 올리는 2단 사슬이었다. 그런데 OnProcessExit 는 성공/실패를 가리지
    # 않으므로, jsb 가 실패해도 turret 은 그대로 올라간다. 실제로 사용자 실행에서
    # 다음 상태가 나왔다.
    #     turret_controller        active
    #     joint_state_broadcaster  unconfigured   <- 활성화 실패
    # 이러면 /joint_states 가 끊기고 robot_state_publisher 가 동적 TF(turret_link,
    # gun_link)를 못 내보내며, FCS 의 base_footprint <- camera_link_optical 조회가
    # 실패해 상태가 TRACK(NO TF) 에 갇힌다. 조준해가 없으니 격발도 거부된다
    # (사수가 방아쇠를 당겨도 SHOTS 가 0 에서 늘지 않는다). 포탑은 움직이고 탐지도
    # 되기 때문에 겉보기에는 멀쩡해 원인 추적이 어렵다.
    #
    # 한 번의 호출로 묶으면 switch_controller 서비스 왕복이 2회에서 1회로 줄어
    # CPU 포화 상태에서 경쟁에 걸릴 확률이 낮아지고, 실패할 때는 둘 다 실패하므로
    # "반쪽만 올라간 조용한 고장" 대신 로그에 분명히 드러난다.
    #
    # 그래도 실패하면 시뮬레이터를 내리지 않고 아래 명령으로 되살릴 수 있다.
    #     ros2 run controller_manager spawner joint_state_broadcaster turret_controller \
    #         -c /controller_manager --controller-manager-timeout 60 \
    #         --switch-timeout 60 --service-call-timeout 60
    # (이미 로드된 컨트롤러는 spawner 가 로드를 건너뛰고 설정·활성화만 이어서 한다.)
    controllers = Node(
        package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster', 'turret_controller',
                   '-c', '/controller_manager', *SPAWNER_TIMEOUTS],
        output='screen',
    )

    # /clock 은 영상 브리지와 분리해 전용 프로세스로 띄운다.
    #
    # 왜 (2026-09-12 실측): 한 parameter_bridge 에 /clock 과 640x640 영상,
    # 1089빔 PointCloud2 를 같이 물리면 시각 갱신이 대용량 메시지 뒤에 줄을 선다.
    # 그러면 sim 시계를 쓰는 노드의 now() 가 실제 시각보다 뒤처진다. FCS 에서
    # 실측(ground truth) 위치 스탬프 대비 최대 -0.106 초까지 벌어졌고, 그 결과
    # Hit/Kill 판정의 20~57 %가 "미래 데이터"로 오인되어 UNVERIFIED 로 버려졌다.
    # 시계는 지연에 민감하고 메시지가 아주 작으므로 따로 두는 것이 맞다.
    clock_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        name='clock_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen',
    )

    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        arguments=['/turret_camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
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
        rsp, clock_bridge, bridge, spawn,
        # 모델 스폰 직후는 Gazebo 가 메시와 센서를 올리느라 가장 바쁜 구간이라
        # 컨트롤러 활성화 서비스가 밀리기 쉽다. 3초 여유를 두고 올린다.
        RegisterEventHandler(OnProcessExit(
            target_action=spawn,
            on_exit=[TimerAction(period=3.0, actions=[controllers])],
        )),
    ])
