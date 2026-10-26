"""smash_scene.launch.py — turret world + 단일 표적 + SMASH FCS 노드.

참고 오픈소스의 단순 YOLO 탐지기·비례제어 추종기·다중 표적(F16, 헬기, 조류) 장면은
제거했다(2026-10-13, 필요하면 git 이력 참고). 단일 드론 표적만 스폰한다.

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
    DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction,
    SetEnvironmentVariable, SetLaunchConfiguration, TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def _apply_operating_envelope(context, *_args, **_kwargs):
    """표적 중심과 재출현 위치를 현재 화각의 포착 가능 거리 안으로 당긴다.

    탐지기는 표적이 화면에서 약 20 px 미만이면 최초 포착이 불안정하다(smash_fcs/operating_envelope.py).
    범위 밖에 두면 시연이 "표적을 못 잡는" 상태로 멈추므로 시작 전에 조정하고 알린다.
    enforce_envelope:=false 로 끄면 범위 밖 시험(한계 측정 등)을 할 수 있다.
    """

    from drone_sim.smash_fcs import operating_envelope as env

    cfg = context.launch_configurations
    hfov = float(cfg['camera_hfov'])
    px = float(cfg['camera_px'])
    limit = env.max_range_m(hfov, px)
    head = f'[envelope] 화각 {hfov:.2f} rad 의 포착 가능 거리 {limit:.1f} m (표적 {env.MIN_TARGET_PX:.0f} px 이상)'
    if str(cfg.get('enforce_envelope', 'true')).lower() != 'true':
        return [LogInfo(msg=head + ' - enforce_envelope:=false, 제한하지 않음')]

    actions = [LogInfo(msg=head)]
    motion = str(cfg.get('target_motion', 'none'))
    center = (float(cfg['target_x']), float(cfg['target_y']), float(cfg['target_z']))
    new, moved = env.clamp_to_envelope(
        center, limit, motion=motion,
        half_length_m=float(cfg.get('target_half_length', 0.0)),
        half_height_m=float(cfg.get('target_half_height', 0.0)))
    if moved:
        actions += [
            SetLaunchConfiguration('target_x', f'{new[0]:.3f}'),
            SetLaunchConfiguration('target_y', f'{new[1]:.3f}'),
            SetLaunchConfiguration('target_z', f'{new[2]:.3f}'),
            LogInfo(msg='[envelope] 표적 중심 ({:.1f}, {:.1f}, {:.1f}) -> ({:.1f}, {:.1f}, {:.1f}) 로 조정'.format(
                *center, *new)),
        ]
    flat = env.parse_flat_positions(str(cfg.get('respawn_positions', '')))
    if flat and len(flat) >= 3:
        clamped, changed = env.clamp_flat_positions(flat, limit)
        if changed:
            actions += [
                SetLaunchConfiguration('respawn_positions', '[' + ', '.join(f'{v:.3f}' for v in clamped) + ']'),
                LogInfo(msg=f'[envelope] 재출현 위치 {len(changed)}곳을 포착 가능 거리 안으로 조정'),
            ]
    return actions


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
        DeclareLaunchArgument(
            'headless', default_value='true',
            description='true 면 Gazebo GUI 없이 서버만 실행. 화면은 smash_scope_viewer 로 확인'),
        DeclareLaunchArgument('device', default_value='cuda:0'),
        DeclareLaunchArgument(
            'camera_px', default_value='640',
            description='조준 카메라 해상도(정사각, 화소). ciws_turret view.launch.py 로 전달'),
        DeclareLaunchArgument(
            'camera_hfov', default_value='0.4',
            description='조준 카메라 수평 화각(rad). ciws_turret view.launch.py 로 전달'),
        DeclareLaunchArgument(
            'enable_lidar', default_value='true',
            description='false면 진단을 위해 Gazebo GPU LiDAR를 생성하지 않음'),
        DeclareLaunchArgument('muzzle_velocity', default_value='880.0'),
        DeclareLaunchArgument('enable_drag', default_value='true'),
        DeclareLaunchArgument('target_x', default_value='10.0'),
        DeclareLaunchArgument('target_y', default_value='0.0'),
        DeclareLaunchArgument('target_z', default_value='4.2'),
        DeclareLaunchArgument(
            'target_sdf', default_value=os.path.join(pkg, 'models', 'drone', 'model.sdf'),
            description='스폰할 표적 모델 SDF'),
        DeclareLaunchArgument(
            'enable_fire_control', default_value='true',
            description='가상 탄환 자동 격발 및 Hit/Kill 판정. false 면 탐지·조준만 한다'),
        DeclareLaunchArgument(
            'ground_truth_pose_topic', default_value='/model/drone/pose',
            description='drone/model.sdf 의 PosePublisher 가 발행하는(가정) gz 토픽. '
                        'WSL2 에서 `gz topic -l` 로 실제 이름을 확인할 것'),
        DeclareLaunchArgument(
            'maneuver_reset', default_value='true',
            description='표적 추정기 기동 감지(방향 전환 시 기록 비우기). A/B 비교용'),
        DeclareLaunchArgument(
            'latency_compensation', default_value='true',
            description='조준해를 발사 시각 기준으로 풀지(true) 촬영 시각 기준으로 풀지(false). 보정 전후 비교용'),
        DeclareLaunchArgument(
            'demo_kill', default_value='false',
            description='격발 제어가 켜져 있을 때, 명중하면 표적을 떨어뜨리고 재출현시키는 시연 연출'),
        # 기본 재출현 위치는 시야 밖(좌우 약 9도)과 거리를 섞어 둔다. 같은 자리에 다시
        # 띄우면 포탑이 이미 그곳을 겨누고 있어 재조준 없이 곧바로 격추되므로 시연이
        # 단조롭다. 순서: 18.3 m 좌측 -> 14 m 우측 -> 17 m 정면 -> 9.6 m 정면(world +y 가 왼쪽).
        # 포는 재출현 때 그 위치로 먼저 돌려 둔다(cue_cmd). 모두 광각(0.4 rad) 포착 가능 거리
        # 19 m 안이다. 더 멀면 표적이 화면에서 28 px 미만이 되어 탐지기가 잡지 못하는 경우가 생긴다
        # (operating_envelope.py). 범위 밖 값은 enforce_envelope 가 켜져 있으면 시작 시 당겨 온다.
        # 앙각은 모두 약 18~20도로 포탑 앙각 범위 안이다. [0.0] 이면 처음 자리로 돌아온다.
        DeclareLaunchArgument(
            'respawn_positions',
            default_value='[17.0, 2.5, 6.3, 13.0, -2.0, 4.8, 16.0, 0.0, 5.9, 9.0, 0.0, 3.4]',
            description='재출현 위치 목록 [x,y,z, x,y,z, ...] (world). [0.0] 이면 처음 자리'),
        # 이동 표적 (target_mover.py). 'none' 이면 정지 호버링. 이동 시험의 기본은 demo_kill=false라
        # 첫 명중 뒤 시야 밖 재출현으로 장기 측정이 끊기지 않는다. 연출이 필요할 때만 명시적으로 켠다.
        DeclareLaunchArgument(
            'target_motion', default_value='none',
            description="'none' | 'straight' (좌우 직선 통과 반복) | 'cross' (좌우 등속 왕복) | "
                        "'maneuver' (8자 기동, 속력 일정) | 'approach' (정면 접근 반복). 중심은 target_x/y/z"),
        DeclareLaunchArgument('target_speed', default_value='5.0',
                              description='이동 속력 (m/s). target_speed_kmh 가 양수면 무시된다'),
        DeclareLaunchArgument('target_speed_kmh', default_value='-1.0',
                              description='이동 속력 (km/h). 양수면 target_speed 대신 쓴다. 예: 30'),
        DeclareLaunchArgument('target_half_length', default_value='6.0', description='좌우 반폭 (m). approach 는 중심에서 앞뒤로 움직이는 거리'),
        DeclareLaunchArgument('target_half_height', default_value='1.8',
                              description="상하 반폭 (m, 'maneuver' 만). 작을수록 좌우 끝 선회가 급하다"),
        DeclareLaunchArgument('target_start_delay', default_value='-1.0',
                              description='첫 모의 시각부터 출발까지 (s). 음수면 /target_mover/start 신호를 기다린다'),
        DeclareLaunchArgument('target_move_rate', default_value='100.0', description='set_pose 호출 주기 (Hz)'),
        DeclareLaunchArgument(
            'target_control', default_value='false',
            description="true 면 target_motion 이 'none' 이어도 target_mover 를 띄워, 스코프 뷰어 키로 실행 중 "
                        "기동 방식(0~4)·속도(+/-)·정지(M)를 바꿀 수 있게 한다"),
        DeclareLaunchArgument(
            'target_start_on_lock', default_value='false',
            description='true 면 이동 표적이 FCS 추적 신뢰도 STABLE 이 1초 유지된 뒤 출발한다(그전엔 중심에서 호버링)'),
        DeclareLaunchArgument(
            'enforce_envelope', default_value='true',
            description='true 면 표적·재출현 위치를 화각별 포착 가능 거리(운용 범위) 안으로 제한한다. '
                        '범위 밖 시험을 하려면 false'),
        DeclareLaunchArgument(
            'launch_viewer', default_value='false',
            description='true 면 대화형 스코프 뷰어(smash_scope_viewer)를 자동으로 함께 기동'),
        DeclareLaunchArgument(
            'publish_annotated', default_value='true',
            description='진단용 HUD 영상 발행. false면 제어 파이프라인만 실행'),
        DeclareLaunchArgument(
            'launch_fcs', default_value='true',
            description='false면 장면·카메라·포탑 제어만 띄우고 FCS/YOLO는 실행하지 않음'),
        DeclareLaunchArgument(
            'spawn_target', default_value='true',
            description='false면 진단을 위해 드론 표적 모델을 스폰하지 않음'),
        DeclareLaunchArgument(
            'launch_turret_command', default_value='true',
            description='false면 진단을 위해 60 Hz 포탑 명령 노드를 실행하지 않음'),
        DeclareLaunchArgument(
            'turret_command_rate', default_value='60.0',
            description='진단용 포탑 명령 발행 주기(Hz)'),
        DeclareLaunchArgument(
            'aim_max_goal_jump_rad', default_value='0.012',
            description='검출기 갱신 한 번에 허용하는 포 목표각 변화 한도(rad). 시험 비교용'),
        DeclareLaunchArgument(
            'operator_follow', default_value='false',
            description='true 면 사수 에뮬레이션: 표적 실측 위치로 포를 돌려 표적을 화면 중심 부근에 둔다(지향에만 사용)'),
        DeclareLaunchArgument(
            'operator_lead_s', default_value='0.16',
            description='사수 에뮬레이션이 표적 앞을 겨냥하는 시간(s). 0 이면 표적 위치 그대로'),
        DeclareLaunchArgument(
            'trigger_mode', default_value='auto',
            description='격발 방식. auto=READY 면 자동, manual=READY 일 때 사용자가 클릭/스페이스로 한 번에 한 발'),
        DeclareLaunchArgument(
            'motion_measurement_lag_s', default_value='0.17',
            description='FCS 가 이동 표적 리드에 더하는 영상/추적 위치 지연(s). 기본 0.17 은 smash_fcs.yaml 값과 같다. 시뮬 비교용'),
        DeclareLaunchArgument(
            'motion_ready_max_fit_rmse_m', default_value='0.20',
            description='이동 표적 궤적 적합 잔차가 이 값(m)을 넘으면 READY 를 막는다("MOTION FIT"). 기본 0.20 은 '
                        'smash_fcs.yaml 값과 같다. 곡선 기동(8자) 시험 비교용이며 실장비 값이 아니다'),
        DeclareLaunchArgument(
            'motion_ready_max_velocity_disagreement_mps', default_value='0.75',
            description='장기 OLS 속도와 최근 6프레임 속도의 차가 이 값(m/s)을 넘으면 READY 를 막는다("MOTION FIT"). '
                        '기본 0.75 는 smash_fcs.yaml 값과 같다. 곡선 기동(8자) 시험 비교용이며 실장비 값이 아니다'),
        DeclareLaunchArgument(
            'operator_fcs_lead', default_value='true',
            description='true 면 사수 에뮬레이션이 조준선을 FCS 리드점에 올린다(표적 방향 + 리드-bbox 각도). 격발은 정렬 때만 나간다'),
        DeclareLaunchArgument(
            'operator_lateral_m', default_value='0.21',
            description='사수 에뮬레이션의 좌우 평행 보정(m). 포탑 역기구학 기본 상수(GUN_TO_CAMERA_OPTICAL_M 좌우 0.21)와 URDF(0)의 차이를 상쇄한다'),
        DeclareLaunchArgument(
            'aim_acquire_envelope_rad', default_value='0.08',
            description='포 잠금(READY 연속)이 확정되기 전에 시작 자세에서 허용하는 최대 이탈각(rad). 시험 비교용'),
        DeclareLaunchArgument(
            'aim_follow_mode', default_value='lead',
            description='포 추종 입력. lead(기본)=리드점, bbox=bbox 중심 + 평활한 리드 성분(시험용)'),
        DeclareLaunchArgument(
            'aim_offset_smoothing', default_value='0.8',
            description='bbox 모드에서 (리드-bbox) 각도 차에 거는 EMA 이전값 가중치'),
        DeclareLaunchArgument(
            'aim_follower_rate_smoothing', default_value='0.0',
            description='피드포워드에 쓰는 팬 각속도의 EMA 이전값 가중치(0=원시값)'),
        DeclareLaunchArgument(
            'pan_dither_rad', default_value='0.0005',
            description='gz_ros2_control 정지 방지용 pan dither 진폭(rad)'),
    ]

    share_parent = os.path.dirname(pkg)
    existing = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    paths = ':'.join(p for p in [existing, share_parent, os.path.join(pkg, 'models')] if p)
    set_res = SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', paths)

    turret = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turret_pkg, 'launch', 'view.launch.py')),
        launch_arguments={'headless': LaunchConfiguration('headless'),
                          'camera_px': LaunchConfiguration('camera_px'),
                          'camera_hfov': LaunchConfiguration('camera_hfov'),
                          'enable_lidar': LaunchConfiguration('enable_lidar')}.items())

    drone = Node(
        package='ros_gz_sim', executable='create', output='screen',
        condition=IfCondition(LaunchConfiguration('spawn_target')),
        arguments=['-file', LaunchConfiguration('target_sdf'), '-name', 'drone',
                   '-x', LaunchConfiguration('target_x'),
                   '-y', LaunchConfiguration('target_y'),
                   '-z', LaunchConfiguration('target_z')])

    fcs = Node(
        package='drone_sim', executable='smash_fcs', name='smash_fcs', output='screen',
        condition=IfCondition(LaunchConfiguration('launch_fcs')),
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
                'latency_compensation': LaunchConfiguration('latency_compensation'),
                'maneuver_reset': LaunchConfiguration('maneuver_reset'),
                'trigger_mode': LaunchConfiguration('trigger_mode'),
                'motion_measurement_lag_s': PythonExpression(['float(', LaunchConfiguration('motion_measurement_lag_s'), ')']),
                'motion_ready_max_fit_rmse_m': PythonExpression(
                    ['float(', LaunchConfiguration('motion_ready_max_fit_rmse_m'), ')']),
                'motion_ready_max_velocity_disagreement_mps': PythonExpression(
                    ['float(', LaunchConfiguration('motion_ready_max_velocity_disagreement_mps'), ')']),
                'publish_annotated': LaunchConfiguration('publish_annotated'),
                'auto_fire_requires_target_move': PythonExpression([
                    "'", LaunchConfiguration('target_motion'), "' != 'none'"
                ]),
            },
        ],
    )

    # Gazebo 포탑 구동은 YOLO/FCS와 별도 프로세스에서 실행한다. 그래야 영상 추론이
    # 느려져도 관절 명령 주기가 함께 멈추지 않는다.
    turret_command = Node(
        package='drone_sim', executable='turret_command', name='turret_command', output='screen',
        condition=IfCondition(LaunchConfiguration('launch_turret_command')),
        parameters=[params_file, {
            'use_sim_time': True,
            'command_rate_hz': LaunchConfiguration('turret_command_rate'),
            'pan_dither_rad': LaunchConfiguration('pan_dither_rad'),
            'aim_max_goal_jump_rad': LaunchConfiguration('aim_max_goal_jump_rad'),
            'aim_acquire_envelope_rad': LaunchConfiguration('aim_acquire_envelope_rad'),
            'aim_follow_mode': LaunchConfiguration('aim_follow_mode'),
            'aim_offset_smoothing': LaunchConfiguration('aim_offset_smoothing'),
            'aim_follower_rate_smoothing': LaunchConfiguration('aim_follower_rate_smoothing'),
        }],
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

    # 시연 연출: 명중 시 격추 낙하 + 재출현 (demo_director.py). 격발 제어가 켜져 있어야
    # 명중 판정이 나오므로 두 인자가 모두 참일 때만 띄운다. 표적 이동은 Gazebo 의
    # set_pose 서비스로 하므로 그 서비스만 따로 브리지한다.
    demo_on = IfCondition(PythonExpression([
        "'", LaunchConfiguration('enable_fire_control'), "' == 'true' and '",
        LaunchConfiguration('demo_kill'), "' == 'true'"]))
    motion_on = IfCondition(PythonExpression([
        "'", LaunchConfiguration('target_motion'), "' != 'none' or '",
        LaunchConfiguration('target_control'), "' == 'true'"]))
    set_pose_needed = IfCondition(PythonExpression([
        "('", LaunchConfiguration('enable_fire_control'), "' == 'true' and '",
        LaunchConfiguration('demo_kill'), "' == 'true') or '",
        LaunchConfiguration('target_motion'), "' != 'none' or '",
        LaunchConfiguration('target_control'), "' == 'true'"]))
    set_pose_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', name='set_pose_bridge',
        output='screen', condition=set_pose_needed,
        arguments=['/world/turret_world/set_pose@ros_gz_interfaces/srv/SetEntityPose'],
    )
    director = Node(
        package='drone_sim', executable='demo_director', name='demo_director',
        output='screen', condition=demo_on,
        parameters=[{
            'use_sim_time': True,
            'ground_truth_pose_topic': LaunchConfiguration('ground_truth_pose_topic'),
            'respawn_positions': LaunchConfiguration('respawn_positions'),
        }],
    )

    mover = Node(
        package='drone_sim', executable='target_mover', name='target_mover',
        output='screen', condition=motion_on,
        parameters=[{
            'use_sim_time': True,
            'motion': LaunchConfiguration('target_motion'),
            'center_x': PythonExpression(['float(', LaunchConfiguration('target_x'), ')']),
            'center_y': PythonExpression(['float(', LaunchConfiguration('target_y'), ')']),
            'center_z': PythonExpression(['float(', LaunchConfiguration('target_z'), ')']),
            'speed_mps': PythonExpression([
                'float(', LaunchConfiguration('target_speed_kmh'), ') / 3.6 if float(',
                LaunchConfiguration('target_speed_kmh'), ') > 0 else float(',
                LaunchConfiguration('target_speed'), ')']),
            'half_length_m': PythonExpression(['float(', LaunchConfiguration('target_half_length'), ')']),
            'half_height_m': PythonExpression(['float(', LaunchConfiguration('target_half_height'), ')']),
            'start_delay_s': PythonExpression(['float(', LaunchConfiguration('target_start_delay'), ')']),
            'rate_hz': PythonExpression(['float(', LaunchConfiguration('target_move_rate'), ')']),
            'start_on_lock': LaunchConfiguration('target_start_on_lock'),
        }],
    )

    viewer_node = Node(
        package='drone_sim', executable='smash_scope_viewer',
        name='smash_scope_viewer', output='screen',
        condition=IfCondition(LaunchConfiguration('launch_viewer')),
        parameters=[{
            'trigger_mode': LaunchConfiguration('trigger_mode'),
            # 즉시 생성 버튼은 demo_director 가 떠 있을 때(demo_on 과 같은 조건)만 동작한다
            'spawn_enabled': PythonExpression([
                "'", LaunchConfiguration('enable_fire_control'), "' == 'true' and '",
                LaunchConfiguration('demo_kill'), "' == 'true'"]),
        }],
    )

    operator = Node(
        package='drone_sim', executable='operator_follow', name='operator_follow', output='screen',
        condition=IfCondition(LaunchConfiguration('operator_follow')),
        parameters=[{
            'use_sim_time': True,
            'ground_truth_pose_topic': LaunchConfiguration('ground_truth_pose_topic'),
            'lead_s': PythonExpression(['float(', LaunchConfiguration('operator_lead_s'), ')']),
            'lateral_offset_m': PythonExpression(['float(', LaunchConfiguration('operator_lateral_m'), ')']),
            'fcs_lead_follow': LaunchConfiguration('operator_fcs_lead'),
        }],
    )

    delayed = TimerAction(period=6.0, actions=[drone, turret_command, fcs, ground_truth_bridge, set_pose_bridge, director, mover, viewer_node, operator])

    return LaunchDescription(args + [OpaqueFunction(function=_apply_operating_envelope), set_res, turret, delayed])
