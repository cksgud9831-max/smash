from setuptools import setup
import os
from glob import glob

package_name = 'drone_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name, package_name + '.smash_fcs'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
        (os.path.join('share', package_name, 'models', 'drone'),
            glob('models/drone/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ajay',
    maintainer_email='ajay@example.com',
    description='SMASH FCS simulation: target drone, FCS node and turret driver for the CIWS turret.',
    license='MIT',
    entry_points={
        'console_scripts': [
            # SMASH 지능형 사격통제 통합 노드
            'smash_fcs = drone_sim.smash_fcs_node:main',
            # Gazebo 포탑 구동 전용 프로세스 (영상 추론과 제어 주기 분리)
            'turret_command = drone_sim.turret_command_node:main',
            # SMASH 스코프 영상 중계 서버 (HTTP MJPEG 9999, 뷰어 명령 중계)
            'smash_scope_viewer = drone_sim.smash_scope_viewer:main',
            # 시연 연출: 명중 시 표적 격추(낙하) 후 재출현
            'demo_director = drone_sim.demo_director:main',
            # 이동 표적 시나리오 (지연 보정 검증, 로드맵 4단계 선행)
            'target_mover = drone_sim.target_mover:main',
            # 사수 에뮬레이션: 표적 실측 위치로 포를 돌려 표적을 화면 중심 부근에 둔다
            'operator_follow = drone_sim.operator_follow:main',
        ],
    },
)
