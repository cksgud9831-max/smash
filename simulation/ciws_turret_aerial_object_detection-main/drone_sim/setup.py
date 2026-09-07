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
        (os.path.join('share', package_name, 'models', 'bird'),
            glob('models/bird/*')),
        (os.path.join('share', package_name, 'models', 'plane'),
            glob('models/plane/*')),
        (os.path.join('share', package_name, 'models', 'heli'),
            glob('models/heli/*')),
        (os.path.join('share', package_name, 'weights'), glob('weights/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ajay',
    maintainer_email='ajay@example.com',
    description='Flying drone target + YOLO11 detector for the CIWS turret.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'drone_flight = drone_sim.drone_flight:main',
            'drone_detector = drone_sim.drone_detector:main',
            'turret_tracker = drone_sim.turret_tracker:main',
            'camera_zoom = drone_sim.camera_zoom:main',
            'bird_wander = drone_sim.bird_wander:main',
            'wander = drone_sim.wander:main',
            # SMASH 지능형 사격통제 통합 노드 (drone_detector + turret_tracker 대체)
            'smash_fcs = drone_sim.smash_fcs_node:main',
            # SMASH 대화형 스코프 뷰어 (마우스 좌클릭 및 스페이스바 격발 지원)
            'smash_scope_viewer = drone_sim.smash_scope_viewer:main',
        ],
    },
)
