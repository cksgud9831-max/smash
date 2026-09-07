"""SMASH FCS 지원 모듈: 포탑 기구학, 거리 추정, 조준 코어 부트스트랩.

ROS 2 의존성이 없는 순수 파이썬 모듈만 모아 두었다. 덕분에 ROS 없이도
scripts/08_stage1_fcs_offline_validation.py 로 단독 검증할 수 있다.
"""

from . import range_estimator, turret_kinematics

__all__ = ["turret_kinematics", "range_estimator"]
