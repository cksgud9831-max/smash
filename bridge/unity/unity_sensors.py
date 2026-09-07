import time
import numpy as np
from typing import Optional

from bridge.range_sensor import RangeSensor, RangeSample
from bridge.pose_source import PoseSource, PoseSample
from .unity_server import UnityServer

class UnityRangeSensor(RangeSensor):
    """유니티에서 전송된 Raycast 거리를 반환하는 가상 레이저 거리계"""
    def __init__(self, server: UnityServer, timeout_s: float = 0.5):
        self.server = server
        self.timeout_s = timeout_s

    def read(self, timestamp: float) -> Optional[RangeSample]:
        _, distance, _, sample_timestamp = self.server.get_latest_data()
        
        if distance is None:
            return None
            
        is_stale = (time.time() - sample_timestamp) > self.timeout_s
        # 유니티에서 0 이하의 거리를 무효로 처리하도록 약속할 수 있음
        valid = (distance > 0.0) and not is_stale
        
        return RangeSample(
            distance_m=distance,
            timestamp=sample_timestamp,
            valid=valid
        )

    def close(self) -> None:
        pass


class UnityPoseSource(PoseSource):
    """유니티에서 전송된 카메라 Transform 행렬을 반환하는 가상 IMU/자세 센서"""
    def __init__(self, server: UnityServer, timeout_s: float = 0.5):
        self.server = server
        self.timeout_s = timeout_s

    def read(self, timestamp: float) -> PoseSample:
        _, _, pose_list, sample_timestamp = self.server.get_latest_data()
        
        if pose_list is None or len(pose_list) != 16:
            return PoseSample(
                extrinsic=np.eye(4, dtype=np.float64),
                timestamp=timestamp,
                valid=False
            )
            
        is_stale = (time.time() - sample_timestamp) > self.timeout_s
        
        # 유니티에서 보낸 16 float 배열을 4x4 변환 행렬로 재조립
        extrinsic = np.array(pose_list, dtype=np.float64).reshape(4, 4)
        
        return PoseSample(
            extrinsic=extrinsic,
            timestamp=sample_timestamp,
            valid=not is_stale
        )

    def close(self) -> None:
        pass
