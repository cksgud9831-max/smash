import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aiming_engine import AimingManager
from bridge.config import load_bridge_config
from bridge.frame_builder import TrackerFrameBuilder
from bridge.range_sensor import RangeSensor, RangeSample
from bridge.pose_source import PoseSource, PoseSample

class DummyRangeSensor(RangeSensor):
    def __init__(self, fixed_distance: float):
        self.fixed_distance = fixed_distance
    def read(self, timestamp: float) -> RangeSample:
        return RangeSample(timestamp=time.time(), distance_m=self.fixed_distance, valid=True)
    def close(self): pass

class DummyPoseSource(PoseSource):
    def read(self, timestamp: float) -> PoseSample:
        return PoseSample(timestamp=time.time(), extrinsic=np.eye(4, dtype=np.float64), valid=True)
    def close(self): pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="테스트할 동영상 파일 경로")
    parser.add_argument("--distance", type=float, default=50.0, help="가상으로 주입할 거리(m)")
    parser.add_argument("--bridge-config", type=str, default=None)
    parser.add_argument("--aiming-config", type=str, default=None)
    args = parser.parse_args()

    bridge_config = load_bridge_config(args.bridge_config)
    
    # 1. 가상 센서 (고정값)
    range_sensor = DummyRangeSensor(args.distance)
    pose_source = DummyPoseSource()
    
    frame_builder = TrackerFrameBuilder(bridge_config, range_sensor, pose_source)
    manager = AimingManager(config_path=args.aiming_config)

    # 2. 동영상 불러오기
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[Error] 동영상 파일을 열 수 없습니다: {args.video}")
        return

    print(f"동영상 테스트 시작! 가상 거리: {args.distance}m")
    frame_idx = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("동영상 재생이 끝났습니다.")
                break
                
            timestamp = time.time()
            
            # 조준 파이프라인 실행
            tracker_frame = frame_builder.build(frame, timestamp)
            
            if tracker_frame is not None:
                output = manager.update(tracker_frame)
                
                # 로그 출력 (15프레임마다)
                if frame_idx % 15 == 0:
                    print(f"[Video Pipeline] Frame {frame_idx} | State: {output.state.value} | "
                          f"Dist: {output.debug['distance_m']:.1f}m | Ready: {output.aim_ready}")
                          
                # 영상 위에 상태 출력
                cv2.putText(frame, f"State: {output.state.value}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                if frame_idx % 15 == 0:
                    print(f"[Video Pipeline] Frame {frame_idx} | State: SEARCH (Target not found)")
                cv2.putText(frame, "State: SEARCH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Video Aiming Test", frame)
            
            # 동영상 재생 속도 조절 (대략 30fps)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
                
            frame_idx += 1
            
    except KeyboardInterrupt:
        print("테스트 종료 명령 수신...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
