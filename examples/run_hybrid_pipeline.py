import argparse
import sys
import time
import socket
import threading
import json
import struct
from pathlib import Path
import cv2
import numpy as np

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

# 글로벌 클라이언트 소켓 변수 (유니티 전송용)
unity_client = None

def tcp_server_loop(host='0.0.0.0', port=8080):
    global unity_client
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(1)
    print(f"[Hybrid Server] 유니티 클라이언트 접속 대기 중... ({host}:{port})")
    
    while True:
        try:
            conn, addr = server.accept()
            print(f"[Hybrid Server] 유니티 클라이언트가 연결되었습니다! {addr}")
            unity_client = conn
        except Exception as e:
            print(f"Server error: {e}")
            break

def send_to_unity(output):
    global unity_client
    if unity_client is None:
        return
        
    azimuth = output.aim_solution.azimuth if output.aim_solution else 0.0
    elevation = output.aim_solution.elevation if output.aim_solution else 0.0
    feedback_data = {
        "state": output.state.value,
        "aim_azimuth": azimuth,
        "aim_elevation": elevation,
        "aim_ready": output.aim_ready
    }
    
    try:
        json_str = json.dumps(feedback_data)
        json_bytes = json_str.encode('utf-8')
        # 4바이트 리틀엔디안으로 길이 패킹
        len_bytes = struct.pack('<I', len(json_bytes))
        
        unity_client.sendall(len_bytes + json_bytes)
    except Exception as e:
        print(f"[Hybrid Server] 유니티 전송 오류, 연결을 끊습니다. ({e})")
        unity_client.close()
        unity_client = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="테스트할 동영상 파일 경로")
    parser.add_argument("--distance", type=float, default=50.0, help="가상 거리(m)")
    args = parser.parse_args()

    bridge_config = load_bridge_config()
    range_sensor = DummyRangeSensor(args.distance)
    pose_source = DummyPoseSource()
    frame_builder = TrackerFrameBuilder(bridge_config, range_sensor, pose_source)
    manager = AimingManager()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[Error] 동영상 열기 실패: {args.video}")
        return

    # TCP 서버 스레드 시작
    server_thread = threading.Thread(target=tcp_server_loop, daemon=True)
    server_thread.start()

    print(f"하이브리드 시뮬레이션 시작! 가상 거리: {args.distance}m")
    
    # 유니티 연결 대기
    print("유니티에서 [Play] 버튼을 눌러 서버에 접속할 때까지 기다립니다...")
    while unity_client is None:
        time.sleep(1.0)
    
    print("유니티 연결 확인! 3초 후 동영상 테스트를 시작합니다...")
    time.sleep(3)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("동영상 재생 종료. 처음부터 다시 시작합니다.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            timestamp = time.time()
            tracker_frame = frame_builder.build(frame, timestamp)
            
            if tracker_frame is not None:
                output = manager.update(tracker_frame)
                
                if frame_idx % 15 == 0:
                    print(f"Frame {frame_idx} | State: {output.state.value} | "
                          f"Dist: {output.debug['distance_m']:.1f}m | Ready: {output.aim_ready}")
                          
                cv2.putText(frame, f"State: {output.state.value} Ready:{output.aim_ready}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if output.aim_ready else (0, 165, 255), 2)
                
                # 유니티로 결과 쏘기
                send_to_unity(output)
            else:
                cv2.putText(frame, "State: SEARCH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Hybrid Aiming (Python AI -> Unity Physics)", frame)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
                
            frame_idx += 1
            
    except KeyboardInterrupt:
        print("시뮬레이션 종료.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if unity_client:
            unity_client.close()

if __name__ == "__main__":
    main()
