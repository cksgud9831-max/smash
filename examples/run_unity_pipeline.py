import argparse
import sys
import time
from pathlib import Path
import cv2

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aiming_engine import AimingManager
from bridge.config import load_bridge_config
from bridge.frame_builder import TrackerFrameBuilder
from bridge.unity.unity_server import UnityServer
from bridge.unity.unity_sensors import UnityRangeSensor, UnityPoseSource

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080, help="유니티 통신 수신 포트")
    parser.add_argument("--bridge-config", type=str, default=None)
    parser.add_argument("--aiming-config", type=str, default=None)
    parser.add_argument("--show-ui", action="store_true", help="수신된 영상 데스크톱에 표시")
    args = parser.parse_args()

    bridge_config = load_bridge_config(args.bridge_config)
    
    print("1. 유니티 통신 서버 시작 중...")
    server = UnityServer(port=args.port)
    server.start()
    
    print("2. 가상 센서 모듈 연결 중...")
    range_sensor = UnityRangeSensor(server)
    pose_source = UnityPoseSource(server)
    
    # YOLO/옵티컬플로우 트래커가 내장된 빌더 인스턴스 생성
    frame_builder = TrackerFrameBuilder(bridge_config, range_sensor, pose_source)
    manager = AimingManager(config_path=args.aiming_config)

    print("3. 유니티 시뮬레이션 파이프라인 대기 중...")
    
    frame_idx = 0
    
    try:
        while True:
            # 유니티에서 보낸 가장 최근 프레임/센서 데이터를 가져옴
            frame, dist, pose, timestamp = server.get_latest_data()
            
            if frame is None:
                # 아직 유니티가 프레임을 전송하지 않았음
                time.sleep(0.01)
                continue
                
            # 디스플레이 옵션이 켜져 있으면 보여줌
            if args.show_ui:
                cv2.imshow("Unity Simulation View", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            # 1) 탐지 + 추적 + 센서 융합
            tracker_frame = frame_builder.build(frame, timestamp)
            
            if tracker_frame is None:
                # 타겟이 없거나 센서 데이터 유효기간 초과
                server.set_feedback(state="SEARCH", aim_ready=False)
                
                if frame_idx % 30 == 0:
                    print(f"[Unity Pipeline] Frame {frame_idx} | State: SEARCH (Target not found)")
                frame_idx += 1
                
                time.sleep(0.01) # 무한 루프 폭주 방지
                continue
                
            # 2) 조준 계산
            output = manager.update(tracker_frame)
            
            # 3) 유니티로 결과 전송
            # output.state.value: SEARCH, TRACK, AIM, READY 중 하나
            # output.aim_ready: 최종 조준 신뢰성 여부
            server.set_feedback(
                state=output.state.value,
                aim_azimuth=output.aim_azimuth,
                aim_elevation=output.aim_elevation,
                aim_ready=output.aim_ready
            )
            
            if frame_idx % 30 == 0:
                print(f"[Unity Pipeline] Frame {frame_idx} | State: {output.state.value} | "
                      f"Dist: {output.debug['distance_m']:.1f}m | "
                      f"Hit Prob: {output.hit_probability:.2f} | "
                      f"Ready: {output.aim_ready}")
            
            frame_idx += 1
            
            # 루프가 너무 빨리 돌지 않도록 최소한의 sleep (약 60fps 제한)
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("시뮬레이션 종료 명령 수신...")
    finally:
        server.stop()
        range_sensor.close()
        pose_source.close()
        if args.show_ui:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
