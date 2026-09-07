import socket
import threading
import json
import struct
import time
import numpy as np
import cv2
from typing import Optional, List, Tuple

class UnityDataCache:
    """유니티에서 수신한 가장 최근의 데이터를 스레드 안전하게 보관하는 캐시"""
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_distance: Optional[float] = None
        self.latest_pose: Optional[List[float]] = None
        self.timestamp: float = 0.0

class UnityServer:
    """
    유니티 클라이언트와 TCP 소켓으로 통신하는 백그라운드 서버.
    수신 프로토콜:
    - 4 Bytes (Little Endian Int): JSON 페이로드 바이트 길이 (N)
    - N Bytes: JSON 페이로드 (distance, pose 행렬, timestamp, image_size 포함)
    - M Bytes (image_size): JPEG 인코딩된 이미지 바이트 배열
    """
    def __init__(self, host='0.0.0.0', port=8080):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.is_running = False
        self.cache = UnityDataCache()
        self.thread: Optional[threading.Thread] = None

        # 유니티로 보낼 결과 피드백 데이터
        self.feedback_lock = threading.Lock()
        self.latest_feedback = {"status": "idle"}

    def start(self):
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.is_running = True
        self.thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.thread.start()
        print(f"[UnityServer] Started on {self.host}:{self.port}. Waiting for Unity...")

    def stop(self):
        self.is_running = False
        try:
            self.server_socket.close()
        except:
            pass
        if self.thread:
            self.thread.join(timeout=1.0)

    def set_feedback(self, state: str, aim_azimuth: float = 0.0, aim_elevation: float = 0.0, aim_ready: bool = False):
        """조준 엔진의 계산 결과를 유니티로 반환하기 위해 세팅"""
        with self.feedback_lock:
            self.latest_feedback = {
                "state": state,
                "aim_azimuth": aim_azimuth,
                "aim_elevation": aim_elevation,
                "aim_ready": aim_ready
            }

    def get_latest_data(self) -> Tuple[Optional[np.ndarray], Optional[float], Optional[List[float]], float]:
        """캐시에서 최신 데이터를 꺼냄"""
        with self.cache.lock:
            return (
                self.cache.latest_frame,
                self.cache.latest_distance,
                self.cache.latest_pose,
                self.cache.timestamp
            )

    def _accept_loop(self):
        while self.is_running:
            try:
                self.server_socket.settimeout(1.0)
                client_sock, addr = self.server_socket.accept()
                print(f"[UnityServer] Unity client connected from {addr}")
                self._handle_client(client_sock)
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    print(f"[UnityServer] Accept exception: {e}")

    def _recv_all(self, sock: socket.socket, n: int) -> bytearray:
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                raise ConnectionError("Client disconnected")
            data.extend(packet)
        return data

    def _handle_client(self, client_sock: socket.socket):
        client_sock.settimeout(10.0)  # 타임아웃 10초
        try:
            while self.is_running:
                step = "read_header"
                # 1. Read JSON length (4 bytes little-endian)
                length_bytes = self._recv_all(client_sock, 4)
                json_length = struct.unpack('<I', length_bytes)[0]

                if json_length > 1000000:
                    print(f"[UnityServer] Warning: JSON length suspiciously large ({json_length} bytes)")
                    
                step = "read_json"
                # 2. Read JSON string
                json_bytes = self._recv_all(client_sock, json_length)
                json_data = json.loads(json_bytes.decode('utf-8'))
                
                image_size = json_data.get('image_size', 0)
                
                step = "read_image"
                # 3. Read JPEG bytes
                frame = None
                if image_size > 0:
                    img_bytes = self._recv_all(client_sock, image_size)
                    img_np = np.frombuffer(img_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

                step = "update_cache"
                # 4. Update Cache
                with self.cache.lock:
                    if frame is not None:
                        self.cache.latest_frame = frame
                    if 'distance' in json_data:
                        self.cache.latest_distance = json_data['distance']
                    if 'pose' in json_data:
                        self.cache.latest_pose = json_data['pose']
                    self.cache.timestamp = json_data.get('timestamp', time.time())

                step = "send_feedback"
                # 5. Send Feedback
                with self.feedback_lock:
                    feedback_json = json.dumps(self.latest_feedback).encode('utf-8')
                
                client_sock.sendall(struct.pack('<I', len(feedback_json)) + feedback_json)

        except socket.timeout:
            print(f"[UnityServer] Timed out at step: {step}")
        except Exception as e:
            print(f"[UnityServer] Connection lost/error at step {step}: {e}")
        finally:
            client_sock.close()
