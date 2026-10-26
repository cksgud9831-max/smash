#!/usr/bin/env python3
"""smash_scope_viewer.py — SMASH 스코프 영상 중계 서버 (WSL 측)

FCS 가 그린 HUD 영상(/turret_camera/image_annotated)을 HTTP MJPEG 로 내보낸다(포트 9999).
Windows 쪽 scripts/win_scope_viewer.py 가 이 스트림을 받아 창에 띄우고 녹화한다.
브라우저로 http://127.0.0.1:9999 에 접속해도 같은 화면을 볼 수 있다.

격발은 FCS 가 READY 에서 자동으로 한다. 사용자 조작은 다음 명령 중계뿐이다.
    /slew?pan=&tilt=   포탑 증분 선회/프리셋(888 = 탐색 자세 복귀). 표적을 놓쳤을 때 쓴다.
    /cmd?name=         HUD 상세 토글·통계 초기화, 이동 표적 기동 방식/속도/출발 명령
    /status            이동 표적 상태(JSON)
"""

import http.server
import json
import threading
import time
import urllib.parse

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, String

from drone_sim.smash_fcs.spawn_command import parse_spawn

PORT = 9999
SLEW_TOPIC = "/smash_fcs/slew_cmd"
HUD_SUMMARY_STALE_S = 1.5  # 이 시간 넘게 HUD 요약이 없으면 /status 의 hud_* 를 비운다

ALLOWED_COMMANDS = (
    "toggle_debug", "reset_stats", "toggle_motion",
    "motion:none", "motion:straight", "motion:cross", "motion:maneuver", "motion:approach",
    "speed:+5", "speed:-5",
    "fire",
)

INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SMASH Smart Scope FCS</title>
<style>
body { background: #0f1117; color: #fff; text-align: center; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 12px; }
h2 { margin: 6px; color: #00ff88; font-size: 22px; letter-spacing: 1px; }
.desc { color: #8892b0; font-size: 13px; margin-bottom: 10px; }
#scope-img { border: 2px solid #00ff88; border-radius: 8px; max-width: 680px; width: 90vw; height: auto; box-shadow: 0 0 20px rgba(0,255,136,0.3); }
</style>
</head>
<body>
<h2>SMASH Smart Scope FCS</h2>
<div class="desc">격발은 READY 에서 자동 | R: 탐색 자세 복귀 | 0~4: 표적 기동 방식 | +/-: 속도 | M: 출발/정지 | H: HUD 상세 | C: 통계 초기화</div>
<div><img id="scope-img" src="/stream" alt="Connecting to SMASH Stream..." /></div>
<script>
const cmd = (name) => fetch('/cmd?name=' + encodeURIComponent(name));
window.addEventListener('keydown', (e) => {
  if (e.key === 'r' || e.key === 'R') fetch('/slew?pan=888&tilt=0');
  else if (e.key === 'h' || e.key === 'H') cmd('toggle_debug');
  else if (e.key === 'c' || e.key === 'C') cmd('reset_stats');
  else if (e.code === 'KeyM') cmd('toggle_motion');
  else if (/^(Digit|Numpad)[0-4]$/.test(e.code)) {
    const m = ['none', 'straight', 'cross', 'maneuver', 'approach'][Number(e.code.slice(-1))];
    cmd('motion:' + m);
  }
  else if (e.key === '+' || e.key === '=' || e.code === 'NumpadAdd') cmd('speed:+5');
  else if (e.key === '-' || e.code === 'NumpadSubtract') cmd('speed:-5');
});
</script>
</body>
</html>"""


class ScopeHttpHandler(http.server.BaseHTTPRequestHandler):
    viewer_instance = None

    def log_message(self, format, *args):
        pass  # 묵음

    def _reply(self, body: bytes, content_type: str = "text/plain", cors: bool = True) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        if cors:
            self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        viewer = ScopeHttpHandler.viewer_instance

        if path == "/slew":
            try:
                d_pan = float(query.get("pan", [0.0])[0])
                d_tilt = float(query.get("tilt", [0.0])[0])
            except ValueError:
                d_pan = d_tilt = 0.0
            if viewer:
                viewer.slew(d_pan, d_tilt)
            self._reply(b"OK")
        elif path == "/cmd":
            name = query.get("name", [""])[0]
            if name not in ALLOWED_COMMANDS and parse_spawn(name) is None:
                # 오타·잘못된 이름이 OK 로 보이면 클라이언트가 동작했다고 오해한다.
                self.send_response(400)
                self.end_headers()
                return
            if viewer:
                viewer.ui_command(name)
            self._reply(b"OK")
        elif path == "/status":
            # 이동 표적 상태 (target_mover 의 /target_mover/status JSON). 없으면 빈 객체
            body = viewer.status_json() if viewer else "{}"
            self._reply(body.encode("utf-8"), "application/json")
        elif path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            # 새로 받은 프레임일 때만 압축해 내보낸다(객체 동일성으로 판별). 같은 프레임을
            # 반복 압축하면 CPU 를 낭비하고 수신 측이 못 따라갈 만큼 밀어내 지연이 커진다.
            last_sent = None
            while rclpy.ok():
                frame = viewer.rendered_frame if viewer else None
                if frame is not None and frame is not last_sent:
                    last_sent = frame
                    ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        try:
                            self.wfile.write(b"--frame\r\n")
                            self.send_header("Content-Type", "image/jpeg")
                            self.send_header("Content-Length", str(len(jpeg)))
                            self.end_headers()
                            self.wfile.write(jpeg.tobytes())
                            self.wfile.write(b"\r\n")
                        except Exception:
                            break
                time.sleep(0.005)
        elif path in ("/", "/index.html"):
            self._reply(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8", cors=False)
        else:
            self.send_response(404)
            self.end_headers()


class SmashScopeViewer(Node):
    def __init__(self):
        super().__init__("smash_scope_viewer")

        self._bridge = CvBridge()
        # 윈도우 뷰어 안내용: 수동이면 READY 자막이 "클릭/스페이스로 발사" 로 바뀐다(FCS 의 trigger_mode 와 같은 값)
        self.declare_parameter("trigger_mode", "auto")
        self.trigger_mode = str(self.get_parameter("trigger_mode").value).strip().lower()
        # 표적 즉시 생성 버튼은 격추 연출(demo_director)이 떠 있어야 동작한다. 패널이 꺼진 버튼을 표시하게 알린다.
        self.declare_parameter("spawn_enabled", True)
        self.spawn_enabled = bool(self.get_parameter("spawn_enabled").value)
        self.rendered_frame = None
        self.target_status = None

        self._slew_pub = self.create_publisher(Float64MultiArray, SLEW_TOPIC, 10)
        self._ui_pub = self.create_publisher(String, "/smash_fcs/ui_command", 10)
        self.create_subscription(String, "/target_mover/status", self._on_target_status, 10)
        # 윈도우 뷰어의 한글 자막용: FCS 상태 문자열과 격추/재출현 이벤트를 /status 에 실어 보낸다.
        self.fcs_state = ""
        self.reliability = ""
        self.last_event = None
        self.last_event_time = 0.0
        self.create_subscription(String, "/smash_fcs/state", self._on_fcs_state, 10)
        self.create_subscription(String, "/smash_fcs/reliability_state", self._on_reliability, 10)
        self.create_subscription(String, "/smash_fcs/target_event", self._on_target_event, 10)
        # HUD 에 그려진 거리·격추 수. 패널이 HUD 와 같은 숫자를 보이도록 /status 에 싣는다.
        self.hud_range_m = None
        self.hud_kills = None
        self.hud_summary_time = 0.0
        self.create_subscription(String, "/smash_fcs/hud_summary", self._on_hud_summary, 10)
        self.create_subscription(
            Image,
            "/turret_camera/image_annotated",
            self._on_image,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST),
        )

        ScopeHttpHandler.viewer_instance = self
        try:
            self._server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), ScopeHttpHandler)
            threading.Thread(target=self._server.serve_forever, daemon=True).start()
            self.get_logger().info(f"SMASH 스코프 HTTP 스트리밍 서버 가동: http://127.0.0.1:{PORT}")
        except Exception as e:
            self.get_logger().warn(f"HTTP 스트리밍 서버 포트 바인딩 실패: {e}")

    def _on_image(self, msg: Image):
        try:
            self.rendered_frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"영상 디코딩 실패: {e}")

    def _on_target_status(self, msg: String):
        self.target_status = msg.data

    def _on_fcs_state(self, msg: String):
        self.fcs_state = msg.data

    def _on_reliability(self, msg: String):
        self.reliability = msg.data

    def _on_hud_summary(self, msg: String):
        try:
            data = json.loads(msg.data)
            range_m, kills = data.get("range_m"), data.get("kills")
        except (TypeError, ValueError, AttributeError):
            return
        self.hud_range_m = None if range_m is None else float(range_m)
        self.hud_kills = None if kills is None else int(kills)
        self.hud_summary_time = time.monotonic()

    def _on_target_event(self, msg: String):
        try:
            event = json.loads(msg.data).get("event")
        except (TypeError, ValueError, AttributeError):
            return
        if event in ("destroyed", "respawn"):
            self.last_event = event
            self.last_event_time = time.monotonic()

    def status_json(self) -> str:
        """이동 표적 상태(target_mover)에 FCS 상태·최근 이벤트를 합친 JSON."""

        try:
            status = json.loads(self.target_status) if self.target_status else {}
        except ValueError:
            status = {}
        if not isinstance(status, dict):
            status = {}
        # range_m 는 target_mover 의 시나리오 값이라 HUD 의 FCS 측정 거리와 다르다. 패널은 hud_* 를 쓴다.
        fresh = self.hud_kills is not None and time.monotonic() - self.hud_summary_time < HUD_SUMMARY_STALE_S
        status["hud_range_m"] = self.hud_range_m if fresh else None
        status["hud_kills"] = self.hud_kills if fresh else None
        status["fcs_state"] = self.fcs_state
        status["trigger_mode"] = self.trigger_mode
        status["spawn_enabled"] = self.spawn_enabled
        status["reliability"] = self.reliability
        status["event"] = self.last_event
        status["event_age_s"] = (
            None if self.last_event is None else round(time.monotonic() - self.last_event_time, 2)
        )
        return json.dumps(status)

    def ui_command(self, name: str):
        msg = String()
        msg.data = name
        self._ui_pub.publish(msg)

    def slew(self, d_pan: float, d_tilt: float):
        msg = Float64MultiArray()
        msg.data = [float(d_pan), float(d_tilt)]
        self._slew_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    viewer = SmashScopeViewer()
    try:
        rclpy.spin(viewer)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
