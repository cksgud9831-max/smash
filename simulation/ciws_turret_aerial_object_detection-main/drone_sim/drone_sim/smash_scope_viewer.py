#!/usr/bin/env python3
"""smash_scope_viewer.py — SMASH 실시간 대화형 스코프 뷰어
사수 격발 지원 및 키보드 방향키 / WASD / 화면 상하좌우 버튼 수동 조준 지원
+ Windows 네이티브 뷰어 및 웹 브라우저 직결 HTTP MJPEG 스트리밍 서버 탑재 (포트 9999)
"""

import http.server
import threading
import time
import urllib.parse
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64MultiArray
from cv_bridge import CvBridge


class ScopeHttpHandler(http.server.BaseHTTPRequestHandler):
    viewer_instance = None

    def log_message(self, format, *args):
        pass  # 묵음

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/fire":
            if ScopeHttpHandler.viewer_instance:
                ScopeHttpHandler.viewer_instance._fire()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b"OK")
        elif path == "/slew":
            d_pan = float(query.get("pan", [0.0])[0])
            d_tilt = float(query.get("tilt", [0.0])[0])
            if ScopeHttpHandler.viewer_instance:
                ScopeHttpHandler.viewer_instance._slew(d_pan, d_tilt)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b"OK")
        elif path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            while rclpy.ok():
                frame = ScopeHttpHandler.viewer_instance._rendered_frame
                if frame is not None:
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
                time.sleep(0.033)
        elif path in ("/", "/index.html"):
            html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SMASH Smart Scope FCS</title>
<style>
body { background: #0f1117; color: #fff; text-align: center; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 12px; }
h2 { margin: 6px; color: #00ff88; font-size: 22px; letter-spacing: 1px; }
.desc { color: #8892b0; font-size: 13px; margin-bottom: 10px; }
#scope-img { border: 2px solid #00ff88; border-radius: 8px; max-width: 680px; width: 90vw; height: auto; cursor: crosshair; box-shadow: 0 0 20px rgba(0,255,136,0.3); }
.btn-panel { margin-top: 14px; display: flex; justify-content: center; align-items: center; gap: 8px; flex-wrap: wrap; }
button { background: #1a2230; color: #00ff88; border: 1px solid #00ff88; padding: 8px 16px; font-size: 15px; border-radius: 6px; cursor: pointer; transition: all 0.15s ease; }
button:hover { background: #00ff88; color: #000; }
.btn-fire { background: #d90429; color: #fff; border-color: #ef233c; font-weight: bold; font-size: 17px; padding: 10px 28px; }
.btn-fire:hover { background: #ff4d6d; color: #fff; }
</style>
</head>
<body>
<h2>SMASH CIWS Smart Scope FCS</h2>
<div class="desc">화면 클릭 또는 스페이스바: 격발 | 방향키 또는 WASD: 포탑 수동 조준 | T: 대공 프리셋 | R: 수평 리셋</div>
<div><img id="scope-img" src="/stream" onclick="fire()" alt="Connecting to SMASH Stream..." /></div>
<div class="btn-panel">
  <button style="background: #005f73; color: #00ffcc; border-color: #00ffcc; font-weight: bold;" onclick="slew(888, 0)">대공 프리셋 (T)</button>
  <button onclick="slew(0, 0.05)">상 (W / UP)</button>
  <button onclick="slew(0, -0.05)">하 (S / DN)</button>
  <button onclick="slew(-0.05, 0)">좌 (A / LT)</button>
  <button onclick="slew(0.05, 0)">우 (D / RT)</button>
  <button onclick="slew(999, 0)">리셋 (R)</button>
  <button class="btn-fire" onclick="fire()">[ BANG! 격발 (SPACE) ]</button>
</div>
<script>
function fire() { fetch('/fire'); }
function slew(p, t) { fetch(`/slew?pan=${p}&tilt=${t}`); }
window.addEventListener('keydown', (e) => {
  if (e.code === 'Space') { fire(); e.preventDefault(); }
  else if (e.key === 't' || e.key === 'T') slew(888, 0);
  else if (e.key === 'w' || e.key === 'W' || e.key === 'ArrowUp') slew(0, 0.05);
  else if (e.key === 's' || e.key === 'S' || e.key === 'ArrowDown') slew(0, -0.05);
  else if (e.key === 'a' || e.key === 'A' || e.key === 'ArrowLeft') slew(-0.05, 0);
  else if (e.key === 'd' || e.key === 'D' || e.key === 'ArrowRight') slew(0.05, 0);
  else if (e.key === 'r' || e.key === 'R') slew(999, 0);
});
</script>
</body>
</html>"""
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


class SmashScopeViewer(Node):
    def __init__(self):
        super().__init__("smash_scope_viewer")

        self._bridge = CvBridge()
        self._latest_frame = None
        self._rendered_frame = None
        self._bang_timer = 0.0

        self._trigger_pub = self.create_publisher(Bool, "/smash_fcs/trigger", 10)
        self._slew_pub = self.create_publisher(Float64MultiArray, "/smash_fcs/manual_slew", 10)
        self._image_sub = self.create_subscription(
            Image, "/turret_camera/image_annotated", self._on_image, 10
        )

        self._window_name = "SMASH Smart Scope FCS (Click or Space to FIRE)"
        self._gui_available = False  # WSL 내부 X11 GUI 비활성화 -> Remote Desktop CPU 12% 완전 소멸 (Windows 네이티브 뷰어 전담)

        # D_Pad 버튼 영역 정의 (x, y, w, h)
        self._btn_up = (0, 0, 0, 0)
        self._btn_down = (0, 0, 0, 0)
        self._btn_left = (0, 0, 0, 0)
        self._btn_right = (0, 0, 0, 0)
        self._btn_reset = (0, 0, 0, 0)

        # HTTP 스트리밍 서버 가동 (Windows 직접 뷰어 및 브라우저 직결)
        ScopeHttpHandler.viewer_instance = self
        try:
            self._server = http.server.ThreadingHTTPServer(("0.0.0.0", 9999), ScopeHttpHandler)
            self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._server_thread.start()
            self.get_logger().info("SMASH 스코프 HTTP 스트리밍 서버 가동: http://localhost:9999 (Windows 직접 연동)")
        except Exception as e:
            self.get_logger().warn(f"HTTP 스트리밍 서버 포트 바인딩 실패: {e}")

        self.get_logger().info("SMASH 스코프 뷰어 기동 완료 (격발 + 키보드 방향키/WASD/화면 버튼 수동 조준 지원)")

    def _fire(self):
        msg = Bool()
        msg.data = True
        self._trigger_pub.publish(msg)
        self._bang_timer = 1.0
        self.get_logger().info("[BANG!] 사수 방아쇠 격발 명령 전송 (/smash_fcs/trigger)")

    def _slew(self, d_pan: float, d_tilt: float):
        msg = Float64MultiArray()
        msg.data = [float(d_pan), float(d_tilt)]
        self._slew_pub.publish(msg)

    def _is_inside(self, pt, rect):
        x, y = pt
        rx, ry, rw, rh = rect
        return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pt = (x, y)
            step_rad = 0.035  # 약 2.0도

            if self._is_inside(pt, self._btn_up):
                self._slew(0.0, step_rad)
                self.get_logger().info("[BUTTON] 포탑 상방(TILT UP) 조준")
            elif self._is_inside(pt, self._btn_down):
                self._slew(0.0, -step_rad)
                self.get_logger().info("[BUTTON] 포탑 하방(TILT DOWN) 조준")
            elif self._is_inside(pt, self._btn_left):
                self._slew(-step_rad, 0.0)
                self.get_logger().info("[BUTTON] 포탑 좌측(PAN LEFT) 조준")
            elif self._is_inside(pt, self._btn_right):
                self._slew(step_rad, 0.0)
                self.get_logger().info("[BUTTON] 포탑 우측(PAN RIGHT) 조준")
            elif self._is_inside(pt, self._btn_reset):
                self._slew(999.0, 0.0)
                self.get_logger().info("[BUTTON] 포탑 기본 조준 리셋")
            else:
                self._fire()

    def _on_image(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self._latest_frame = frame
        except Exception as e:
            self.get_logger().warn(f"영상 디코딩 실패: {e}")

    def _draw_button(self, img, rect, label, active=False):
        x, y, w, h = rect
        bg = (40, 45, 55) if not active else (0, 180, 80)
        border = (0, 255, 140)
        cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), border, 1)
        font_scale = 0.38
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx = x + (w - tw) // 2
        ty = y + (h + th) // 2
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    def run_loop(self):
        step_rad = 0.035  # 약 2.0도

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

            if self._latest_frame is not None:
                display = self._latest_frame.copy()
                h, w = display.shape[:2]

                # 하단 정보 바
                cv2.rectangle(display, (10, h - 38), (w - 10, h - 8), (20, 20, 25), -1)
                cv2.rectangle(display, (10, h - 38), (w - 10, h - 8), (0, 255, 120), 1)
                cv2.putText(
                    display,
                    "[ FIRE: CLICK / SPACE ] [ AIM: ARROWS / WASD / D_PAD ] [ RESET: R ] [ QUIT: Q ]",
                    (18, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (0, 255, 120),
                    1,
                )

                # 우측 하단 D_Pad 수동 조준 버튼 UI (상하좌우 버튼)
                bw, bh = 38, 28
                cx = w - 85
                cy = h - 110

                self._btn_up = (cx, cy - bh - 4, bw, bh)
                self._btn_down = (cx, cy + bh + 4, bw, bh)
                self._btn_left = (cx - bw - 4, cy, bw, bh)
                self._btn_right = (cx + bw + 4, cy, bw, bh)
                self._btn_reset = (cx, cy, bw, bh)

                # D_Pad 배경 패널
                cv2.rectangle(display, (cx - bw - 10, cy - bh - 10), (cx + 2 * bw + 10, cy + 2 * bh + 10), (15, 18, 22), -1)
                cv2.rectangle(display, (cx - bw - 10, cy - bh - 10), (cx + 2 * bw + 10, cy + 2 * bh + 10), (70, 80, 95), 1)
                cv2.putText(display, "AIM PAD", (cx - bw - 4, cy - bh - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 200, 220), 1)

                self._draw_button(display, self._btn_up, "UP")
                self._draw_button(display, self._btn_down, "DN")
                self._draw_button(display, self._btn_left, "LT")
                self._draw_button(display, self._btn_right, "RT")
                self._draw_button(display, self._btn_reset, "RST")

                if self._bang_timer > 0.0:
                    self._bang_timer -= 0.05
                    cv2.putText(
                        display,
                        "FIRE TRIGGERED!",
                        (w // 2 - 110, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 255),
                        2,
                    )
                    cv2.circle(display, (w // 2, h // 2), 28, (0, 255, 255), 2)

                self._rendered_frame = display
                if self._gui_available:
                    try:
                        cv2.imshow(self._window_name, display)
                    except Exception:
                        pass
            else:
                loading = np.zeros((720, 720, 3), dtype=np.uint8)
                cv2.putText(
                    loading,
                    "SMASH Smart Scope FCS",
                    (160, 320),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 120),
                    2,
                )
                cv2.putText(
                    loading,
                    "Connecting to camera stream...",
                    (180, 380),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (180, 180, 180),
                    1,
                )
                self._rendered_frame = loading
                if self._gui_available:
                    try:
                        cv2.imshow(self._window_name, loading)
                    except Exception:
                        pass

            if self._gui_available:
                try:
                    key_ex = cv2.waitKeyEx(1)
                    key = key_ex & 0xFF

                    is_up = (key in (ord("w"), ord("W"))) or (key_ex in (65362, 2490368, 0x260000))
                    is_down = (key in (ord("s"), ord("S"))) or (key_ex in (65364, 2621440, 0x280000))
                    is_left = (key in (ord("a"), ord("A"))) or (key_ex in (65361, 2424832, 0x250000))
                    is_right = (key in (ord("d"), ord("D"))) or (key_ex in (65363, 2555904, 0x270000))

                    if is_up:
                        self._slew(0.0, step_rad)
                    elif is_down:
                        self._slew(0.0, -step_rad)
                    elif is_left:
                        self._slew(-step_rad, 0.0)
                    elif is_right:
                        self._slew(step_rad, 0.0)
                    elif key in (ord("r"), ord("R")):
                        self._slew(999.0, 0.0)
                    elif key == 32:
                        self._fire()
                    elif key in (27, ord("q"), ord("Q")):
                        break
                except Exception:
                    pass

        if self._gui_available:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    viewer = SmashScopeViewer()
    try:
        viewer.run_loop()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
