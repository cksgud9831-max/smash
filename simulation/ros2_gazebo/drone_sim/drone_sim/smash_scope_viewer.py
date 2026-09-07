#!/usr/bin/env python3
"""smash_scope_viewer.py — SMASH 실시간 대화형 스코프 뷰어 (마우스 좌클릭 및 스페이스바 격발 지원)"""

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge


class SmashScopeViewer(Node):
    def __init__(self):
        super().__init__("smash_scope_viewer")

        self._bridge = CvBridge()
        self._latest_frame = None
        self._bang_timer = 0.0

        self._trigger_pub = self.create_publisher(Bool, "/smash_fcs/trigger", 10)
        self._image_sub = self.create_subscription(
            Image, "/turret_camera/image_annotated", self._on_image, 10
        )

        self._window_name = "SMASH Smart Scope FCS (Click or Space to FIRE)"
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window_name, 720, 720)
        cv2.setMouseCallback(self._window_name, self._on_mouse)

        self.get_logger().info("SMASH 스코프 뷰어 기동 완료 (마우스 좌클릭 / 스페이스바 격발 지원)")

    def _fire(self):
        msg = Bool()
        msg.data = True
        self._trigger_pub.publish(msg)
        self._bang_timer = 1.0
        self.get_logger().info("[BANG!] 사수 방아쇠 격발 명령 전송 (/smash_fcs/trigger)")

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._fire()

    def _on_image(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self._latest_frame = frame
        except Exception as e:
            self.get_logger().warn(f"영상 디코딩 실패: {e}")

    def run_loop(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

            if self._latest_frame is not None:
                display = self._latest_frame.copy()
                h, w = display.shape[:2]

                cv2.rectangle(display, (10, h - 38), (w - 10, h - 8), (20, 20, 25), -1)
                cv2.rectangle(display, (10, h - 38), (w - 10, h - 8), (0, 255, 120), 1)
                cv2.putText(
                    display,
                    "[ TRIGGER: MOUSE L-CLICK or SPACEBAR ] [ Q: QUIT ]",
                    (25, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 120),
                    1,
                )

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

                cv2.imshow(self._window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                self._fire()
            elif key in (27, ord("q"), ord("Q")):
                break

        cv2.destroyAllWindows()


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
