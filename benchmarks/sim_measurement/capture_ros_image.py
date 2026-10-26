#!/usr/bin/env python3
"""Capture one latest-only ROS Image for detector diagnostics."""

import argparse

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class Capture(Node):
    def __init__(self, topic: str, output: str):
        super().__init__("capture_ros_image")
        self.output = output
        self.done = False
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Image, topic, self.on_image, qos)

    def on_image(self, msg: Image) -> None:
        frame = CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if not cv2.imwrite(self.output, frame):
            raise RuntimeError(f"failed to write {self.output}")
        self.done = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--topic", default="/turret_camera/image_raw")
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args()
    rclpy.init()
    node = Capture(args.topic, args.output)
    deadline = node.get_clock().now().nanoseconds * 1e-9 + args.timeout
    while rclpy.ok() and not node.done and node.get_clock().now().nanoseconds * 1e-9 < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    return 0 if node.done else 2


if __name__ == "__main__":
    raise SystemExit(main())
