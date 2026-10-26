#!/usr/bin/env python3
"""Capture a bounded raw Gazebo camera sequence to an MJPEG AVI file."""

import argparse
import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class Capture(Node):
    def __init__(self, output: str, frames: int) -> None:
        super().__init__("raw_sequence_capture")
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        self.output = output
        self.limit = frames
        self.bridge = CvBridge()
        self.writer = None
        self.count = 0
        self.create_subscription(Image, "/turret_camera/image_raw", self.on_image, qos_profile_sensor_data)

    def on_image(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if self.writer is None:
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(
                self.output, cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (w, h)
            )
            if not self.writer.isOpened():
                raise RuntimeError(f"cannot open video writer: {self.output}")
        self.writer.write(frame)
        self.count += 1
        if self.count >= self.limit:
            rclpy.shutdown()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    rclpy.init()
    node = Capture(args.output, args.frames)
    started = time.monotonic()
    try:
        while rclpy.ok() and time.monotonic() - started < args.timeout:
            rclpy.spin_once(node, timeout_sec=0.5)
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if node.count < args.frames:
        raise SystemExit(f"captured only {node.count}/{args.frames} frames")


if __name__ == "__main__":
    main()
