#!/usr/bin/env python3
"""
drone_detector.py  -  run the YOLO11 drone model on the turret camera.

Subscribes:  /turret_camera/image_raw   (sensor_msgs/Image)
Publishes :  /drone_detections          (vision_msgs/Detection2DArray)
             /drone_detections/image     (sensor_msgs/Image, annotated)

Params:
  weights    : path to the .pt file           (required)
  conf       : confidence threshold           (default 0.25)
  imgsz      : inference size                  (default 640)
  device     : 'cpu' or 'cuda:0'              (default cpu)
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np


class DroneDetector(Node):
    def __init__(self):
        super().__init__('drone_detector')
        self.declare_parameter('weights', '')
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('device', 'cpu')

        weights = self.get_parameter('weights').value
        self.conf = float(self.get_parameter('conf').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.device = self.get_parameter('device').value
        if not weights:
            self.get_logger().fatal('set the "weights" parameter to your .pt file')
            raise SystemExit(1)

        self.get_logger().info(f'loading YOLO weights: {weights}')
        self.model = YOLO(weights)
        self.names = self.model.names
        self.get_logger().info(f'classes: {self.names}')

        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image, '/turret_camera/image_raw', self.on_image, 10)
        self.pub_det = self.create_publisher(Detection2DArray, '/drone_detections', 10)
        self.pub_img = self.create_publisher(Image, '/drone_detections/image', 10)
        # also publish the boxed image onto a topic you can open directly
        self.pub_overlay = self.create_publisher(Image, '/turret_camera/image_annotated', 10)

    def on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                 device=self.device, verbose=False)[0]

        out = Detection2DArray()
        out.header = msg.header
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls = int(b.cls[0]); score = float(b.conf[0])
            d = Detection2D()
            d.header = msg.header
            d.bbox.center.position.x = (x1 + x2) / 2.0
            d.bbox.center.position.y = (y1 + y2) / 2.0
            d.bbox.size_x = x2 - x1
            d.bbox.size_y = y2 - y1
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(self.names.get(cls, cls))
            hyp.hypothesis.score = score
            d.results.append(hyp)
            out.detections.append(d)
        self.pub_det.publish(out)

        # annotated image (YOLO draws boxes+labels)
        annotated = np.asarray(res.plot())
        if annotated.ndim == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
        elif annotated.shape[2] == 4:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGRA2BGR)
        annotated = np.ascontiguousarray(annotated, dtype=np.uint8)

        # build sensor_msgs/Image by hand (no cv_bridge -> no cvtype KeyError)
        img_msg = Image()
        img_msg.header = msg.header
        img_msg.height = annotated.shape[0]
        img_msg.width = annotated.shape[1]
        img_msg.encoding = 'bgr8'
        img_msg.is_bigendian = 0
        img_msg.step = annotated.shape[1] * 3
        img_msg.data = annotated.tobytes()
        self.pub_img.publish(img_msg)
        self.pub_overlay.publish(img_msg)

        if out.detections:
            top = max(out.detections,
                      key=lambda dd: dd.results[0].hypothesis.score)
            r = top.results[0].hypothesis
            self.get_logger().info(
                f'{len(out.detections)} det | best={r.class_id} {r.score:.2f}',
                throttle_duration_sec=1.0)


def main():
    rclpy.init()
    node = DroneDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
