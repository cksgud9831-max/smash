#!/usr/bin/env python3
"""
turret_tracker.py  -  slew the turret to keep a detected target centred.

Subscribes:  /drone_detections        (vision_msgs/Detection2DArray)
Publishes :  /turret_controller/commands  (std_msgs/Float64MultiArray [pan, tilt])

Visual servo: the box centre's offset from image centre becomes a pan/tilt
rate. Pan and tilt are integrated and clamped, then published. When the
target is lost for a while the turret holds its last aim.

Params:
  image_w / image_h : camera resolution        (default 640 x 640)
  kp                : proportional gain (rad per normalised error) (0.6)
  deadband          : ignore error smaller than this (norm.)       (0.03)
  max_rate          : max slew rate [rad/s]                         (1.2)
  tilt_min / tilt_max : elevation limits [rad]      (-0.17 .. 1.48)
  classes           : which class_ids to follow ('' = any)         ('')
  lost_timeout      : seconds before declaring target lost          (1.5)
"""
import math
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Float64MultiArray


class TurretTracker(Node):
    def __init__(self):
        super().__init__('turret_tracker')
        self.declare_parameter('image_w', 640)
        self.declare_parameter('image_h', 640)
        self.declare_parameter('kp', 0.6)
        self.declare_parameter('deadband', 0.03)
        self.declare_parameter('max_rate', 1.2)
        self.declare_parameter('tilt_min', -0.17)
        self.declare_parameter('tilt_max', 1.48)
        self.declare_parameter('classes', '')      # comma list, '' = any
        self.declare_parameter('lost_timeout', 1.5)

        self.w = float(self.get_parameter('image_w').value)
        self.h = float(self.get_parameter('image_h').value)
        self.kp = float(self.get_parameter('kp').value)
        self.db = float(self.get_parameter('deadband').value)
        self.max_rate = float(self.get_parameter('max_rate').value)
        self.tilt_min = float(self.get_parameter('tilt_min').value)
        self.tilt_max = float(self.get_parameter('tilt_max').value)
        cl = str(self.get_parameter('classes').value).strip()
        self.classes = set(c.strip() for c in cl.split(',') if c.strip())
        self.lost_timeout = float(self.get_parameter('lost_timeout').value)

        self.pan = 0.0
        self.tilt = 0.4
        self.last_seen = self.get_clock().now()
        self.last_t = self.get_clock().now()

        self.pub = self.create_publisher(
            Float64MultiArray, '/turret_controller/commands', 10)
        self.sub = self.create_subscription(
            Detection2DArray, '/drone_detections', self.on_det, 10)
        # steady output at 30 Hz so the controller always has a target
        self.timer = self.create_timer(1.0 / 30.0, self.publish_cmd)
        self.get_logger().info(
            f'tracker up | follow classes: {self.classes or "ANY"}')

    def on_det(self, msg):
        target = self.pick_target(msg)
        if target is None:
            return
        cx = target.bbox.center.position.x
        cy = target.bbox.center.position.y

        # normalised error, -1..1 (image centre = 0)
        ex = (cx - self.w / 2.0) / (self.w / 2.0)
        ey = (cy - self.h / 2.0) / (self.h / 2.0)

        now = self.get_clock().now()
        dt = (now - self.last_t).nanoseconds * 1e-9
        self.last_t = now
        dt = min(max(dt, 1e-3), 0.2)

        # deadband so it settles instead of jittering
        if abs(ex) < self.db:
            ex = 0.0
        if abs(ey) < self.db:
            ey = 0.0

        # proportional step toward centre (kp maps normalised error -> radians)
        pan_step = clamp(-self.kp * ex, -self.max_rate * dt, self.max_rate * dt)
        # image +y is DOWN; camera looks forward, so target high in frame
        # (ey negative) should RAISE tilt -> subtract kp*ey
        tilt_step = clamp(-self.kp * ey, -self.max_rate * dt, self.max_rate * dt)

        self.pan += pan_step
        self.tilt += tilt_step
        self.tilt = clamp(self.tilt, self.tilt_min, self.tilt_max)
        self.last_seen = now

    def pick_target(self, msg):
        best = None
        best_score = -1.0
        for d in msg.detections:
            if not d.results:
                continue
            hyp = d.results[0].hypothesis
            if self.classes and hyp.class_id not in self.classes:
                continue
            if hyp.score > best_score:
                best_score = hyp.score
                best = d
        return best

    def publish_cmd(self):
        m = Float64MultiArray()
        m.data = [self.pan, self.tilt]
        self.pub.publish(m)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    rclpy.init()
    node = TurretTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
