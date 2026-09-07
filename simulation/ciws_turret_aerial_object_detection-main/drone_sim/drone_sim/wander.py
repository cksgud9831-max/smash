#!/usr/bin/env python3
"""
wander.py  -  fly ONE named model on a random-wander path.

Generic version of the bird wander for a single entity (drone, plane, heli).
Kinematic teleport via /world/<world>/set_pose; noses along travel direction.

Params:
  world      : gz world name        (default: turret_world)
  model      : entity name in gz    (required, e.g. 'drone')
  area       : half-width XY box [m] (default: 12.0)
  z_min/z_max: altitude band [m]     (default: 4.0 .. 9.0)
  speed      : cruise speed [m/s]    (default: 3.0)
"""
import math
import random
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity


class Wander(Node):
    def __init__(self):
        super().__init__('wander')
        self.declare_parameter('world', 'turret_world')
        self.declare_parameter('model', 'drone')
        self.declare_parameter('area', 12.0)
        self.declare_parameter('z_min', 4.0)
        self.declare_parameter('z_max', 9.0)
        self.declare_parameter('speed', 3.0)

        world = self.get_parameter('world').value
        self.model = self.get_parameter('model').value
        self.area = float(self.get_parameter('area').value)
        self.z_min = float(self.get_parameter('z_min').value)
        self.z_max = float(self.get_parameter('z_max').value)
        self.speed = float(self.get_parameter('speed').value)

        self.x = random.uniform(-self.area, self.area)
        self.y = random.uniform(-self.area, self.area)
        self.z = random.uniform(self.z_min, self.z_max)
        self.yaw = 0.0
        self.pick_target()

        srv = f'/world/{world}/set_pose'
        self.cli = self.create_client(SetEntityPose, srv)
        self.ready = False
        self.get_logger().info(f'[{self.model}] connecting to {srv} ...')

        self.last = self.get_clock().now()
        self.timer = self.create_timer(1.0 / 15.0, self.tick)

    def _ensure_ready(self):
        if self.ready:
            return True
        if self.cli.service_is_ready():
            self.ready = True
            self.get_logger().info(f'[{self.model}] wandering')
            return True
        return False

    def pick_target(self):
        self.tx = random.uniform(-self.area, self.area)
        self.ty = random.uniform(-self.area, self.area)
        self.tz = random.uniform(self.z_min, self.z_max)

    def tick(self):
        if not self._ensure_ready():
            return
        now = self.get_clock().now()
        dt = (now - self.last).nanoseconds * 1e-9
        self.last = now
        dt = min(max(dt, 1e-3), 0.2)

        dx, dy, dz = self.tx - self.x, self.ty - self.y, self.tz - self.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 0.6:
            self.pick_target()
        else:
            step = min(self.speed * dt, dist)
            self.x += dx / dist * step
            self.y += dy / dist * step
            self.z += dz / dist * step
            self.yaw = math.atan2(dy, dx)

        req = SetEntityPose.Request()
        req.entity = Entity(name=self.model, type=Entity.MODEL)
        p = Pose()
        p.position.x = self.x
        p.position.y = self.y
        p.position.z = self.z
        p.orientation.z = math.sin(self.yaw / 2.0)
        p.orientation.w = math.cos(self.yaw / 2.0)
        req.pose = p
        self.cli.call_async(req)
        self.get_logger().info(
            f'[{self.model}] -> ({self.x:.1f},{self.y:.1f},{self.z:.1f})',
            throttle_duration_sec=1.0)


def main():
    rclpy.init()
    node = Wander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
