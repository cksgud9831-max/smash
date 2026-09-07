#!/usr/bin/env python3
"""
bird_wander.py  -  fly N birds on independent random-wander paths.

Each bird drifts toward a random waypoint; on arrival it picks a new one.
Kinematic motion via /world/<world>/set_pose. Birds nose along their
travel direction.

Params:
  world      : gz world name              (default: turret_world)
  count      : number of birds            (default: 4)
  name_prefix: model name prefix          (default: bird)  -> bird_1..bird_N
  area       : half-width of wander box XY [m]   (default: 12.0)
  z_min/z_max: altitude band [m]          (default: 3.0 .. 8.0)
  speed      : cruise speed [m/s]         (default: 2.0)
"""
import math
import random
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity


class Bird:
    def __init__(self, name, area, z_min, z_max):
        self.name = name
        self.area = area
        self.z_min, self.z_max = z_min, z_max
        self.x = random.uniform(-area, area)
        self.y = random.uniform(-area, area)
        self.z = random.uniform(z_min, z_max)
        self.yaw = random.uniform(-math.pi, math.pi)
        self.pick_target()

    def pick_target(self):
        self.tx = random.uniform(-self.area, self.area)
        self.ty = random.uniform(-self.area, self.area)
        self.tz = random.uniform(self.z_min, self.z_max)

    def step(self, speed, dt):
        dx, dy, dz = self.tx - self.x, self.ty - self.y, self.tz - self.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 0.5:
            self.pick_target()
            return
        step = min(speed * dt, dist)
        self.x += dx / dist * step
        self.y += dy / dist * step
        self.z += dz / dist * step
        self.yaw = math.atan2(dy, dx)


class BirdWander(Node):
    def __init__(self):
        super().__init__('bird_wander')
        self.declare_parameter('world', 'turret_world')
        self.declare_parameter('count', 4)
        self.declare_parameter('name_prefix', 'bird')
        self.declare_parameter('area', 12.0)
        self.declare_parameter('z_min', 3.0)
        self.declare_parameter('z_max', 8.0)
        self.declare_parameter('speed', 2.0)

        world = self.get_parameter('world').value
        n = int(self.get_parameter('count').value)
        prefix = self.get_parameter('name_prefix').value
        area = float(self.get_parameter('area').value)
        z_min = float(self.get_parameter('z_min').value)
        z_max = float(self.get_parameter('z_max').value)
        self.speed = float(self.get_parameter('speed').value)

        self.birds = [Bird(f'{prefix}_{i+1}', area, z_min, z_max) for i in range(n)]

        srv = f'/world/{world}/set_pose'
        self.cli = self.create_client(SetEntityPose, srv)
        self.ready = False
        self.get_logger().info(f'connecting to {srv} for {n} birds ...')

        self.last = self.get_clock().now()
        self.timer = self.create_timer(1.0 / 15.0, self.tick)

    def tick(self):
        if not self.ready:
            if not self.cli.service_is_ready():
                return
            self.ready = True
            self.get_logger().info('wandering birds')
        now = self.get_clock().now()
        dt = (now - self.last).nanoseconds * 1e-9
        self.last = now
        dt = min(max(dt, 1e-3), 0.2)
        for b in self.birds:
            b.step(self.speed, dt)
            req = SetEntityPose.Request()
            req.entity = Entity(name=b.name, type=Entity.MODEL)
            p = Pose()
            p.position.x = b.x
            p.position.y = b.y
            p.position.z = b.z
            p.orientation.z = math.sin(b.yaw / 2.0)
            p.orientation.w = math.cos(b.yaw / 2.0)
            req.pose = p
            self.cli.call_async(req)


def main():
    rclpy.init()
    node = BirdWander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
