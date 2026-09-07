#!/usr/bin/env python3
"""
drone_flight.py  -  fly one drone in a horizontal circle overhead.

Publishes the drone's pose to Gazebo via /world/<world>/set_pose so the
drone model (spawned separately) moves along the circle.  Kinematic motion
(no physics/thrust) -- perfect as a moving target for the turret camera.

Params (ros2 param / launch):
  world        : gz world name              (default: turret_world)
  model        : drone model name in gz     (default: drone)
  radius       : circle radius [m]          (default: 6.0)
  altitude     : height above ground [m]    (default: 5.0)
  period       : seconds per lap            (default: 20.0)
  center_x/y   : circle centre [m]          (default: 0, 0)
"""
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity


class DroneFlight(Node):
    def __init__(self):
        super().__init__('drone_flight')
        self.declare_parameter('world', 'turret_world')
        self.declare_parameter('model', 'drone')
        self.declare_parameter('radius', 6.0)
        self.declare_parameter('altitude', 5.0)
        self.declare_parameter('period', 20.0)
        self.declare_parameter('center_x', 0.0)
        self.declare_parameter('center_y', 0.0)

        self.world = self.get_parameter('world').value
        self.model = self.get_parameter('model').value
        self.radius = self.get_parameter('radius').value
        self.alt = self.get_parameter('altitude').value
        self.period = self.get_parameter('period').value
        self.cx = self.get_parameter('center_x').value
        self.cy = self.get_parameter('center_y').value

        srv = f'/world/{self.world}/set_pose'
        self.cli = self.create_client(SetEntityPose, srv)
        self.get_logger().info(f'waiting for {srv} ...')
        self.cli.wait_for_service()
        self.get_logger().info('set_pose service ready, flying drone')

        self.t0 = self.get_clock().now()
        self.timer = self.create_timer(1.0 / 30.0, self.tick)

    def tick(self):
        t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        ang = 2.0 * math.pi * (t / self.period)
        x = self.cx + self.radius * math.cos(ang)
        y = self.cy + self.radius * math.sin(ang)
        z = self.alt
        # yaw so the drone noses along its direction of travel
        yaw = ang + math.pi / 2.0

        req = SetEntityPose.Request()
        req.entity = Entity(name=self.model, type=Entity.MODEL)
        p = Pose()
        p.position.x = x
        p.position.y = y
        p.position.z = z
        p.orientation.z = math.sin(yaw / 2.0)
        p.orientation.w = math.cos(yaw / 2.0)
        req.pose = p
        self.cli.call_async(req)


def main():
    rclpy.init()
    node = DroneFlight()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
