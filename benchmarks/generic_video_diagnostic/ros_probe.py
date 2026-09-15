"""Synthetic ROS Image transfer diagnostic, independent of project nodes."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

OUT = Path('/mnt/d/Aiming/results/generic_video_diagnostic')
OUT.mkdir(parents=True, exist_ok=True)
TOPIC = '/generic_video_diagnostic/image'

def pack(values):
    return {'median': float(np.median(values)), 'p95': float(np.percentile(values, 95)), 'mean': float(np.mean(values))}

if '--publisher' in sys.argv:
    width, height = map(int, sys.argv[-2:])
    rclpy.init()
    node = Node('generic_image_source')
    pub = node.create_publisher(Image, TOPIC, qos_profile_sensor_data)
    msg = Image(height=height, width=width, encoding='bgr8', step=width*3, data=bytes(width*height*3))
    def tick():
        stamp = time.monotonic_ns()
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 1000000000)
        pub.publish(msg)
    node.create_timer(1/30, tick)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
else:
    rclpy.init()
    results = []
    for width, height in [(640,480),(1280,720)]:
        node = Node('generic_image_receiver')
        bridge = CvBridge()
        samples = []
        def receive(msg):
            start = time.monotonic_ns()
            sent = msg.header.stamp.sec*1000000000 + msg.header.stamp.nanosec
            image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            done = time.monotonic_ns()
            samples.append({'arrival_ns': start, 'delivery_ms': (start-sent)/1e6, 'cv_bridge_ms': (done-start)/1e6})
        sub = node.create_subscription(Image, TOPIC, receive, qos_profile_sensor_data)
        p = subprocess.Popen([sys.executable, __file__, '--publisher', str(width), str(height)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic()+20
            while len(samples)<140 and time.monotonic()<deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
        finally:
            p.terminate()
            p.wait(timeout=5)
        measured = samples[20:]
        if len(measured)<2:
            raise RuntimeError('Not enough ROS samples')
        row = {'size': f'{width}x{height}', 'samples': len(measured),
               'receive_hz': (len(measured)-1)*1e9/(measured[-1]['arrival_ns']-measured[0]['arrival_ns']),
               'delivery_ms': pack([s['delivery_ms'] for s in measured]),
               'cv_bridge_ms': pack([s['cv_bridge_ms'] for s in measured])}
        results.append(row)
        (OUT / f'ros_{width}x{height}_samples.json').write_text(json.dumps(measured, indent=2))
        print(json.dumps(row), flush=True)
        node.destroy_node()
    rclpy.shutdown()
    (OUT/'ros_summary.json').write_text(json.dumps(results, indent=2))
