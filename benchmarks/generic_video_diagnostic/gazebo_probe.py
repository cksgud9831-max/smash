"""Stock Gazebo camera -> ROS image arrival measurement. No project assets."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

OUT = Path('/mnt/d/Aiming/results/generic_video_diagnostic')
OUT.mkdir(parents=True, exist_ok=True)
partition = 'generic_camera_probe_' + str(os.getpid())
env = os.environ.copy()
env.update(GZ_PARTITION=partition, GALLIUM_DRIVER='d3d12', MESA_D3D12_DEFAULT_ADAPTER_NAME='NVIDIA')
env.pop('LIBGL_ALWAYS_SOFTWARE', None)
world = '/opt/ros/jazzy/opt/gz_sim_vendor/share/gz/gz-sim8/worlds/camera_sensor.sdf'
processes = []
logs = []

def start(args, name):
    log = (OUT/name).open('w')
    logs.append(log)
    p = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    processes.append(p)
    return p

def owned():
    match = ('GZ_PARTITION=' + partition).encode()
    found = []
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                if match in (p/'environ').read_bytes().split(b'\0'):
                    found.append(int(p.name))
            except (OSError, PermissionError):
                pass
    return found

rclpy.init()
node = Node('generic_stock_camera_receiver')
cv = CvBridge()
samples = []
def receive(msg):
    now = time.monotonic_ns()
    img = cv.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    done = time.monotonic_ns()
    samples.append({'arrival_ns': now, 'sim_stamp_ns': msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec,
                    'conversion_ms': (done-now)/1e6, 'width': msg.width, 'height': msg.height, 'encoding': msg.encoding})
node.create_subscription(Image, '/camera', receive, qos_profile_sensor_data)
try:
    start(['gz','sim','-s','-r','-v','3',world], 'stock_camera_server.log')
    time.sleep(8)
    start(['ros2','run','ros_gz_bridge','parameter_bridge','/camera@sensor_msgs/msg/Image[gz.msgs.Image'], 'stock_camera_bridge.log')
    deadline = time.monotonic()+35
    first = None
    while time.monotonic()<deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if samples and first is None:
            first = time.monotonic()
        if first is not None and time.monotonic()-first>=25:
            break
    measured = samples[30:]
    if len(measured)<2:
        raise RuntimeError(f'Not enough camera samples: {len(samples)}')
    wall = (measured[-1]['arrival_ns']-measured[0]['arrival_ns'])/1e9
    sim = (measured[-1]['sim_stamp_ns']-measured[0]['sim_stamp_ns'])/1e9
    gaps = np.diff([s['arrival_ns'] for s in measured])/1e6
    report = {'samples':len(measured), 'wall_seconds':wall, 'sim_seconds':sim,
              'wall_receive_hz':(len(measured)-1)/wall, 'sim_receive_hz':(len(measured)-1)/sim,
              'stamp_progress_per_wall_second':sim/wall,
              'gap_ms_median':float(np.median(gaps)), 'gap_ms_p95':float(np.percentile(gaps,95)),
              'conversion_ms_median':float(np.median([s['conversion_ms'] for s in measured])),
              'conversion_ms_p95':float(np.percentile([s['conversion_ms'] for s in measured],95)),
              'width':measured[0]['width'], 'height':measured[0]['height'], 'encoding':measured[0]['encoding']}
    (OUT/'stock_camera_summary.json').write_text(json.dumps(report,indent=2))
    (OUT/'stock_camera_samples.json').write_text(json.dumps(measured,indent=2))
    print(json.dumps(report),flush=True)
finally:
    node.destroy_node()
    rclpy.shutdown()
    for pid in owned():
        try: os.kill(pid, signal.SIGTERM)
        except ProcessLookupError: pass
    time.sleep(2)
    for pid in owned():
        try: os.kill(pid, signal.SIGKILL)
        except ProcessLookupError: pass
    for p in processes:
        p.wait(timeout=5)
    for log in logs:
        log.close()
    print('remaining_owned_processes', owned(), flush=True)
