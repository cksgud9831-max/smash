#!/usr/bin/env python3
"""
camera_zoom.py  -  live digital zoom for the turret camera.

Subscribes:  /turret_camera/image_raw   (sensor_msgs/Image)
Publishes :  /turret_camera/image_zoom  (sensor_msgs/Image)

A Tk slider sets the zoom factor (1x..10x). Each frame is centre-cropped
by 1/zoom and scaled back to full size, so the drone appears larger.
View /turret_camera/image_zoom in rqt_image_view.
"""
import threading
import tkinter as tk

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraZoom(Node):
    def __init__(self):
        super().__init__('camera_zoom')
        self.zoom = 1.0
        self.sub = self.create_subscription(
            Image, '/turret_camera/image_raw', self.on_image, 10)
        self.pub = self.create_publisher(Image, '/turret_camera/image_zoom', 10)

    def on_image(self, msg):
        # ROS Image -> numpy (assume bgr8/rgb8, 3 channels)
        h, w = msg.height, msg.width
        buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, -1)

        z = max(1.0, float(self.zoom))
        # centre crop of size (w/z, h/z)
        cw, ch = int(w / z), int(h / z)
        x0 = (w - cw) // 2
        y0 = (h - ch) // 2
        crop = buf[y0:y0 + ch, x0:x0 + cw]
        # scale back up to full frame
        zoomed = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        zoomed = np.ascontiguousarray(zoomed, dtype=np.uint8)

        out = Image()
        out.header = msg.header
        out.height = h
        out.width = w
        out.encoding = msg.encoding
        out.is_bigendian = msg.is_bigendian
        out.step = w * zoomed.shape[2]
        out.data = zoomed.tobytes()
        self.pub.publish(out)


def main():
    rclpy.init()
    node = CameraZoom()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    root = tk.Tk()
    root.title('Camera Zoom')
    root.geometry('340x120')

    tk.Label(root, text='ZOOM (x)', font=('TkDefaultFont', 10, 'bold')).pack()

    def on_zoom(v):
        node.zoom = float(v)

    s = tk.Scale(root, from_=1.0, to=10.0, resolution=0.1,
                 orient=tk.HORIZONTAL, length=300, command=on_zoom)
    s.set(1.0)
    s.pack()
    tk.Label(root, text='view /turret_camera/image_zoom in rqt').pack()

    def close():
        node.destroy_node()
        rclpy.shutdown()
        root.destroy()

    root.protocol('WM_DELETE_WINDOW', close)
    root.mainloop()


if __name__ == '__main__':
    main()
