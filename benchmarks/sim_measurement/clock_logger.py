#!/usr/bin/env python3
"""모의 시각 <-> 실제 시각 대응(/clock)과 HUD 영상·격발 이벤트 도착 시각을 기록한다 (지연 분석용).

5초마다 JSON 을 덮어쓰므로 강제 종료돼도 최대 5초만 잃는다. analyze_latency.py 의 입력이다.

사용 예:
    python3 clock_logger.py runs/case.lat_clock.json
"""

from __future__ import annotations

import json
import os
import struct
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from std_msgs.msg import String


class Logger(Node):
    def __init__(self, out: str):
        super().__init__("latency_clock_logger")
        self.out = out
        self.clock: list = []  # (wall, sim)
        self.ann: list = []    # (wall_arrival, image_stamp)
        self.fire: list = []   # (wall, fire_time_s)
        self.lock = threading.Lock()
        self._last_clock_wall = 0.0
        # 최신 값 1개만 받는다. 깊이가 크면 처리가 밀릴 때 오래된 시각이 쌓여 대응이 틀어진다.
        q = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Clock, "/clock", self.on_clock, q)
        self.create_subscription(Image, "/turret_camera/image_annotated", self.on_ann, 5, raw=True)
        self.create_subscription(String, "/smash_fcs/fire_event", self.on_fire, 50)

    def on_clock(self, m):
        w = time.time()
        if w - self._last_clock_wall < 0.005:  # 약 200 Hz 로 솎는다
            return
        self._last_clock_wall = w
        with self.lock:
            self.clock.append((w, m.clock.sec + m.clock.nanosec * 1e-9))

    def on_ann(self, raw):
        w = time.time()
        # 역직렬화 비용을 피하려고 raw CDR 에서 header.stamp 만 읽는다:
        # 4바이트 캡슐화 헤더 다음이 int32 sec, uint32 nanosec.
        sec, nsec = struct.unpack_from("<iI", raw, 4)
        with self.lock:
            self.ann.append((w, sec + nsec * 1e-9))

    def on_fire(self, m):
        w = time.time()
        with self.lock:
            self.fire.append((w, json.loads(m.data)["fire_time_s"]))

    def dump(self):
        with self.lock:
            data = {"clock": self.clock, "ann": self.ann, "fire": self.fire}
        with open(self.out + ".tmp", "w") as f:
            json.dump(data, f)
        os.replace(self.out + ".tmp", self.out)


def main() -> None:
    rclpy.init()
    n = Logger(sys.argv[1])
    n.create_timer(5.0, n.dump)
    try:
        rclpy.spin(n)
    except BaseException:
        pass
    n.dump()


if __name__ == "__main__":
    main()
