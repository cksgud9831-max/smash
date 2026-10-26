"""측정 도구용 모의 시계 감시.

2026-09-22 실측: 측정 도구와 target_mover(파이썬 노드)가 /clock 을 46분 동안 받지 못하다가
한꺼번에 풀린 일이 있었다. 같은 동안 FCS 는 시계를 정상으로 받았다(Fast DDS 공유메모리 수신이
막힌 것으로 추정). rclpy 의 use_sim_time 시계에 기대면 그동안 타이머와 대기 루프가 전부 멈추고,
측정 결과는 멀쩡해 보이는 채로 틀어진다.

그래서 /clock 을 전용 노드·전용 스레드에서 최신 값 1개만 받고(밀린 메시지를 쌓지 않는다),
마지막으로 값이 바뀐 실제 시각을 기억해 멈춤을 잰다. 멈춤이 길면 그 측정은 무효로 표시한다.
"""

from __future__ import annotations

import threading
import time


class StallTracker:
    """값 갱신 간격(실제 시각)을 재어 가장 긴 멈춤을 기록한다. ROS 없이 시험할 수 있다."""

    def __init__(self, stall_threshold_s: float = 1.0):
        self.stall_threshold_s = float(stall_threshold_s)
        self.latest = None
        self.last_update_wall = None
        self.max_gap_s = 0.0
        self.stalls = 0
        self._in_stall = False

    def update(self, value: float, wall: float) -> None:
        if self.latest is not None and value <= self.latest:
            return  # 같은 값(또는 되감긴 값)은 진행이 아니다
        self.check(wall)
        self.latest = float(value)
        self.last_update_wall = float(wall)
        self._in_stall = False

    def check(self, wall: float) -> float:
        """지금까지의 멈춤 길이(초)를 돌려주고 최대치·횟수를 갱신한다."""
        if self.last_update_wall is None:
            return 0.0
        gap = wall - self.last_update_wall
        self.max_gap_s = max(self.max_gap_s, gap)
        if gap >= self.stall_threshold_s and not self._in_stall:
            self.stalls += 1
            self._in_stall = True
        return gap

    def reset_stats(self) -> None:
        self.max_gap_s = 0.0
        self.stalls = 0


class SimClockWatcher:
    """/clock 을 전용 스레드에서 받는다. now() 는 마지막으로 받은 모의 시각."""

    def __init__(self, stall_threshold_s: float = 1.0):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from rosgraph_msgs.msg import Clock

        self.tracker = StallTracker(stall_threshold_s)
        self._lock = threading.Lock()
        self._node = Node("sim_clock_watcher")
        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._node.create_subscription(Clock, "/clock", self._on_clock, qos)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()
        self._rclpy = rclpy

    def _on_clock(self, msg) -> None:
        with self._lock:
            self.tracker.update(msg.clock.sec + msg.clock.nanosec * 1e-9, time.monotonic())

    def now(self) -> float:
        with self._lock:
            return self.tracker.latest if self.tracker.latest is not None else 0.0

    def gap_s(self) -> float:
        with self._lock:
            return self.tracker.check(time.monotonic())

    def stats(self) -> dict:
        with self._lock:
            self.tracker.check(time.monotonic())
            return {"max_gap_s": round(self.tracker.max_gap_s, 2), "stalls": self.tracker.stalls}

    def reset_stats(self) -> None:
        with self._lock:
            self.tracker.reset_stats()

    def shutdown(self) -> None:
        self._executor.shutdown()
        self._thread.join(timeout=2.0)
        self._node.destroy_node()
