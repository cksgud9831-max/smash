#!/usr/bin/env python3
"""Measure the raw Gazebo camera rate without FCS, YOLO, HUD, or viewer load."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import struct
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from std_msgs.msg import String


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def gap_stats(times: list[float]) -> dict:
    gaps = [(b - a) * 1000.0 for a, b in zip(times, times[1:]) if b >= a]
    return {
        "count": len(gaps),
        "median_ms": round(statistics.median(gaps), 3) if gaps else None,
        "p95_ms": round(percentile(gaps, 0.95), 3) if gaps else None,
        "max_ms": round(max(gaps), 3) if gaps else None,
    }


def summarize(wall_times: list[float], stamps: list[float], clock: list[tuple[float, float]]) -> dict:
    wall_elapsed = wall_times[-1] - wall_times[0] if len(wall_times) >= 2 else 0.0
    sim_elapsed = stamps[-1] - stamps[0] if len(stamps) >= 2 else 0.0
    clock_wall = clock[-1][0] - clock[0][0] if len(clock) >= 2 else 0.0
    clock_sim = clock[-1][1] - clock[0][1] if len(clock) >= 2 else 0.0
    clock_progress_walls = [w for (w0, s0), (w, s) in zip(clock, clock[1:]) if s > s0]
    clock_progress_walls.insert(0, clock[0][0] if clock else 0.0)
    clock_gaps = gap_stats(clock_progress_walls) if clock else gap_stats([])
    return {
        "frames": len(wall_times),
        "wall_elapsed_s": round(wall_elapsed, 3),
        "sim_elapsed_s": round(sim_elapsed, 3),
        "arrival_fps_wall": round((len(wall_times) - 1) / wall_elapsed, 3) if wall_elapsed > 0 else None,
        "sensor_fps_sim": round((len(stamps) - 1) / sim_elapsed, 3) if sim_elapsed > 0 else None,
        "frame_gap_wall": gap_stats(wall_times),
        "frame_gap_sim": gap_stats(stamps),
        "clock_samples": len(clock),
        "rtf": round(clock_sim / clock_wall, 4) if clock_wall > 0 else None,
        "clock_progress_gap_wall": clock_gaps,
        "clock_stalls_over_0_5s": sum(
            1 for (a, b) in zip(clock_progress_walls, clock_progress_walls[1:]) if b - a > 0.5
        ),
    }


class CameraProbe(Node):
    def __init__(self) -> None:
        super().__init__("camera_only_probe")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.wall_times: list[float] = []
        self.stamps: list[float] = []
        self.clock: list[tuple[float, float]] = []
        self.state_times: list[float] = []
        self.annotated_times: list[float] = []
        self.last_clock_wall = 0.0
        self.create_subscription(Image, "/turret_camera/image_raw", self.on_image, qos, raw=True)
        self.create_subscription(Clock, "/clock", self.on_clock, qos)
        self.create_subscription(String, "/smash_fcs/state", self.on_state, 10)
        self.create_subscription(Image, "/turret_camera/image_annotated", self.on_annotated, qos, raw=True)

    def on_image(self, raw) -> None:
        now = time.time()
        sec, nsec = struct.unpack_from("<iI", raw, 4)
        self.wall_times.append(now)
        self.stamps.append(sec + nsec * 1e-9)

    def on_clock(self, msg: Clock) -> None:
        now = time.time()
        if now - self.last_clock_wall < 0.005:
            return
        self.last_clock_wall = now
        self.clock.append((now, msg.clock.sec + msg.clock.nanosec * 1e-9))

    def on_state(self, _msg: String) -> None:
        self.state_times.append(time.time())

    def on_annotated(self, _raw) -> None:
        self.annotated_times.append(time.time())


def write_report(data: dict, report_path: str, cleanup_status: str) -> None:
    m = data.get("metrics", {})
    mode = data.get("mode", "camera-only")
    resource = {}
    resource_path = data.get("resource_path")
    if resource_path and os.path.isfile(resource_path):
        with open(resource_path, encoding="utf-8") as f:
            resource = json.load(f).get("summary", {})
    stage = {
        "camera-only": "P0-1 camera-only 기준선",
        "fcs-raw": "P1-1 FCS raw-only",
        "fcs-annotated": "P2-1 annotated/HUD 발행",
        "fcs-viewer": "P2-2 viewer/MJPEG 발행",
        "scene-only": "P1-2 scene-only",
        "turret-command-only": "P1-3 turret_command 증분",
        "drone-only": "P1-4 드론 모델 증분",
    }.get(mode, mode)
    hypothesis = {
        "camera-only": "FCS, YOLO, HUD, viewer를 제거해도 raw 카메라가 10 FPS 미만이면 Gazebo 렌더링/센서 생성이 현재 6 FPS 병목의 주원인이라는 가설을 검증한다.",
        "fcs-raw": "FCS와 CUDA YOLO를 추가하되 annotated/HUD/viewer를 끈 상태에서 raw 입력률과 FCS callback 처리율이 유지되는지 검증한다.",
        "fcs-annotated": "FCS의 annotated/HUD 영상 생성과 ROS 발행을 추가했을 때 전체 입력률과 처리율이 유지되는지 검증한다.",
        "fcs-viewer": "annotated/HUD에 웹 viewer/MJPEG까지 추가했을 때 실제 운용 표시 경로의 처리율이 유지되는지 검증한다.",
        "scene-only": "전체 smash_scene과 드론은 유지하되 FCS/YOLO만 끄고 장면 구성 자체의 비용을 측정한다.",
        "turret-command-only": "드론과 FCS를 끄고 turret_command만 추가해 60 Hz 명령 노드의 증분 비용을 측정한다.",
        "drone-only": "FCS와 turret_command를 끄고 드론 모델만 추가해 표적 렌더링·물리·PosePublisher 비용을 측정한다.",
    }.get(mode, mode)
    fps = m.get("arrival_fps_wall")
    valid = (
        m.get("frames", 0) >= 30
        and fps is not None
        and m.get("clock_stalls_over_0_5s", 0) == 0
    )
    if not valid:
        verdict = "INVALID"
        branch = "카메라 표본 부족 또는 `/clock` stall이 있어 같은 설정으로 재시험한다."
    elif fps >= 20.0:
        verdict = "PASS"
        branch = (
            "FCS raw-only 부하에서도 raw 입력이 20 FPS 이상 유지된다. 다음 분기는 annotated/HUD를 추가한다."
            if mode == "fcs-raw" else
            "annotated/HUD 또는 viewer를 추가해도 20 FPS 이상 유지된다. 실제 기동·격발 회귀 시험으로 진행한다."
            if mode in ("fcs-annotated", "fcs-viewer") else
            "전체 장면만 추가해도 20 FPS 이상 유지된다. FCS/PyTorch/CUDA가 주 병목 후보다."
            if mode == "scene-only" else
            "단일 구성요소 추가 후에도 20 FPS 이상 유지된다. 이 요소 단독은 1차 병목이 아니다."
            if mode in ("turret-command-only", "drone-only") else
            "카메라 생성은 20 FPS 이상이다. 다음 분기는 bridge/DDS 및 FCS 입출력 경로 계측이다."
        )
    else:
        verdict = "FAIL"
        if mode in ("fcs-raw", "fcs-annotated", "fcs-viewer"):
            branch = (
                "FCS/CUDA 추가 후 raw 벽시계 FPS와 RTF가 함께 하락했다. 센서 모의 시각 FPS와 FCS callback률을 "
                "대조해 프레임 유실이 아닌 시스템 자원 경합인지 확인하고, 다음 분기에서 CPU/no-op FCS와 비교한다."
            )
        elif mode == "scene-only":
            branch = (
                "FCS/YOLO 없이 전체 장면만 추가해도 raw 벽시계 FPS와 RTF가 하락했다. 드론 렌더링, "
                "ros2_control 및 추가 노드 부하를 분리하는 장면 축소 시험으로 진행한다."
            )
        elif mode in ("turret-command-only", "drone-only"):
            branch = "이 구성요소 하나만으로 20 FPS 아래로 하락했다. 반복 결과와 자원 사용률로 증분 비용을 확정한다."
        else:
            branch = (
                "raw camera-only가 10 FPS 미만이므로 Gazebo 렌더링/센서가 P0 병목으로 확정된다."
                if fps < 10.0 else
                "raw camera-only가 10~20 FPS이다. 렌더링 백엔드 A/B 시험으로 원인을 더 분리한다."
            )
    data["verdict"] = verdict
    data["cleanup_status"] = cleanup_status
    with open(data["result_path"], "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    def val(path: list[str], suffix: str = "") -> str:
        x = m
        for key in path:
            x = x.get(key) if isinstance(x, dict) else None
        return "N/A" if x is None else f"{x}{suffix}"

    def resource_val(path: list[str], suffix: str = "") -> str:
        x = resource
        for key in path:
            x = x.get(key) if isinstance(x, dict) else None
        return "N/A" if x is None else f"{x}{suffix}"

    lines = [
        f"# 분기 보고서 — {data['run_id']}", "",
        f"- 판정: **{verdict}**", f"- 실행일: {data['started_at']}",
        f"- 단계: {stage}", "",
        "## 1. 목적과 가설", "",
        hypothesis, "",
        "## 2. 실행 환경과 설정", "",
        f"- 렌더링 백엔드: `{data['backend']}`", f"- 해상도: {data['camera_px']}×{data['camera_px']}",
        f"- HFOV: {data['camera_hfov']} rad", f"- 측정시간: {data['duration_s']} s (벽시계)",
        f"- 실행 모드: `{mode}`", f"- 호스트: `{data['host']}`", "",
        "## 3. 실행 명령", "", f"```bash\n{data['command']}\n```", "",
        "## 4. 원본 파일", "",
        f"- JSON: [{os.path.basename(data['result_path'])}]({data['result_path'].replace(os.sep, '/')})",
        f"- 실행 로그: [{os.path.basename(data['log_path'])}]({data['log_path'].replace(os.sep, '/')})", "",
        "## 5. 측정 결과", "",
        "| 지표 | 값 |", "|---|---:|",
        f"| 수신 프레임 | {val(['frames'])} |", f"| raw 도착률(벽시계) | {val(['arrival_fps_wall'], ' FPS')} |",
        f"| FCS state/callback 추정률 | {val(['state_fps_wall'], ' FPS')} |",
        f"| annotated 도착률 | {val(['annotated_fps_wall'], ' FPS')} |",
        f"| 센서 timestamp 기준 | {val(['sensor_fps_sim'], ' FPS')} |",
        f"| 프레임 간격 P50 | {val(['frame_gap_wall', 'median_ms'], ' ms')} |",
        f"| 프레임 간격 P95 | {val(['frame_gap_wall', 'p95_ms'], ' ms')} |",
        f"| 프레임 최대 공백 | {val(['frame_gap_wall', 'max_ms'], ' ms')} |",
        f"| RTF | {val(['rtf'])} |", f"| `/clock` 0.5초 초과 stall | {val(['clock_stalls_over_0_5s'])} |", "",
        "### 자원 사용률", "",
        "| 프로세스/자원 | 평균 | P95 | 최대 |", "|---|---:|---:|---:|",
        f"| Gazebo CPU | {resource_val(['process_cpu_pct', 'gazebo', 'mean'], '%')} | {resource_val(['process_cpu_pct', 'gazebo', 'p95'], '%')} | {resource_val(['process_cpu_pct', 'gazebo', 'max'], '%')} |",
        f"| FCS CPU | {resource_val(['process_cpu_pct', 'smash_fcs', 'mean'], '%')} | {resource_val(['process_cpu_pct', 'smash_fcs', 'p95'], '%')} | {resource_val(['process_cpu_pct', 'smash_fcs', 'max'], '%')} |",
        f"| turret_command CPU | {resource_val(['process_cpu_pct', 'turret_command', 'mean'], '%')} | {resource_val(['process_cpu_pct', 'turret_command', 'p95'], '%')} | {resource_val(['process_cpu_pct', 'turret_command', 'max'], '%')} |",
        f"| GPU 사용률 | {resource_val(['gpu_util_pct', 'mean'], '%')} | {resource_val(['gpu_util_pct', 'p95'], '%')} | {resource_val(['gpu_util_pct', 'max'], '%')} |",
        f"| GPU 메모리 | {resource_val(['gpu_memory_mb', 'mean'], ' MB')} | {resource_val(['gpu_memory_mb', 'p95'], ' MB')} | {resource_val(['gpu_memory_mb', 'max'], ' MB')} |", "",
        "## 6. 판정 및 병목 귀속", "", branch, "",
        ("이 시험만으로 FCS 계산, CUDA/PyTorch, 추가 ROS 노드, 드론 모델 렌더링의 자원 비용을 서로 분리할 수 없다."
         if mode in ("fcs-raw", "fcs-annotated", "fcs-viewer") else
         "이 시험만으로 software renderer 내부의 rasterization 비용과 카메라 센서 플러그인 비용은 분리할 수 없다."), "",
        "## 7. 다음 분기", "", branch, "",
        "## 8. 종료 확인", "", f"- 잔류 프로세스 확인: `{cleanup_status}`", "",
    ]
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--run-id")
    parser.add_argument("--backend", default="software")
    parser.add_argument("--camera-px", type=int, default=640)
    parser.add_argument("--camera-hfov", type=float, default=0.2)
    parser.add_argument("--log-path", default="")
    parser.add_argument("--command", default="")
    parser.add_argument("--mode", default="camera-only", choices=(
        "camera-only", "scene-only", "fcs-raw", "fcs-annotated", "fcs-viewer",
        "turret-command-only", "drone-only"
    ))
    parser.add_argument("--resource-path", default="")
    parser.add_argument("--render-report", nargs=3, metavar=("JSON", "REPORT", "CLEANUP"))
    args = parser.parse_args()
    if args.render_report:
        json_path, report_path, cleanup = args.render_report
        with open(json_path, encoding="utf-8") as f:
            write_report(json.load(f), report_path, cleanup)
        return 0

    rclpy.init()
    node = CameraProbe()
    started = time.monotonic()
    try:
        while time.monotonic() - started < args.duration:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    metrics = summarize(node.wall_times, node.stamps, node.clock)

    def observed_rate(times: list[float]) -> float | None:
        if len(times) < 2 or times[-1] <= times[0]:
            return None
        return round((len(times) - 1) / (times[-1] - times[0]), 3)

    metrics["state_messages"] = len(node.state_times)
    metrics["state_fps_wall"] = observed_rate(node.state_times)
    metrics["annotated_frames"] = len(node.annotated_times)
    metrics["annotated_fps_wall"] = observed_rate(node.annotated_times)
    data = {
        "run_id": args.run_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "backend": args.backend,
        "camera_px": args.camera_px,
        "camera_hfov": args.camera_hfov,
        "duration_s": args.duration,
        "host": platform.platform(),
        "command": args.command,
        "mode": args.mode,
        "resource_path": os.path.abspath(args.resource_path) if args.resource_path else "",
        "result_path": os.path.abspath(args.output),
        "log_path": os.path.abspath(args.log_path),
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(json.dumps(data["metrics"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
