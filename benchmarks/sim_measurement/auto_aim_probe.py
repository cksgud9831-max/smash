#!/usr/bin/env python3
"""SMASH Gazebo 시뮬레이터 자동 조준 + 측정 프로브 (자동 격발 전제).

순서:
    1. 표적 기하(목표 좌표)로 포탑을 돌린다.
    2. 포착을 기다린다. 좁은 화각에서 초기 조준이 빗나가면 tilt 를 반화각보다 작은 간격으로 훑는다.
    3. 정렬. FCS 는 오차의 크기(|err| px)만 발행하므로 축마다 탐침 이동 두 번으로 오차 벡터를 복원한다.
    4. window 초 동안 측정: 상태 분포, 카메라·FCS 처리율, 격발 수, 명중 판정, 빗나감 거리.

결과는 JSON 한 개로 저장한다. 측정 전용 도구이며 FCS 동작 자체는 바꾸지 않는다
(정렬 단계에서만 align_tolerance_max_px 를 잠시 줄였다가 되돌린다).

사용 예:
    python3 auto_aim_probe.py out.json 33.0 0.0 12.5 --window 60
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float64MultiArray, String

from alignment import axis_offset_from_probe, clamp_correction, probe_is_plausible
from simclock import SimClockWatcher
from tracking import LeadTracker, az_el, wrap

ALIGN_RE = re.compile(r"ALIGN\((\d+)px\)")


class Probe(Node):
    def __init__(self, subscribe_images: bool):
        # 모의 시각은 rclpy use_sim_time 대신 전용 스레드의 SimClockWatcher 로 받는다(simclock.py 참고).
        super().__init__("auto_aim_probe")
        self.clock_watcher = None
        self.lock = threading.Lock()
        self.state = None
        self.state_hist: dict[str, int] = {}
        self.recent: list[tuple[float, str]] = []
        self.fx = None
        self.width = None
        self.cam_n = 0
        self.ann_n = 0
        self.raw_n = 0
        self.hits: list[dict] = []
        self.shots = 0
        self.fires: list[dict] = []
        self.stats = None
        self.counting = False
        self.track_msg = None
        self.track_errs: list[tuple[float, float]] = []  # 측정 구간의 (정렬 오차 px, 허용오차 px)
        self.track_samples: list[dict] = []
        self.create_subscription(String, "/smash_fcs/aim_track", self.on_track, 50)
        # 표적 실측 위치(격발 판정용 브리지 토픽). --track 에서 표적을 잃었을 때 다시 겨누는 데만 쓴다.
        self.gt: list[tuple[float, tuple[float, float, float]]] = []
        self.create_subscription(PoseStamped, "/model/drone/pose", self.on_gt, qos_profile_sensor_data)
        self.pub = self.create_publisher(Float64MultiArray, "/smash_fcs/slew_cmd", 10)
        self.motion_pub = self.create_publisher(Bool, "/target_mover/start", 10)
        self.create_subscription(String, "/smash_fcs/state", self.on_state, 50)
        self.create_subscription(CameraInfo, "/turret_camera/camera_info", self.on_info, qos_profile_sensor_data)
        if subscribe_images:
            # 영상 구독 자체가 640 전송 부하를 늘리므로 전송·처리율 측정 때는 끈다(--no-images).
            self.create_subscription(Image, "/turret_camera/image_annotated", self.on_ann, 10, raw=True)
            self.create_subscription(Image, "/turret_camera/image_raw", self.on_raw, qos_profile_sensor_data, raw=True)
        self.create_subscription(String, "/smash_fcs/hit_result", self.on_hit, 200)
        self.create_subscription(String, "/smash_fcs/fire_event", self.on_fire, 200)
        self.create_subscription(String, "/smash_fcs/engagement_stats", self.on_stats, 10)
        from rcl_interfaces.srv import SetParameters
        self.param_cli = self.create_client(SetParameters, "/smash_fcs/set_parameters")

    def on_state(self, m):
        with self.lock:
            self.state = m.data
            self.recent.append((time.monotonic(), m.data))
            self.recent = self.recent[-200:]
            if self.counting:
                key = "ALIGN" if m.data.startswith("ALIGN") else m.data
                self.state_hist[key] = self.state_hist.get(key, 0) + 1

    def on_info(self, m):
        self.fx = float(m.k[0])
        self.width = int(m.width)
        if self.counting:
            self.cam_n += 1

    def on_raw(self, _m):
        if self.counting:
            self.raw_n += 1

    def on_ann(self, _m):
        if self.counting:
            self.ann_n += 1

    def on_hit(self, m):
        if self.counting:
            self.hits.append(json.loads(m.data))

    def on_fire(self, m):
        if self.counting:
            self.shots += 1
            self.fires.append(json.loads(m.data))

    def on_track(self, m):
        d = json.loads(m.data)
        with self.lock:
            self.track_msg = d
            if self.counting:
                self.track_errs.append((d["align_err_px"], d["align_tol_px"]))
                self.track_samples.append({
                    key: d.get(key) for key in (
                        "stamp", "align_err_px", "align_tol_px", "reliability",
                        "motion_solution_ready",
                        "detector_refresh", "bbox_xywh", "detector_score",
                        "reliability_reason", "tf_exact", "measurement_admitted",
                        "target_speed_mps", "target_position_world_m",
                        "target_velocity_world_mps", "motion_fit_rmse_m",
                        "motion_recent_velocity_world_mps", "motion_velocity_disagreement_mps",
                        "motion_history_count", "motion_maneuver_resets",
                        "motion_speed_collapsed", "meas_pos_world_m", "observer_world_m",
                        "motion_lateral_velocity_long_mps", "motion_lateral_velocity_recent_mps",
                    )
                })
                if self.gt:
                    self.track_samples[-1]["ground_truth_world_m"] = list(self.gt[-1][1])
                    self.track_samples[-1]["ground_truth_stamp"] = self.gt[-1][0]

    def on_gt(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        p_ = m.pose.position
        with self.lock:
            self.gt.append((t, (p_.x, p_.y, p_.z)))
            self.gt = self.gt[-20:]

    def sim_now(self) -> float:
        return self.clock_watcher.now()

    def gt_latest(self):
        with self.lock:
            return self.gt[-1] if self.gt else None

    def on_stats(self, m):
        self.stats = json.loads(m.data)

    def slew(self, dpan, dtilt):
        msg = Float64MultiArray()
        msg.data = [float(dpan), float(dtilt)]
        self.pub.publish(msg)


def set_align_tolerance(p: Probe, value_px: float) -> None:
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
    from rcl_interfaces.srv import SetParameters

    req = SetParameters.Request()
    req.parameters = [Parameter(
        name="align_tolerance_max_px",
        value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(value_px)))]
    for _ in range(5):
        if p.param_cli.wait_for_service(timeout_sec=5.0):
            fut = p.param_cli.call_async(req)
            t = time.monotonic()
            while not fut.done() and time.monotonic() - t < 10:
                time.sleep(0.05)
            if fut.done() and fut.result().results[0].successful:
                return
    raise RuntimeError("set_parameters failed")


def get_align_tolerance(p: Probe) -> float | None:
    from rcl_interfaces.srv import GetParameters

    cli = p.create_client(GetParameters, "/smash_fcs/get_parameters")
    if not cli.wait_for_service(timeout_sec=5.0):
        return None
    req = GetParameters.Request()
    req.names = ["align_tolerance_max_px"]
    fut = cli.call_async(req)
    t = time.monotonic()
    while not fut.done() and time.monotonic() - t < 10:
        time.sleep(0.05)
    if not fut.done() or not fut.result().values:
        return None
    return float(fut.result().values[0].double_value)


def move(p: Probe, axis: int, d: float) -> None:
    if abs(d) < 1e-9:
        return
    p.slew(d, 0.0) if axis == 0 else p.slew(0.0, d)


def measure(p: Probe, settle: float = 1.5, dur: float = 2.0):
    """dur 초 동안의 정렬 오차(px) 중앙값. 추적 중이 아니면 None."""
    time.sleep(settle)
    t0 = time.monotonic()
    time.sleep(dur)
    with p.lock:
        labels = [l for t, l in p.recent if t >= t0]
    errs = sorted(float(m.group(1)) for m in (ALIGN_RE.match(l) for l in labels) if m)
    if len(errs) < max(3, len(labels) // 2):
        return None, labels[-1] if labels else None
    return errs[len(errs) // 2], labels[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="결과 JSON 경로")
    ap.add_argument("x", type=float)
    ap.add_argument("y", type=float)
    ap.add_argument("z", type=float)
    ap.add_argument("--window", type=float, default=60.0, help="측정 구간 길이(초)")
    ap.add_argument("--cam-height", type=float, default=0.7, help="초기 조준용 카메라 높이 근사값(m)")
    ap.add_argument("--no-images", action="store_true", help="영상 토픽을 구독하지 않는다(전송 부하 측정용)")
    ap.add_argument("--skip-align", action="store_true", help="정렬 단계를 건너뛴다(수동 조준 시험용)")
    ap.add_argument("--track", action="store_true",
                    help="이동 표적용: 정적 정렬 대신 리드 레티클을 계속 따라가는 모의 사수로 조준한다")
    ap.add_argument("--servo-lag", type=float, default=0.1, help="--track: 명령 → 포탑 도달 지연 추정(초)")
    ap.add_argument("--offset-alpha", type=float, default=0.1, help="--track: 총구-명령 오프셋 학습률")
    ap.add_argument("--start-motion", type=float, default=None, metavar="LEAD_S",
                    help="정렬(--track 이면 추종 안정) 뒤 표적 이동 출발 신호를 보내고, 모의 시각 LEAD_S 초 뒤 측정 시작")
    ap.add_argument("--acquire-ready-hold", type=float, default=0.0, metavar="SECONDS",
                    help="표적 출발 전 READY가 이 시간만큼 연속 유지될 것을 요구한다")
    ap.add_argument("--window-marker", help="측정 구간 시작 순간 생성할 동기화 marker 파일")
    args = ap.parse_args()
    no_images = args.no_images or os.environ.get("NOIMG") == "1"

    rclpy.init()
    p = Probe(subscribe_images=not no_images)
    p.clock_watcher = SimClockWatcher()
    ex = rclpy.executors.SingleThreadedExecutor()
    ex.add_node(p)
    def spin_executor():
        try:
            ex.spin()
        except Exception:
            # 정상 종료 중 guard condition이 먼저 파괴되는 rclpy race는 결과에
            # 영향을 주지 않는다. 실행 중 예외만 다시 올려 진짜 오류를 보인다.
            if rclpy.ok():
                raise

    spin_thread = threading.Thread(target=spin_executor, daemon=True)
    spin_thread.start()

    def shutdown():
        # 실행기 스레드가 도는 중에 rclpy.shutdown() 을 부르면 종료 시 abort(코어 덤프)가 난다.
        ex.shutdown()
        spin_thread.join(timeout=5.0)
        p.clock_watcher.shutdown()
        p.destroy_node()
        rclpy.shutdown()

    log: dict = {"target": [args.x, args.y, args.z],
                 "slant_range_m": round(math.dist((0.0, 0.0, args.cam_height), (args.x, args.y, args.z)), 2),
                 "events": [], "invalid_reasons": []}
    T0 = time.monotonic()

    def invalid(reason):
        log["invalid_reasons"].append(reason)
        ev(f"INVALID: {reason}")

    def ev(msg):
        log["events"].append([round(time.monotonic() - T0, 1), msg])
        print(f"[{time.monotonic() - T0:6.1f}] {msg}", flush=True)

    def dump():
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=1, ensure_ascii=False)

    while p.state is None and time.monotonic() - T0 < 120:
        time.sleep(0.5)
    if p.state is None:
        ev("no state topic")
        dump()
        shutdown()
        return
    time.sleep(2.0)

    # 1. 목표 기하로 선회. slew_cmd 는 증분 명령이고 pan 999 는 FCS 의 "수평(0, 0) 리셋" 약속값이다.
    pan0 = math.atan2(args.y, args.x)
    tilt0 = math.atan2(args.z - args.cam_height, math.hypot(args.x, args.y))
    p.slew(999.0, 0.0)
    time.sleep(0.3)
    p.slew(pan0, tilt0)
    ev(f"slew to pan {math.degrees(pan0):.1f} tilt {math.degrees(tilt0):.1f}; fx={p.fx}")
    time.sleep(3.0)

    # 2. 포착. 카메라 높이를 근사값으로 쓰므로 근거리·좁은 화각에서는 표적이 화면 밖일 수 있다.
    def acquired():
        s = p.state
        return bool(s) and s != "SEARCH" and not s.startswith("TRACK(NO")

    t_acq = None
    ta = time.monotonic()
    half_fov = math.atan((p.width or 640) / 2.0 / (p.fx or 1578.0))
    step = 0.8 * half_fov
    sweep = [0.0]
    for k in range(1, 6):
        sweep += [-k * step, k * step]
    cur = 0.0
    log["sweep_offset_deg"] = None
    for off in sweep:
        p.slew(0.0, off - cur)
        cur = off
        t_end = time.monotonic() + (20.0 if off == 0.0 else 6.0)
        while time.monotonic() < t_end:
            if acquired():
                t_acq = time.monotonic() - ta
                log["sweep_offset_deg"] = round(math.degrees(off), 2)
                break
            time.sleep(0.2)
        if t_acq is not None:
            break
    log["acquire_s"] = t_acq
    ev(f"acquired after {t_acq} s, state={p.state}")
    if t_acq is None:
        invalid("target never acquired")

    if t_acq is not None and args.acquire_ready_hold > 0.0:
        ready_since = None
        ready_deadline = time.monotonic() + 60.0
        while time.monotonic() < ready_deadline:
            if p.state == "READY":
                ready_since = ready_since or time.monotonic()
                if time.monotonic() - ready_since >= args.acquire_ready_hold:
                    break
            else:
                ready_since = None
            time.sleep(0.05)
        if ready_since is None or time.monotonic() - ready_since < args.acquire_ready_hold:
            invalid(f"READY did not hold for {args.acquire_ready_hold:.1f}s before motion (state={p.state})")
            t_acq = None
        else:
            ev(f"READY held for {args.acquire_ready_hold:.1f} s before motion")

    def wait_solved(timeout_s: float, hold_s: float = 2.0) -> bool:
        """조준해가 나와 상태 라벨이 ALIGN(n px) 또는 READY 로 hold_s 초 유지될 때까지 기다린다.
        포착 직후에는 신뢰도가 HOLD/LOST 라 수치 오차가 발행되지 않는다(2026-09-22: 이 상태에서
        정렬을 시작해 정렬이 통째로 건너뛰어졌다)."""
        t_end = time.monotonic() + timeout_s
        since = None
        while time.monotonic() < t_end:
            s_ = p.state or ""
            if s_ == "READY" or s_.startswith("ALIGN("):
                since = since or time.monotonic()
                if time.monotonic() - since >= hold_s:
                    return True
            else:
                since = None
            time.sleep(0.1)
        return False

    # 3. 정렬. READY 상한을 작게 줄여 상태 라벨이 항상 수치 오차를 내게 한 뒤 최소 오차로 수렴시키고,
    #    측정 전에 원래 상한으로 되돌린다.
    tol_orig = get_align_tolerance(p)
    log["align_tolerance_max_px"] = tol_orig
    tracker_stop = threading.Event()
    tracker_thread = None
    if args.track and t_acq is not None:
        # 모의 사수. 포탑 명령은 (pan0, tilt0 + 탐색 오프셋)에서 시작한다(999 리셋 후 증분 합).
        tracker = LeadTracker(servo_lag_s=args.servo_lag, offset_alpha=args.offset_alpha)
        tracker.set_command(p.sim_now(), pan0, tilt0 + cur)
        log["track"] = {"servo_lag_s": args.servo_lag, "offset_alpha": args.offset_alpha}

        def gt_direction():
            """카메라 높이에서 본 표적 방향과 그 방위각·앙각 변화율 (실측 두 샘플로)."""
            with p.lock:
                g = list(p.gt[-2:])
            if len(g) < 2 or g[1][0] <= g[0][0]:
                return None
            (t0, a), (t1, b) = g
            d0 = (a[0], a[1], a[2] - args.cam_height)
            d1 = (b[0], b[1], b[2] - args.cam_height)
            az0, el0 = az_el(d0)
            az1, el1 = az_el(d1)
            return d1, ((wrap(az1 - az0)) / (t1 - t0), (el1 - el0) / (t1 - t0))

        def track_loop():
            last_stamp = None
            last_stable = time.monotonic()
            coarse_n = 0
            pending = [0.0, 0.0]
            last_send = [0.0]
            while not tracker_stop.is_set():
                time.sleep(1.0 / 60.0)
                now = p.sim_now()
                with p.lock:
                    m = p.track_msg
                state = p.state or ""
                if m is not None and m["stamp"] != last_stamp and m.get("reliability") == "STABLE":
                    last_stamp = m["stamp"]
                    last_stable = time.monotonic()
                    inc = tracker.correction(now, m["stamp"], m["fire_time"], m["lead_dir_base"], m["bore_dir_base"])
                elif (state == "SEARCH" or state.startswith("TRACK(REL LOST")) and time.monotonic() - last_stable > 0.3:
                    # 표적을 잃었다. 사람이 눈으로 다시 찾듯 실제 위치로 대강 겨눈다(격발 판정과는 무관).
                    g = gt_direction()
                    inc = tracker.coarse(now, g[0], g[1]) if g is not None else tracker.feedforward(now)
                    coarse_n += 1
                    log.setdefault("track", {})["coarse_ticks"] = coarse_n
                else:
                    inc = tracker.feedforward(now)
                # 증분은 모아서 30 Hz 로 보낸다(FCS 명령 타이머와 같은 주기, 대기열 부담 절반).
                pending[0] += inc[0]
                pending[1] += inc[1]
                if time.monotonic() - last_send[0] >= 1.0 / 30.0 and (abs(pending[0]) > 1e-7 or abs(pending[1]) > 1e-7):
                    p.slew(pending[0], pending[1])
                    pending[0] = pending[1] = 0.0
                    last_send[0] = time.monotonic()

        tracker_thread = threading.Thread(target=track_loop, daemon=True)
        tracker_thread.start()
        # 추종이 자리 잡을 때까지(READY 가 3초 이상 유지) 최대 60초 기다린다.
        t_settle = time.monotonic()
        ready_since = None
        while time.monotonic() - t_settle < 60.0:
            if p.state == "READY":
                ready_since = ready_since or time.monotonic()
                if time.monotonic() - ready_since >= 3.0:
                    break
            else:
                ready_since = None
            time.sleep(0.1)
        log["track_settle_s"] = round(time.monotonic() - t_settle, 1)
        ev(f"tracking settled after {log['track_settle_s']} s; state={p.state}")
    elif not args.skip_align and t_acq is not None:
        if not wait_solved(60.0):
            invalid(f"no aim solution before alignment (state={p.state})")
        set_align_tolerance(p, 0.5)
        t_al = time.monotonic()
        off = [0.0, 0.0]  # 초기 선회 이후 누적 pan/tilt 이동량

        def align_pass(tag: str):
            """탐침 정렬 한 바퀴. 가장 작았던 오차 지점으로 돌아가고 그 지점의 (오차, pan, tilt) 를 돌려준다."""
            best = (1e9, off[0], off[1])
            e0, lab = measure(p)
            ev(f"{tag} start err={e0} ({lab})")
            for it in range(8):
                if e0 is None:
                    e0, lab = measure(p, settle=2.0)
                    if e0 is None:
                        continue
                if e0 < best[0]:
                    best = (e0, off[0], off[1])
                if e0 <= 1.0:
                    break
                fx = p.fx or 800.0
                for ax in (0, 1):
                    k = fx * (math.cos(tilt0 + off[1]) if ax == 0 else 1.0)
                    step_px = max(8.0, min(e0, 40.0))
                    s = step_px / k
                    move(p, ax, s)
                    off[ax] += s
                    ep, _ = measure(p)
                    if ep is not None and not probe_is_plausible(e0, ep, step_px):
                        ev(f"  {tag} it{it} ax{ax}: implausible probe {e0} -> {ep} px, re-measuring")
                        ep, _ = measure(p, settle=1.0)
                        if ep is not None and not probe_is_plausible(e0, ep, step_px):
                            ep = None
                    if ep is None:
                        move(p, ax, -s)
                        off[ax] -= s
                        e0, _ = measure(p)
                        ev(f"  {tag} it{it} ax{ax} probe unusable, reverted")
                        if e0 is None:
                            break
                        continue
                    d = clamp_correction(axis_offset_from_probe(e0, ep, s, k), e0, k)
                    move(p, ax, d - s)
                    off[ax] += d - s
                    e1, lab = measure(p)
                    ev(f"  {tag} it{it} ax{ax}: e0={e0} probe={ep} -> {d * 1000:+.2f} mrad, err={e1}")
                    if e1 is None:
                        break
                    e0 = e1
                    if e0 < best[0]:
                        best = (e0, off[0], off[1])
            move(p, 0, best[1] - off[0])
            move(p, 1, best[2] - off[1])
            off[0], off[1] = best[1], best[2]
            return best

        best = align_pass("pass1")
        ef, lab = measure(p, settle=2.0, dur=3.0)
        ev(f"pass1 best err={best[0]} final err={ef}")
        # 최종 확인이 가장 작았던 오차보다 크게 나쁘면(추적 흔들림 등) 한 번 더 정렬한다.
        if ef is None or ef > max(3.0, 2.0 * best[0]) + 5.0:
            best = align_pass("pass2")
            ef, lab = measure(p, settle=2.0, dur=3.0)
            ev(f"pass2 best err={best[0]} final err={ef}")
            log["align_passes"] = 2
        log["align_best_px"] = best[0]
        log["align_final_px"] = ef
        log["align_s"] = round(time.monotonic() - t_al, 1)
        ev(f"best err={best[0]} final err={ef}")
        set_align_tolerance(p, tol_orig if tol_orig is not None else 35.0)
        time.sleep(2.0)
        with p.lock:
            tol_now = (p.track_msg or {}).get("align_tol_px")
        log["align_tol_px"] = tol_now
        if ef is None or (tol_now is not None and ef > tol_now):
            invalid(f"alignment failed (final err {ef} px, tolerance {tol_now} px)")

    # 이동 표적 출발. --track 이 없으면 총은 정렬된 자리에 그대로 두고 표적만 조준점을 지나가게 한다
    # (매복 사격). 지연 보정의 효과는 이 경우에 드러난다: 총이 표적을 따라 돌면 영상 속 정렬이
    # 격발 순간까지 유지되지만, 총이 멈춰 있으면 정렬된 영상 이후 격발까지 표적이 ω×지연만큼 지나간다.
    if args.start_motion is not None and t_acq is not None:
        # 출발은 모의 시계가 아니라 실측 위치가 실제로 움직였는지로 확인한다. 실측 스탬프는 Gazebo 가
        # 찍으므로 이 노드의 /clock 수신이 막혀도 맞다.
        g0 = p.gt_latest()
        for _ in range(5):
            p.motion_pub.publish(Bool(data=True))
            time.sleep(0.1)
        ev(f"motion start signal at sim {p.sim_now():.1f} s")
        moved = None
        t_end = time.monotonic() + 60.0
        while time.monotonic() < t_end and g0 is not None:
            g = p.gt_latest()
            if g is not None and math.dist(g[1], g0[1]) > 0.3:
                moved = g[0]
                break
            time.sleep(0.05)
        if moved is None:
            invalid("target did not start moving within 60 s after the start signal")
        else:
            log["motion_start_sim_s"] = round(moved, 2)
            ev(f"target moving since sim {moved:.1f} s (ground truth)")
            while time.monotonic() < t_end + 60.0:
                g = p.gt_latest()
                if g is not None and g[0] >= moved + args.start_motion:
                    break
                time.sleep(0.05)

    # 4. 측정 구간
    ev(f"window starts at sim {p.sim_now():.1f} s, state={p.state}")
    log["window_start_sim_s"] = round(p.sim_now(), 2)
    if args.window_marker:
        os.makedirs(os.path.dirname(os.path.abspath(args.window_marker)), exist_ok=True)
        with open(args.window_marker, "w", encoding="utf-8") as marker:
            marker.write(f"{time.time():.6f}\n")
    p.clock_watcher.reset_stats()
    with p.lock:
        p.counting = True
    tw = time.monotonic()
    time.sleep(args.window / 2)
    try:
        # 중간 시점 CPU 스냅샷. top 첫 샘플은 기동 이후 평균이라 두 번째 샘플을 남긴다.
        top = subprocess.run(["top", "-b", "-n2", "-d1", "-w", "200"], capture_output=True, text=True).stdout
        second = top.split("top - ")[-1]
        with open(os.path.splitext(args.out)[0] + ".top.txt", "w", encoding="utf-8") as f:
            f.write("top - " + "\n".join(second.splitlines()[:25]))
    except FileNotFoundError:
        pass
    time.sleep(args.window / 2)
    with p.lock:
        p.counting = False
    dt = time.monotonic() - tw
    log["clock"] = p.clock_watcher.stats()
    if log["clock"]["max_gap_s"] > 1.0:
        invalid(f"/clock stalled for {log['clock']['max_gap_s']} s during the window")
    tracker_stop.set()
    if tracker_thread is not None:
        tracker_thread.join(timeout=2.0)  # 종료 뒤에 발행하지 않도록 먼저 멈춘다

    miss = [h["miss_distance_m"] for h in p.hits if h.get("miss_distance_m") is not None]
    verdicts: dict[str, int] = {}
    for h in p.hits:
        verdicts[h["verdict"]] = verdicts.get(h["verdict"], 0) + 1
    total_states = sum(p.state_hist.values()) or 1
    log.update({
        "window_s": round(dt, 1),
        "fx": p.fx,
        "camera_px": p.width,
        "camera_hz": round(p.cam_n / dt, 2),
        "image_raw_hz": None if no_images else round(p.raw_n / dt, 2),
        "fcs_hz": None if no_images else round(p.ann_n / dt, 2),
        "state_frac": {k: round(v / total_states, 3) for k, v in sorted(p.state_hist.items())},
        "ready_frac": round(p.state_hist.get("READY", 0) / total_states, 3),
        "shots": p.shots,
        "verdicts": verdicts,
        "miss_m": ({"mean": round(sum(miss) / len(miss), 3), "min": round(min(miss), 3),
                    "median": round(sorted(miss)[len(miss) // 2], 3), "max": round(max(miss), 3)}
                   if miss else None),
        "hits": p.hits,
        "fires": p.fires,
        "track_samples": p.track_samples,
        "final_stats": p.stats,
    })
    if p.track_errs:
        errs = sorted(e for e, _ in p.track_errs)
        log["track_align_px"] = {
            "median": round(errs[len(errs) // 2], 1),
            "p90": round(errs[int(0.9 * (len(errs) - 1))], 1),
            "within_tol_frac": round(sum(1 for e, tol in p.track_errs if e <= tol) / len(p.track_errs), 3),
        }
    ages = sorted(f["frame_age_s"] for f in p.fires if f.get("frame_age_s") is not None)
    if ages:
        log["fire_frame_age_ms"] = {"median": round(1000 * ages[len(ages) // 2], 1), "max": round(1000 * ages[-1], 1)}
    log["valid"] = not log["invalid_reasons"]
    ev(f"valid={log['valid']} {log['invalid_reasons']}")
    ev(f"window: cam {log['camera_hz']} Hz, fcs {log['fcs_hz']} Hz, shots {p.shots}, {verdicts}, "
       f"states {log['state_frac']}")
    dump()
    shutdown()


if __name__ == "__main__":
    main()
