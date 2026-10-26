#!/usr/bin/env python3
"""target_mover.py — 이동 표적 시나리오 (지연 보정 검증용, 로드맵 4단계 선행).

표적(kinematic 드론)을 Gazebo set_pose 서비스로 움직인다. 물리 없이 위치를 직접 정하므로
속도가 정확히 일정하다.

motion:
    none      움직이지 않는다(호버링). 실행 중 명령으로 다른 방식으로 바꿀 수 있다.
    straight  시선에 수직인 수평 방향으로 등속 직선 통과. 중심에서 출발해 반폭 끝에 닿으면
              반대쪽 끝에서 다시 들어온다(같은 방향으로 반복 통과).
    cross     시선에 수직인 수평 방향으로 등속 왕복. 끝점에서만 방향이 바뀐다(순간 반전).
    maneuver  좌우(시선에 수직인 수평) x 상하(월드 수직) 평면에서 8자 기동. 속력이 일정하고 방향
              전환은 곡선으로만 한다. 중심을 한 주기에 두 번 지나므로 매복 사격 시험에 맞는다.
    approach  포탑을 향해 시선 방향으로 등속 접근(앙각 유지, 거리만 줄어든다). 중심 거리에서 출발해
              반폭만큼 가까워지면 그만큼 먼 곳에서 다시 다가온다.
    궤적은 target_paths.py. 중심은 스폰 위치(center_x/y/z)이고, 격추 연출(demo_director)이
    표적을 다른 자리에 재출현시키면 그 자리가 새 중심이 된다.

크기:
    half_length_m / half_height_m 은 시작 방식의 크기다. 실행 중 다른 방식으로 바꾸면 35 m 기준 크기
    (REFERENCE_SIZE_M)를 표적 거리에 비례해 줄이거나 늘려 쓴다. 재출현으로 거리가 바뀌어도 같은
    비율로 맞춘다. 화각 안에서 보이는 폭이 거리에 비례하기 때문이다.

실행 중 명령 (/smash_fcs/ui_command, 스코프 뷰어 키):
    toggle_motion        M      출발 전이면 출발, 움직이면 그 자리에서 정지(호버링), 멈춰 있으면 재개.
                                호버링(none) 상태면 마지막으로 쓴 방식(없으면 8자)으로 출발.
    motion:<방식>        0~4    그 방식으로 바꿔 중심에서 출발(잠금 후 출발이면 잠길 때까지 호버링).
                                motion:none 은 중심으로 돌아가 호버링.
    speed:<+-km/h>       + / -  속도를 바꾼다(궤적은 끊기지 않고 이어진다). 1 ~ 150 km/h.
    상태는 /target_mover/status (JSON) 로 2 Hz 발행한다(뷰어 화면 표시용).

격추 연출과 함께 쓰기:
    /smash_fcs/target_event 의 destroyed 를 받으면 표적을 건드리지 않고(낙하 연출), respawn 을 받으면
    그 위치를 새 중심으로 삼아 같은 방식을 처음부터 다시 한다.

잠금 후 출발 (start_on_lock, 런처 기본):
    방식을 고르면 바로 떠나지 않고 중심에서 호버링하며, FCS 추적 신뢰도(/smash_fcs/reliability_state)가
    STABLE 로 lock_hold_s 동안 유지된 뒤 출발한다. FCS 는 표적을 처음 잡고 조준점을 내기까지 수 초가
    걸리는데(2026-09-22 실측, 포탑 정지 후 3.3~5.0 s), 30 km/h 직선 통과는 0.3~0.4 s 만에 화면을
    벗어나므로 바로 출발하면 조준점이 나오기 전에 표적이 사라진다. 기다리는 중 M 을 누르면 바로 출발한다.

출발 시점 (측정 도구용):
    start_delay_s >= 0  첫 모의 시각부터 그만큼 뒤 출발.
    start_delay_s <  0  /target_mover/start (std_msgs/Bool, true) 또는 M 키를 받을 때 출발.

위치는 모의 시각으로 계산한다. 속도를 바꿔도 궤적이 이어지도록 시간이 아니라 이동 거리를 쌓는다.
set_pose 는 이산 호출이라 위치가 rate_hz 간격 계단으로 바뀐다.
"""

from __future__ import annotations

import json
import math
import time

import rclpy
from rclpy.clock import Clock, ClockType
from geometry_msgs.msg import Pose
from rclpy.node import Node
from std_msgs.msg import Bool, String
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose

from drone_sim.target_paths import Figure8Path, cross_offset, straight_offset

MOTIONS = ("none", "straight", "cross", "maneuver", "approach")
# 접근(approach) 방향과 거리를 잡을 때 쓰는 포탑 카메라 높이 근사값(m, world)
CAMERA_HEIGHT_M = 0.9
# 실행 중 방식을 바꿀 때의 크기 (35 m 기준 반폭, [좌우 또는 앞뒤, 상하])
REFERENCE_RANGE_M = 35.0
REFERENCE_SIZE_M = {
    "straight": (15.0, 0.0),
    "cross": (3.0, 0.0),
    "maneuver": (3.0, 1.8),
    "approach": (15.0, 0.0),
}
# 하한 1 km/h: 수동 격발 연습용. 협각 35 m 왕복에서 10 km/h 는 READY 창이 한 번 지날 때 약 0.14 s 뿐이다.
SPEED_MIN_MPS = 1.0 / 3.6
SPEED_MAX_MPS = 150.0 / 3.6
# 이보다 긴 모의 시각 간격은 /clock 멈춤으로 보고 그 사이 이동을 넣지 않는다(표적이 튀지 않게)
MAX_STEP_S = 0.5


class TargetMover(Node):
    def __init__(self) -> None:
        super().__init__("target_mover")
        self.declare_parameter("world", "turret_world")
        self.declare_parameter("model", "drone")
        self.declare_parameter("motion", "cross")
        self.declare_parameter("center_x", 10.0)
        self.declare_parameter("center_y", 0.0)
        self.declare_parameter("center_z", 4.2)
        self.declare_parameter("speed_mps", 5.0)
        self.declare_parameter("half_length_m", 6.0)
        self.declare_parameter("half_height_m", 1.8)
        self.declare_parameter("start_delay_s", -1.0)
        self.declare_parameter("rate_hz", 100.0)
        self.declare_parameter("start_on_lock", False)
        self.declare_parameter("lock_hold_s", 1.0)

        motion = str(self.get_parameter("motion").value)
        if motion not in MOTIONS:
            raise ValueError(f"motion 은 {MOTIONS} 중 하나여야 한다: {motion!r}")
        requested_speed = float(self.get_parameter("speed_mps").value)
        self._speed = min(SPEED_MAX_MPS, max(SPEED_MIN_MPS, requested_speed))
        if self._speed != requested_speed:
            self.get_logger().warn(
                f"요청 속도 {requested_speed * 3.6:.1f} km/h를 허용 범위 "
                f"{SPEED_MIN_MPS * 3.6:.0f}~{SPEED_MAX_MPS * 3.6:.0f} km/h로 제한했습니다."
            )
        self._delay = float(self.get_parameter("start_delay_s").value)
        self._set_center([float(self.get_parameter(k).value) for k in ("center_x", "center_y", "center_z")])
        # 시작 방식의 크기는 인자 그대로 쓰고, 35 m 기준 크기로 환산해 둔다(재출현 때 비율 유지용)
        self._ref_size = dict(REFERENCE_SIZE_M)
        if motion != "none":
            k = REFERENCE_RANGE_M / self._range
            self._ref_size[motion] = (float(self.get_parameter("half_length_m").value) * k,
                                      float(self.get_parameter("half_height_m").value) * k)
        self._last_motion = motion if motion != "none" else "maneuver"
        self._select(motion)

        world = str(self.get_parameter("world").value)
        self._model = str(self.get_parameter("model").value)
        self._cli = self.create_client(SetEntityPose, f"/world/{world}/set_pose")
        self._started = False           # 한 번이라도 출발했는가
        self._t_start = None            # 출발 예정 모의 시각 (start_delay_s >= 0 일 때)
        self._start_requested = False
        self._paused = False
        self._down = False              # 격추 연출 중이면 표적을 건드리지 않는다
        self._s = 0.0                   # 궤적 위 이동 거리 (m)
        self._last_step_sim = None
        self._return_to_center = False
        self._start_on_lock = bool(self.get_parameter("start_on_lock").value)
        self._lock_hold = float(self.get_parameter("lock_hold_s").value)
        self._awaiting_lock = False     # 방식을 골랐고 FCS 잠금을 기다리는 중
        self._stable_since = None       # FCS 신뢰도가 STABLE 이 된 모의 시각
        self.create_subscription(String, "/smash_fcs/reliability_state", self._on_reliability, 10)
        if self._start_on_lock and motion != "none" and self._delay < 0.0:
            self._awaiting_lock = True  # 시작 방식도 FCS 가 잡을 때까지 호버링
        self.create_subscription(Bool, "/target_mover/start", self._on_start, 10)
        self.create_subscription(String, "/smash_fcs/ui_command", self._on_ui_command, 10)
        self.create_subscription(String, "/smash_fcs/target_event", self._on_target_event, 10)
        self._status_pub = self.create_publisher(String, "/target_mover/status", 10)
        # 타이머는 실제 시각(steady)으로 돈다. 모의 시각 타이머는 /clock 수신이 막히면 콜백 자체가
        # 멈춘다(2026-09-22: 출발 신호를 받고도 46분 뒤에야 움직였다). 위치는 여전히 모의 시각으로 계산한다.
        self._last_sim = None
        self._last_sim_wall = None
        self._stall_warned = False
        self.create_timer(1.0 / max(1.0, float(self.get_parameter("rate_hz").value)), self._tick,
                          clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.create_timer(0.5, self._publish_status, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(
            f"이동 표적: {self._describe()}, "
            + ("호버링 (뷰어 0~4 키로 방식 선택)" if motion == "none" else
               "FCS 잠금 뒤 출발 (M 이면 바로)" if self._awaiting_lock else
               f"{self._delay:.0f} 초 뒤 출발" if self._delay >= 0 else "/target_mover/start 또는 M 키 대기")
        )

    # ── 설정 ──────────────────────────────────────────────────────────

    def _set_center(self, center) -> None:
        self._center = [float(v) for v in center]
        cx, cy = self._center[0], self._center[1]
        n = math.hypot(cx, cy) or 1.0
        self._axis = (-cy / n, cx / n)  # 시선에 수직인 수평 단위벡터 (world +y 쪽이 왼쪽)
        los = (cx, cy, self._center[2] - CAMERA_HEIGHT_M)
        self._range = math.sqrt(sum(v * v for v in los)) or 1.0
        self._los = tuple(v / self._range for v in los)  # 포탑에서 표적 쪽 단위벡터 (접근용)

    def _select(self, motion: str) -> None:
        """방식과 크기를 정하고 궤적을 중심에서 다시 시작한다."""
        self._motion = motion
        self._s = 0.0
        self._path = None
        if motion == "none":
            self._half = self._half_h = 0.0
            return
        k = self._range / REFERENCE_RANGE_M
        self._half, self._half_h = (v * k for v in self._ref_size[motion])
        if motion == "maneuver":
            self._path = Figure8Path(1.0, self._half, self._half_h)  # 이동 거리로 매개화

    def _describe(self) -> str:
        head = f"{self._motion} {self._speed * 3.6:.0f} km/h, 거리 {self._range:.0f} m, "
        if self._motion == "maneuver":
            return (head + f"8자 좌우 {self._half:.1f} m / 상하 {self._half_h:.1f} m, "
                    f"주기 {self._path.length / self._speed:.1f} s, "
                    f"최대 선회 {self._speed ** 2 * self._path.max_lateral_accel() / 9.81:.1f} g")
        if self._motion == "approach":
            return head + f"{self._range + self._half:.0f} -> {self._range - self._half:.0f} m 반복 접근"
        if self._motion in ("straight", "cross"):
            return head + f"반폭 {self._half:.1f} m"
        return head + "호버링"

    # ── 명령 ──────────────────────────────────────────────────────────

    def _on_start(self, msg: Bool) -> None:
        if msg.data and not self._start_requested:
            self._start_requested = True
            self.get_logger().info("출발 신호 수신")

    def _start_now(self) -> None:
        self._start_requested = True
        self._delay = min(self._delay, 0.0)
        self._t_start = None
        self._paused = False

    def _on_ui_command(self, msg: String) -> None:
        cmd = msg.data.strip()
        if cmd == "toggle_motion":
            if self._awaiting_lock:
                self._awaiting_lock = False
                self._start_now()
                self.get_logger().info("[M] 잠금을 기다리지 않고 바로 출발")
            elif self._motion == "none":
                self._select(self._last_motion)
                self._return_to_center = True
                self._arm()
                self.get_logger().info(f"[M] 기동 선택: {self._describe()}"
                                       + (" - FCS 잠금 뒤 출발" if self._awaiting_lock else ""))
            elif not self._started or (self._t_start is not None and self._now() < self._t_start):
                self._start_now()
                self.get_logger().info(f"[M] 기동 시작: {self._describe()}")
            else:
                self._paused = not self._paused
                self._last_step_sim = None
                self.get_logger().info("[M] 기동 " + ("정지 (그 자리에서 호버링)" if self._paused else "재개"))
        elif cmd.startswith("motion:"):
            motion = cmd.split(":", 1)[1]
            if motion not in MOTIONS:
                return
            if motion == "none":
                self._select("none")
                self._return_to_center = True
                self.get_logger().info("[키] 호버링 (중심으로 복귀)")
                return
            self._last_motion = motion
            self._select(motion)
            self._last_step_sim = None
            self._return_to_center = True
            self._arm()
            self.get_logger().info(f"[키] 방식 변경: {self._describe()}"
                                   + (" - FCS 잠금 뒤 출발" if self._awaiting_lock else ""))
        elif cmd.startswith("speed:"):
            try:
                d_kmh = float(cmd.split(":", 1)[1])
            except ValueError:
                return
            self._speed = min(SPEED_MAX_MPS, max(SPEED_MIN_MPS, self._speed + d_kmh / 3.6))
            self.get_logger().info(f"[키] 속도 {self._speed * 3.6:.0f} km/h")

    def _arm(self) -> None:
        """방식을 골랐을 때: 잠금 후 출발이면 호버링하며 기다리고, 아니면 바로 출발한다."""
        self._started = False
        self._paused = False
        if self._start_on_lock:
            self._awaiting_lock = True
            self._start_requested = False
            self._delay = -1.0
            self._t_start = None
        else:
            self._start_now()

    def _on_reliability(self, msg: String) -> None:
        now = self._now()
        if msg.data.strip() == "STABLE":
            if self._stable_since is None:
                self._stable_since = now
            if self._awaiting_lock and now - self._stable_since >= self._lock_hold:
                self._awaiting_lock = False
                self._start_now()
                self.get_logger().info(f"FCS 잠금 확인 ({self._lock_hold:.1f} s STABLE) -> 출발")
        else:
            self._stable_since = None

    def _on_target_event(self, msg: String) -> None:
        try:
            ev = json.loads(msg.data)
        except ValueError:
            return
        if ev.get("event") == "destroyed":
            self._down = True
        elif ev.get("event") == "respawn":
            pos = ev.get("position")
            spawned = self._apply_spawn_request(ev)  # 뷰어의 즉시 생성: 방식·속도 지정
            if isinstance(pos, list) and len(pos) >= 3:
                self._set_center(pos[:3])
                self._select(self._motion)  # 새 거리에 맞춰 크기를 다시 잡고 처음부터
                if self._motion != "none" and (self._start_on_lock or spawned):
                    self._arm()  # 새 자리에서도 잡힐 때까지 호버링(잠금 후 출발이 아니면 바로 출발)
            self._down = False
            self._last_step_sim = None
            self.get_logger().info(f"재출현 위치를 새 중심으로: {self._describe()}")

    def _apply_spawn_request(self, ev: dict) -> bool:
        """재출현 이벤트에 방식·속도가 실려 있으면(즉시 생성) 그대로 적용한다. 적용했으면 True."""

        motion = ev.get("motion")
        if motion not in MOTIONS or motion == "none":
            return False
        self._motion = self._last_motion = motion
        try:
            self._speed = min(SPEED_MAX_MPS, max(SPEED_MIN_MPS, float(ev["kmh"]) / 3.6))
        except (KeyError, TypeError, ValueError):
            pass  # 속도가 없거나 이상하면 지금 속도를 유지한다
        return True

    def _publish_status(self) -> None:
        moving = self._motion != "none" and self._started and not self._paused and not self._down
        waiting = self._motion != "none" and not self._started
        m = String()
        m.data = json.dumps({
            "motion": self._motion,
            "kmh": round(self._speed * 3.6, 1),
            "state": ("DOWN" if self._down else "MOVE" if moving else
                      "WAIT LOCK" if self._awaiting_lock else "WAIT" if waiting else "HOLD"),
            "range_m": round(self._range, 1),
        })
        self._status_pub.publish(m)

    # ── 주기 갱신 ────────────────────────────────────────────────────

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _tick(self) -> None:
        now = self._now()
        wall = time.monotonic()
        if self._last_sim is None or now > self._last_sim:
            if self._stall_warned:
                self.get_logger().warn(f"모의 시각 수신 재개 ({wall - self._last_sim_wall:.1f} s 멈춤)")
            self._last_sim, self._last_sim_wall, self._stall_warned = now, wall, False
        elif self._started and not self._stall_warned and wall - self._last_sim_wall > 1.0:
            self.get_logger().warn("모의 시각이 1초 넘게 멈춰 있다 (/clock 수신 중단). 그동안 표적은 제자리다")
            self._stall_warned = True
        if now <= 0.0 or not self._cli.service_is_ready() or self._down:
            return  # /clock 이나 서비스 브리지가 아직 없거나, 격추 연출 중이다
        if self._return_to_center:
            self._send_pose(*self._center)  # 방식을 바꾸면 중심에서 다시 시작
            self._return_to_center = False
        if self._motion == "none" or self._awaiting_lock:
            return
        if not self._started:
            if self._delay < 0.0 and not self._start_requested:
                return
            if self._t_start is None:
                self._t_start = now + max(self._delay, 0.0)
                self.get_logger().info(f"모의 시각 {self._t_start:.1f} s 에 출발")
            if now < self._t_start:
                return
            self._started = True
            self._last_step_sim = now
        if self._paused:
            return  # set_pose 를 부르지 않으면 표적은 마지막 위치에 그대로 떠 있다
        if self._last_step_sim is not None and 0.0 < now - self._last_step_sim <= MAX_STEP_S:
            self._s += self._speed * (now - self._last_step_sim)
        self._last_step_sim = now

        s = self._s
        offset = dz = radial = 0.0
        if self._path is not None:
            offset, dz = self._path.offset(s)
        elif self._motion == "cross":
            offset, _ = cross_offset(s, 1.0, self._half)
        elif self._motion == "straight":
            offset = straight_offset(s, 1.0, self._half)
        else:  # approach: 중심에서 가까워지는 쪽으로 출발
            radial = -straight_offset(s, 1.0, self._half)
        self._send_pose(
            self._center[0] + self._axis[0] * offset + self._los[0] * radial,
            self._center[1] + self._axis[1] * offset + self._los[1] * radial,
            self._center[2] + dz + self._los[2] * radial,
        )

    def _send_pose(self, x: float, y: float, z: float) -> None:
        req = SetEntityPose.Request()
        req.entity = Entity(name=self._model, type=Entity.MODEL)
        p = Pose()
        p.position.x, p.position.y, p.position.z = float(x), float(y), float(z)
        p.orientation.w = 1.0
        req.pose = p
        self._cli.call_async(req)


def main() -> None:
    rclpy.init()
    node = TargetMover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
