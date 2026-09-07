"""fire_control.py — SMASH FCS 가상 탄환 격발 및 Hit/Kill 판정기 [로드맵 3단계]

■ 이 모듈이 하는 일
    smash_fcs_node.py 가 매 프레임 조준(1~2단계)을 끝내고 나면, 이 모듈이
    다음 두 가지를 추가로 담당한다.

      1. 격발 판단: READY 점등 시 자동 격발(auto) 또는 사수 트리거 신호의
         상승 엣지에 따른 수동 격발(manual)을 판정하고, 발사 순간의 발사점 /
         발사방향 / 예상 비행시간 / 예상 명중점을 기록한다.
      2. Hit/Kill 판정: 예상 비행시간이 지난 시점에, 그 시점까지 탄환이 실제로
         날아간 위치(재적분)와 "그 시점의 표적 실제 위치"를 비교해 명중 여부를
         판정한다.

    aiming_engine 은 여전히 스스로 발사하지 않는다("there is no firing state
    here", aim_state_machine.py). "지금 쏜다" 라는 결정과 그 결과를 사후 판정하는
    책임은 전부 이 모듈에 있고, aiming_engine 코드는 한 줄도 바꾸지 않는다.

■ 왜 발사 방향을 aiming_engine 에서 다시 꺼내지 않고 smash_fcs_node 가 이미
  계산해 둔 것을 받는가
    AimSolver.solve() 는 세계(=base_footprint) 프레임에서 [t, azimuth, elevation]
    을 직접 푼 뒤(hit_equation.py), scope 프레임으로 변환한 azimuth/elevation 만
    AimSolution 에 담아 돌려준다(HUD 조준선 표시용). 세계 프레임 각도 자체는
    AimSolution 필드에 없다.

    smash_fcs_node.py 는 이미 정확히 같은 값을 다른 경로로 복원해 포탑 서보를
    돌리는 데 쓰고 있다(_aim_direction_in_base): scope 프레임 방위/앙각을 단위
    벡터로 만든 뒤, 좌표변환에 쓰인 것과 동일한 회전행렬(scope_pose 의 회전
    성분)로 되돌리면 솔버가 실제로 푼 세계 프레임 방향이 그대로 복원된다. 회전
    행렬은 직교이므로 전치가 곧 역행렬이고, 이 왕복은 stage1 검증에서 최대 오차
    1.9e-13 m 로 확인되었다(stage1_fcs_integration_report.md 2.2절). 즉 근사가
    아니라 수치오차 수준의 정확한 복원이다.

    따라서 이 모듈은 aiming_engine 내부 상태(예: AimSolver 가 물고 있는
    ProjectileModel 인스턴스)에 접근하지 않는다. 대신 자기 자신의
    ProjectileModel 인스턴스를 독립적으로 하나 더 만들어, smash_fcs_node 가
    넘겨주는 세계 프레임 발사 방향으로 탄도를 처음부터 다시 적분한다. 조준
    로직과 격발/판정 로직이 서로의 내부 구현에 의존하지 않게 하기 위한
    의도적인 분리다. 같은 물리 모델(같은 ProjectileConfig)로 다시 적분하므로,
    표적이 등속 가정을 정확히 지킨다면 재적분한 명중점은 AimSolution 이 이미
    계산해 둔 lead_point_world 와 수치오차 수준까지 일치해야 한다 — 이는
    verify_stage3_fire_control.py 가 자체 검증하는 항목이다.

■ 왜 명중 판정을 FCS 자신의 표적 추정치가 아니라 Gazebo 실측(ground truth)과
  비교하는가 (과학적 타당성 및 객관성)
    발사 시점에 이미 계산해 둔 예상 명중점(predicted_impact_point_world)과
    "판정 시점"을 비교하면, 뉴턴 솔버가 정의상 그 지점을 목표로 풀었으므로
    (해가 수렴하는 한) 사실상 항상 명중으로 나온다. 이는 명중률을 검증하는
    것이 아니라 솔버가 자기 자신과 일치하는지를 확인하는 동어반복이며, 로드맵
    4단계의 회피 기동 표적처럼 표적이 등속 가정을 실제로 위반해 예측과 다르게
    움직이는 상황을 포착하지 못한다.

    그래서 판정은 항상 "그 순간 시뮬레이터가 실제로 표적을 어디에 두었는가"
    (Gazebo PosePublisher 가 world 프레임으로 발행하는 실측 자세, smash_fcs_node
    의 ground_truth_pose_topic)를 기준으로 한다. FCS 의 자체 추정치는 판정에
    전혀 관여하지 않는다.

■ 이 모듈이 답하지 않는 것 (한계, stage1_fcs_integration_report.md 6절과 같은 형식)
    1. Hit == Kill 로 단순화했다. 5.56mm 탄이 소형 드론 동체에 명중하면 통상
       파괴로 이어진다고 가정한 단순화이며, 부위별 치명도나 관통 에너지 모델은
       없다. 정밀한 살상 판정이 필요해지면 hit_probability.py 와는 별개로
       충돌 부위 기반의 치명도 모델을 새로 추가해야 한다.
    2. target_radius_m(명중 판정 반경)은 target_characteristic_size_m 에서
       파생한 가정값이다(기본 target_characteristic_size_m/2). 실제 드론
       동체의 충돌 반경을 실측한 값이 아니다.
    3. Gazebo 실측 위치를 base_footprint 프레임으로 옮기는 데 쓰는
       turret_world_translation_m / turret_world_yaw_rad 는 포탑을 스폰하는
       view.launch.py 의 스폰 인자(-z 0.05, x=y=0, 회전 없음)와 반드시 일치해야
       하는 상수다. 스폰 인자를 바꾸면 이 값도 반드시 함께 바꿔야 하며, 이
       동기화는 코드로 강제되지 않는다(설계 노트 참고, smash_fcs_node.py).
    4. ground_truth_pose_topic 이 실제로 어떤 이름으로 발행되는지(예:
       /model/drone/pose)는 gz-sim 버전에 따라 달라질 수 있어 WSL2 에서
       `gz topic -l` 로 사용자가 직접 확인해야 한다. 이 세션은 실제 gz 런타임을
       실행할 수 없으므로 토픽 이름은 문서화된 관례를 따른 추정값이다.
    5. 이 세션이 실제로 수행한 검증은 verify_stage3_fire_control.py 의 순수
       파이썬 물리/로직 오라클 검증까지다. ROS 2 노드가 실제로 기동해 격발
       이벤트가 흐르는지는 사용자가 WSL2 에서 확인해야 한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


def direction_to_world_azimuth_elevation(direction_world: np.ndarray) -> tuple[float, float]:
    """세계(=base_footprint) 프레임 단위벡터를 (방위각, 앙각) 라디안으로 변환한다.

    aim_solver.py 의 _direction_to_azimuth_elevation 과 정의가 완전히 동일하다
    (azimuth = atan2(y, x), elevation = atan2(z, hypot(x, y))). aiming_engine
    코드를 그대로 재사용하지 않는 이유는 이 모듈이 aiming_engine 내부에
    의존하지 않는다는 원칙을 지키기 위함이며, 정의가 같으므로 결과도 같다.
    """

    d = np.asarray(direction_world, dtype=np.float64)
    horizontal = float(math.hypot(d[0], d[1]))
    azimuth = float(math.atan2(d[1], d[0]))
    elevation = float(math.atan2(d[2], horizontal))
    return azimuth, elevation


@dataclass
class FiredShot:
    """한 발의 격발 기록. base_footprint 프레임, 노드 클럭(초) 기준."""

    shot_id: int
    fire_time: float
    launch_point_world: np.ndarray
    azimuth_world: float
    elevation_world: float
    predicted_tof_s: float
    predicted_impact_point_world: np.ndarray
    target_radius_m: float
    muzzle_velocity_mps: float
    trigger_mode: str
    judged: bool = False
    result: Optional["ShotResult"] = None


@dataclass
class ShotResult:
    shot_id: int
    verdict: str  # "HIT" | "MISS" | "UNVERIFIED"
    impact_time: float
    projectile_point_world: np.ndarray
    ground_truth_point_world: Optional[np.ndarray]
    miss_distance_m: Optional[float]
    ground_truth_available: bool
    ground_truth_age_s: Optional[float]


class FireController:
    """격발 트리거 판단, 탄환 궤적 재적분, Hit/Kill 판정을 한 곳에서 담당한다."""

    def __init__(
        self,
        projectile_config,
        muzzle_velocity_mps: float,
        fire_mode: str,
        fire_rate_rpm: float,
        target_radius_m: float,
        ground_truth_timeout_s: float,
        impact_effect_hold_s: float = 0.35,
        max_tof_multiple: float = 4.0,
    ) -> None:
        from aiming_engine.projectile_model import ProjectileModel  # 지연 임포트

        if fire_mode not in ("auto", "manual"):
            raise ValueError(f"fire_mode 는 'auto' 또는 'manual' 이어야 합니다: {fire_mode!r}")

        self._projectile_model = ProjectileModel(projectile_config)
        self._muzzle_velocity_mps = float(muzzle_velocity_mps)
        self._fire_mode = fire_mode
        self._cooldown_s = 60.0 / max(float(fire_rate_rpm), 1e-3)
        self._target_radius_m = float(target_radius_m)
        self._ground_truth_timeout_s = float(ground_truth_timeout_s)
        self._impact_effect_hold_s = float(impact_effect_hold_s)
        self._max_tof_multiple = float(max_tof_multiple)

        self._next_shot_id = 1
        self._last_fire_time: Optional[float] = None
        self._prev_trigger_pressed = False
        self._pending: list[FiredShot] = []
        self._recent_results: list[ShotResult] = []

        self.shots_fired = 0
        self.hits = 0
        self.misses = 0
        self.unverified = 0

    # ── 상태 초기화 ────────────────────────────────────────────────────

    def reset_trigger_edge(self) -> None:
        """SEARCH 재진입 시 호출. 트리거 엣지 검출 상태만 초기화한다.

        이미 발사된 탄환(``_pending``)은 취소하지 않는다 — 표적 추적을
        일시적으로 잃었더라도 이미 날아간 탄환은 물리적으로 계속 비행 중이며,
        판정은 예정대로 진행되어야 한다(단, 판정 시점에 ground truth 마저
        구할 수 없다면 UNVERIFIED 로 남는다).
        """

        self._prev_trigger_pressed = False

    # ── 격발 판단 ──────────────────────────────────────────────────────

    def maybe_fire(
        self,
        now: float,
        aim_ready: bool,
        trigger_pressed: bool,
        aim_solution,
        launch_point_world: np.ndarray,
        direction_world: Optional[np.ndarray],
    ) -> Optional[FiredShot]:
        """이번 프레임에 격발해야 하면 FiredShot 을 만들어 반환한다.

        auto 모드: aim_ready 가 참인 매 프레임 격발을 "원하는" 것으로 보되,
            fire_rate_rpm 에서 유도한 쿨다운으로 실제 발사 간격을 제한한다
            (완전자동 사격의 근사).
        manual 모드: trigger_pressed 의 상승 엣지에서만 한 발을 격발한다
            (반자동 방아쇠의 근사). 같은 쿨다운이 여전히 적용된다.

        aim_solution 이나 direction_world 가 없으면(SEARCH/TRACK 상태 등, 아직
        조준각이 없음) 격발하지 않는다 — "쏠 방향"이 없는 상태에서 가상
        탄환을 만드는 것은 물리적으로 정의되지 않는다.
        """

        rising_edge = trigger_pressed and not self._prev_trigger_pressed
        self._prev_trigger_pressed = trigger_pressed

        if aim_solution is None or direction_world is None:
            return None

        if self._fire_mode == "auto":
            want_fire = bool(aim_ready)
        else:
            want_fire = bool(rising_edge and aim_ready)

        if not want_fire:
            return None
        if self._last_fire_time is not None and (now - self._last_fire_time) < self._cooldown_s:
            return None

        azimuth_world, elevation_world = direction_to_world_azimuth_elevation(direction_world)
        shot = FiredShot(
            shot_id=self._next_shot_id,
            fire_time=float(now),
            launch_point_world=np.asarray(launch_point_world, dtype=np.float64).copy(),
            azimuth_world=azimuth_world,
            elevation_world=elevation_world,
            predicted_tof_s=float(aim_solution.time_of_flight),
            predicted_impact_point_world=aim_solution.lead_point_world.to_array(),
            target_radius_m=self._target_radius_m,
            muzzle_velocity_mps=self._muzzle_velocity_mps,
            trigger_mode=self._fire_mode,
        )
        self._next_shot_id += 1
        self._last_fire_time = float(now)
        self.shots_fired += 1
        self._pending.append(shot)
        return shot

    # ── 탄도 재적분 ────────────────────────────────────────────────────

    def projectile_position_at(self, shot: FiredShot, t_elapsed: float) -> np.ndarray:
        """발사 후 t_elapsed 초 시점의 탄환 위치(base_footprint, m)."""

        from aiming_engine.types import Vector3

        t_cap = max(shot.predicted_tof_s * self._max_tof_multiple, 1e-3)
        t_eval = max(0.0, min(float(t_elapsed), t_cap))
        pos, _vel = self._projectile_model.state_at(
            t_eval,
            Vector3.from_array(shot.launch_point_world),
            shot.azimuth_world,
            shot.elevation_world,
        )
        return pos.to_array()

    # ── Hit/Kill 판정 ──────────────────────────────────────────────────

    def update(
        self,
        now: float,
        ground_truth_position_world: Optional[np.ndarray],
        ground_truth_timestamp: Optional[float],
    ) -> list[ShotResult]:
        """매 프레임(또는 고정 주기 타이머) 호출. 비행시간이 지난 대기 중
        탄환을 판정하고, 이번 호출에서 새로 판정된 결과 목록을 반환한다."""

        judged_now: list[ShotResult] = []
        still_pending: list[FiredShot] = []

        for shot in self._pending:
            elapsed = now - shot.fire_time
            if elapsed < shot.predicted_tof_s:
                still_pending.append(shot)
                continue

            impact_point = self.projectile_position_at(shot, shot.predicted_tof_s)

            gt_available = ground_truth_position_world is not None and ground_truth_timestamp is not None
            gt_age: Optional[float] = None
            if gt_available:
                gt_age = now - float(ground_truth_timestamp)
                if gt_age > self._ground_truth_timeout_s or gt_age < 0.0:
                    gt_available = False

            if gt_available:
                miss_distance = float(np.linalg.norm(np.asarray(ground_truth_position_world) - impact_point))
                verdict = "HIT" if miss_distance <= shot.target_radius_m else "MISS"
                if verdict == "HIT":
                    self.hits += 1
                else:
                    self.misses += 1
            else:
                miss_distance = None
                verdict = "UNVERIFIED"
                self.unverified += 1

            result = ShotResult(
                shot_id=shot.shot_id,
                verdict=verdict,
                impact_time=shot.fire_time + shot.predicted_tof_s,
                projectile_point_world=impact_point,
                ground_truth_point_world=(
                    None
                    if ground_truth_position_world is None
                    else np.asarray(ground_truth_position_world, dtype=np.float64).copy()
                ),
                miss_distance_m=miss_distance,
                ground_truth_available=gt_available,
                ground_truth_age_s=gt_age,
            )
            shot.judged = True
            shot.result = result
            judged_now.append(result)
            self._recent_results.append(result)

        self._pending = still_pending
        self._recent_results = [
            r for r in self._recent_results if (now - r.impact_time) <= self._impact_effect_hold_s
        ]
        return judged_now

    # ── 조회 ───────────────────────────────────────────────────────────

    @property
    def pending_shots(self) -> list[FiredShot]:
        return list(self._pending)

    @property
    def recent_results(self) -> list[ShotResult]:
        return list(self._recent_results)

    def stats(self) -> dict:
        judged_total = self.hits + self.misses + self.unverified
        verified_total = self.hits + self.misses
        return {
            "shots_fired": self.shots_fired,
            "hits": self.hits,
            "misses": self.misses,
            "unverified": self.unverified,
            "judged_total": judged_total,
            # 명중률은 판정 불능(UNVERIFIED)을 분모에서 제외한다 — ground truth를
            # 못 구한 탄환까지 "명중 실패"로 세면 통계가 편향된다.
            "hit_rate": (self.hits / verified_total) if verified_total > 0 else None,
        }
