"""YAML configuration loading for the AI Smart Scope Aiming Engine.

Every module receives its own typed config dataclass at construction time;
nothing outside this file reads YAML directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "aiming_engine.yaml"


@dataclass(frozen=True)
class ProjectileConfig:
    muzzle_velocity: float
    forces: list[str]
    gravity: float
    integrator_step: float
    # Only read when "drag" is present in `forces` -- see forces.py:DragForce.
    drag_model: str = "G1"  # "G1" | "G7"
    mass_kg: float = 0.0  # projectile mass; must be > 0 if drag is enabled
    diameter_m: float = 0.0  # projectile diameter; must be > 0 if drag is enabled
    air_density_kg_m3: float = 1.225  # ICAO standard sea-level; override for altitude/temperature
    speed_of_sound_mps: float = 340.3  # standard sea-level, 15C


@dataclass(frozen=True)
class ScopeConfig:
    mount_offset_m: tuple[float, float, float]
    mount_rotation_rad: tuple[float, float, float]  # (roll, pitch, yaw)


@dataclass(frozen=True)
class SolverConfig:
    max_iterations: int
    tolerance_m: float
    jacobian_epsilon: float
    warm_start: bool


@dataclass(frozen=True)
class TargetStateConfig:
    estimator: str
    history_length: int
    min_dt: float
    process_noise_std: float = 1.0
    measurement_noise_std: float = 0.1
    # ConstantVelocityEstimator only. 1.0(기본값) = 스무딩 없음, 기존 검증 수치와 완전히
    # 동일한 무보정 OLS 동작. 1.0 미만이면 프레임 간 지수이동평균(EMA)으로 속도 추정치를
    # 완만하게 만든다 (known_issues.md 1.5절의 이중 미분 노이즈 완화 목적, target_state.py 참고).
    velocity_smoothing_alpha: float = 1.0
    # ConstantVelocityEstimator only. 기동(방향 전환) 감지 시 기록 비우기. 기본 꺼짐 = 기존 동작.
    # 새 측정이 직전 추정의 예측과 시선에 수직으로 gate 넘게 maneuver_hits 번 연속 어긋나면 최근
    # maneuver_keep 개 샘플만 남긴다. gate = max(maneuver_gate_min_m, maneuver_gate_mrad * 거리).
    # 근거: 2026-09-22 기록 재생(benchmarks/sim_measurement/estimator_replay.py)에서 1.2 s 마다 방향을
    # 바꾸는 35 m 표적의 리드가 명중 반경 안에 드는 비율이 43 % -> 78 %(10 mrad), 등속·정지 흔들림은 그대로.
    # 문턱 5 / 7 / 10 mrad 중 10 이 가장 좋고 리셋도 가장 적었다(35 m 에서 0.35 m).
    maneuver_reset: bool = False
    maneuver_gate_mrad: float = 10.0
    maneuver_gate_min_m: float = 0.1
    maneuver_hits: int = 2
    maneuver_keep: int = 3


@dataclass(frozen=True)
class HitProbabilityConfig:
    weights: dict[str, float]
    distance_scale_m: float
    velocity_scale_mps: float
    aim_error_scale_mrad: float


@dataclass(frozen=True)
class AimReadinessConfig:
    hit_probability_threshold: float
    max_aim_error_mrad: float
    min_valid_range_m: float
    max_valid_range_m: float


@dataclass(frozen=True)
class AimStateMachineConfig:
    tracking_confidence_threshold: float
    min_stable_frames_for_track: int
    min_stable_frames_for_aim: int
    track_lost_timeout_s: float


@dataclass(frozen=True)
class AimAssistConfig:
    projectile: ProjectileConfig
    scope: ScopeConfig
    solver: SolverConfig
    target_state: TargetStateConfig
    hit_probability: HitProbabilityConfig
    aim_readiness: AimReadinessConfig
    aim_state_machine: AimStateMachineConfig


def _vec3(d: dict[str, Any]) -> tuple[float, float, float]:
    return (float(d["x"]), float(d["y"]), float(d["z"]))


def _rot3(d: dict[str, Any]) -> tuple[float, float, float]:
    return (float(d["roll"]), float(d["pitch"]), float(d["yaw"]))


def load_config(path: str | Path | None = None) -> AimAssistConfig:
    """Load and validate the Aiming Engine YAML config into typed dataclasses."""

    resolved = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(resolved, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return AimAssistConfig(
        projectile=ProjectileConfig(
            muzzle_velocity=float(raw["projectile"]["muzzle_velocity"]),
            forces=list(raw["projectile"]["forces"]),
            gravity=float(raw["projectile"]["gravity"]),
            integrator_step=float(raw["projectile"]["integrator_step"]),
            drag_model=str(raw["projectile"].get("drag_model", "G1")),
            mass_kg=float(raw["projectile"].get("mass_kg", 0.0)),
            diameter_m=float(raw["projectile"].get("diameter_m", 0.0)),
            air_density_kg_m3=float(raw["projectile"].get("air_density_kg_m3", 1.225)),
            speed_of_sound_mps=float(raw["projectile"].get("speed_of_sound_mps", 340.3)),
        ),
        scope=ScopeConfig(
            mount_offset_m=_vec3(raw["scope"]["mount_offset_m"]),
            mount_rotation_rad=_rot3(raw["scope"]["mount_rotation_rad"]),
        ),
        solver=SolverConfig(
            max_iterations=int(raw["solver"]["max_iterations"]),
            tolerance_m=float(raw["solver"]["tolerance_m"]),
            jacobian_epsilon=float(raw["solver"]["jacobian_epsilon"]),
            warm_start=bool(raw["solver"]["warm_start"]),
        ),
        target_state=TargetStateConfig(
            estimator=str(raw["target_state"]["estimator"]),
            history_length=int(raw["target_state"]["history_length"]),
            min_dt=float(raw["target_state"]["min_dt"]),
            process_noise_std=float(raw["target_state"].get("process_noise_std", 1.0)),
            measurement_noise_std=float(raw["target_state"].get("measurement_noise_std", 0.1)),
            velocity_smoothing_alpha=float(raw["target_state"].get("velocity_smoothing_alpha", 1.0)),
            maneuver_reset=bool(raw["target_state"].get("maneuver_reset", False)),
            maneuver_gate_mrad=float(raw["target_state"].get("maneuver_gate_mrad", 10.0)),
            maneuver_gate_min_m=float(raw["target_state"].get("maneuver_gate_min_m", 0.1)),
            maneuver_hits=int(raw["target_state"].get("maneuver_hits", 2)),
            maneuver_keep=int(raw["target_state"].get("maneuver_keep", 3)),
        ),
        hit_probability=HitProbabilityConfig(
            weights={k: float(v) for k, v in raw["hit_probability"]["weights"].items()},
            distance_scale_m=float(raw["hit_probability"]["distance_scale_m"]),
            velocity_scale_mps=float(raw["hit_probability"]["velocity_scale_mps"]),
            aim_error_scale_mrad=float(raw["hit_probability"]["aim_error_scale_mrad"]),
        ),
        aim_readiness=AimReadinessConfig(
            hit_probability_threshold=float(raw["aim_readiness"]["hit_probability_threshold"]),
            max_aim_error_mrad=float(raw["aim_readiness"]["max_aim_error_mrad"]),
            min_valid_range_m=float(raw["aim_readiness"]["min_valid_range_m"]),
            max_valid_range_m=float(raw["aim_readiness"]["max_valid_range_m"]),
        ),
        aim_state_machine=AimStateMachineConfig(
            tracking_confidence_threshold=float(raw["aim_state_machine"]["tracking_confidence_threshold"]),
            min_stable_frames_for_track=int(raw["aim_state_machine"]["min_stable_frames_for_track"]),
            min_stable_frames_for_aim=int(raw["aim_state_machine"]["min_stable_frames_for_aim"]),
            track_lost_timeout_s=float(raw["aim_state_machine"]["track_lost_timeout_s"]),
        ),
    )
