"""SMASH 조준 코어(aiming_engine, bridge)를 ROS 2 패키지에서 임포트하기 위한 부트스트랩.

aiming_engine 과 bridge 는 ciws_turret_aerial_object_detection 저장소 밖(SMASH 코어
저장소 루트)에 있으므로 colcon 이 설치해 주지 않는다. 그래서 실행 시점에 sys.path 로
붙인다. 경로는 다음 순서로 결정한다.

    1. 노드 파라미터 core_path
    2. 환경변수 SMASH_CORE_PATH
    3. 이 파일 기준 상대 경로 후보들 (개발 트리에서 바로 실행하는 경우)

WSL2 에서 D 드라이브를 쓰는 경우 SMASH_CORE_PATH=/mnt/d/Aiming 형태가 된다.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

_REQUIRED_PACKAGES = ("aiming_engine", "bridge")


def _looks_like_core_root(path: Path) -> bool:
    return all((path / name / "__init__.py").is_file() for name in _REQUIRED_PACKAGES)


def _candidate_roots(explicit: Optional[str]) -> Iterable[Path]:
    if explicit:
        yield Path(explicit).expanduser()
    env = os.environ.get("SMASH_CORE_PATH")
    if env:
        yield Path(env).expanduser()
    here = Path(__file__).resolve()
    # 개발 트리 배치: <core>/ciws_turret_aerial_object_detection-main/drone_sim/drone_sim/smash_fcs/
    for up in range(3, 7):
        if len(here.parents) > up:
            yield here.parents[up]


def resolve_core_root(explicit: Optional[str] = None) -> Path:
    tried: list[str] = []
    for candidate in _candidate_roots(explicit):
        tried.append(str(candidate))
        if _looks_like_core_root(candidate):
            return candidate
    raise RuntimeError(
        "SMASH 조준 코어(aiming_engine, bridge)를 찾지 못했습니다. "
        "smash_fcs 노드의 core_path 파라미터나 환경변수 SMASH_CORE_PATH 를 "
        "코어 저장소 루트로 지정하세요. 시도한 경로: " + ", ".join(tried)
    )


def ensure_core_on_path(explicit: Optional[str] = None) -> Path:
    root = resolve_core_root(explicit)
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)
    return root


def build_aim_config(
    core_root: Path,
    config_path: Optional[str],
    muzzle_velocity_mps: float,
    scope_mount_offset_m: tuple[float, float, float],
    enable_drag: bool,
    projectile_mass_kg: float,
    projectile_diameter_m: float,
    drag_model: str = "G1",
):
    """config/aiming_engine.yaml 을 불러온 뒤 포탑 전용 값으로 덮어쓴다.

    덮어쓰는 항목과 이유:

    1. scope.mount_offset_m
       기본 YAML 값은 견착식 스코프용 자리표시자다. CIWS 포탑에서는 이 값이
       "카메라 광학중심 -> 포구" 오프셋이어야 한다. aiming_engine 은 scope 프레임
       원점을 발사점(launch point)으로 삼기 때문에(aiming_manager.py 참고), 이 값을
       올바로 넣어야 카메라와 포구 사이 시차(약 0.68 m, 35 m 거리에서 최대 약 19 mrad)가
       탄도 해에 반영된다. 이 시차는 4단계 READY 판정 임계값(0.5도 = 8.7 mrad)보다
       크므로 무시할 수 없다.

    2. projectile.muzzle_velocity / forces / mass_kg / diameter_m
       20mm CIWS 탄 기준으로 바꾼다. 기본 YAML 은 850 m/s 에 중력만 켜져 있다.

    공기저항 주의: aiming_engine/forces.py 의 G1/G7 Cd(Mach) 표는 해당 파일 주석에
    명시된 대로 아직 1차 출처로 검증되지 않은 근사치다. enable_drag 를 켜면 탄도
    계산이 중력만 쓸 때보다 물리적으로는 더 타당해지지만, 절대 정확도는 이 표의
    품질에 종속된다는 점을 보고서에 반드시 병기해야 한다.
    """

    from aiming_engine.config import load_config  # 지연 임포트: sys.path 설정 이후여야 한다

    resolved = Path(config_path) if config_path else (core_root / "config" / "aiming_engine.yaml")
    config = load_config(resolved)

    forces = list(config.projectile.forces)
    if enable_drag:
        if projectile_mass_kg <= 0.0 or projectile_diameter_m <= 0.0:
            raise ValueError(
                "enable_drag 가 참이면 projectile_mass_kg 와 projectile_diameter_m 가 "
                "모두 0보다 커야 합니다 (aiming_engine/forces.py:DragForce 요구사항)"
            )
        if "drag" not in forces:
            forces.append("drag")
    else:
        forces = [f for f in forces if f != "drag"]

    projectile = replace(
        config.projectile,
        muzzle_velocity=float(muzzle_velocity_mps),
        forces=forces,
        drag_model=str(drag_model),
        mass_kg=float(projectile_mass_kg),
        diameter_m=float(projectile_diameter_m),
    )
    scope = replace(
        config.scope,
        mount_offset_m=(
            float(scope_mount_offset_m[0]),
            float(scope_mount_offset_m[1]),
            float(scope_mount_offset_m[2]),
        ),
    )
    return replace(config, projectile=projectile, scope=scope)
