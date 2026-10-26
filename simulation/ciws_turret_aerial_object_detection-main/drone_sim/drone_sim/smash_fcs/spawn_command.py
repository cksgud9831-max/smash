"""표적 즉시 생성 명령(/smash_fcs/ui_command 의 "spawn:<방식>:<km/h>") 형식.

뷰어가 보내고(smash_scope_viewer 가 형식을 검사해 통과시킨다) demo_director 가 받아 표적을 바로
다시 띄운다. 방식과 속도는 재출현 이벤트에 실려 target_mover 가 적용한다. ROS 에 의존하지 않는다.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# 정지(none)는 생성 대상이 아니다. 정지 표적은 뷰어의 0 키(호버링)로 만든다.
SPAWN_MOTIONS = ("straight", "cross", "maneuver", "approach")
SPAWN_KMH_MIN = 1.0
SPAWN_KMH_MAX = 150.0

_PATTERN = re.compile(r"^spawn:(" + "|".join(SPAWN_MOTIONS) + r"):(\d{1,3}(?:\.\d{1,2})?)$")


def format_spawn(motion: str, kmh: float) -> str:
    if motion not in SPAWN_MOTIONS:
        raise ValueError(f"생성 방식은 {SPAWN_MOTIONS} 중 하나여야 한다: {motion!r}")
    return f"spawn:{motion}:{min(SPAWN_KMH_MAX, max(SPAWN_KMH_MIN, float(kmh))):g}"


def parse_spawn(command: str) -> Optional[Tuple[str, float]]:
    """(방식, km/h). 형식이 틀리거나 속도가 1~150 km/h 밖이면 None."""

    match = _PATTERN.match(command.strip())
    if match is None:
        return None
    kmh = float(match.group(2))
    if not SPAWN_KMH_MIN <= kmh <= SPAWN_KMH_MAX:
        return None
    return match.group(1), kmh
