"""Fixed-size raw history for future reliability metric calculations."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator

from .types import ReliabilityInput


class ReliabilityBuffer:
    """Keep recent inputs and observation counters without judging quality."""

    def __init__(self, maxlen: int) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be greater than zero")

        self._items: deque[ReliabilityInput] = deque(maxlen=maxlen)
        self._consecutive_result_frames = 0
        self._consecutive_missing_frames = 0
        self._frames_since_last_yolo: int | None = None

    @property
    def maxlen(self) -> int:
        return self._items.maxlen  # type: ignore[return-value]

    @property
    def consecutive_result_frames(self) -> int:
        return self._consecutive_result_frames

    @property
    def consecutive_missing_frames(self) -> int:
        return self._consecutive_missing_frames

    @property
    def frames_since_last_yolo(self) -> int | None:
        """Frames elapsed since YOLO ran, or ``None`` if not yet observed."""

        return self._frames_since_last_yolo

    @property
    def history(self) -> tuple[ReliabilityInput, ...]:
        """Return an immutable, oldest-to-newest snapshot of retained input."""

        return tuple(self._items)

    def append(self, item: ReliabilityInput) -> None:
        self._items.append(item)

        if item.result_present:
            self._consecutive_result_frames += 1
            self._consecutive_missing_frames = 0
        else:
            self._consecutive_result_frames = 0
            self._consecutive_missing_frames += 1

        if item.used_yolo:
            self._frames_since_last_yolo = 0
        elif self._frames_since_last_yolo is not None:
            self._frames_since_last_yolo += 1

    def latest(self) -> ReliabilityInput | None:
        return self._items[-1] if self._items else None

    def clear(self) -> None:
        self._items.clear()
        self._consecutive_result_frames = 0
        self._consecutive_missing_frames = 0
        self._frames_since_last_yolo = None

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[ReliabilityInput]:
        return iter(self._items)
