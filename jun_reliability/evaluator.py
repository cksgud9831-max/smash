"""Manual-ground-truth evaluation for reliability-state predictions.

Prediction CSV rows need ``frame_index``, ``timestamp``, and
``reliability_state``. Ground truth must use exactly one interval schema:
``start_frame,end_frame,state`` or ``start_time,end_time,state``. Intervals are
inclusive. No labels are inferred and no score is produced without GT.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable

from .state_machine import TrackingReliabilityState


_STATES = tuple(state.value for state in TrackingReliabilityState)


@dataclass(frozen=True, slots=True)
class StateMetrics:
    precision: float | None
    recall: float | None
    f1: float | None
    support: int


@dataclass(frozen=True, slots=True)
class TransitionDelay:
    previous_state: str
    current_state: str
    ground_truth_position: float
    predicted_position: float | None
    delay: float | None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    sample_count: int
    position_unit: str
    per_state: tuple[tuple[str, StateMetrics], ...]
    macro_f1: float | None
    confusion_matrix: tuple[tuple[int, ...], ...]
    confusion_labels: tuple[str, ...]
    false_stable_count: int
    false_stable_rate: float | None
    transition_delays: tuple[TransitionDelay, ...]
    mean_transition_delay: float | None
    stable_to_hold_delay: float | None
    hold_to_lost_delay: float | None
    lost_to_hold_reacquisition_delay: float | None


@dataclass(frozen=True, slots=True)
class _Prediction:
    frame_index: int
    timestamp: float
    state: str


@dataclass(frozen=True, slots=True)
class _Interval:
    start: float
    end: float
    state: str


def evaluate_files(prediction_csv: str | Path, ground_truth_csv: str | Path) -> EvaluationResult:
    predictions = _load_predictions(Path(prediction_csv))
    intervals, unit = _load_ground_truth(Path(ground_truth_csv))
    truth = _expand_truth(predictions, intervals, unit)
    predicted = tuple(item.state for item in predictions)
    positions = tuple(
        float(item.frame_index) if unit == "frame" else item.timestamp
        for item in predictions
    )
    return evaluate_sequences(truth, predicted, positions, unit)


def evaluate_sequences(
    ground_truth: Iterable[str],
    predictions: Iterable[str],
    positions: Iterable[float],
    position_unit: str,
) -> EvaluationResult:
    truth = tuple(ground_truth)
    predicted = tuple(predictions)
    retained_positions = tuple(float(value) for value in positions)
    if not truth:
        raise ValueError("ground truth must contain at least one labeled sample")
    if len(truth) != len(predicted) or len(truth) != len(retained_positions):
        raise ValueError("ground truth, predictions, and positions must have equal lengths")
    if position_unit not in ("frame", "second"):
        raise ValueError("position_unit must be 'frame' or 'second'")
    _validate_states(truth, "ground truth")
    _validate_states(predicted, "predictions")
    if any(not math.isfinite(value) for value in retained_positions):
        raise ValueError("positions must be finite")
    if any(current <= previous for previous, current in zip(retained_positions, retained_positions[1:])):
        raise ValueError("positions must be strictly increasing")

    matrix = [[0 for _ in _STATES] for _ in _STATES]
    index = {state: idx for idx, state in enumerate(_STATES)}
    for actual, estimate in zip(truth, predicted):
        matrix[index[actual]][index[estimate]] += 1

    per_state: list[tuple[str, StateMetrics]] = []
    f1_values: list[float] = []
    for state in _STATES:
        idx = index[state]
        tp = matrix[idx][idx]
        fp = sum(matrix[row][idx] for row in range(len(_STATES)) if row != idx)
        fn = sum(matrix[idx][column] for column in range(len(_STATES)) if column != idx)
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0.0
            else None
        )
        if f1 is not None:
            f1_values.append(f1)
        per_state.append((state, StateMetrics(precision, recall, f1, sum(matrix[idx]))))

    false_stable_count = sum(
        actual != "STABLE" and estimate == "STABLE"
        for actual, estimate in zip(truth, predicted)
    )
    actual_nonstable = sum(actual != "STABLE" for actual in truth)
    delays = _transition_delays(truth, predicted, retained_positions)
    return EvaluationResult(
        sample_count=len(truth),
        position_unit=position_unit,
        per_state=tuple(per_state),
        macro_f1=sum(f1_values) / len(f1_values) if f1_values else None,
        confusion_matrix=tuple(tuple(row) for row in matrix),
        confusion_labels=_STATES,
        false_stable_count=false_stable_count,
        false_stable_rate=_ratio(false_stable_count, actual_nonstable),
        transition_delays=delays,
        mean_transition_delay=_mean_delay(delays),
        stable_to_hold_delay=_mean_delay(delays, "STABLE", "HOLD"),
        hold_to_lost_delay=_mean_delay(delays, "HOLD", "LOST"),
        lost_to_hold_reacquisition_delay=_mean_delay(delays, "LOST", "HOLD"),
    )


def _load_predictions(path: Path) -> tuple[_Prediction, ...]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"frame_index", "timestamp", "reliability_state"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"prediction CSV must contain {sorted(required)}")
        rows = tuple(
            _Prediction(
                frame_index=_integer(row["frame_index"], f"prediction row {line}.frame_index"),
                timestamp=_number(row["timestamp"], f"prediction row {line}.timestamp"),
                state=_state(row["reliability_state"], f"prediction row {line}.reliability_state"),
            )
            for line, row in enumerate(reader, start=2)
        )
    if not rows:
        raise ValueError("prediction CSV contains no rows")
    return rows


def _load_ground_truth(path: Path) -> tuple[tuple[_Interval, ...], str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = tuple(reader.fieldnames or ())
        if fields == ("start_frame", "end_frame", "state"):
            unit, start_name, end_name = "frame", "start_frame", "end_frame"
        elif fields == ("start_time", "end_time", "state"):
            unit, start_name, end_name = "second", "start_time", "end_time"
        else:
            raise ValueError(
                "GT fields must be exactly start_frame,end_frame,state or "
                "start_time,end_time,state"
            )
        intervals = []
        for line, row in enumerate(reader, start=2):
            start = _number(row[start_name], f"GT row {line}.{start_name}")
            end = _number(row[end_name], f"GT row {line}.{end_name}")
            if unit == "frame" and (not start.is_integer() or not end.is_integer()):
                raise ValueError(f"GT row {line}: frame bounds must be integers")
            if end < start:
                raise ValueError(f"GT row {line}: end precedes start")
            intervals.append(_Interval(start, end, _state(row["state"], f"GT row {line}.state")))
    if not intervals:
        raise ValueError("ground-truth CSV contains no intervals")
    ordered = tuple(intervals)
    for previous, current in zip(ordered, ordered[1:]):
        if current.start <= previous.end:
            raise ValueError("ground-truth intervals overlap or are not strictly ordered")
    return ordered, unit


def _expand_truth(
    predictions: tuple[_Prediction, ...], intervals: tuple[_Interval, ...], unit: str
) -> tuple[str, ...]:
    labels = []
    for prediction in predictions:
        position = float(prediction.frame_index) if unit == "frame" else prediction.timestamp
        matches = [interval.state for interval in intervals if interval.start <= position <= interval.end]
        if len(matches) != 1:
            raise ValueError(
                f"ground truth must cover prediction {unit} position {position} exactly once"
            )
        labels.append(matches[0])
    return tuple(labels)


def _transition_delays(
    truth: tuple[str, ...], predicted: tuple[str, ...], positions: tuple[float, ...]
) -> tuple[TransitionDelay, ...]:
    predicted_transitions = [
        (predicted[index - 1], predicted[index], positions[index])
        for index in range(1, len(predicted))
        if predicted[index] != predicted[index - 1]
    ]
    delays = []
    used: set[int] = set()
    for index in range(1, len(truth)):
        if truth[index] == truth[index - 1]:
            continue
        previous, current, gt_position = truth[index - 1], truth[index], positions[index]
        match = next(
            (
                (candidate_index, position)
                for candidate_index, (pred_previous, pred_current, position) in enumerate(predicted_transitions)
                if candidate_index not in used
                and pred_previous == previous
                and pred_current == current
                and position >= gt_position
            ),
            None,
        )
        predicted_position = None
        delay = None
        if match is not None:
            used.add(match[0])
            predicted_position = match[1]
            delay = predicted_position - gt_position
        delays.append(TransitionDelay(previous, current, gt_position, predicted_position, delay))
    return tuple(delays)


def _mean_delay(
    delays: tuple[TransitionDelay, ...], previous: str | None = None, current: str | None = None
) -> float | None:
    values = [
        item.delay
        for item in delays
        if item.delay is not None
        and (previous is None or item.previous_state == previous)
        and (current is None or item.current_state == current)
    ]
    return sum(values) / len(values) if values else None


def _validate_states(values: tuple[str, ...], label: str) -> None:
    invalid = sorted(set(values) - set(_STATES))
    if invalid:
        raise ValueError(f"{label} contains invalid states: {invalid}")


def _state(value: str | None, location: str) -> str:
    if value not in _STATES:
        raise ValueError(f"{location}: expected one of {_STATES}")
    return value


def _number(value: str | None, location: str) -> float:
    try:
        converted = float(value) if value is not None else math.nan
    except ValueError as error:
        raise ValueError(f"{location}: expected finite number") from error
    if not math.isfinite(converted):
        raise ValueError(f"{location}: expected finite number")
    return converted


def _integer(value: str | None, location: str) -> int:
    converted = _number(value, location)
    if not converted.is_integer():
        raise ValueError(f"{location}: expected integer")
    return int(converted)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = evaluate_files(args.predictions, args.ground_truth)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
