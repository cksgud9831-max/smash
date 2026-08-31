"""Facade wiring the independent tracking-reliability stages together.

The pipeline accepts an already-created ``ReliabilityInput``.  It does not run
a tracker and does not produce aiming, firing, actuator, or control commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .buffer import ReliabilityBuffer
from .hard_gate import GateConfig, GateResult, ReliabilityHardGate
from .metrics import ReliabilityMetricCalculator, ReliabilityMetrics
from .quality import (
    QualityConfig,
    ReliabilityQuality,
    ReliabilityQualityCalculator,
)
from .state_machine import (
    ReliabilityStateMachine,
    StateTransition,
    TrackingReliabilityState,
)
from .temporal import (
    ReliabilityTemporalValidator,
    TemporalConfig,
    TemporalEvidence,
)
from .types import ReliabilityInput


@dataclass(frozen=True, slots=True)
class ReliabilityPipelineConfig:
    """Composition of stage-specific configs.

    ``buffer_size`` is an initial experimental default, not a validated
    deployment window size.  Each nested config retains its own responsibility.
    """

    buffer_size: int = 30
    quality: QualityConfig = field(default_factory=QualityConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)

    def __post_init__(self) -> None:
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be greater than zero")


@dataclass(frozen=True, slots=True)
class ReliabilityFrameResult:
    """Complete, inspectable result of processing one reliability input."""

    input: ReliabilityInput
    metrics: ReliabilityMetrics
    quality: ReliabilityQuality
    gate: GateResult
    temporal: TemporalEvidence
    transition: StateTransition
    state: TrackingReliabilityState


class ReliabilityPipeline:
    """Apply each reliability stage once, in dependency order."""

    def __init__(self, config: ReliabilityPipelineConfig | None = None) -> None:
        self._config = config if config is not None else ReliabilityPipelineConfig()
        self._buffer = ReliabilityBuffer(self._config.buffer_size)
        self._metric_calculator = ReliabilityMetricCalculator()
        self._quality_calculator = ReliabilityQualityCalculator(self._config.quality)
        self._hard_gate = ReliabilityHardGate(self._config.gate)
        self._temporal_validator = ReliabilityTemporalValidator(self._config.temporal)
        self._state_machine = ReliabilityStateMachine()

    @property
    def config(self) -> ReliabilityPipelineConfig:
        return self._config

    @property
    def state(self) -> TrackingReliabilityState:
        return self._state_machine.state

    def update(self, reliability_input: ReliabilityInput) -> ReliabilityFrameResult:
        self._buffer.append(reliability_input)
        metrics = self._metric_calculator.calculate(self._buffer)
        quality = self._quality_calculator.calculate(metrics)
        gate = self._hard_gate.evaluate(metrics, quality)
        temporal = self._temporal_validator.update(metrics, quality, gate)
        transition = self._state_machine.update(
            temporal,
            timestamp=reliability_input.timestamp,
        )

        return ReliabilityFrameResult(
            input=reliability_input,
            metrics=metrics,
            quality=quality,
            gate=gate,
            temporal=temporal,
            transition=transition,
            state=self._state_machine.state,
        )

    def reset(self) -> None:
        self._buffer.clear()
        self._temporal_validator.reset()
        self._state_machine.reset()
