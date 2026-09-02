# Tracking Reliability State Estimator Design

This document defines tracking reliability only. `STABLE`, `HOLD`, and
`LOST` are not aiming, firing, trigger, actuator, threat, or other operational
permissions.

All thresholds described here are **initial experimental defaults**. They are
not validated, optimal, or deployment-ready values.

## Input semantics

`ReliabilityInput.result_present` means only that the existing hybrid tracker
returned a `FollowerResult`. The tracker can retain a previous bbox and YOLO
score, so:

```text
result_present == True  does not imply  current target observation
high overall quality    does not imply  STABLE
```

The estimator never reads tracker-private state. It uses only the public
`FollowerResult` fields, frame dimensions, timestamp, and measured update
latency.

## State semantics

### STABLE

Fresh observation evidence is sufficient, geometry and temporal behavior are
consistent with trusted tracking, strong stale evidence is absent, the hard
gate passes, and these conditions persist long enough to establish tracking
reliability. No single quality threshold can establish STABLE.

### HOLD

Tracking is uncertain or degraded. A result or recent trusted history may
exist, but current evidence is insufficient for STABLE and has not persisted
long enough to establish LOST. Examples include short occlusion, temporary
detector miss, recovery, stale suspicion, consistency degradation, and short
runtime anomalies. HOLD is an uncertainty state, not a middle score.

### LOST

Trustworthy target-tracking evidence has remained absent long enough, or
missing/stale/invalid evidence has persisted long enough that tracking
reliability is no longer established. A persistent stale result can therefore
produce LOST even while raw `result_present` remains true.

## Architecture and evidence axes

```text
Existing tracker public output
  -> ReliabilityInput
  -> rolling raw buffer
  -> raw metrics and existing quality components
  -> ObservationEvidence
  -> ConsistencyEvidence
  -> FreshnessEvidence
  -> RuntimeEvidence
  -> ReliabilityEvidence fusion
  -> stale evidence with trusted-history gating
  -> temporal persistence and hysteresis
  -> STABLE / HOLD / LOST
```

### Observation evidence

Answers whether there is current, independently refreshed target evidence.
It retains result availability, YOLO execution, confidence, confirmation age,
recovery context, and visibility separately. `used_yolo` alone is never a
fresh confirmation. The score combines only available observation components.

### Consistency evidence

Answers whether current tracking behavior is geometrically and temporally
plausible. It uses continuity, center displacement, bbox area/aspect change,
visibility, and tracking duration. Stationary geometry is not intrinsically
bad; it becomes suspicious only when combined with frozen quality and missing
fresh confirmation.

### Freshness evidence

Separates the retained YOLO score from its age. A detector confirmation
requires a current YOLO invocation, an available acceptable confidence, fresh
age, and valid observation/consistency. Runtime latency is intentionally not
part of observation confirmation. The stale detector may still reject a
candidate using its independent stale precheck. Raw result presence is not a
confirmation.

### Runtime evidence

Represents whole `OpticalFlowTracker.update()` latency health. It does not
separate CPU flow from YOLO work. A latency spike can fail a hard gate or add a
supporting diagnostic, but runtime evidence alone cannot produce STALE/LOST.

## Temporal consistency adaptation

The estimator adapts the temporal-consistency-deviation idea to available
signals; it is not an exact reproduction of the ISA Transactions method
because this tracker exposes no PSR or response map.

For newest-first trusted qualities `q_j`:

```text
weight_j = exp(-decay * j)
trusted_quality_baseline = sum(weight_j * q_j) / sum(weight_j)
temporal_deviation_ratio = trusted_quality_baseline / max(current_quality, epsilon)
```

A ratio above one indicates degradation relative to trusted history. It is
one evidence item, never a state decision by itself.

## Trusted-history gating

To prevent stale output becoming the new normal, suspect, stale, gate-failed,
observation-invalid, or consistency-invalid frames are excluded. The initial
conservative policy admits only explicit fresh detector confirmations that
also pass observation, consistency, gate, and stale-precheck requirements.
This is a degradation-aware memory gate adapted from MACTrack's philosophy,
not an implementation of its network architecture.

## Frozen/stale detection

A window retains actual historical overall-quality values and raw bbox
geometry. It measures quality range and standard deviation, normalized bbox
center span, relative area span, and relative aspect span.

Stale candidacy requires multiple independent categories:

1. abnormal current-vs-trusted quality deviation;
2. simultaneous quality and geometry plateau;
3. absence of acceptable fresh detector confirmation.

Latency and recovery are supporting context. Quality alone, latency alone,
stationary bbox alone, or detector age alone cannot confirm stale tracking.
Candidate evidence must persist for the configured suspect duration and then
the longer confirmation duration.

Internal evidence states are `FRESH`, `SUSPECT`, and `STALE`. They do not
replace the three public reliability states.

```text
effective_tracking_present = result_present and not stale_confirmed
```

The raw field is never overwritten.

## Evidence fusion

The frame result retains separate observation, consistency, freshness, and
runtime scores plus their reasons. Available components may be summarized for
diagnostics, but state transitions require categorical evidence and temporal
persistence in addition to existing quality and hard-gate results.

## State-transition rules

```text
Initial LOST
  -- fresh observation + consistency --> HOLD
HOLD
  -- persistent stable-entry evidence --> STABLE
  -- persistent missing/stale/effective-absence evidence --> LOST
STABLE
  -- persistent degradation or stale suspicion --> HOLD
  -- persistent confirmed stale/missing evidence --> LOST
LOST
  -- validated fresh reacquisition --> HOLD
```

Direct `LOST -> STABLE` is structurally impossible. Stable-entry accumulation
does not proceed during LOST before fresh reacquisition. Entry and exit use
different thresholds to preserve hysteresis.

LOST-to-HOLD reacquisition uses a latency-independent fresh observation that
also survives stale validation and has effective tracking presence. The global
hard gate, including its runtime-latency requirement, remains mandatory for
stable-entry accumulation. Thus an expensive detector frame may establish
HOLD without claiming that full tracking reliability is already STABLE.

Short uncertainty follows the SiamMDTP-derived principle: it may enter HOLD
without immediately becoming LOST. Persistent unresolved uncertainty can
become LOST. Reacquisition must be validated through HOLD before STABLE.

## Failure cases and limitations

- A high-confidence detector false positive can still appear fresh without
  class identity, appearance embedding, or ground-truth association signals.
- Camera motion and truly stationary targets can resemble geometric patterns;
  multiple evidence and periodic confirmation reduce but cannot eliminate the
  ambiguity.
- Device-dependent latency is not a target-observation signal.
- Rolling windows limit observable duration and depend on timestamp validity.
- Default thresholds require calibration across multiple labeled videos,
  target scales, motions, occlusions, devices, and frame rates.
- No claim of a perfect or validated classifier is made.

## Evaluation

Synthetic validation covers normal tracking, isolated runtime spikes, short
detector misses and occlusions, persistent missing and stale results,
stationary targets, reacquisition ordering, history contamination, and
temporary quality degradation.

Manual ground-truth evaluation accepts explicit frame/time state intervals.
Without a GT file it produces no accuracy score. Reports include per-state
precision/recall/F1, macro F1, confusion matrix, false-STABLE rate, and
transition-delay diagnostics where matching GT/predicted transitions exist.
