# AI Smart Scope Aiming Engine

An AI-assisted aiming module for a smart optical scope. It consumes
SmartTracker's per-frame output and produces a real-time aim
recommendation — lead point, azimuth/elevation, hit probability — for
display on a HUD. Does not perform detection or tracking. Pipeline:

```
Camera -> YOLO11s-GA -> SmartTracker -> Smart Scope Aiming Engine -> HUD -> Human Operator
```

**This is an aim-assist system, not an autonomous weapon.** There is no
fire command anywhere in this codebase, and nothing here ever triggers an
action. Every value in `AimAssistOutput` is advisory information for the
HUD; the human operator always reads it and decides independently whether
and when to fire.

## Usage

```python
from aiming_engine import AimingManager

manager = AimingManager()  # loads config/aiming_engine.yaml
output = manager.update(tracker_frame)  # one TrackerFrame per SmartTracker frame

output.state            # AimState (SEARCH/TRACK/AIM/READY) — no firing state
output.aim_solution     # AimSolution: lead point, scope-frame azimuth/elevation, ballistic offset
output.hit_probability  # float in [0, 1] — current aim's estimated shot quality, advisory only
output.aim_ready        # bool, AimReadinessLogic verdict: "aim is currently trustworthy"
output.debug            # dict of diagnostic values (distance, aim error, target speed, ...)
```

Run `python examples/run_aiming_engine_demo.py` for a synthetic end-to-end
smoke test (no SmartTracker/detector involved) that prints per-frame state
and latency.

## Design summary

See `C:\Users\cksgu\.claude\plans\parsed-tickling-narwhal.md` for the
original architecture writeup (written when this was scoped as an
autonomous fire-control system; the underlying math is unchanged, only the
decision layer around it — no more automatic firing — and naming have
been reworked since). Key points:

- All target-motion physics (`TargetState`, `LeadPrediction`, `HitEquation`,
  `ProjectileModel`) runs in **World frame** (inertial, Z-up). Scope frame
  (FLU: X-forward/boresight, Y-left, Z-up) is only used at the very end of
  `AimSolver` to produce azimuth/elevation relative to the scope's own
  boresight, for the HUD reticle.
- `HitEquation` solves for `[time_of_flight, azimuth, elevation]` jointly
  (3 equations, 3 unknowns) — flight time and target motion are coupled,
  not pre-baked into a static lead point. `LeadPrediction` only supplies a
  fast initial guess. The "projectile" here is a simulated trajectory used
  purely to compute where the reticle should sit — nothing is launched.
- `NewtonSolver` only calls `equation.residual(x)`; it knows nothing about
  ballistics. `AimSolver` warm-starts it from the previous frame's
  converged solution when the target is moving smoothly.
- `AimStateMachine` tracks aim *quality* (SEARCH -> TRACK -> AIM -> READY)
  so the HUD can show a lock/confidence indicator. It has no state beyond
  READY and cannot trigger anything.
- `AimReadinessLogic` computes a separate, per-frame `aim_ready` boolean
  (hit probability, distance, aim error, tracking/follower stability) —
  this is the "green reticle" signal for the operator, not a permission
  gate for an action.

## Extending the projectile model (drag, wind, G1/G7 ballistic coefficient)

`ProjectileModel` integrates acceleration from a list of `ForceModel`
plugins (`forces.py`) via RK4. To add a new force:

1. Subclass `ForceModel` in `forces.py` (or a new module) and implement
   `acceleration(position, velocity, t) -> np.ndarray`.
2. Register it in `FORCE_REGISTRY` under a name string.
3. Add that name to `projectile.forces` in `config/aiming_engine.yaml`
   (e.g. `forces: [gravity, drag]`).

No changes are needed in `ProjectileModel`, `HitEquation`, `NewtonSolver`,
or `AimSolver` — they only ever see the combined acceleration/position via
`ProjectileModel.state_at()`.

Note: `integrator_step` doesn't affect accuracy for the gravity-only v1
model (RK4 is exact for constant acceleration at any step size). Once a
nonlinear force like drag is added, `integrator_step` becomes a real
accuracy/latency tradeoff — tighten it if the trajectory curves quickly
relative to the step, and re-check the per-frame latency budget.

## Extending target motion estimation (Kalman, constant acceleration)

`TargetState` delegates to a `MotionEstimator` (`target_state.py`). To add
one:

1. Subclass `MotionEstimator` and implement `update()`/`predict()`.
2. Register it in `build_estimator()` under a name string.
3. Set `target_state.estimator` in the YAML config to that name.

`HitEquation`, `LeadPrediction`, and `AimSolver` only ever call
`TargetState.predict()` / `TargetState.update()` — they never depend on
which estimator is behind it.

## Configuration

All tunable parameters live in `config/aiming_engine.yaml`, loaded once by
`config.py` into typed dataclasses. Nothing outside `config.py` reads YAML
directly.
