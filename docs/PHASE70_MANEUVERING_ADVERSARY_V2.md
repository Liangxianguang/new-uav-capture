# Phase 70: Maneuvering Adversary v2

Phase 70 adds a reproducible, physically constrained target maneuver policy
for development and validation experiments. It is deliberately separate from
the frozen locked-test target modes.

## Behavior contract

At a replan step, the target evaluates short candidate rollouts and selects
the candidate with the largest survival-oriented score. The candidate set
contains six maneuver families:

- `straight_flee`: move away from the estimated defender centroid;
- `lateral_jink`: combine fleeing with a positive or negative lateral move;
- `obstacle_bypass`: select a left, right, or top waypoint around known
  obstacle geometry;
- `reverse_lane_change`: trade forward progress for a lateral lane change;
- `vertical_escape`: climb or descend while fleeing;
- `speed_burst`: use a bounded short speed burst.

The default development configuration uses a 12-step horizon, replans every
8 steps, and holds a selected mode for at least 6 steps. The valid ranges are
8--16 horizon steps and 6--10 replanning steps, so the policy does not inject
independent white noise at every control step.

The target command is constrained by the target speed and acceleration limits,
a 1.05 rad/s maximum turn rate, and a 12 m/s^3 jerk limit. The target's
defender tracks are private delayed/noisy measurements with dropout and
constant-velocity propagation during missing packets. The maneuver selector
therefore does not read `defender_positions` or `defender_velocities` directly.
The existing `adaptive_adversarial` mode remains available as an explicit
oracle stress-test mode and is not used as the main v2 contract.

The obstacle route candidates use obstacle shape extents and generate left,
right, and top bypass directions before local obstacle repulsion becomes
dominant. The selector scores predicted defender distance, terminal distance,
obstacle clearance, boundary clearance, the angular escape gap, estimated
capture time, and direction-change smoothness.

## Versioned implementation

- Environment implementation: `src/encirclement3d/pursuit_env.py`
- Showcase override: `scripts/run_mixed_obstacle_showcase.py`
- Dataset-mode registration: `scripts/collect_prediction_dataset.py`
- Validation TensorBoard logging: `scripts/evaluate_random_central_mixed_obstacles.py`
- Environment configuration: `configs/phase70_maneuvering_adversary_v2.yaml`
- Development protocol: `configs/phase70_maneuvering_adversary_v2_protocol.yaml`

The selected mode, route, switch count, estimated capture time, escape gap,
observation ages, and physical limits are emitted in episode diagnostics and
the target velocity/acceleration/mode are retained in the optional replay
history. The policy-safe defender observation remains unchanged.

## Smoke evidence

A four-episode validation smoke was run with the development dynamic
encirclement baseline and local CBF:

```powershell
python scripts\evaluate_random_central_mixed_obstacles.py `
  --baseline dynamic_encirclement `
  --protocol configs\phase70_maneuvering_adversary_v2_protocol.yaml `
  --environment-config configs\phase70_maneuvering_adversary_v2.yaml `
  --split validation --episodes 4 `
  --output-dir results\phase70_maneuvering_adversary_v2_baseline_smoke `
  --use-cbf --device cpu
```

The smoke produced:

| Metric | Value |
| --- | ---: |
| Episodes | 4 |
| Safe capture | 1/4 |
| Collision/safety-failure episodes | 3/4 |
| Mean target maneuver switches | 2.5 |
| Mean estimated capture time | 2.496 s |
| Mean estimated escape gap | 5.032 rad |
| Transit-route feasibility | 100% |

This is a difficulty and implementation smoke, not a claim that the
defender model has improved. It shows that the target is switching between
maneuvers under a reproducible private-information contract, while the
baseline defender is not yet robust to this new behavior. The final run's
per-episode scalars and configuration text are retained in
`results/phase70_maneuvering_adversary_v2_baseline_smoke_final/tensorboard/`.

The prediction-data path was also smoke-tested with two stationary-observer
episodes, producing 112 windows. Its metadata records
`uses_target_truth: false` for the input contract and uses target truth only
for the future supervised label, as required by the partial-observation
protocol.

## Reproducible showcase override

For a single rendered replay, the showcase runner accepts
`--target-motion-mode adaptive_maneuvering`. This is a development-only
override and must not be supplied when reproducing any locked-test result.

```powershell
python scripts\run_mixed_obstacle_showcase.py `
  --method capture `
  --checkpoint models\v5_development_exact_reactive_seed661606.pt `
  --seed 642130 `
  --output-dir results\phase70_maneuvering_showcase_s1 `
  --initial-side-distance 7.0 --scenario s1 --layout mixed `
  --detection-range 14.0 --target-speed-scale 0.65 `
  --target-motion-mode adaptive_maneuvering `
  --use-cbf --device cpu --fps 12 --frame-stride 1
```

## Required follow-up before a paper claim

The next experiment is a fresh, pre-registered comparison of
`flee_persistence`, `adaptive_maneuvering`, and `adaptive_adversarial` on the
same validation scene manifest. It should report safe capture, collision,
timeout, target mode-switch count, turn-rate/acceleration/jerk percentiles,
obstacle-bypass rate, escape gap, and predictor/planner/QDR/safety/total
p50/p95/p99. The locked test remains closed until the validation result is
stable across seeds and the target behavior contract is frozen.
