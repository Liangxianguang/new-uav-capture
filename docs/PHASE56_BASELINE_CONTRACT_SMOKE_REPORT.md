# Phase 56 baseline contract smoke report

## Scope

This is an interface and instrumentation smoke, not a performance claim. It
uses the 12-episode Phase 56 scene-generator smoke manifest and evaluates two
episodes with the same GRU `both` checkpoint, local CBF, CPU Torch threads
`1/1`, `num_samples=1`, and `sampling_steps=1`.

The formal development matrix remains separate and must use the complete
360-episode manifest and three seeds.

## Command

```powershell
python scripts\evaluate_s4_closed_loop.py `
  --scenes results\phase56_strong_baseline_scenes_smoke\scenes.jsonl `
  --protocol configs\phase15_s4_branching_pilot.yaml `
  --environment-config configs\capture_radius_pursuit_central_v4_flee.yaml `
  --mpc-config configs\innovation_mpc.yaml `
  --checkpoint results\phase16_gru_both_seed727201\checkpoint.pt `
  --output-dir results\phase56_baseline_contract_smoke `
  --candidate-source checkpoint `
  --methods B0_current_state_delayed_mpc B1_qdr_mpc B5_asynchronous_distributed_mpc `
  --episodes 2 --num-samples 1 --sampling-steps 1 `
  --projection-iterations 1 --prediction-refresh-interval-steps 1 `
  --device cpu --torch-num-threads 1 --torch-num-interop-threads 1 `
  --safety-layer local_cbf --decision validation_confirmation
```

The command writes `config.yaml`, `episodes.jsonl`, `steps.jsonl`,
`summary.json`, and TensorBoard events under each method directory.

## Descriptive output

| Method | Safe capture | Collision | Timeout | Predictor p95 (ms) | Planner p95 (ms) | QDR/tube p95 (ms) | Safety p95 (ms) | Total p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 current-state delayed-MPC | 100% | 0% | 0% | 20.23 | 26.42 | 0.00 | 4.89 | 53.75 |
| B1 QDR-MPC | 100% | 0% | 0% | 20.79 | 71.69 | 4.39 | 4.56 | 109.15 |
| B5 asynchronous distributed-MPC | 100% | 0% | 0% | 20.31 | 36.33 | 0.00 | 5.01 | 64.22 |

The machine-readable source is local and intentionally ignored by Git:
`results/phase56_baseline_contract_smoke/summary.json`.

## Interpretation boundary

The smoke confirms that the baseline aliases, deterministic asynchronous
communication path, fixed component timing fields, JSONL logging, and
TensorBoard logging are wired correctly. Two episodes cannot establish a
capture or safety effect. It also does not validate the complete 360-episode
matrix, mirror-group bootstrap statistics, or any locked-diagnostic claim.

local CBF remains an empirical filter. No R-CLBF-QP safety proof is claimed.

