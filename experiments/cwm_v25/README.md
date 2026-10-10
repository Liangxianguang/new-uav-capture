# V25: fail-closed optional response boundary

V23 is fully audited but all four response configurations failed research
eligibility. V25 records the deployment boundary needed before any later model
can touch DN-MPC: default `off`, `shadow` never controls, guarded mode requires
both offline qualification and every online contract. This boundary returns
TARGET forecasts (N,8,3), never UAV commands (N,8,4,3). Empty forecast on
off/refusal/failure leaves the caller's original action route available.

This is an adapter contract, not a promoted model or an active local-selector
integration. It does not generate candidates, change the planner, run CBF,
change safety margins, or use holdout data. The original DN-MPC + local CBF
remains the only active controller.

`original_entry_boundary.py` exercises the real historical `run_episode`
entry after the unchanged DN-MPC plan has been produced. Its off,
qualification-refusal, controlled loader/inference-failure and synthetic shadow
paths must reproduce the plain entry AND archived historical defender/target
trajectory bytes, DN-MPC plan bytes and post-CBF commanded-action bytes.
Fault-injection uses noncontrolling shadow and zero history/relative/CV fixtures:
it is NOT a real learned-model public context or sequential selector integration.
Refusal uses V23's actual failed status. No synthetic all-True record qualifies
the entry harness. The real V23 models are never loaded here.
This representative CPU/runtime regression is not 8-level capture-gain or
latency qualification. The archive preserves both separate-process runs.

Loading is lazy; off/refusal never inspect inputs or invoke the loader. Public
inputs are isolated copies. CPU Torch/Python/NumPy RNG is restored around both
loading and inference, without CUDA initialization. Anchored response must be
exactly zero; calibrated public common reference is explicit and exactly
candidate-invariant. Full 8-level/new-scene closed-loop qualification,
sequential/self-generated candidates, latency, and audit-record binding of
future eligibility remain unproved.

```powershell
python -m pytest experiments/cwm_v21 experiments/cwm_v22 experiments/cwm_v23 experiments/cwm_v24 experiments/cwm_v25 -q
python experiments/cwm_v25/original_entry_boundary.py --capsule experiments/cwm_v1/baseline/capsule.zip --output results/cwm_v25/fresh_entry_primary --episodes 2
python experiments/cwm_v25/original_entry_boundary.py --capsule experiments/cwm_v1/baseline/capsule.zip --output results/cwm_v25/fresh_entry_repeated --episodes 2
python experiments/cwm_v25/release_entry_audit.py --primary results/cwm_v25/fresh_entry_primary --repeated results/cwm_v25/fresh_entry_repeated --output results/cwm_v25/fresh_entry_release.zip --report results/cwm_v25/fresh_entry_release.json
python experiments/cwm_v25/release_entry_audit.py --verify experiments/cwm_v25/artifacts/entry_fault_replay_20261010.zip
```
