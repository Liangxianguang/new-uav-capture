# Phase 117 IPPO + CBF training record

Status: running. This record is a checkpointed experiment log, not evidence of
convergence.

## Objective

Train a decentralized IPPO baseline with the empirical local CBF execution
filter on repaired, contract-valid target trajectories. Target contract faults
are reported separately from defender collision and defender boundary failure.

## Launch configuration

- Run directory: `models/phase117_ippo_cbf_resumable_seed11701`
- Algorithm: IPPO, seed `11701`, CUDA, hidden dimension `128`
- Optimizer: PPO, learning rate `3e-5`, 4 PPO epochs, minibatch size `256`
- Rollouts: one episode per update; 20,000 update budget
- Safety: `--use-cbf-train --cbf-train-iterations 1 --use-cbf-eval`
- Training split: 300 records from
  `results/phase86_nominal_repaired_development_calibration/scenes.jsonl`
- Configuration: `configs/phase108_ippo_expert_contract.yaml`
- TensorBoard: `models/phase117_ippo_cbf_resumable_seed11701/tensorboard`
- Full checkpoint interval: 500 updates; `progress.json` is atomically
  refreshed after each update.

## Independent validation protocol

Each full checkpoint at updates 500, 1,000, 2,000, 5,000, 10,000, 15,000, and
20,000 is evaluated on the separate 300-scene manifest
`results/phase86_nominal_repaired_development_validation/scenes.jsonl` using
CUDA and the CBF filter. The reported outcomes are safe capture, defender
collision, defender boundary violation, target invalidity, and timeout. A
training return alone is never used as a convergence claim.

## Early observation (2026-09-19 14:00 +08:00)

At update 81 (20,167 environment steps), the latest rollout ended in timeout;
safe capture, defender collision, defender boundary violation, and target
invalidity were all zero. This is an exploratory-stage observation and does not
justify a reward or hyperparameter change. The first independent validation is
pending update 500.

## Reward-contract correction

Commit `f3d961a` changes the safety reward so a target contract fault terminates
and is logged, but does not impose a defender collision penalty. That preserves
the intended evaluation semantics for subsequent resumes and comparison runs.
