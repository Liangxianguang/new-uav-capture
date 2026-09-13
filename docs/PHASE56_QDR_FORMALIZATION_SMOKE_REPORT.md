# Phase 56：QDR 形式化时间索引 Smoke 报告

## 结果

独立 checker 已在固定 seed `20260913`、4 架无人机、horizon `8` 和队列长度
`0/2/4/8` 上运行。四种队列长度全部通过：

| Queue length | First controllable step | Max position error | Max velocity error | Terminal index error | Double delay |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 0.0 m | 0.0 m/s | 0 | false |
| 2 | 2 | 0.0 m | 0.0 m/s | 0 | false |
| 4 | 4 | 0.0 m | 0.0 m/s | 0 | false |
| 8 | 8 | 0.0 m | 0.0 m/s | 0 | false |

候选动作施加时刻为 `t+d+k`，对应状态首次可见时刻为 `t+d+k+1`。该 smoke
只证明 deterministic toy dynamics 下的时间索引组合等价性，不证明真实执行器
安全、连续时间 forward invariance 或 local CBF 的安全性。

## 场景生成器 smoke

`generate_phase56_strong_baseline_scenes.py` 在 `groups-per-block=2` 下成功生成
12 个 episodes / 6 个 mirror groups，包含：

- `id_reference`；
- `delay_noise_grid`；
- `communication_execution_grid`；
- development-calibration、development-confirmation、locked-diagnostic 三类
  split；
- 每个 mirror group 的 upper/lower 成对记录、共享 layout seed 和障碍物。

正式矩阵最低规模为 360 episodes / 180 mirror groups；smoke 结果不能替代正式
三种子闭环实验。

## 复现工件

- checker：`scripts/check_qdr_time_index.py`；
- checker module：`src/encirclement3d/qdr_formalization.py`；
- scene generator：`scripts/generate_phase56_strong_baseline_scenes.py`；
- config：`configs/phase56_qdr_formalization.yaml`、
  `configs/phase56_strong_baseline_protocol.yaml`；
- local result：`results/phase56_qdr_formalization_smoke_rerun/`；
- local TensorBoard：`results/phase56_qdr_formalization_smoke_rerun/tensorboard/`。

