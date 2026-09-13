# Phase 52：QDR 经验执行可达管标定与闭环诊断

## 结论

本阶段实现了一个可复现的 QDR suffix 经验执行误差管：独立 calibration
样本先冻结逐步半径，再由 distributed DN-MPC 在 obstacle、boundary 和
inter-agent suffix gate 中使用该半径。实现、配置快照、source hash、step/episode
JSONL 和 TensorBoard 均保留。

本阶段不通过 promotion gate。经验管显著提高了 nominal suffix 的几何裕量，
但在当前 immutable queue 合同下造成严重 liveness 损失：40 个 validation
development episode 的 safe capture 从 100% 降至 15%，timeout 从 0% 升至 85%。
因此该版本冻结为可复现的 safety--liveness negative ablation；没有访问
locked-test，也不能把经验管写成形式化安全证明。

## 1. 实现和信息边界

新增的 `DistributedDNMPCConfig` 字段为：

```text
qdr_execution_tube_enabled
qdr_execution_tube_multiplier
qdr_execution_tube_radius_m_by_step
```

当没有显式开启时，半径为零，历史 QDR action-selection 路径保持不变。当
开启并提供 calibration artifact 时，在线规划使用逐 horizon step 的冻结半径，
而不是把解析式最坏上界直接乘到末端。诊断字段包括启用率、倍率、平均半径和
最大半径。该管只使用执行合同和离线 calibration artifact；target truth 不进入
在线 planner。

相关源文件和配置：

- `src/encirclement3d/distributed_dn_mpc.py`
- `scripts/evaluate_s4_closed_loop.py`
- `configs/phase52_qdr_execution_contract.yaml`
- `configs/phase52_qdr_execution_tube_calibration.yaml`
- `configs/phase52_qdr_execution_tube_closed_loop.yaml`

## 2. 独立 calibration

标定使用 `2048` 个样本、8-step horizon、4 架防御机、目标覆盖率 `0.99` 和
分位数 `0.99`。固定参数块与执行参数随机化压力块分开记录：

| variant | 基础管覆盖率 | 校准后覆盖率 | simultaneous multiplier | 校准半径最大值 |
| --- | ---: | ---: | ---: | ---: |
| `qdr_delay4_noise008` | 100.00% | 99.02% | 0.7790（在线下限取 1.0） | 0.179 m error-quantile |
| `qdr_delay4_noise008_randomized` | 43.80% | 99.02% | **2.0343** | 0.327 m error-quantile |

最终闭环固定使用压力块的 `2.0342518477112503` 和其逐步 calibrated radius
vector；8 个 step 的最大冻结半径为 `1.4086 m`。这里的 multiplier 和 radius
是经验校准量，不是 reachable-set theorem，也不提供 forward invariance。

calibration artifact：

```text
results/phase52_qdr_execution_tube_calibration_v2/summary.json
SHA-256 is recorded in the artifact config.yaml.
```

## 3. 配对闭环设置

两臂使用同一 Phase 48 scene prefix 的前 40 个 episode、同一 GRU checkpoint
`727201`、同一采样 seed `745102`、same planner、distributed delayed DN-MPC、
local CBF、CPU 单进程和 Torch intra/inter-op threads `1/1`。显式执行合同为：

```text
action_delay_steps = 4
command_noise_std = 0.08 m/s
command_noise_bound = 3 sigma
pending_command_authority = immutable
```

比较项只有 QDR execution tube：

- `nominal`: QDR 开启、经验管关闭；
- `empirical`: QDR 开启、加载 calibration JSON 的逐步经验管。

该实验使用 `validation_selection`，不是 confirmation，也没有读取 locked-test。

## 4. 闭环结果

| 指标 | nominal QDR | empirical tube QDR |
| --- | ---: | ---: |
| safe capture | 100.00% | **15.00%** |
| capture event | 100.00% | 15.00% |
| collision | 0.00% | 0.00% |
| boundary violation | 0.00% | 0.00% |
| timeout | 0.00% | **85.00%** |
| mean minimum clearance | 0.513 m | 1.601 m |
| QDR suffix admissible rate | 80.50% | 99.43% |
| suffix gate exhaustion rate | 23.49% | 89.25% |
| maximum exhaustion streak | 23 steps | 250 steps |
| mean QDR radius | 0 m | 0.504 m |
| maximum QDR radius | 0 m | 1.409 m |

40-episode paired bootstrap（10,000 resamples，candidate minus nominal）为：

- safe-capture delta：`-85.00 pp [-95.00,-72.50]`；
- timeout delta：`+85.00 pp [+72.50,+95.00]`；
- collision 和 boundary delta：`0 pp`。

这说明经验管确实改变了 planner 的 suffix feasibility 判定并增加了几何裕量，
但当前“不可变 prefix + 最小违反候选”合同没有提供恢复机制，最终表现为大量
超时。碰撞为 0% 不能单独解释为安全收益，因为 safe capture 同时降至 15%。

## 5. 延迟分解（p50 / p95 / p99，ms）

| stage | nominal QDR | empirical tube QDR |
| --- | ---: | ---: |
| predictor | 5.41 / 6.80 / 7.82 | 5.50 / 7.17 / 8.36 |
| planner | 12.46 / 16.10 / 17.96 | 9.17 / 12.60 / 15.07 |
| QDR | 0.63 / 0.85 / 1.07 | 0.64 / 0.85 / 1.07 |
| safety | 0.62 / 0.82 / 1.01 | 0.62 / 0.84 / 1.08 |
| total | **21.37 / 25.84 / 28.36** | **18.26 / 22.75 / 26.47** |

本阶段没有把 `100 ms` 当作硬门槛；所有组件和总闭环延迟均按 p50/p95/p99
报告。经验管没有造成计算延迟上升，失败原因主要是 liveness/feasibility，而
不是 runtime。

## 6. TensorBoard 与复现资产

两臂均保留：

```text
results/phase52_qdr_tube_nominal_seed727201_v2/
results/phase52_qdr_tube_empirical_calibrated_seed727201/
results/phase52_qdr_execution_tube_calibration_v2/
results/phase52_qdr_execution_tube_paired_aggregate.json
```

TensorBoard 中记录了：

```text
Summary/qdr_execution_tube_enabled_rate
Summary/mean_qdr_execution_tube_multiplier
Summary/mean_qdr_execution_tube_radius_m
Summary/max_qdr_execution_tube_radius_m
Summary/PlannerLatency/p{50,95,99}_ms
Summary/PredictorLatency/p{50,95,99}_ms
Summary/QDR/latency_p{50,95,99}_ms
Summary/SafetyLatency/p{50,95,99}_ms
Summary/TotalControlLatency/p{50,95,99}_ms
```

## 7. 决策与后续计划

- [x] 完成独立 calibration，并冻结压力块 multiplier 和逐步 radius vector；
- [x] 完成默认关闭、显式开启、配置快照、source hash、TensorBoard 和单元测试；
- [x] 在同一 validation prefix 完成 nominal/empirical 配对闭环；
- [x] 识别 empirical tube 的主要失败模式：`gate exhaustion -> timeout`；
- [x] 将当前版本冻结为负消融，不进入 locked-test；
- [ ] 不在当前 validation prefix 上继续扫描 multiplier；
- [ ] 若继续 QDR，优先研究“tube radius 随剩余可恢复时间和局部障碍距离衰减”的
      gated margin，并在新的 development split 预注册 timeout 和 safe-capture gate；
- [ ] 在任何重新开放 UAKR/RNIC/full composition 前，先证明 QDR margin 不会把
      liveness 牺牲到不可接受水平；
- [ ] 不把本阶段的 0% collision 写成安全证明或无条件性能提升。
