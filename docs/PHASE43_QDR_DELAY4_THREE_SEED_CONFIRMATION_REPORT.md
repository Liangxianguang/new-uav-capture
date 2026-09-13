# Phase 43：QDR delay4 三种子 fresh confirmation 报告

> 日期：2026-09-13
> 实验级别：validation-confirmation，三种子执行延迟轴
> 仓库：[Liangxianguang/new-uav-capture](https://github.com/Liangxianguang/new-uav-capture)

## 1. 目的与审计边界

Phase 42 在一个 checkpoint seed 上观察到 delay4 下 QDR 的明显安全收益。Phase 43
在同一独立 fresh manifest 上补齐 checkpoint seeds `727202/727203`，并重新执行每个
seed 的 QDR-off 对照，形成三种子、同 episode_index 的配对 confirmation。

该确认只改变 QDR 开关；command delay 固定为 4，authority 固定为 immutable，
predictor、采样 seed、MPC、local CBF、场景和执行动力学全部冻结。QDR-off 使用显式
`--no-queue-aware-rollout --no-queue-aware-safety-projection`，因为共享 YAML 的
默认值是 QDR-on。该确认不使用 locked-test，也不包含 UAKR/RNIC。

## 2. 可复现设置

| 项目 | 设置 |
| --- | --- |
| manifest | `results/phase34_qdr_fresh_validation_scenes/scenes.jsonl` |
| 规模 | 3 seeds × 80 episodes = 240 episodes per arm / 40 mirror groups per seed |
| split | `validation_confirmation`，不含 locked-test |
| manifest SHA-256 | `15652f76157157c8823dfd7086a4978f4f5de2aed108b4ebef18817f4cc03646` |
| seeds | `727201/727202/727203` |
| predictor | GRU `both` |
| planner | distributed delayed DN-MPC，worst-case，horizon 8，control horizon 3 |
| execution | enabled，command delay 4，immutable，noise/tracking/drag 0 |
| safety | local CBF |
| sampling | `num_samples=1`，`sampling_steps=8`，`sampling_seed=745102` |
| committed config | `configs/phase42_qdr_delay4_fresh_validation.yaml` |
| aggregate artifact | `results/phase42_qdr_delay4_three_seed_aggregate.json` |

聚合使用 `scripts/aggregate_closed_loop_seed_results.py`，bootstrap `10,000` 次，
seed `20260913`，重采样单位为 matched predictor seed + frozen-scene episode。

## 3. 三种子结果

### 3.1 Aggregated episode outcome

| 指标 | QDR-off | QDR-on | on - off |
| --- | ---: | ---: | ---: |
| safe capture | 77.08% [71.25, 82.50] | 96.67% [94.17, 98.75] | +19.58 pp [13.75, 25.42] |
| collision | 22.92% [17.50, 28.75] | 1.25% [0.00, 3.33] | -21.67 pp [-27.92, -15.83] |
| boundary violation | 5.00% [2.50, 7.92] | 1.25% [0.00, 3.33] | -3.75 pp [-7.08, -0.42] |
| timeout | 0.00% [0.00, 0.00] | 2.08% [0.00, 5.00] | +2.08 pp [0.00, 5.00] |
| mean capture time | 2.499 s [2.354, 2.677] | 5.826 s [5.296, 6.349] | +2.940 s [2.316, 3.633]* |
| mean minimum clearance | 0.268 m [0.240, 0.295] | 0.533 m [0.519, 0.546] | +0.265 m [0.234, 0.294] |

`*` 捕获时间使用成功捕获样本，因两臂成功数不同，只作描述性 paired summary。

三种子确认支持以下结论：在该固定 delay4 轴上，QDR 的 safe-capture 与 collision
改善方向跨 seed 保持一致，且 paired bootstrap 区间不跨零；同时 QDR 引入了少量
timeout，不能把安全收益写成无代价提升。

### 3.2 Per-seed outcome and total latency

下表保留每个 seed 的完整 total latency `p50/p95/p99`（ms），避免只依赖 pooled
step-level percentile：

| seed | arm | safe capture | collision | boundary | timeout | total p50/p95/p99 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 727201 | off | 77.5% | 22.5% | 5.0% | 0.0% | 29.19 / 54.29 / 61.93 |
| 727202 | off | 75.0% | 25.0% | 5.0% | 0.0% | 54.00 / 70.35 / 90.87 |
| 727203 | off | 78.75% | 21.25% | 5.0% | 0.0% | 53.78 / 69.33 / 89.16 |
| 727201 | on | 97.5% | 2.5% | 2.5% | 0.0% | 45.88 / 78.65 / 95.10 |
| 727202 | on | 96.25% | 0.0% | 0.0% | 3.75% | 51.23 / 102.90 / 126.13 |
| 727203 | on | 96.25% | 1.25% | 1.25% | 2.5% | 45.92 / 88.86 / 100.98 |

三种子 summary percentile 的均值（仅作描述性汇总）如下，顺序均为
`p50/p95/p99`（ms）：

| arm | predictor | planner | QDR | safety | total |
| --- | --- | --- | --- | --- | --- |
| QDR-off | 21.14 / 30.83 / 38.78 | 19.07 / 28.67 / 36.06 | 0 / 0 / 0 | 2.25 / 3.51 / 4.63 | 45.65 / 64.65 / 80.66 |
| QDR-on | 12.52 / 26.04 / 31.20 | 27.52 / 51.54 / 63.63 | 1.30 / 2.43 / 3.12 | 1.41 / 2.86 / 3.51 | 47.68 / 90.14 / 107.40 |

QDR-on 的平均 summary total p95 比 off 高约 `39.4%`；p99 也高约 `33.2%`。虽不把
100 ms 作为硬门槛，QDR-on 的平均 p99 已超过 100 ms 参考值，说明效率代价不能
忽略。

### 3.3 QDR 机制指标

QDR-on 三种子 summary percentile/episode 指标的均值为：

- queue length：`4.0`，first controllable step：`4.0`；
- prefix admissible：`95.75%`；
- suffix admissible：`84.62%`；
- suffix gate exhaustion：`19.02%`；
- endpoint check coverage：`90.30%`；
- effective-plan rate：约 `98.7%`。

这表明 QDR 确实进入延迟队列执行路径，但近五分之一的 gate step 仍无完全可行
suffix，且 immutable prefix 不能由后续控制事后修复。

## 4. TensorBoard 与结果完整性

六个 run 的根配置、source hashes、episode/step JSONL、summary 和 TensorBoard
event files均已保留：

```text
results/phase42_qdr_delay4_off_seed727201/
results/phase42_qdr_delay4_off_seed727202_v2/
results/phase42_qdr_delay4_off_seed727203_v2/
results/phase42_qdr_delay4_on_seed727201/
results/phase42_qdr_delay4_on_seed727202/
results/phase42_qdr_delay4_on_seed727203/
```

每个 run 都有 80 episode rows 和 2 个 TensorBoard event files。off/on 的 scalar
tag 数分别为 `187/221`；每个 run 的 predictor/planner/QDR/safety/total latency
p50/p95/p99 必需标签均存在。三种子聚合 artifact 记录了 manifest hash、seed 列表、
bootstrap 参数和 paired deltas。

## 5. 阶段判定

**QDR delay4 安全效果：三种子 confirmation 通过（条件性）。** safe capture
提升 `19.58 pp [13.75,25.42]`，collision 降低 `21.67 pp`，boundary 也改善；
这是目前 QDR 最有说服力的实证轴。

**QDR promotion：仍为 No-Go。** QDR-on 仍有 `1.25%` collision、`2.08%` timeout、
`19.02%` suffix gate exhaustion，并带来约 `39.4%` 的 summary total p95 增幅。因而
当前最准确的论文表述是“在较长执行延迟下显著改善安全--捕获 outcome，但以延迟和
liveness 代价为代价的可审计 QDR 模块”，不是“无条件提升”或“安全证明”。

## 6. 下一步 TodoList

- [x] 三种子 delay4 QDR-off/QDR-on 配对 confirmation；
- [x] 分层 bootstrap 与逐 seed total/predictor/planner/QDR/safety latency 记录；
- [ ] 在 delay4 固定条件下，只加入 bounded execution noise，完成三种子 paired test；
- [ ] 单独测试 `replace_nonexecuting`/`flush_pending` authority，记录 safety--liveness
  trade-off，禁止与 noise 同时改变；
- [ ] 若至少一个 stress block 同时满足 safe-capture non-inferiority、collision/
  timeout 约束和相对 p95 gate，再开放 QDR×UAKR/RNIC 交互；
- [ ] 在上述 gate 通过前不运行 locked-test Full 组合，也不把 QDR gate 写成形式化
  safety certificate。
