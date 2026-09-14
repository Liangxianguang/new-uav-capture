# Phase 63：QueueToken / QDR fresh calibration 报告

## 1. 阶段定位

本阶段在一份全新的 delay/noise/communication/execution calibration manifest
上，比较 strong delayed-MPC、immutable QDR 和 token-matched bounded
`replace_nonexecuting`。目标是验证 Phase 62 的 QueueToken/ACK 合同是否能在
真实闭环中支持可审计的队列恢复，同时检查恢复是否改善 QDR liveness。

本阶段只使用 `development_calibration` split，不读取或调参
`locked_diagnostic` split；因此结果不是 locked-test promotion。

## 2. 场景与协议

- canonical manifest：360 episodes / 180 mirror groups；
- 四个 block：`id_reference`、`delay_noise_factorial`、
  `communication_execution_factorial`、`joint_stress_transfer`；
- 每个 block 90 episodes / 45 mirror groups；
- development calibration：120 episodes / 60 mirror groups；
- 三个 Diagonal-SSM predictor seeds：727201、727202、727203；
- 场景 manifest SHA-256：
  `f5cab1e7eae0f4bf92686d7e1b586d9d67c1e7cd9a871fd435e53fe724fbd2c4`；
- calibration split SHA-256：
  `56f1ed9d8dd3e5b46e9b974bd21d1d0cad7f1e4f514c0bec61372e225b4b8e90`；
- 所有运行均使用相同场景、相同 predictor seeds、local CBF、每步刷新预测和
  1/1 Torch 线程；
- 所有实验均保留 config snapshot、episode/step JSONL 和 TensorBoard event。
- fixed-K=8 arm 使用真正的 Diagonal-SSM sample set，不将重复的单均值轨迹计作
  多模态候选。

## 3. 主要结果

区间为按 mirror group 和 predictor seed 的分层 bootstrap 95% CI。

| 方法 | Safe capture | Collision | Boundary | Timeout | Total p50/p95/p99 (ms) |
|---|---:|---:|---:|---:|---:|
| Strong delayed-MPC | 57.78% [50.83, 64.44] | 42.22% [35.56, 48.89] | 10.00% [5.83, 14.44] | 0.00% | 18.505 / 24.168 / 27.687 |
| Immutable QDR + token | 87.78% [82.50, 92.50] | 0.28% [0.00, 1.11] | 0.28% [0.00, 1.11] | 11.94% [6.94, 17.50] | 31.264 / 42.647 / 61.948 |
| Bounded replace + token | 74.17% [67.50, 80.28] | 1.11% [0.00, 3.61] | 0.56% [0.00, 1.94] | 24.72% [19.44, 30.56] | 31.443 / 41.430 / 48.113 |
| Fixed-K=8 immutable QDR + token | 85.00% [80.56, 89.17] | 5.28% [2.78, 8.06] | 3.89% [1.67, 6.67] | 9.72% [6.11, 13.61] | 64.591 / 80.739 / 91.569 |

相对 strong delayed-MPC，immutable QDR 的 paired safe-capture delta 为
`+30.00 pp [+23.33, +37.22]`，collision delta 为
`-41.94 pp [-48.89, -35.28]`；但 timeout 增加
`+11.94 pp [+6.94, +17.50]`。这支持一个条件性的 safety-axis 改善，不能解释为
无代价的完整闭环提升。

相对 immutable QDR，bounded replacement 的 safe-capture delta 为
`-13.61 pp [-19.44, -8.33]`，timeout delta 为
`+12.78 pp [+8.06, +18.06]`，最大 exhaustion streak 从 113 增加到 192。
因此，当前 recovery authority 不通过 promotion gate。

fixed-K=8 相对 immutable K=1 QDR 的 paired safe-capture delta 为
`-2.78 pp [-9.44, +3.06]`，collision delta 为
`+5.00 pp [+2.50, +7.78]`，timeout delta 为
`-2.22 pp [-8.06, +4.17]`；最大 exhaustion streak 为 78。候选数增加因此
没有在本 calibration 上带来可确认的安全或 liveness 增益，且计算代价显著增加。

## 4. QueueToken / ACK 结果

| 方法 | Recovery request | Token present | ACK accepted | ACK applied | 最大 exhaustion streak |
|---|---:|---:|---:|---:|---:|
| Strong delayed-MPC | 0.00% | 0.00% | 0.00% | 0.00% | 0 |
| Immutable QDR + token | 0.00% | 0.00% | 100.00% | 0.00% | 113 |
| Bounded replace + token | 14.26% | 14.26% | 100.00% | 14.26% | 192 |
| Fixed-K=8 immutable QDR + token | 0.00% | 0.00% | 100.00% | 0.00% | 78 |

所有需要 mutable recovery 的请求都携带了当前 token，并获得 accepted ACK；
因此 QueueToken freshness 合同在真实闭环中没有出现 stale/missing-token
失败。但 accepted/applied 只说明队列合同执行成功，不说明 suffix 可行、捕获率
或几何安全。

## 5. 五类延迟分位数

完整分位数如下，单位为 ms：

| 方法 | predictor p50/p95/p99 | planner p50/p95/p99 | QDR/tube p50/p95/p99 | safety p50/p95/p99 | total p50/p95/p99 |
|---|---:|---:|---:|---:|---:|
| Strong delayed-MPC | 10.847 / 14.789 / 17.374 | 5.462 / 7.699 / 9.506 | 0 / 0 / 0 | 0.834 / 1.419 / 1.766 | 18.505 / 24.168 / 27.687 |
| Immutable QDR + token | 10.025 / 15.005 / 24.104 | 16.215 / 23.112 / 30.698 | 1.174 / 2.046 / 3.074 | 0.789 / 1.362 / 1.893 | 31.264 / 42.647 / 61.948 |
| Bounded replace + token | 10.077 / 14.565 / 17.555 | 16.265 / 22.761 / 27.278 | 1.113 / 1.811 / 2.479 | 0.798 / 1.305 / 1.718 | 31.443 / 41.430 / 48.113 |
| Fixed-K=8 immutable QDR + token | 10.412 / 14.545 / 17.277 | 49.121 / 61.191 / 69.447 | 1.394 / 2.101 / 2.890 | 0.795 / 1.215 / 1.616 | 64.591 / 80.739 / 91.569 |

100 ms 仍是描述性参考而非硬门槛；同时保留 p99，不用 p50 单独宣称实时部署。

## 6. Gate decision

- QueueToken/ACK freshness：通过；三种子 valid request accepted rate 为 100%；
- immutable QDR 相对 strong delayed-MPC 的 ID safe-capture 非劣方向：通过；
- timeout `<=5%`：immutable 和 bounded replacement 均失败；
- timeout delta `<=5 pp`：两种 QDR arm 均失败；
- maximum exhaustion streak `<=24`：immutable 113、bounded replacement 192、
  fixed-K=8 78，均失败；
- promotion：**No-Go**；
- confirmation / locked-test：不开启。

结论是：QueueToken/ACK 已经成为可复现的执行队列一致性机制，但当前
`replace_nonexecuting` 没有解决 QDR liveness，反而引入更高 timeout 和更长
exhaustion streak。后续应冻结该 recovery authority 负结果，优先修正 suffix
可恢复性或重新设计 planner 的可行性判定，而不是继续扫描 authority 参数。

local CBF 仍只称为经验过滤器；robust CBF-QP/R-CLBF-QP 仍为 diagnostic
No-Go，不宣称安全证明。

## 7. 可复现工件

- 计划与配置：`configs/phase63_queue_token_calibration.yaml`；
- split 提取工具：`scripts/extract_scene_split.py`；
- 聚合工具：`scripts/aggregate_phase63_queue_token_calibration.py`；
- 聚合报告：`results/phase63_queue_token_calibration_aggregate.md`；
- 三组运行目录：
  `results/phase63_strong_delayed_seed*`、
  `results/phase63_qdr_token_immutable_seed*`、
  `results/phase63_qdr_token_replace_seed*`、
  `results/phase63_qdr_fixed_k8_seed*`；
- `results/` 依照仓库约定被 Git 忽略，但本地结果、配置快照和 TensorBoard
  event 保留。
