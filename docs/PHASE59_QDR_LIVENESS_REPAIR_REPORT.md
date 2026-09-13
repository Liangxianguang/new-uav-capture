# Phase 59：QDR liveness repair calibration report

## 1. 阶段定位

本阶段是独立的 `development_calibration` 修复实验，不是 confirmation，也不
访问 locked-test。目标是回答两个问题：

1. 将 local CBF 评价点放到 first-controllable delayed state，是否能减少 QDR
   repair contract 下的碰撞；
2. K=4/K=8 是否真正执行了多候选 sample set，以及候选预算是否值得其额外开销。

本报告只使用 `results/phase59_repair_calibration_selection/scenes.jsonl` 的
80 episodes / 40 mirror groups、3 个 Diagonal-SSM predictor seeds
`727201/727202/727203`。场景文件 SHA-256 为
`0f227daa36dead54f9b7ebea6f958989e6d8c359a476cabea2654d915831b720`。

local CBF 在全文中只称为**经验动作过滤器**。本阶段没有建立 R-CLBF-QP 或
闭环安全证明。

## 2. 预算真实性审计

此前 GRU repair run 的 R2/R3 配置虽然写入 K=4/K=8，但 GRU 前向只输出一条
均值轨迹，实际 `candidate_count=1`，因此不能作为多候选结果。本阶段改用支持
`sample_set` 的 Diagonal-SSM，并将 requested/realized budget 写入每个 step 和
episode：

| 方法 | 请求 K | 实际 K 范围 | realized rate | 是否可用于 K 消融 |
|---|---:|---:|---:|---|
| R0 QDR baseline | 1 | 1--1 | 100% | 是，K=1参考 |
| R1 queue-aware local CBF | 1 | 1--1 | 100% | 是，K=1参考 |
| R2 queue-aware CBF | 4 | 4--4 | 100% | 是 |
| R3 queue-aware CBF | 8 | 8--8 | 100% | 是 |

因此本报告的 K4/K8 结果具有实现层面的预算真实性；旧 GRU R2/R3 只保留为
无效实现诊断，不与本表混合。

## 3. 闭环结果

总体为 80 episodes / 40 mirror groups，区间为 matched predictor seed +
mirror-group bootstrap 95% CI：

| 方法 | Safe capture | Collision | Boundary | Timeout | Min clearance |
|---|---:|---:|---:|---:|---:|
| R0 QDR baseline, K1 | 76.25% [67.92,83.33] | 9.17% | 4.58% | 14.58% | 0.397 m |
| R1 queue-aware local CBF, K1 | 85.42% [79.17,90.83] | 0.42% | 0.42% | 14.17% | 0.480 m |
| R2 queue-aware local CBF, K4 | 86.67% [80.00,92.92] | 1.25% | 1.25% | 12.08% | 0.478 m |
| R3 queue-aware local CBF, K8 | 88.33% [82.92,92.92] | 1.67% | 1.67% | 10.00% | 0.490 m |

相对于 R0，R1 的总体 paired safe-capture 增量为 `+9.17 pp`
`[+3.75,+15.42]`，collision 增量为 `-8.75 pp`
`[-12.92,-5.00]`，timeout 增量为 `-0.42 pp`
`[-5.00,+4.17]`。这支持“queue-aware local CBF 是有希望的经验修复方向”，
但不是安全保证。

相对于 R1，R2 的 safe-capture 增量为 `+1.25 pp [-5.83,+8.33]`，R3 为
`+2.92 pp [-4.17,+9.58]`；两者区间均跨零。总体 collision 反而分别增加
`+0.83 pp [0,+2.50]` 和 `+1.25 pp [0,+3.33]`，说明多候选不是无条件收益。

## 4. 分场景块结果

| Block | R0 safe/coll/timeout | R1 safe/coll/timeout | R2 safe/coll/timeout | R3 safe/coll/timeout |
|---|---|---|---|---|
| ID reference | 100.00/0.00/0.00% | 100.00/0.00/0.00% | 100.00/0.00/0.00% | 100.00/0.00/0.00% |
| delay-noise | 70.00/16.67/13.33% | 81.67/1.67/16.67% | 76.67/3.33/20.00% | 73.33/6.67/20.00% |
| communication-execution | 98.33/1.67/0.00% | 100.00/0.00/0.00% | 98.33/1.67/0.00% | 100.00/0.00/0.00% |
| joint stress | 36.67/18.33/45.00% | 60.00/0.00/40.00% | 71.67/0.00/28.33% | 80.00/0.00/20.00% |

表中每格依次为 `safe capture / collision / timeout`。joint stress 中 K8 相对
K1 的点估计继续提高 safe capture `20 pp` 并降低 timeout `20 pp`，但 paired
safe-capture CI 为 `[-1.67,+40.00] pp`，尚不足以宣称稳定 superiority。反之在
delay-noise block，K1 比 K4/K8 更好，说明预算应该由风险状态触发或受 liveness
约束，而不是固定增大。

## 5. QDR exhaustion 与 liveness

各 block 的最大 exhaustion streak（step）如下：

| 方法 | ID | delay-noise | communication-execution | joint stress | 全局最大 |
|---|---:|---:|---:|---:|---:|
| R0 | 19 | 22 | 18 | 87 | 87 |
| R1 | 12 | 90 | 11 | 65 | 90 |
| R2 | 12 | 21 | 11 | 15 | 21 |
| R3 | 12 | 31 | 11 | 23 | 31 |

预注册的 liveness 参考是最大 streak `<=24`。因此 R2 在本次 calibration 达到
该指标，但 R1/R3 仍存在长时间 exhaustion，尤其 R1 的 delay-noise 与 R3 的
delay-noise 超过阈值。这说明“降低碰撞”与“保持可持续控制”必须联合评价，不能
只看 safe capture。

## 6. 完整运行时分位数

以下为 pooled step raw samples 的 `p50/p95/p99`，单位 ms；顺序固定为
predictor / planner / QDR / safety / total：

| 方法 | Predictor | Planner | QDR | Safety | Total |
|---|---:|---:|---:|---:|---:|
| R0 | 7.60/13.96/18.73 | 12.21/21.00/27.10 | 1.01/1.97/3.21 | 0.63/1.37/2.10 | 24.06/38.46/47.91 |
| R1 | 7.81/14.67/19.98 | 12.71/21.80/28.26 | 1.01/1.92/3.23 | 0.64/1.43/2.14 | 25.00/40.00/50.31 |
| R2 | 8.00/15.23/20.38 | 19.96/31.92/39.98 | 1.08/2.16/3.42 | 0.64/1.59/2.19 | 32.18/50.45/62.16 |
| R3 | 7.98/14.90/20.05 | 19.29/31.20/39.59 | 1.12/2.12/3.48 | 0.64/1.44/2.14 | 31.90/49.54/60.81 |

100 ms 不是本项目硬门槛；本表用于完整披露尾延迟。K4/K8 的额外 planner/total
开销清晰可见，因此后续 confirmation 必须同时验证其 liveness 收益是否足以
抵偿计算成本。

## 7. 本阶段判定

- **实现审计：通过。** Diagonal-SSM 的 K1/K4/K8 预算均真实实现，聚合脚本按
  block 切片，TensorBoard 和 JSONL 工件均生成。
- **R1 repair：有希望但未晋级。** 总体 collision 显著降低，但 exhaustion
  streak 在 delay-noise 仍失败，且还没有在 confirmation holdout 上验证。
- **K4：候选方向最稳健。** 本次全局最大 streak 为 21，joint stress 的点估计
  优于 K1；但总体 paired CI 跨零，不能称作已证实增益。
- **K8：有 joint-stress liveness 信号但代价更高。** 全局最大 streak 为 31，
  不满足 liveness 参考；不能作为默认预算。
- **安全表述：受限。** local CBF 仅是经验过滤器；本阶段没有 R-CLBF-QP 安全
 证明，也没有 collision-free/forward-invariance 结论。

## 8. 下一步

1. 不访问 locked-test；使用 Phase60 计划中的全新 `>=360 episodes / 180 mirror
   groups` 矩阵，补齐更强 delayed-MPC、fixed/queue-aware tube-MPC 和同步/异步
   分布式基线。
2. 先在新的 development calibration 冻结 K4 candidate contract、liveness
   gate 和 tube 参数，再运行 confirmation。
3. 将 R1 的 queue-aware safety projection 与 K4 的 candidate budget 作为两条
   独立干预，不能把 R2/R3 的联合变化归因于单一模块。
4. confirmation 的首要 gate 为 ID safe-capture non-inferiority、joint collision/
   timeout、最大 exhaustion streak `<=24`、budget realized rate `100%` 和五类
   latency 完整性；任一失败就保留 No-Go 结果，不开放 locked-test。

