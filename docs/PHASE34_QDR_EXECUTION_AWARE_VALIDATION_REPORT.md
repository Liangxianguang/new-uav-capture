# Phase 34：执行感知 QDR 开发验证报告

> 日期：2026-09-13
> 代码提交：`8e219d1`、`33d1f6e`
> 仓库：[Liangxianguang/new-uav-capture](https://github.com/Liangxianguang/new-uav-capture)

## 1. 目的与实验边界

本阶段验证 Queue-Aware Delayed-State Rollout（QDR）在真实命令队列延迟下的
实现是否与执行动力学一致。实验只使用 validation-development 场景，不读取
locked-test，也没有用本结果调节 predictor、MPC 权重或安全阈值。

本阶段还修复了一个实现错误：QDR 分支中的单个队友轨迹为 `[H,1,3]`，普通分支
为 `[H,3]`，导致分布式局部代价计算错误广播并在每步 fallback。修复后增加了
执行感知完整动作序列 rollout、QDR 前视障碍范围以及 planner fallback 原因日志。

## 2. 可复现设置

| 项目 | 设置 |
| --- | --- |
| 场景 | `results/phase34_qdr_development_scenes_v2/scenes.jsonl` |
| 场景规模 | 40 episodes / 20 mirror groups |
| 场景 split | `validation_development`，`locked_test_included=false` |
| 场景 SHA-256 | `4d99d829157145a6f747a4afde6dcbe7c0846e2bc44c5c0a5de5dbe65bd600d3` |
| predictor | 固定 GRU `both` checkpoint，seed `727201` |
| planner | distributed delayed DN-MPC，worst-case，horizon 8，control horizon 3 |
| safety | local CBF |
| execution | enabled，command delay 2，immutable authority，noise 0，tracking/drag 0 |
| sampling | `num_samples=1`，`sampling_steps=8`，`sampling_seed=745102` |
| 评测决策 | `validation_selection` |

所有运行目录都保留 `config.yaml`、source hashes、episode/step JSONL、summary 和
TensorBoard event files。

## 3. 同版本结果

### 3.1 主对照：QDR-off 与 QDR-on

| 指标 | QDR-off | QDR-on，immutable，当前状态安全投影 | 差异 |
| --- | ---: | ---: | ---: |
| safe capture | 100.0% | 85.0% | -15.0 pp |
| ordinary capture | 100.0% | 87.5% | -12.5 pp |
| collision | 0.0% | 15.0% | +15.0 pp |
| boundary violation | 0.0% | 0.0% | 0 pp |
| timeout | 0.0% | 0.0% | 0 pp |
| mean capture time | 1.963 s | 2.121 s | +0.159 s |
| mean minimum clearance | 0.446 m | 0.398 m | -0.047 m |
| worst minimum clearance | 0.222 m | -0.133 m | degraded |
| QDR prefix admissible rate | N/A | 95.93% | — |
| QDR suffix admissible rate | N/A | 44.63% | — |
| queue length / first controllable step | 0 / 0 | 2 / 2 | — |

时延统计为 predictor / planner / QDR / safety / total 的
`p50 / p95 / p99`（ms）：

| 运行 | predictor | planner | QDR | safety | total |
| --- | --- | --- | --- | --- | --- |
| QDR-off | 11.03 / 22.04 / 23.80 | 12.26 / 22.51 / 25.27 | 0 / 0 / 0 | 1.16 / 2.52 / 2.85 | 27.84 / 46.53 / 51.59 |
| QDR-on | 10.66 / 21.79 / 24.33 | 68.55 / 95.72 / 110.12 | 0.76 / 1.43 / 1.73 | 1.16 / 2.61 / 2.96 | 87.83 / 118.60 / 135.02 |

因此当前 QDR-on 不满足预注册的 ID safe-capture 非劣门槛，也不满足碰撞不增加
和 total p95 增幅不超过 15% 的工程门槛。100 ms 不是硬门槛，但 QDR-on 的总 p95
和 planner p95 均已超过该参考值，必须在论文中如实报告。

### 3.2 safety projection 消融

在相同 40 集开发块上，QDR-on 的 queue-aware safety projection 将安全捕获率从
此前受 planner shape bug 影响的 55.0% 重新验证为当前实现的 85.0%，并相对
QDR-on、非 queue-aware safety projection 的 82.5% 有 +2.5 pp 描述性改善；但
碰撞仍为 15.0%，不能作为通过证据。该消融说明安全层锚点确实存在时序影响，但
不能替代对不可变队列 prefix 的可行性处理。

### 3.3 QDR precondition 诊断

当前 QDR-on（queue-aware safety projection）step-level 诊断为：

- `prefix_safe_suffix_safe`：388 steps；
- `prefix_safe_suffix_unsafe`：449 steps；
- `prefix_unsafe_unrecoverable`：51 steps；
- 未出现 `prefix_unsafe_recoverable`，因为本实验 authority 是 `immutable`；
- planner status 全部为 `success`，fallback 原因计数为 0。

这表明失败已经从“代码广播错误”转化为可观测的控制合同问题：大量候选 suffix
在 nominal geometry 下仍不可行，而不可变 queue 中已经提交的 prefix 一旦不安全，
新规划动作不能事后修复。当前 precondition audit 是诊断器，不是安全证明。

## 4. TensorBoard 与日志审计

以下运行均检测到 TensorBoard event files，并包含配置、源码 hash 和四级时延标签：

```text
results/phase34_qdr_dev_v6_off_delay2_current_seed727201/distributed_delayed/tensorboard/
results/phase34_qdr_dev_v5_on_delay2_immutable_qsafety_seed727201/distributed_delayed/tensorboard/
```

QDR-on event file 包含 `Summary/QDR/*`、`Summary/PlannerLatency/*`、
`Summary/PredictorLatency/*`、`Summary/SafetyLatency/*` 和
`Summary/TotalControlLatency/*`；QDR-off 也保留同名字段，便于同一仪表板比较。

## 5. 阶段判定

**实现修复：通过。** 全量测试 `264 passed`；QDR 分布式 planner 不再因队友轨迹
维度错误而 fallback，执行感知 rollout 与日志可复现。

**性能晋级：No-Go。** 在当前开发块上，QDR-on 的 safe capture 低于 QDR-off，
碰撞增加，suffix 可行率不足，且 planner/total p95 明显增加。不得进入 fresh
confirmation 或 locked-test，也不能把 QDR 当前结果写成延迟鲁棒性提升。

## 6. 后续实验计划

1. 在新的 development-calibration 场景上实现“suffix feasibility gate”：候选
   生成或分布式 best-response 阶段直接拒绝 nominal suffix 不可行的序列，不能
   只在 plan 结束后审计；同时记录 gate 是否耗尽和 fallback 原因。
2. 将 gate 与 queue-aware safety projection 分成正交消融，保持 command authority
   为 immutable；若需要 `replace_nonexecuting` 或 `flush_pending`，必须单独作为
   执行器能力实验，不能与 QDR 主结果混写。
3. 在不调 locked-test 的前提下，按单因素运行 delay `{0,1,2,4}`、bounded
   execution noise 和 message/observation age；每个块固定 mirror groups、seed、
   checkpoint、sampling seed，并报告 predictor/planner/safety/total p50/p95/p99。
4. 只有当 ID safe capture 的 paired CI 下界达到 `-2 pp`、collision/boundary
   不增加、且至少一个 delay/execution OOD 轴有改善时，才做三 seed confirmation。
5. QDR 通过之前不开放 QDR×UAKR×RNIC Full 组合，也不重新接入 robust CBF-QP；
   robust CBF-QP 仍是独立诊断 No-Go，不构成闭环安全证明。
