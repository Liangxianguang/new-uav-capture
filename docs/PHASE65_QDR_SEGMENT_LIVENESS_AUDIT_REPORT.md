# Phase 65：QDR prefix/suffix/terminal liveness audit

## 1. 阶段定位与结论

本阶段在全新 tail-stress calibration manifest 上，把 QDR 闭环拆成
`prefix admissibility → suffix admissibility → terminal public-candidate progress`
三段进行诊断。审计只使用队列对齐后的 public belief、选定动作序列和预测候选
轨迹，不读取 simulator target，也不改变 immutable queue、planner 选择或 local
CBF。它是 failure taxonomy 工具，不是安全证明。

在更强延迟/噪声矩阵上，immutable QDR 将 safe capture 从 strong delayed-MPC 的
`36.39%` 提高到 `70.28%`，collision 从 `63.61%` 降到 `0.83%`，但 timeout
上升到 `28.89%`。paired safe-capture delta 为
`+33.89 pp [+28.06,+40.00]`，timeout delta 为
`+28.89 pp [+22.50,+35.28]`。QDR 最大 exhaustion streak 为 `153` 步，
说明极端尾部条件下的主要瓶颈仍是队列后的可恢复性和持续进度，而不是碰撞过滤
本身。本阶段不设 promotion，继续保持 confirmation/locked-test 封存。

## 2. 新场景与协议

- canonical manifest：`360 episodes / 180 mirror groups`；
- block：`id_reference`、`execution_tail`、`communication_tail`、
  `joint_tail_transfer`，每个 90 episodes / 45 mirror groups；
- development calibration：`120 episodes / 60 mirror groups`；
- 三个 Diagonal-SSM predictor seeds：`727201`、`727202`、`727203`；
- manifest SHA-256：
  `9c7bef956d6a19bfba39e434aadd684246c73359c867355c3d1e31e54fd2ddf5`；
- calibration split SHA-256：
  `ec717ad6be69b0b53f63f1478e426f407606809324fe5b78d8763818d18beffb`；
- 新增压力范围包括 action delay `12`、command noise `0.20`、message delay
  `8`、dropout `0.40`、tracking `0.30` 和 drag `0.10`；
- 所有方法使用相同场景、checkpoint seed、每步 prediction refresh、Torch
  `1/1` 线程和 local CBF；bootstrap 使用 `10,000` 次、匹配
  `predictor seed + mirror_group`；
- locked diagnostic 未读取、未调参。

## 3. 闭环结果

区间为分层 bootstrap 95% CI；百分比四舍五入到两位小数。

| 方法 | Safe capture | Collision | Boundary | Timeout | Capture time (s) | Min clearance (m) | Total p50/p95/p99 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strong current-state delayed-MPC | 36.39% [30.00,42.78] | 63.61% [56.94,70.00] | 8.33% [5.28,11.67] | 0.00% | 3.444 | 0.078 | 16.972 / 22.306 / 25.456 |
| Immutable QDR | 70.28% [63.89,76.39] | 0.83% [0.00,2.22] | 0.83% [0.00,2.22] | 28.89% [22.50,35.28] | 8.348 | 0.428 | 30.836 / 40.115 / 45.944 |

安全轴明显改善，但 timeout 代价很大；因此不能将该结果表述为完整端到端性能
提升。

## 4. 五类延迟分位数

单位为 ms，顺序均为 p50/p95/p99。QDR/tube 对 strong baseline 为零，因为
该方法没有 QDR/tube 阶段。

| 方法 | predictor | planner | QDR/tube | safety | total |
|---|---:|---:|---:|---:|---:|
| Strong current-state delayed-MPC | 9.927 / 13.720 / 16.318 | 5.064 / 6.744 / 8.770 | 0 / 0 / 0 | 0.780 / 1.250 / 1.613 | 16.972 / 22.306 / 25.456 |
| Immutable QDR | 9.970 / 14.036 / 16.713 | 15.243 / 21.197 / 24.882 | 1.441 / 2.096 / 2.821 | 0.781 / 1.198 / 1.637 | 30.836 / 40.115 / 45.944 |

100 ms 不是硬门槛；报告保留 p99 以展示长尾。

## 5. Segment 诊断

以下 terminal 指标基于 public candidate set；本阶段 K=1，因此 any/all candidate
capture 数值相同，这不是多模态鲁棒保证。

| block | finite progress | terminal capture | best progress (m) | best terminal distance (m) | max exhaustion streak |
|---|---:|---:|---:|---:|---:|
| id_reference | 80.37% | 30.84% | 0.327 | 0.245 | 16 |
| execution_tail | 67.40% | 26.36% | 0.077 | 0.309 | 85 |
| communication_tail | 91.52% | 38.87% | 0.480 | 0.201 | 16 |
| joint_tail_transfer | 38.88% | 12.40% | -0.544 | 0.530 | 153 |

`joint_tail_transfer` 同时表现出最低 finite-progress rate、负的平均 best progress
和最长 exhaustion streak，与 timeout 上升方向一致；这支持下一步优先修复
suffix recoverability，而不是继续增加安全惩罚权重。完整 status counts 保存在
聚合 JSON 和 TensorBoard summary text 中。

## 6. 阶段判定与下一步

- 三段式审计实现与 JSONL/TensorBoard 记录：通过；
- 新场景数量与 mirror-disjoint split 审计：通过；
- immutable QDR 的安全轴：在该压力矩阵上保持优势；
- QDR liveness：最大 streak `153`，仍未通过 `24` 步目标；
- promotion / confirmation：**No-Go**，不访问 locked-test。

下一阶段固定为：

1. 使用 segment status 对 timeout episode 做逐步归因，区分 prefix infeasible、
   suffix infeasible 和 terminal progress 缺失；
2. 设计有限恢复步数上界的候选生成/筛选，不把软回退当作安全证书；
3. 新增至少 300 个场景，继续保持一次只改变一个压力因素的 block 结构；
4. 只有 timeout、最大 exhaustion streak、safe capture 和延迟尾部同时满足预注册
   门槛后，才允许 confirmation；
5. `local CBF` 继续标注为经验过滤器，robust CBF-QP/R-CLBF-QP 继续保持
   diagnostic No-Go。

## 7. 可复现工件

- 场景生成：`scripts/generate_phase65_tail_stress_scenes.py`；
- 配置：`configs/phase65_qdr_segment_liveness_audit.yaml`；
- 聚合：`scripts/aggregate_phase65_qdr_segment_liveness.py`；
- 聚合结果：`results/phase65_qdr_segment_liveness_aggregate.json` 及同名 `.md`；
- 三个 seed 的运行目录均保留 episode/step JSONL、effective config 和
  `tensorboard/` event files。

正式闭环使用的是与本阶段 operational settings 等价的 Phase 64 repair 配置；本阶段新增配置
已将 segment-audit tolerance 固定为 `1e-9`，用于后续复现实验和 smoke 校验。正式结果仍以
各 run 目录中的 effective config snapshot 和本报告列出的 calibration manifest hash 为准。
