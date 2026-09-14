# Phase 64：QDR exhausted-candidate liveness repair 校准报告

## 1. 阶段结论

Phase 64 在全新开发校准集上测试了一个非常窄的 planner-side 修复：当
QDR 候选全部不可行时，不再只按硬的最小违反度排序，而是使用归一化的
“基础任务/安全代价 + QDR 违反度”软进度排序。该分支只改变 exhausted-candidate
的选择，不改变队列 authority、时间索引、预测器、场景、local CBF 或执行合同。

结果是：软进度策略相对 immutable hard-QDR 将 safe capture 从 `84.17%` 提高到
`86.94%`，collision/boundary 从 `1.11%/1.11%` 降到 `0.28%/0.28%`，timeout
从 `14.72%` 降到 `12.78%`。但是 paired safe-capture CI 为
`+2.78 pp [-2.50,+8.89]`，timeout CI 为 `-1.94 pp [-8.06,+3.33]`，均未形成
统计确认；更重要的是最大 QDR exhaustion streak 从 `93` 增加到 `198`，远超
预注册的 `24` 步 liveness 门槛。因此本阶段判定为 **No-Go / 冻结为诊断修复**，
不进入 confirmation 或 locked-test。

该结果支持一个明确诊断：软回退可以在当前校准矩阵中恢复部分安全捕获机会，但它
没有证明 suffix 可恢复，也没有解决 QDR 长时间耗尽。后续应优先重新定义可恢复
suffix 的可行性判定或设计带有限进度证书的候选生成器，而不是继续扫描软权重。

## 2. 数据、协议与可复现性

- 新建 canonical manifest：`360 episodes / 180 mirror groups`；
- 本阶段只使用 `development_calibration`：`120 episodes / 60 mirror groups`，
  每个 predictor seed 运行 120 episodes；
- 三个 Diagonal-SSM predictor seeds：`727201`、`727202`、`727203`；
- 四个场景 block：`id_reference`、`delay_noise_factorial`、
  `communication_execution_factorial`、`joint_stress_transfer`；
- calibration split SHA-256：
  `ff4f5c1fbe891322ee5cec99ca6d3c6a5749b93a2f8a09d3eb7004f2f31a7dab`；
- 所有方法使用相同场景、相同 checkpoint seed、`local_cbf`、每步预测刷新和
  Torch `1/1` 线程；
- 所有运行均保存 effective config、episode/step JSONL 与 TensorBoard event；
- bootstrap 单位为匹配的 predictor training seed 与 `mirror_group`，样本数
  `10,000`，seed `20260914`；
- 本报告不读取、不调参 `locked_diagnostic` split。

## 3. 闭环结果

区间为分层 bootstrap 95% CI；百分比四舍五入到两位小数。

| 方法 | Safe capture | Collision | Boundary | Timeout | Capture time (s) | Min clearance (m) | Total p50/p95/p99 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strong current-state delayed-MPC | 61.11% [54.17, 67.78] | 38.89% | 5.83% | 0.00% | 2.291 | 0.194 | 17.231 / 22.791 / 26.089 |
| Immutable hard-QDR | 84.17% [78.89, 89.17] | 1.11% | 1.11% | 14.72% | 7.536 | 0.488 | 31.175 / 41.322 / 47.602 |
| Normalized soft-progress QDR | 86.94% [80.56, 92.22] | 0.28% | 0.28% | 12.78% | 7.467 | 0.486 | 30.348 / 39.679 / 45.514 |

相对 immutable hard-QDR：

- safe-capture paired delta：`+2.78 pp [-2.50,+8.89]`；
- collision paired delta：`-0.83 pp [-1.94,0.00]`；
- timeout paired delta：`-1.94 pp [-8.06,+3.33]`；
- capture-time paired delta：`-0.169 s [-0.958,+0.611]`。

因此不能把软进度写成已经显著提升捕获率的主方法；更准确的表述是“安全轴方向
有改善迹象，但 liveness gate 未通过”。

## 4. 完整延迟分位数

单位为 ms。`QDR/tube` 对 strong baseline 为零，因为该方法没有 QDR/tube 阶段。

| 方法 | predictor p50/p95/p99 | planner p50/p95/p99 | QDR/tube p50/p95/p99 | safety p50/p95/p99 | total p50/p95/p99 |
|---|---:|---:|---:|---:|---:|
| Strong current-state delayed-MPC | 10.052 / 14.079 / 16.420 | 5.146 / 7.006 / 8.848 | 0 / 0 / 0 | 0.788 / 1.284 / 1.623 | 17.231 / 22.791 / 26.089 |
| Immutable hard-QDR | 10.041 / 14.526 / 17.201 | 16.045 / 22.568 / 26.501 | 1.240 / 1.941 / 2.668 | 0.794 / 1.278 / 1.702 | 31.175 / 41.322 / 47.602 |
| Normalized soft-progress QDR | 9.776 / 13.699 / 16.492 | 15.658 / 21.781 / 25.417 | 1.179 / 1.766 / 2.434 | 0.778 / 1.175 / 1.605 | 30.348 / 39.679 / 45.514 |

100 ms 仍是描述性参考，不是硬门槛；论文和仓库报告同时保留 p50/p95/p99，避免
用平均值或 p50 掩盖长尾。

## 5. QDR exhaustion 诊断

| 方法 | exhaustion policy | 平均 soft fallback/episode | soft fallback 总数 | 最大 exhaustion streak |
|---|---|---:|---:|---:|
| Strong delayed-MPC | 不适用 | 0.00 | 0 | 0 |
| Immutable hard-QDR | `hard_min_violation` | 0.00 | 0 | 93 |
| Normalized soft-progress QDR | `normalized_soft_progress` | 71.03 | 25,572 | 198 |

软策略只在“所有候选均 exhausted”的分支触发，并且不修改 immutable queue，
也不构成安全证书。`198` 步的最大 streak 直接否定了当前 liveness 门槛；因此
该策略不能作为 Queue-Aware Reachable-Tube Robust CBF-QP 的完成证据。

## 6. Gate 与论文边界

预注册门槛为：相对 hard-QDR 的 safe-capture paired lower bound 不低于 `-2 pp`、
timeout paired upper bound 不高于 `+5 pp`、最大 exhaustion streak 不超过 `24`
步。前两个区间在本次校准中没有被统计性否定，但第三项明确失败，因此总体
**No-Go**。

- 不开放 confirmation 或 locked-test；
- 不把 soft fallback 称为 formal reachability guarantee；
- `local CBF` 仅是经验安全过滤器，不能称为 R-CLBF-QP 安全证明；
- robust CBF-QP / R-CLBF-QP 仍为 diagnostic No-Go；
- 不宣称已达到部署级实时性，只报告上述分项和总延迟尾部。

## 7. 下一步 TodoList

1. **冻结当前失败证据**：保留 hard/soft 两套 event、JSONL、config snapshot 和
   本报告，不再在本 calibration 或 locked-test 上扫描 soft 权重。
2. **重做 suffix feasibility**：把候选评估拆成 prefix admissibility、suffix
   recoverability 和 terminal capture 三个可审计阶段；在 planner 内先拒绝没有
   有限步恢复上界的候选，再做任务代价排序。
3. **加入有限进度诊断**：记录每一步的 best reachable distance、remaining
   queue depth、可行候选数和预计恢复步数；先做 20--40 mirror groups 的 smoke，
   只检查诊断是否与 timeout/streak 对齐。
4. **扩展新 calibration block**：至少新增 300 episodes，单独改变通信延迟、执行
   噪声、目标机动强度和 obstacle geometry，每个 block 保留 mirror-disjoint split。
5. **预注册 promotion gate**：在新 calibration 上验证 safe capture、collision、
   boundary、timeout、最大 streak 与五类 p50/p95/p99；只有 liveness 通过后才允许
   confirmation，locked-test 继续封存。
6. **维持 TensorBoard 审计**：每个方法必须记录 effective YAML、source hash、
   policy、candidate K、queue age、exhaustion streak 和 predictor/planner/QDR/
   safety/total scalars；缺任一项则实验无效。

## 8. 工件

- 配置：`configs/phase64_qdr_liveness_repair.yaml`；
- 聚合脚本：`scripts/aggregate_phase64_qdr_liveness_repair.py`；
- 聚合结果：`results/phase64_qdr_liveness_repair_aggregate.json` 和同名 `.md`；
- 三个运行目录：`results/phase64_qdr_liveness_repair_seed727201`、
  `...seed727202`、`...seed727203`；
- TensorBoard event 位于每个方法目录下的 `tensorboard/`。
