# Phase 55：Queue-Prefix Risk Augmented UAKR 实验报告

## 1. 结论

Phase 55 在新的 `validation_selection` 开发集上完成了三种子、100
episodes、50 个 mirror groups 的配对实验。Queue-prefix risk 已正确进入
UAKR 的加性触发分数，并且其离线 next-state violation 排序在 calibration /
confirmation 上分别达到 AUROC `0.706/0.604`。但是闭环安全捕获率相对固定
`K=8` 下降 `7.00 pp`，timeout 增加 `4.67 pp`，没有通过预注册的安全/活性
非劣 gate。因此该候选判定为 **No-Go**，仅保留为可复现的诊断/负消融；不访问
locked-test，也不开放 QDR×UAKR×RNIC 完整组合。

该结果不表示 queue-prefix risk 没有信息：它能够排序一部分即将发生的局部
违例，但当前“风险分数 → 增大 K/刷新频率”的干预策略没有把排序优势转化为
闭环收益。尤其是 current-state confirmation AUROC 只有 `0.559`，不能把该
特征写成安全触发器或安全证书。

## 2. 冻结实验协议

- 场景：独立 Phase 55 manifest，100 episodes / 50 mirror groups，50 upper /
  50 lower；manifest SHA-256：
  `f05ed4fbdf4e8c2f9bdda0b96ed6f5f73cab561b367848b45ea7735ffa0680d7`；
- 预测器：GRU `both`，三个 predictor seeds：`727201/727202/727203`；
- 控制：distributed delayed DN-MPC、nominal QDR、immutable delay `4`、
  bounded command noise `0.08`、local CBF；
- fixed-K reference：`K=8`，每步刷新；
- original UAKR：冻结 Phase 54 的阈值、预算和 cache 权重；
- queue-risk UAKR：在原始 UAKR 分数上加性加入
  `weight=0.30`、`buffer=0.15 m`、`scale=0.50 m`；风险为 0 时严格退化为
  原始 UAKR；
- 所有运行使用 CPU Torch threads `1/1`。阈值和权重没有在 locked-test 上
  选择或调整。

风险特征只读取 immutable queue prefix 的公开几何诊断：最小 clearance 和
safety-margin violation；不读取 target truth、终局标签或未来 episode
outcome，也没有 command authority。

## 3. 闭环结果

括号为 bootstrap 95% CI；safe capture 是目标捕获且没有 collision、boundary
或其他安全失败污染的主指标。

| 方法 | Safe capture | Collision | Boundary | Timeout | Mean capture time | Mean min clearance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-K=8 | `98.33% [96.33,99.67]` | `1.00%` | `1.00%` | `0.67%` | `5.967 s` | `0.528 m` |
| original UAKR | `90.67% [87.00,94.00]` | `3.33%` | `3.33%` | `6.00%` | `8.624 s` | `0.484 m` |
| queue-risk UAKR | `91.33% [87.67,94.67]` | `3.33%` | `3.33%` | `5.33%` | `8.869 s` | `0.485 m` |

相对 fixed-K=8 的 paired 结果为：

- original UAKR safe capture `-7.67 pp [-11.67,-3.67]`，timeout
  `+5.33 pp [+1.67,+8.67]`；
- queue-risk UAKR safe capture `-7.00 pp [-11.00,-3.33]`，collision/boundary
  `+2.33 pp [0.00,+4.67]`，timeout `+4.67 pp [+1.67,+8.33]`；
- queue-risk 相对 original UAKR 只有描述性的 `+0.67 pp` safe-capture 差异，
  未做 promotion-level 显著性解释。

因此 queue risk 相比原始 UAKR 有轻微改善，但没有接近固定预算参考，不能
称为成功的 UAKR 修复。

## 4. 自适应行为和可靠性

| 方法 | Mean K | Refresh rate | Mean cache age | Mean queue risk |
| --- | ---: | ---: | ---: | ---: |
| fixed-K=8 | `8.000` | `100.00%` | `0.000` | `0.000` |
| original UAKR | `2.106` | `30.62%` | `1.280` | `0.000` |
| queue-risk UAKR | `2.125` | `30.75%` | `1.275` | `0.0126` |

queue risk 的平均值很低，且只使 mean K 增加约 `0.019`、refresh rate 增加约
`0.13 pp`；本实验中它没有形成足够强的预算干预。QDR suffix gate 的
per-step exhaustion rate 约为 `20.39%`，exhausted-episode rate 为 `100%`，
也说明“局部 prefix 风险”不能单独代表后缀可恢复性。

离线、按 mirror group 划分的 step-level reliability audit 如下：

| Score | Calibration next violation | Confirmation next violation | Calibration current violation | Confirmation current violation |
| --- | ---: | ---: | ---: | ---: |
| uncertainty | `0.633` | `0.513` | `0.604` | `0.488` |
| prediction residual | `0.472` | `0.502` | `0.472` | `0.498` |
| queue-prefix risk | `0.706` | `0.604` | `0.659` | `0.559` |

AUROC 仅是排序诊断，不是概率校准，也不是安全证明。queue-prefix risk 对
next-state violation 在两个 split 上刚好达到预设 `0.60` 方向，但没有在
current-state confirmation 上稳定转移，且闭环干预仍失败；故不能据此开放
更激进的阈值或预算策略。

## 5. 延迟

以下为三个 seed 的 pooled step latency，单位 ms，顺序为 p50/p95/p99。自适应
方法在未刷新步骤的 predictor 延迟为 0 是缓存语义，不代表模型推理为零；必须
同时查看 refresh rate 和 cache age。

| 方法 | Predictor | Planner | QDR | Safety | Total control |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed-K=8 | `6.29/8.29/10.51` | `14.95/20.09/23.45` | `0.74/1.08/1.46` | `0.73/1.08/1.46` | `25.55/32.32/37.50` |
| original UAKR | `0.00/7.42/9.51` | `14.69/20.16/23.69` | `0.70/1.03/1.32` | `0.73/1.08/1.43` | `20.77/30.13/35.18` |
| queue-risk UAKR | `0.00/7.09/8.66` | `14.34/19.17/22.63` | `0.67/0.97/1.27` | `0.72/1.01/1.41` | `20.71/29.70/33.98` |

100 ms 没有作为硬门槛；这里完整报告五类延迟，以便后续比较缓存收益和尾部
延迟。queue-risk 的较低总延迟主要来自缓存/step 分布，不足以抵消捕获和
liveness 损失。

## 6. 复现工件

- 配置：`configs/phase55_uakr_queue_risk_development.yaml`；
- 代码：`src/encirclement3d/qdr_precondition.py`、
  `src/encirclement3d/adaptive_prediction.py`、
  `scripts/evaluate_minimax_mpc.py`；
- 每步审计：`adaptive_queue_prefix_risk`；
- 聚合：`results/phase55_uakr_queue_risk_aggregate.json`；
- 可靠性：`results/phase55_uakr_queue_risk_reliability.json` 及其 Markdown；
- TensorBoard：`results/phase55_uakr_queue_risk_reliability_tensorboard` 以及
  三个闭环 run 目录中的 event 文件；
- 三个修正版 run：
  `results/phase55_queue_risk_uakr_seed{727201,727202,727203}_threads1_fix1`。

结果目录被 Git 忽略，正式结论、配置、测试和源代码进入仓库；本地结果保留
用于复现和独立重算。

## 7. 后续计划

1. 冻结 queue-prefix risk 的当前参数与 No-Go 结论，不再扫描
   `buffer/scale/weight`，也不访问 locked-test。
2. 保留 fixed-K=8 作为 UAKR 的当前开发参考，original/queue-risk UAKR 作为
   efficiency/negative ablation。
3. 若要重新挽救 UAKR，下一实验必须以“干预后的失败风险”而不是瞬时几何违例
   作为 calibration label，预先固定 intervention quota、K 上限、刷新上限和
   safety/liveness gate，并在新的 mirror-disjoint confirmation 上验证。
4. 在 UAKR 没有通过独立 gate 之前，不运行 Full factorial，不把三个点写成
   已完成的联合安全方法；论文主线应优先采用已通过的 QDR 条件性证据、DN-MPC
   模块化结果和可审计的失败边界。

