# Phase 54：UAKR fresh development 与局部可靠性审计

## 1. 阶段结论

本阶段在全新的 `validation_development` 场景块上，比较固定 `K=8` 与冻结的
原始 Uncertainty-Triggered Adaptive K and Replanning（UAKR）。结果确认：UAKR
确实减少了扩散候选预算和刷新次数，但当前公开 uncertainty/residual 信号不能
可靠地区分下一步安全违例；在相同 QDR、DN-MPC、执行延迟和 local CBF 合同下，
safe capture 下降、timeout 增加。因此原始 UAKR 在本阶段判定 **No-Go for
closed-loop promotion**，冻结为可复现的 efficiency/negative ablation。

该结论不访问 locked-test，不用于调阈值，也不构成安全证明。100 ms 不是硬门槛，
仍完整报告 predictor、planner、QDR、安全层和 total 的 p50/p95/p99。

## 2. 冻结协议

| 项目 | 设置 |
| --- | --- |
| 场景 | 100 episodes、50 mirror groups、50 upper + 50 lower |
| 场景 split | `validation_development`；评估决策标签为 `validation_selection` |
| 场景 seed-start | `851101` |
| predictor seeds | `727201/727202/727203` |
| manifest SHA-256 | `10521ad12ee30071b7e946c7e83889bd348b0d1e9fbd357bc48e1c4bb9c5764b` |
| target speed | `0.65/0.75` |
| QDR | enabled；immutable queue；delay `4` steps |
| execution | command noise `0.08 m/s`，`3σ` clipping，tracking/drag 为 0 |
| predictor | GRU `both` checkpoint；`K` 上限 8；8 diffusion steps；4 projection iterations |
| planner | distributed delayed DN-MPC，worst-case，3 consensus iterations |
| safety | local CBF；robust CBF-QP 未接入闭环主结果 |
| 统计 | 3 个训练 seed，episode 配对 bootstrap 95% CI，bootstrap samples `10000` |

固定参考和 UAKR 使用完全相同的场景、checkpoint、采样 seed、planner、执行合同
和 safety layer；唯一受试因素是是否启用 adaptive K/refresh policy。

## 3. 闭环结果

| 指标 | fixed K=8 | original UAKR | UAKR − fixed |
| --- | ---: | ---: | ---: |
| safe capture | 98.33% [95.67, 100.00] | 95.00% [92.00, 97.67] | −3.33 pp [−7.67, +1.00] |
| collision | 0.67% [0.00, 2.33] | 0.67% [0.00, 2.00] | 0.00 pp [−2.33, +2.00] |
| boundary violation | 0.67% [0.00, 2.33] | 0.67% [0.00, 2.00] | 0.00 pp [−2.33, +2.00] |
| timeout | 1.00% [0.00, 2.67] | 4.33% [2.00, 7.33] | +3.33 pp [+0.33, +7.00] |
| mean capture time | 5.566 s [4.526, 6.474] | 8.915 s [8.182, 9.653] | +3.439 s [+2.587, +4.295] |
| mean minimum clearance | 0.523 m [0.503, 0.541] | 0.470 m [0.458, 0.482] | −0.052 m [−0.073, −0.034] |

UAKR 的 paired safe-capture CI 下界为 `−7.67 pp`，低于预注册的 `−2 pp`
非劣界；timeout 增量的 CI 也不跨 0。三个 predictor seed 的预算下降方向
一致，但闭环安全和活性没有保持。

## 4. 预算与缓存行为

三种子平均值如下：

| 预算指标 | fixed K=8 | original UAKR |
| --- | ---: | ---: |
| mean selected K | 8.000 | 2.145 |
| prediction refresh rate | 100.00% | 30.98% |
| mean refresh interval | 1.000 step | 3.240 steps |
| forced refresh rate | 0% | 20.55% |
| mean prediction/cache age | 0.000 step | 1.264 steps |
| suffix admissible rate | 84.11% | 78.43% |

因此，UAKR 的计算节省不是伪影：平均候选数约下降 `73.2%`，预测刷新率约
下降 `69.0%`。但是缓存年龄上升，suffix admissibility 下降约 `5.68 pp`，
与 timeout、捕获时间和 clearance 的退化方向一致；不能只报告预算节省。

## 5. 延迟分解

以下为三种子 pooled step-level latency（ms）：

| 模块 | fixed K=8 p50/p95/p99 | original UAKR p50/p95/p99 |
| --- | --- | --- |
| predictor | 5.06 / 6.45 / 7.56 | 0.00 / 5.69 / 6.68 |
| planner | 11.55 / 15.14 / 17.56 | 11.28 / 14.48 / 16.06 |
| QDR | 约 0.59 / 0.77 / 0.98 | 约 0.54 / 0.73 / 0.90 |
| safety | 0.58 / 0.78 / 0.98 | 0.57 / 0.75 / 0.94 |
| total control | 19.88 / 24.84 / 29.04 | 15.79 / 22.51 / 24.94 |

QDR 数值为三种子 summary percentiles 的算术均值；总延迟使用 pooled samples，
不应把各列简单相加。UAKR 的 pooled predictor p50 为 0 是因为缓存命中时预测
调用不执行，不能解释为真实 predictor 推理耗时为零；p95/p99 和 total latency
必须与 cache age、refresh rate 一起阅读。

## 6. uncertainty 是否真的能触发正确干预

对 UAKR 三种子日志按 canonical mirror group 划分为 25 个 calibration groups
和 25 个 confirmation groups，审计标签为公开几何导致的 current/next-state
violation。没有使用 target truth 或终局结果作为在线输入。

| split | score | next violation AUROC | current violation AUROC |
| --- | --- | ---: | ---: |
| calibration | uncertainty | 0.495 | 0.497 |
| calibration | prediction residual | 0.534 | 0.521 |
| confirmation | uncertainty | 0.520 | 0.510 |
| confirmation | prediction residual | 0.513 | 0.509 |

这些值接近随机排序，说明当前 uncertainty 与 residual 虽然可记录、可解释，
但还不是可用于安全预算分配的 intervention-effect 信号。该审计结果解释了
为什么继续做未经校准的阈值网格没有价值。

## 7. 可复现产物与审计

- 场景生成：`scripts/generate_phase48_qdr_liveness_scenes.py`；Phase 54 元数据
  和配置：`configs/phase54_uakr_development.yaml`；
- 执行配置：`configs/phase50_qdr_uakr_development.yaml`；
- 聚合结果：`results/phase54_uakr_development_aggregate.json`；
- 局部可靠性审计：`results/phase54_uakr_local_reliability.json`；
- 每个 seed 均保留 `config.yaml`、`summary.json`、episode/step JSONL 和
  TensorBoard event；可靠性审计 event 位于
  `results/phase54_uakr_local_reliability_tensorboard`；
- 三种子闭环 TensorBoard 位于各 run 的
  `distributed_delayed/tensorboard`；
- source/config/manifest hash 已写入每个 run 的 `config.yaml`。

## 8. 决策与下一步

- [x] 完成 fixed-K=8 与 original UAKR 的三种子 fresh development 配对；
- [x] 完成 paired bootstrap、预算、缓存年龄、suffix admissibility 和五级延迟统计；
- [x] 完成 calibration/confirmation mirror-group reliability audit；
- [x] 判定 original UAKR 未通过 `−2 pp` safe-capture 非劣和 timeout gate；
- [x] 冻结当前 UAKR 结果为 efficiency/negative ablation，不访问 locked-test；
- [ ] 不扫描当前阈值，不在该结果上重新选择权重；
- [ ] 若继续研究 UAKR，新建独立 development block，先加入公开的 queue/QDR
  prefix feasibility 或 recoverable-margin 特征，再做 intervention-effect
  calibration；
- [ ] 如果新信号在 calibration 与 confirmation 仍接近随机，则停止 UAKR 主线，
  将论文主线收敛为 QDR 的延迟状态对齐与可复现实验边界，并把 UAKR 作为负消融；
- [ ] 只有新策略通过独立 confirmation 的安全、活性和预算 gate，才允许一次性
  访问 locked-test 或重新开放 QDR×UAKR 组合。

完整原始结果保留在本地 `results/`，不提交到 Git；源码、配置和本报告提交到
`new-uav-capture` 仓库。
