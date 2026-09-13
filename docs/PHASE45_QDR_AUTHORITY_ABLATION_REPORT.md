# Phase 45：QDR prefix recovery authority 三种子消融报告

> 日期：2026-09-13
> 实验级别：validation-confirmation，authority safety--liveness 消融
> 仓库：[Liangxianguang/new-uav-capture](https://github.com/Liangxianguang/new-uav-capture)

## 1. 目的与边界

Phase 44 在 delay4 + bounded execution noise 下使用 immutable authority，取得了较
高的安全捕获率，但仍有少量 collision/timeout 和较大延迟。Phase 45 固定同一
manifest、delay4、noise、checkpoint、采样 seed、MPC 和 local CBF，只改变 QDR
prefix recovery authority：

- `immutable`：已执行 queue prefix 不可取消，作为参考；
- `replace_nonexecuting`：仅替换尚未执行的 pending commands；
- `flush_pending`：清空 pending commands 后注入 recovery action。

这不是安全证明实验。后两种 authority 改变了系统执行契约，结果只用于判断安全和
liveness 的 trade-off，不能与 immutable 主结果合并成一个方法。

## 2. 可复现设置

| 项目 | 设置 |
| --- | --- |
| manifest | `results/phase34_qdr_fresh_validation_scenes/scenes.jsonl` |
| 规模 | 3 seeds × 80 episodes = 240 episodes / authority |
| split | `validation_confirmation`，不含 locked-test |
| manifest SHA-256 | `15652f76157157c8823dfd7086a4978f4f5de2aed108b4ebef18817f4cc03646` |
| seeds | `727201/727202/727203` |
| execution | delay 4，command-noise std `0.08 m/s`，bound `3σ`，tracking/drag 0 |
| predictor/planner | GRU `both` / distributed delayed DN-MPC |
| safety | local CBF |
| aggregate | `results/phase45_qdr_authority_three_seed_aggregate.json` |

## 3. 结果

### 3.1 Aggregated episode outcome

| authority | safe capture | collision | boundary | timeout | mean capture time | mean min clearance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| immutable | 97.92% [95.83,99.58] | 0.83% [0,2.92] | 0.83% [0,2.92] | 1.25% [0,3.33] | 5.812 s | 0.522 m |
| replace_nonexecuting | 81.25% [75.42,87.08] | 0.00% [0,0] | 0.00% [0,0] | 18.75% [12.92,24.58] | 5.203 s | 0.521 m |
| flush_pending | 85.42% [78.75,91.67] | 0.00% [0,0] | 0.00% [0,0] | 14.58% [8.33,21.25] | 5.471 s | 0.519 m |

相对 immutable 的 paired bootstrap delta 为：

| authority | safe-capture delta | collision delta | boundary delta | timeout delta |
| --- | ---: | ---: | ---: | ---: |
| replace_nonexecuting | -16.67 pp [-22.50,-10.83] | -0.83 pp [-2.92,0] | -0.83 pp [-2.92,0] | +17.50 pp [11.25,24.17] |
| flush_pending | -12.50 pp [-19.17,-5.83] | -0.83 pp [-2.92,0] | -0.83 pp [-2.92,0] | +13.33 pp [6.25,20.42] |

两种可取消 pending 的 authority 都消除了本 block 的碰撞，但以显著增加 timeout
为代价；`replace_nonexecuting` 的 safe capture 下降更大。因而 authority 改动不能
被描述为“更安全”，而应描述为 collision--liveness trade-off。

### 3.2 Authority/queue diagnostics

三种子均保持 queue length/first controllable step 为 `4.0/4.0`。相对 immutable，
平均 prefix recovery apply rate、override slots、prefix/suffix admissible 和 gate
exhaustion 为：

| authority | recovery apply | override slots | prefix admissible | suffix admissible | gate exhaustion |
| --- | ---: | ---: | ---: | ---: | ---: |
| immutable | 0.00% | 0.000 | 96.35% | 86.01% | 19.99% |
| replace_nonexecuting | 14.84% | 0.445 | 85.16% | 77.86% | 19.31% |
| flush_pending | 11.26% | 0.450 | 88.74% | 81.31% | 20.25% |

本地 CBF 模式下 `safety_abort_required_rate` 不适用（记录为 NaN），因此不能根据
本表宣称“无 safety abort”；可确认的是 episode-level collision/boundary/timeout
结果和 authority override 日志。

### 3.3 Latency

下表是三个 seed 的 summary percentile 均值，顺序均为 `p50/p95/p99`（ms）：

| authority | predictor | planner | QDR | safety | total |
| --- | --- | --- | --- | --- | --- |
| immutable | 20.78 / 39.36 / 53.58 | 42.40 / 70.90 / 88.51 | 1.86 / 3.13 / 4.30 | 2.17 / 3.78 / 5.49 | 74.92 / 123.91 / 154.77 |
| replace_nonexecuting | 28.16 / 46.16 / 61.62 | 53.36 / 77.52 / 94.58 | 2.18 / 3.47 / 4.79 | 2.66 / 4.27 / 6.50 | 96.71 / 136.54 / 168.29 |
| flush_pending | 28.69 / 46.92 / 61.87 | 54.28 / 78.17 / 95.21 | 2.21 / 3.53 / 4.94 | 2.69 / 4.34 / 6.88 | 97.92 / 138.55 / 169.45 |

两种 authority 都比 immutable 增加 total p95，分别约 `10.2%` 和 `11.8%`；更重要
的是 timeout 上升到 `18.75%/14.58%`，因此没有形成可晋级的 authority 修复。
100 ms 仍只是参考值。

## 4. TensorBoard 与完整性

六个 authority run 均保留 80 episode rows、step JSONL、effective config、source
hashes、summary 和 TensorBoard event files。每个 run 有 2 个 event files；QDR-on
的必需 queue/authority/gate 与 predictor/planner/QDR/safety/total latency
p50/p95/p99 标签均存在。聚合 JSON 记录了三种子列表、manifest hash、bootstrap
参数和相对 immutable 的 paired delta。

## 5. 阶段判定

**Authority 消融：完成。** `replace_nonexecuting` 和 `flush_pending` 都实现并完成
三种子可复现实验；两者将 collision/boundary 压到 0，但分别引入 `18.75%` 和
`14.58%` timeout，并降低 safe capture。

**Authority promotion：No-Go。** 当前最可靠的执行契约仍是 immutable；后两种模式
应冻结为 safety--liveness negative ablation，不作为 QDR 主分支，也不能与 R-CLBF-QP
安全证明混写。

## 6. 下一步 TodoList

- [x] 固定 delay4 + noise008，完成 immutable/replace/flush 三种子 authority 消融；
- [x] 保存 authority override、prefix/suffix、timeout、collision 和全套延迟指标；
- [ ] 保持 immutable 主分支，单独分析 timeout/collision hard cases 与 suffix
  exhaustion 的因果路径；
- [ ] 若继续推进 QDR，只能做候选早停或执行 rollout 增量化等不改变 authority 的
  runtime 优化；
- [ ] 暂不开放 QDR×UAKR/RNIC Full 组合，先完成失败轴汇总和论文限制说明。

