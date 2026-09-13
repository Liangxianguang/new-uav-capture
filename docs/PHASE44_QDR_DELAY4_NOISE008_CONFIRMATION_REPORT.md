# Phase 44：QDR delay4 + bounded execution noise 三种子确认报告

> 日期：2026-09-13
> 实验级别：validation-confirmation，单因素 bounded execution-noise 轴
> 仓库：[Liangxianguang/new-uav-capture](https://github.com/Liangxianguang/new-uav-capture)

## 1. 目的与控制变量

Phase 43 在 delay4、零执行噪声下确认了 QDR 的安全收益，但仍存在效率和少量
liveness 代价。Phase 44 固定 delay4 和 immutable authority，只加入有界 command
noise：高斯标准差 `0.08 m/s`、`3σ` clipping（最大单轴噪声 `0.24 m/s`）。off/on
仍使用同一 fresh manifest、checkpoint、采样 seed、MPC 和 local CBF。

该实验不改变目标策略、几何、通信 dropout 或 QDR authority；因此它是独立的
execution-noise stress axis。它不使用 locked-test，也不包含 UAKR/RNIC。

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
| execution | delay 4，immutable，noise std `0.08 m/s`，bound `3σ`，tracking/drag 0 |
| safety | local CBF |
| sampling | `num_samples=1`，`sampling_steps=8`，`sampling_seed=745102` |
| committed config | `configs/phase44_qdr_delay4_noise008_fresh_validation.yaml` |
| aggregate artifact | `results/phase44_qdr_delay4_noise008_three_seed_aggregate.json` |

实际运行使用了 Phase42 等价配置并通过 CLI 显式覆盖 delay/noise；每个 root
`config.yaml` 保存了最终 effective execution mapping，因此结果不依赖隐含默认值。

## 3. 三种子结果

### 3.1 Aggregated episode outcome

| 指标 | QDR-off | QDR-on | on - off |
| --- | ---: | ---: | ---: |
| safe capture | 77.92% [72.08, 83.33] | 97.92% [95.83, 99.58] | +20.00 pp [14.17, 25.83] |
| collision | 22.08% [16.67, 27.92] | 0.83% [0.00, 2.92] | -21.25 pp [-27.50, -15.00] |
| boundary violation | 4.58% [2.08, 7.92] | 0.83% [0.00, 2.92] | -3.75 pp [-7.50, 0.00] |
| timeout | 0.00% [0.00, 0.00] | 1.25% [0.00, 3.33] | +1.25 pp [0.00, 3.33] |
| mean capture time | 2.596 s [2.435, 2.781] | 5.812 s [5.127, 6.468] | +2.727 s [2.139, 3.323]* |
| mean minimum clearance | 0.263 m [0.236, 0.291] | 0.522 m [0.506, 0.538] | +0.259 m [0.229, 0.288] |

`*` 捕获时间仅在成功捕获 episode 上统计，因两臂成功数不同，只作描述性比较。

三种子 paired bootstrap（10,000 次，seed `20260913`）显示，QDR 在 delay4 + noise
轴上仍然保持约 `+20 pp` safe-capture 和约 `-21 pp` collision 的改善方向；同时
QDR-on 出现 `1.25%` timeout，liveness 代价不能忽略。

### 3.2 Per-seed outcome and latency

下表为各 seed summary 中的 total control latency `p50/p95/p99`（ms）：

| seed | arm | safe capture | collision | boundary | timeout | total p50/p95/p99 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 727201 | off | 80.0% | 20.0% | 3.75% | 0.0% | 61.20 / 91.60 / 115.23 |
| 727202 | off | 75.0% | 25.0% | 6.25% | 0.0% | 62.41 / 93.40 / 116.77 |
| 727203 | off | 78.75% | 21.25% | 3.75% | 0.0% | 61.62 / 92.48 / 121.62 |
| 727201 | on | 97.5% | 2.5% | 2.5% | 0.0% | 87.65 / 124.37 / 154.25 |
| 727202 | on | 97.5% | 0.0% | 0.0% | 2.5% | 49.78 / 122.05 / 154.34 |
| 727203 | on | 98.75% | 0.0% | 0.0% | 1.25% | 87.32 / 125.32 / 155.73 |

三种子 summary percentile 均值，顺序均为 `p50/p95/p99`（ms）：

| arm | predictor | planner | QDR | safety | total |
| --- | --- | --- | --- | --- | --- |
| QDR-off | 29.49 / 49.25 / 66.80 | 25.16 / 38.40 / 47.59 | 0 / 0 / 0 | 2.67 / 4.26 / 5.76 | 61.75 / 92.49 / 117.87 |
| QDR-on | 20.78 / 39.36 / 53.58 | 42.40 / 70.90 / 88.51 | 1.86 / 3.13 / 4.30 | 2.17 / 3.78 / 5.49 | 74.92 / 123.91 / 154.77 |

QDR-on 的 summary total p95 比 off 高约 `34.0%`，p99 高约 `31.3%`；100 ms 仍然
只是参考值，不能替代相对 latency gate。

### 3.3 QDR 机制诊断

QDR-on 三种子均值为：queue length/first controllable step `4.0/4.0`，prefix
admissible `96.35%`，suffix admissible `86.01%`，suffix gate exhaustion `19.99%`，
endpoint check coverage `90.03%`，effective-plan rate `98.35%`。与 delay4-noise0
相比，噪声没有消除 suffix infeasibility；QDR 仍属于“改善失败 outcome，但计算和
liveness 有代价”的模块。

## 4. TensorBoard 与结果完整性

六个 run 均保存了 effective config、source hashes、episode/step JSONL、summary 和
TensorBoard event files：

```text
results/phase44_qdr_delay4_noise008_off_seed727201/
results/phase44_qdr_delay4_noise008_off_seed727202/
results/phase44_qdr_delay4_noise008_off_seed727203/
results/phase44_qdr_delay4_noise008_on_seed727201/
results/phase44_qdr_delay4_noise008_on_seed727202/
results/phase44_qdr_delay4_noise008_on_seed727203/
```

每个 run 有 80 episode rows 和 2 个 TensorBoard event files；off/on 的 scalar tag
数量分别为 `187/221`，predictor/planner/QDR/safety/total latency 的 p50/p95/p99
标签全部存在。聚合 JSON 保存 manifest hash、三种子列表、bootstrap 参数和 paired
delta。

## 5. 阶段判定

**bounded-noise 安全效果：条件性通过。** 在固定 delay4 + `0.08 m/s` noise 轴上，
QDR 仍把 safe capture 提升 `20.00 pp [14.17,25.83]`，collision 降低
`21.25 pp [-27.50,-15.00]`；该效果跨三个 checkpoint seed 保持。

**QDR promotion：仍为 No-Go。** QDR-on 仍有 `0.83%` collision、`1.25%` timeout、
约 `20%` suffix gate exhaustion，并带来约 `34%` total p95 增幅。当前证据足以支持
“QDR 在长延迟和有界执行扰动下改善安全 outcome”的条件性论文结果，但不足以支持
无条件实时性、完整安全证明或 Full 组合成功。

## 6. 下一步 TodoList

- [x] 固定 delay4，只加入 bounded command noise 的三种子 off/on confirmation；
- [x] 记录 noise bound、队列状态、suffix gate 和全套 latency 分位数；
- [ ] 单独测试 `replace_nonexecuting` authority，使用相同 delay4 + noise，不改变
  其他因素；
- [ ] 单独测试 `flush_pending` authority，并报告 timeout/collision trade-off；
- [ ] 在 authority/noise 各轴完成后，再决定是否值得做 QDR×UAKR/RNIC interaction；
- [ ] 在效率 gate 通过前继续关闭 locked-test Full 组合，不把当前 QDR 结果写成
  safety certificate。
