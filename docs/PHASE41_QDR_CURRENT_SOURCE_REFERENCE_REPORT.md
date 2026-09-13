# Phase 41：当前源码 QDR-off 配对基线报告

> 日期：2026-09-13
> 实验级别：validation-development，QDR efficiency gate 配对基线
> 仓库：[Liangxianguang/new-uav-capture](https://github.com/Liangxianguang/new-uav-capture)

## 1. 目的

Phase 38--40 的运行时压缩实验沿用了较早的 QDR-off reference。Phase 41 在缓存和
候选去重已经合入当前源码后，重新执行同一 40-episode development block 的 QDR-off
臂，确认比较基线与当前实现一致。

本实验显式传入 `--no-queue-aware-rollout` 和
`--no-queue-aware-safety-projection`。这是必要的，因为 protocol 文件中的
`phase17.queue_aware_rollout` 默认值为 true；不传 CLI 开关不能代表 QDR-off。

该实验不使用 locked-test，不改变模型、预测采样、MPC cost、执行条件或 local CBF。

## 2. 可复现设置

| 项目 | 设置 |
| --- | --- |
| 场景 | `results/phase34_qdr_development_scenes_v2/scenes.jsonl` |
| 规模 | 40 episodes / 20 mirror groups |
| split | `validation-development`，不含 locked-test |
| 场景 SHA-256 | `4d99d829157145a6f747a4afde6dcbe7c0846e2bc44c5c0a5de5dbe65bd600d3` |
| predictor | GRU `both`，checkpoint seed `727201` |
| planner | distributed delayed DN-MPC，worst-case，horizon 8，control horizon 3 |
| execution | enabled，command delay 2，immutable，noise/tracking/drag 0 |
| safety | local CBF |
| sampling | `num_samples=1`，`sampling_steps=8`，`sampling_seed=745102` |
| QDR | explicitly off |
| run | `results/phase41_qdr_off_current_source_dev_seed727201_v2` |

## 3. 结果

### 3.1 Episode-level outcome

| 指标 | QDR-off current source |
| --- | ---: |
| safe capture | 100.0% |
| collision | 0.0% |
| boundary violation | 0.0% |
| timeout | 0.0% |
| mean capture time | 1.9625 s |
| mean minimum clearance | 0.4457 m |
| worst minimum clearance | 0.2220 m |
| solver success / effective plan | 100.0% / 100.0% |
| queue-aware rollout rate | 0.0% |
| QDR latency | 0 / 0 / 0 ms |

### 3.2 与 Phase40 QDR-on 的配对参考

Phase40 QDR-on 使用同一场景、checkpoint、采样 seed、执行延迟和 safety layer；唯一
主变量是 QDR 开关及其必要的 suffix gate/recovery 机制。

| 指标 | QDR-off | Phase40 QDR-on | QDR-on - off |
| --- | ---: | ---: | ---: |
| safe capture | 100.0% | 100.0% | 0 pp |
| collision | 0.0% | 0.0% | 0 pp |
| mean capture time | 1.9625 s | 3.9150 s | +1.9525 s |
| mean minimum clearance | 0.4457 m | 0.5562 m | +0.1106 m |

所有延迟均为 `p50 / p95 / p99`（ms）：

| 运行 | predictor | planner | QDR | safety | total |
| --- | --- | --- | --- | --- | --- |
| QDR-off current source | 10.51 / 21.42 / 23.20 | 10.04 / 19.76 / 22.01 | 0 / 0 / 0 | 1.13 / 2.47 / 2.88 | 24.44 / 43.36 / 47.23 |
| Phase40 QDR-on | 10.76 / 21.79 / 23.83 | 24.51 / 40.43 / 46.77 | 0.77 / 1.43 / 1.74 | 1.17 / 2.58 / 2.94 | 43.61 / 63.77 / 72.40 |

QDR-on 的 total p95 增加 `20.42 ms`，相对增幅约 `47.1%`；预注册的 `+15%`
上限对应约 `49.86 ms`，因此仍未通过效率 gate。100 ms 仅作为参考值，不能替代
相对基线门槛。

### 3.3 TensorBoard

事件目录为：

```text
results/phase41_qdr_off_current_source_dev_seed727201_v2/distributed_delayed/tensorboard/
```

审计得到 2 个 event files、187 个 scalar tags；predictor/planner/QDR/safety/
total latency 的 p50/p95/p99 标签均存在。root `config.yaml` 同时保存了显式
QDR-off flags、effective config、source hashes 和场景 hash。

## 4. 阶段判定

**当前源码配对基线：通过。** QDR-off outcome 与预期一致，且确认缓存/候选去重没有
改变非 QDR 分支的安全结果。

**QDR efficiency promotion：No-Go。** 在同一 development block 上，QDR-on 虽然
保持了 100% 安全捕获，但 total p95 比当前源码 QDR-off 高 `47.1%`，并伴随
`21.76%` suffix gate exhaustion。该结果支持继续做执行延迟失败分析，不支持立即
进入三种子 confirmation、OOD 扩展或完整 QDR×UAKR×RNIC 组合。

## 5. 后续计划

- [ ] 在 fresh validation 上分别冻结 command delay、authority、bounded execution
  noise，逐轴验证 QDR 的安全 outcome，而不是继续在同一 block 调参；
- [ ] 保留 QDR-off/current-source 与 QDR-on 的 paired episode/step traces；
- [ ] 若任一 stress axis 的 safe-capture non-inferiority 和相对 latency gate 同时
  通过，再进行三种子 confirmation；
- [ ] 在 QDR 通过前不开放 UAKR、RNIC 或 Full 组合，避免把失败原因混在一起。
