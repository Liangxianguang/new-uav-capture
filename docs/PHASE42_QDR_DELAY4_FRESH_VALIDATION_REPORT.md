# Phase 42：QDR delay4 fresh validation 报告

> 日期：2026-09-13
> 实验级别：validation-confirmation，单因素执行延迟轴
> 仓库：[Liangxianguang/new-uav-capture](https://github.com/Liangxianguang/new-uav-capture)

## 1. 目的与预注册比较

Phase 41 已确认当前源码的 QDR-off baseline。Phase 42 转到独立 fresh validation
manifest，只改变 command delay，从 development block 的 2 steps 提高到 4 steps，
并对 QDR-off/QDR-on 做同场景配对。该轴检验 QDR 是否能在实际待执行队列更长时降低
状态错位造成的碰撞和越界。

QDR-off 和 QDR-on 使用同一 checkpoint、场景、target branch、采样 seed、MPC、
execution dynamics 和 local CBF；QDR-on 额外启用 queue-aware rollout、suffix gate、
recovery candidates 和 queue-aware safety projection。该实验只有一个 checkpoint
seed，属于 fresh validation evidence，不替代三种子确认，也不开放 locked-test。

## 2. 可复现设置

| 项目 | 设置 |
| --- | --- |
| manifest | `results/phase34_qdr_fresh_validation_scenes/scenes.jsonl` |
| 规模 | 80 episodes / 40 mirror groups |
| split | `validation_confirmation`，不含 locked-test |
| manifest SHA-256 | `15652f76157157c8823dfd7086a4978f4f5de2aed108b4ebef18817f4cc03646` |
| predictor | GRU `both`，checkpoint seed `727201` |
| planner | distributed delayed DN-MPC，worst-case，horizon 8，control horizon 3 |
| execution | enabled，command delay 4，immutable，noise/tracking/drag 0 |
| safety | local CBF |
| sampling | `num_samples=1`，`sampling_steps=8`，`sampling_seed=745102` |
| committed reproduction config | `configs/phase42_qdr_delay4_fresh_validation.yaml` |
| executed mpc config | `configs/phase34_qdr_precondition_validation.yaml` + explicit delay4/QDR CLI flags |
| QDR-off run | `results/phase42_qdr_delay4_off_seed727201` |
| QDR-on run | `results/phase42_qdr_delay4_on_seed727201` |

## 3. 结果

### 3.1 Episode-level outcome

| 指标 | QDR-off | QDR-on | on - off |
| --- | ---: | ---: | ---: |
| safe capture | 77.5% (62/80) | 97.5% (78/80) | +20.0 pp |
| collision | 22.5% (18/80) | 2.5% (2/80) | -20.0 pp |
| boundary violation | 5.0% (4/80) | 2.5% (2/80) | -2.5 pp |
| timeout | 0.0% | 0.0% | 0 pp |
| mean capture time | 2.513 s | 6.024 s | +3.512 s* |
| mean minimum clearance | 0.268 m | 0.523 m | +0.256 m |
| worst minimum clearance | -0.166 m | 0.149 m | +0.315 m |

`*` 捕获时间只在成功捕获 episode 上统计，因两臂成功 episode 数不同，差值仅作
描述性比较。

按 episode_index 对齐的 10,000 次 paired bootstrap（seed `20260913`）得到：

- safe-capture delta `+20.0 pp [11.25, 30.00]`；
- collision delta `-20.0 pp [-30.00, -11.25]`；
- boundary delta `-2.5 pp [-8.75, +3.75]`。

这说明在该 delay4 fresh block、固定一个 checkpoint seed 下，QDR 对碰撞/捕获结果有
明确的改善方向；但它仍是单 seed、单 delay 轴，不能外推到任意 delay、噪声或目标
策略，也不能写成安全证明。

### 3.2 QDR 机制诊断

QDR-on 的 queue length 为 `4.0`，first controllable step 为 `4.0`；prefix/suffix
admissible rate 为 `95.65%/83.67%`，suffix gate exhaustion 为 `19.16%`，endpoint
check coverage 为 `91.01%`。solver/effective-plan rate 为 `98.69%`。QDR-off 的
queue-aware rollout rate 为 `0%`，QDR latency 为 `0/0/0 ms`，因此两臂的信息边界
和计算增量是明确可审计的。

### 3.3 延迟

所有延迟均为 `p50 / p95 / p99`（ms）：

| 运行 | predictor | planner | QDR | safety | total |
| --- | --- | --- | --- | --- | --- |
| QDR-off | 12.85 / 25.41 / 28.64 | 12.65 / 22.98 / 28.53 | 0 / 0 / 0 | 1.44 / 2.83 / 3.46 | 29.19 / 54.29 / 61.93 |
| QDR-on | 11.99 / 23.44 / 26.36 | 25.88 / 44.52 / 56.58 | 1.22 / 2.25 / 2.64 | 1.30 / 2.62 / 3.10 | 45.88 / 78.65 / 95.10 |

QDR-on total p95 增加 `24.36 ms`、相对增加约 `44.9%`；虽然 p99 仍低于 100 ms
参考值，但相对 latency gate 仍未通过。这里的延迟代价与 safe-capture/collision
改善共同构成 safety--efficiency trade-off，不能只报告捕获率。

## 4. TensorBoard 与结果审计

两个运行都保存了 effective config、source/config hash、episode/step JSONL、summary
和 TensorBoard event files：

```text
results/phase42_qdr_delay4_off_seed727201/distributed_delayed/tensorboard/
results/phase42_qdr_delay4_on_seed727201/distributed_delayed/tensorboard/
```

off/on 分别审计得到 187/221 个 scalar tags，各有 2 个 event files。两个 run 的
predictor/planner/QDR/safety/total latency p50/p95/p99 标签均存在；QDR-on 还记录了
queue length、first controllable step、prefix/suffix admissibility、gate exhaustion
和 endpoint coverage。

## 5. 阶段判定

**delay4 单因素 fresh validation：有条件通过。** 在一个独立 80 集 manifest 上，
QDR 将 safe capture 从 `77.5%` 提升到 `97.5%`，collision 从 `22.5%` 降到 `2.5%`，
并显著增加 clearance；这是当前 QDR 最强的失败轴证据。

**QDR promotion：仍为 No-Go。** 该结果只有一个 checkpoint seed；QDR-on 仍有 2 个
collision episode、19.16% suffix gate exhaustion，并带来 total p95 `44.9%` 相对
增幅。因此不能直接宣称 QDR 已完成，也不能跳过 authority/noise confirmation。

## 6. 下一步 TodoList

- [x] 在独立 fresh manifest 上完成 delay4 QDR-off/QDR-on 配对；
- [x] 保存 step-level queue、prefix/suffix、gate 和 latency 诊断；
- [ ] 用相同 fresh manifest 完成至少 `727202/727203` 两个 checkpoint seed 的 delay4
  confirmation；
- [ ] 在保持 delay4 不变的前提下，单独加入 bounded execution noise，不能同时改
  authority 或 target behavior；
- [ ] 以 safe-capture non-inferiority、collision、timeout 和相对 p95 为联合 gate；
- [ ] 三种子联合 gate 通过前，不开放 locked-test、UAKR/RNIC 或 Full 组合。
