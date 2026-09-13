# Phase 53：QDR 可恢复重规划窗口诊断

## 结论

Phase 52 的全 horizon 经验执行管将 40 个 validation development episode 的
safe capture 降至 15%、timeout 提高到 85%。Phase 53 保持同一 calibration、同一
执行合同和同一场景前缀，只把经验管限制在当前 immediate controllable window 的
前 2 个 step，后续 suffix 留给下一次 replanning。

该修改恢复了本开发块的 liveness：safe capture `100%`、timeout `0%`、collision
和 boundary `0%`。这是一个“可恢复性窗口”机制的 development pass，但只有一个
checkpoint seed、40 个 episode，不能替代三种子 confirmation，也不是安全证明。

## 1. 固定条件

两臂均使用：

```text
Phase 48 validation scene prefix 前 40 episodes
GRU both checkpoint seed 727201
distributed delayed DN-MPC + local CBF
action delay = 4
command noise std = 0.08 m/s, clipped at 3 sigma
pending command authority = immutable
Torch intra/inter-op threads = 1/1
```

唯一变量是 QDR execution tube：

- `nominal`：经验管关闭；
- `windowed`：使用 Phase 52 randomized execution calibration artifact，
  multiplier `2.0342518477112503`，但只作用于前 2 个 suffix steps。

## 2. 闭环结果

| 指标 | nominal QDR | 2-step windowed tube |
| --- | ---: | ---: |
| safe capture | 100.00% | **100.00%** |
| capture event | 100.00% | 100.00% |
| collision | 0.00% | 0.00% |
| boundary violation | 0.00% | 0.00% |
| timeout | 0.00% | 0.00% |
| mean capture time | 6.273 s | 5.935 s |
| mean minimum clearance | 0.513 m | **0.551 m** |
| QDR suffix admissible rate | 80.50% | **81.34%** |
| suffix gate exhaustion rate | 23.49% | 25.64% |
| maximum exhaustion streak | 23 steps | **18 steps** |
| mean/max applied tube radius | 0 / 0 m | 0.011 / 0.068 m |

40-episode paired bootstrap（10,000 resamples）给出：

- safe-capture delta：`0.00 pp [0.00,0.00]`；
- timeout delta：`0.00 pp [0.00,0.00]`；
- mean capture-time delta：`-0.3375 s [-2.2525,1.5301]`；
- minimum-clearance delta：`+0.0372 m [+0.0091,+0.0643]`。

因此本阶段的价值是修复 Phase 52 的 liveness 失败并增加短期几何裕量，不能
宣称 safe-capture superiority。

## 3. 延迟分解（p50 / p95 / p99，ms）

| stage | nominal QDR | 2-step windowed tube |
| --- | ---: | ---: |
| predictor | 5.56 / 7.43 / 8.76 | 5.47 / 6.93 / 7.77 |
| planner | 12.80 / 17.45 / 20.96 | 12.44 / 15.93 / 17.50 |
| QDR | 0.65 / 0.88 / 1.12 | 0.63 / 0.84 / 1.05 |
| safety | 0.64 / 0.88 / 1.18 | 0.62 / 0.81 / 0.97 |
| total | **22.02 / 28.81 / 34.11** | **21.43 / 25.88 / 28.04** |

该延迟结果来自同硬件、同一进程、固定线程的 development diagnostic；100 ms
仍只是参考值。

## 4. 机制解释

全 horizon 管的问题不是管本身没有覆盖率，而是把未来仍可由下一次 MPC 更新的
动作提前判成不可行。2-step window 将经验裕量限制在近期可控动作上，使后续状态
能够由新观测和新队列状态重新计算。该解释与 `qdr_execution_tube_active_steps`
的 step-level 记录一致，但仍属于实验性机制解释。

## 5. 复现资产和决策

配置与日志：

```text
configs/phase53_qdr_recoverability_window.yaml
results/phase53_qdr_recoverability_window_nominal_seed727201/
results/phase53_qdr_recoverability_window_empirical_seed727201/
results/phase53_qdr_recoverability_window_paired_aggregate.json
```

TensorBoard 已记录 `execution_tube_enabled_rate`、multiplier、active steps、
mean/max radius 以及 predictor/planner/QDR/safety/total 的 p50/p95/p99。

决策：

- [x] Phase 52 全 horizon 经验管的 timeout 失败被可重复定位；
- [x] 2-step recoverability window 在 development block 恢复 liveness；
- [ ] 在新的 confirmation manifest 上用 `727201/727202/727203` 三种子复现；
- [ ] confirmation 前不访问 locked-test，也不同时引入 UAKR/RNIC；
- [ ] 若 confirmation 通过，再测试 active window `1/2/4` 的预注册比较；
- [ ] 若 confirmation 失败，保留 nominal QDR，冻结经验管为负消融。

