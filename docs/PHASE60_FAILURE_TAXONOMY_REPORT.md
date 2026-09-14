# Phase 60：Joint-Stress Failure Taxonomy

本报告对 Phase 60 calibration 的 5 个方法（M0、M5、M6、M7、M8）、3 个
checkpoint seed、4 个 block，共 `5 × 3 × 120 = 1,800` 个 episode summary
做公开日志归因。字段是重叠诊断标记，不是互斥类别，也不构成因果证明或安全
证明。完整 artifact：

```text
results/phase60_failure_taxonomy.json
results/phase60_failure_taxonomy.md
results/phase60_failure_taxonomy_tensorboard/
```

## 关键归因表

| Method | Block | Safe | Collision | Boundary | Timeout | Timeout + QDR exhaustion | Prefix unrecoverable | Max streak |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M6 | id_reference | 90/90 | 0/90 | 0/90 | 0/90 | 0/90 | 19/90 | 12 |
| M6 | delay_noise | 71/90 | 8/90 | 8/90 | 11/90 | 11/90 | 43/90 | 66 |
| M6 | communication_execution | 85/90 | 4/90 | 4/90 | 1/90 | 1/90 | 45/90 | 16 |
| M6 | joint_stress | 46/90 | 14/90 | 2/90 | 30/90 | 30/90 | 83/90 | **168** |
| M7 | id_reference | 90/90 | 0/90 | 0/90 | 0/90 | 0/90 | 30/90 | 15 |
| M7 | delay_noise | 71/90 | 4/90 | 3/90 | 15/90 | 15/90 | 43/90 | 91 |
| M7 | communication_execution | 85/90 | 5/90 | 5/90 | 0/90 | 0/90 | 55/90 | 16 |
| M7 | joint_stress | 45/90 | 12/90 | 4/90 | 33/90 | 33/90 | 77/90 | **112** |
| M8 fixed-K8 | id_reference | 87/90 | 3/90 | 3/90 | 0/90 | 0/90 | 68/90 | 41 |
| M8 fixed-K8 | delay_noise | 47/90 | 40/90 | 26/90 | 3/90 | 3/90 | 60/90 | 23 |
| M8 fixed-K8 | communication_execution | 68/90 | 22/90 | 3/90 | 0/90 | 0/90 | 64/90 | 39 |
| M8 fixed-K8 | joint_stress | 44/90 | 44/90 | 16/90 | 2/90 | 2/90 | 88/90 | 36 |

M6/M7 的 QDR suffix gate 在几乎所有 episode 中都至少发生过一次 exhaustion，
所以 `ever exhausted` 不是可靠的 episode failure label；真正区分 liveness 的
是连续 exhaustion streak、timeout attribution 和 prefix 是否已经不可恢复。

## 主要发现

1. M6/M7 在 joint-stress 中把大部分 baseline collision 转化为了 timeout：
   M6 为 `14 collision / 30 timeout`，M7 为 `12 collision / 33 timeout`。
   这解释了“安全 outcome 变好但不能晋级”的原因。
2. joint-stress 的 prefix-unrecoverable 标记很高（M6 `83/90`、M7 `77/90`），
   与 immutable queue authority 下“已经执行的 prefix 无法事后修复”一致。
3. M8 fixed-K8 并没有自动解决 liveness：joint-stress 仍有 `44/90` collision，
   且 planner p95/total p95 明显高于 M6/M7。增加固定候选数不能替代可行 suffix
   与恢复策略。
4. M7 只有 `1` 个 planner-fallback episode，M6 为 `0`；因此本阶段的主要瓶颈
   不是未解释 solver failure，而是 QDR 可行性/恢复与不可变队列的 liveness。

## 后续实验入口

- 固定当前 checkpoint、阈值和场景 split，单独测试 prefix-unrecoverable 时的
  recovery authority；预注册 collision、timeout、max streak 和 total p95；
- 统计 recovery 后是否真正恢复 suffix admissibility，而不是只统计“曾经耗尽”；
- 若 recovery 仍以 timeout 换 collision，冻结为 safety--liveness negative ablation；
- 不把 local CBF、QDR 时间索引或 failure taxonomy 写成 R-CLBF-QP 安全证明。

