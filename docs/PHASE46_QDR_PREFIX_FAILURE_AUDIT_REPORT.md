# Phase 46：QDR 前缀失败路径与 authority 诊断报告

> 日期：2026-09-13  
> 实验级别：validation-confirmation 后验机制诊断  
> 仓库：[Liangxianguang/new-uav-capture](https://github.com/Liangxianguang/new-uav-capture)

## 1. 目的与边界

Phase 43--45 已确认：在 `delay=4` 和有界执行噪声下，immutable QDR 能显著降低
collision，但 suffix gate 仍有约 20% 的耗尽率；更换为
`replace_nonexecuting` 或 `flush_pending` 虽然消除了本批次的 episode-level
collision/boundary，却明显增加 timeout。本阶段对这些已完成运行的
`steps.jsonl` 做前缀失败路径诊断，回答两个问题：

1. QDR 的 nominal prefix violation 主要来自哪类公开几何约束；
2. authority 改变后，前缀不可行和 suffix 不可行是否真正减少。

该分析只读取 evaluator 已记录的公开几何字段，不使用 target truth、终止原因或
locked-test 来反推前缀分类。因此它是机制诊断，不是安全证明，也不改变 Phase 43--45
的 episode-level 结论。

## 2. 可复现输入

| 项目 | 设置 |
| --- | --- |
| 输入 | Phase 44 immutable 3 seeds；Phase 45 `replace_nonexecuting`/`flush_pending` 各 3 seeds |
| 每个输入 | `distributed_delayed/steps.jsonl` |
| 总量 | immutable 14,545 rows；replace 21,378 rows；flush 19,951 rows |
| 分类字段 | `qdr_prefix_*` 共 9 个字段，全部输入行均完整 |
| 分析脚本 | `scripts/analyze_qdr_prefix_failures.py` |
| 聚合工件 | `results/phase46_qdr_prefix_failure_audit.json` |
| 工件 SHA-256 | `20EE56ECB8FEC5C66FEC51B4FB65C6E316A84A2ED6EC6C5B7E6951C01366EBD3` |
| locked-test | 未使用 |

## 3. 结果

### 3.1 Prefix 与 suffix 诊断

| authority | prefix admissible | mean violating-step ratio | suffix-inadmissible rows | gate-exhausted rows | 首次 prefix violation |
| --- | ---: | ---: | ---: | ---: | --- |
| immutable | 93.48% | 4.79% | 18.05% | 19.97% | boundary 741、obstacle 202、inter-agent 5 |
| replace_nonexecuting | 60.33% | 38.77% | 43.55% | 18.23% | boundary 7,716、obstacle 529、inter-agent 235 |
| flush_pending | 67.73% | 31.14% | 34.95% | 21.28% | boundary 5,412、obstacle 499、inter-agent 527 |

这里的 `prefix admissible` 是控制步级比例，不是 episode 安全率；
`suffix-inadmissible rows` 统计的是 `qdr_suffix_admissible=false` 的控制步，
`gate-exhausted rows` 统计候选 gate 没有找到可行 suffix 的控制步。三者不能互相
替代，也不能直接解释为 collision 概率。

### 3.2 失败 episode 的追踪

在 240 个 immutable episode 中，235 个以 safe capture 结束，3 个 timeout，2 个
以 safety failure 结束。共有 238 个 episode 至少出现一次 gate exhaustion，说明
“是否曾耗尽”本身过于宽泛；真正需要优化的是耗尽的持续长度、首次耗尽时的 prefix
状态和后续是否恢复。immutable 运行中有 70 个 safe-capture episode 出现过
prefix violation，但仍成功捕获，说明 prefix 几何告警与最终 episode failure 不是
一一对应关系。

两个 authority ablation 都把 collision/boundary 降至 0%，但其 prefix violation
比例显著高于 immutable，并以 timeout 换取了物理安全余量。这与 Phase 45 的
safety--liveness trade-off 一致：简单地替换或清空 pending queue 不是无代价修复，
也不能作为真实飞控 authority 的默认假设。

## 4. 机制结论

1. immutable QDR 的主要公开几何风险是 boundary，其次是 obstacle；inter-agent
   只占极少数首次 violation。当前优先级应是 boundary-aware suffix proposal 与
   首次 violation 后的短 horizon recovery，而不是继续扫描 authority。
2. authority ablation 没有降低 suffix gate exhaustion：replace 的耗尽率略低，
   flush 反而略高；二者却显著降低 prefix admissibility，并造成 timeout。因此
   “允许修改队列”不能被写成 QDR 的核心收益。
3. 大多数 episode 都曾遇到至少一次 gate exhaustion，但只有少数最终失败，说明
   gate exhaustion 更像一个需要做持续性/恢复性分层的过程指标，而不是直接的
   failure label。下一版实验应记录 consecutive exhaustion length、首次耗尽距
   capture/timeout 的步数和 recovery candidate 的选择结果。
4. 当前最安全的主分支仍是 `immutable`。可以研究不改变 queue authority 的
   **bounded candidate early-stop / incremental rollout**，并要求逐步 action
   selection 与 episode outcome 与当前实现等价；不能为了降低 latency 而改变
   prefix 语义。

## 5. 下一步 TodoList

- [x] 对 Phase 44 immutable 和 Phase 45 两种 authority 完成公开几何前缀分类审计；
- [x] 校验九个 classifier 字段在 9 个输入文件中全部存在；
- [x] 汇总 boundary/obstacle/inter-agent 首次 violation 计数；
- [x] 在既有 40-episode development block 上对 full scorer 与
  `qdr_feasibility_first=true` 做一对一 smoke；两者 episode outcome、step-level
  gate 字段和测试用例中的 selected action/cost 保持一致；
- [x] 检查两条 smoke 的 TensorBoard：均为 2 个 event files、221 个 scalar tags，
  predictor/planner/QDR/safety/total 的 p50/p95/p99 全部存在；
- [x] 记录当前实现的运行时结果：本次单种子 smoke 中 feasibility-first 没有降
  低 planner/total latency，因此不晋级为 runtime promotion；
- [ ] 在 evaluator 中加入连续 gate-exhaustion length、首次耗尽步、恢复步数和
  episode 终止前剩余步数；
- [ ] 若后续继续优化，先减少 feasibility-first 预计算开销，再在新的 development
  block 重做等价性和 p50/p95/p99 runtime smoke；
- [ ] 只有在 runtime smoke 证实真实收益且 episode outcome 仍完全等价时，才考虑
  fresh confirmation；
- [ ] 在上述等价性与 confirmation 通过前，不重开 UAKR/RNIC Full factorial，也不
  把 QDR gate 写成 safety certificate。

## 6. Feasibility-first smoke 结果

使用 Phase 40 的 40-episode development block、seed `727201`、GRU `both`、
delay4、noise008、immutable authority 和 local CBF。full scorer 与
`qdr_feasibility_first` 使用同一 protocol、checkpoint、sampling seed 和
`distributed_delayed` 方法。

| 指标 | full scorer | feasibility-first | 差异/判定 |
| --- | ---: | ---: | --- |
| safe capture | 95.00% | 95.00% | 0 episode mismatch |
| collision | 5.00% | 5.00% | 0 episode mismatch |
| boundary | 5.00% | 5.00% | 0 episode mismatch |
| timeout | 0.00% | 0.00% | 0 episode mismatch |
| planner p50/p95/p99 (ms) | 30.41 / 58.22 / 66.37 | 34.46 / 62.97 / 77.66 | 当前未改善 |
| total p50/p95/p99 (ms) | 51.22 / 98.19 / 111.71 | 55.12 / 103.06 / 119.69 | 当前未改善 |

两条运行均为 40 episode、2,579 step rows；step-level 的 episode index、step、
gate exhaustion、rejected-candidate count、planner status/fallback/valid 和
prefix/suffix admissibility 字段逐行一致。focused unit test 还验证了带有可行与
不可行候选的场景中 selected action 和 selected scenario cost 完全一致。

因此这不是性能成功结果：当前 feasibility-first 预计算开销抵消了被跳过的 full
cost 计算。它保留为可审计的等价性实现和负 runtime ablation，不替换 Phase 40
参考，也不开放 locked-test。

## 6. Exhaustion persistence logging schema smoke

已在 evaluator 中加入以下不改变控制逻辑的过程字段：

```text
qdr_suffix_gate_exhausted_once
qdr_suffix_gate_first_exhaustion_step
qdr_suffix_gate_exhaustion_streak_steps
qdr_suffix_gate_max_exhaustion_streak_steps
qdr_suffix_gate_recovered_after_exhaustion
qdr_suffix_gate_recovery_count
```

在同一 development block 的 2-episode schema smoke 中，新增 summary 能记录
`exhausted_once=1.0`、平均首次耗尽步 `14.0`、最大连续耗尽长度 `5.0` 和平均恢复
次数 `7.5`；两条 TensorBoard run 均包含新增 episode 字段和原有完整 latency
分位数标签。该数值仅用于验证日志契约，不能作为新的性能估计。

## 7. 阶段判定

**诊断完成；QDR promotion 状态不变。** Phase 46 找到了可执行的优化方向：对
immutable authority 保持不变的前提下，压缩无望候选的 full-cost 计算并增加连续耗尽
诊断。新增持久性字段已经通过 schema smoke，但候选压缩尚未提速；它没有证明新的
捕获收益，也没有为 replace/flush 提供升级依据。
