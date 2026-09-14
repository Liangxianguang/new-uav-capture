# Phase 60b：Adaptive-K 一致性与强制刷新 Smoke 报告

## 1. 目的与结论

本阶段只做 validation-selection smoke，不做性能 promotion、confirmation 或
locked-test。目标是验证三个实现事实：

1. uncertainty policy 是否真正产生 `K=1/4/8`；
2. budget 改变时缓存 candidate set 是否同步刷新；
3. realized candidate count 是否与当步期望 K 一致。

结论：修复后的 v3 smoke 通过实现一致性检查。M6/M7 都出现了低、中、高三个
budget bucket，平均 K 约 `1.69--1.78`，平均 prediction refresh 约 `42%--43%`，
平均 cache age 约 `1.0` step；全部 `candidate_budget_realized_rate=100%`、
`candidate_budget_mismatch_steps=0`。这只证明调度器和审计字段一致，不证明
adaptive-K 能提高捕获率。

## 2. 修复内容

旧版本的问题是：policy 在缓存仍存在时切换 K，但 runtime 沿用旧 candidate set，
造成“决策 K=4/8、实际候选数仍为 1/4”的不一致。当前实现向
`AdaptivePredictionPolicy.decide` 传入 cached candidate count；当预算变化时
强制刷新，并记录 `forced_refresh_reason=budget_change`。同时 evaluator 的
candidate audit 改为：

- 固定-K：按固定请求 K 检查；
- adaptive-K：按当步 `adaptive_num_samples` 检查，而不是把最大 K=8 当成每一步
  的期望值。

因此 v1/v2 smoke 只作为被丢弃的工具链诊断，v3 才是修复后的有效 smoke。

## 3. 配置与规模

| 项目 | 结果 |
| --- | --- |
| scene source | Phase 60 calibration selection 前 20 个 scene |
| checkpoint seeds | 727201、727202、727203 |
| 方法 | M6 QDR synchronous、M7 QDR asynchronous |
| 总 episode | 每方法 60（20 scenes × 3 seeds） |
| 最大预算 | K=8；adaptive levels 为 1/4/8 |
| policy | frozen original adaptive budget，未调阈值 |
| safety layer | local CBF empirical filter |
| decision | validation_selection |
| TensorBoard | 每个 seed/method 的 evaluator 输出目录 |

配置文件为 `configs/phase60_uakr_adaptive_k_smoke.yaml`。结果目录被 Git
忽略，v3 路径模式为
`results/phase60_uakr_adaptive_k_smoke_v3_seed*/`。

## 4. 结果

表中 latency 为 `total p50/p95/p99`，单位 ms；safe/collision/timeout 只作
smoke 描述。

| Seed | Method | Safe | Collision | Timeout | Mean K | Refresh | Cache age | Total p50/p95/p99 | Budget audit |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 727201 | M6 | 100% | 0% | 0% | 1.687 | 41.49% | 1.048 | 27.55/45.59/54.23 | 100%, mismatch 0 |
| 727202 | M6 | 100% | 0% | 0% | 1.782 | 43.60% | 0.993 | 27.89/45.44/53.02 | 100%, mismatch 0 |
| 727203 | M6 | 100% | 0% | 0% | 1.726 | 42.75% | 1.019 | 27.07/45.61/52.83 | 100%, mismatch 0 |
| 727201 | M7 | 100% | 0% | 0% | 1.746 | 43.16% | 1.001 | 25.69/43.58/51.66 | 100%, mismatch 0 |
| 727202 | M7 | 95% | 5% | 0% | 1.710 | 42.31% | 1.032 | 25.96/43.38/51.34 | 100%, mismatch 0 |
| 727203 | M7 | 100% | 0% | 0% | 1.707 | 41.99% | 1.040 | 26.00/44.03/52.98 | 100%, mismatch 0 |

三个 seed 合计的 bucket 计数为：M6 `low/medium/high=2992/839/15`，M7
`3603/1025/15`。所有 v3 rows 的期望 K 集合为 `{1,4,8}`，实际 candidate
count 与期望 K 一致。

## 5. 解释边界与后续

该 smoke 说明 Queue-Aware Delayed-State Rollout 与 Uncertainty-Triggered
Adaptive K 在代码层面已经可审计、可复现；但样本来自同一 calibration selection
前缀，不能宣称泛化、捕获率提升或实时部署优势。下一步若继续研究 adaptive-K，
必须在 untouched development confirmation 预冻结 K/refresh/timeout/exhaustion
门槛；否则保留为实现诊断。local CBF 仍不是 R-CLBF-QP 安全证明。

