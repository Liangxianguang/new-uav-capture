# Phase 62b：QueueToken / ACK 闭环接入冒烟报告

## 1. 阶段定位

本阶段验证 Phase 62 的队列 token/ACK 合同是否真正贯穿
`evaluator → environment → delayed execution queue`，不再只停留在独立的
确定性单元审计。它是实现集成 smoke，不是新的性能、泛化或安全结论。

场景复用了 Phase 60 calibration-selection manifest，因此不能称为 fresh
calibration，也不能用于调参、confirmation 或 locked-test promotion。

## 2. 设置

- 场景：8 episodes，来自 `results/phase60_calibration_selection/scenes.jsonl`；
- predictor：Diagonal-SSM checkpoint `seed=727201`；
- planner：`M6_qdr_synchronous_mpc`；
- authority：`replace_nonexecuting`，最多覆盖 1 个可修改队列槽位；
- token：`τ=(generation, issued_step, pending_length)`；
- safety：local CBF；
- CPU Torch 线程：intra/inter-op 均为 1；
- 每个控制步刷新预测，`num_samples=1`；
- TensorBoard、配置快照、episode/step JSONL 均保留。

配置快照：`configs/phase62_queue_token_closed_loop_smoke.yaml`。

## 3. 集成结果

| 指标 | 结果 |
|---|---:|
| safe capture | 87.50% |
| collision | 0.00% |
| boundary violation | 0.00% |
| timeout | 12.50% |
| mean capture time | 7.543 s |
| token-present rate on recovery requests | 10.50% steps |
| ACK accepted rate | 100.00% |
| ACK applied rate | 10.50% steps |
| ACK reason | bounded recovery 252；no mutation 576 |

这里的 token-present/apply rate 是按控制步统计；恢复只在 QDR 前缀不可行且
需要 authority 时请求，因此不应期待接近 100%。ACK accepted rate 为 100% 说明
本次有效请求均携带与当前队列一致的 token；它不等价于恢复成功率或安全性。

## 4. 五类延迟分位数

单位为 ms，来自本次 828 个控制步。100 ms 不是硬门槛；这里同时保留 p50、
p95、p99 以便之后和完整 calibration 对照。

| component | p50 | p95 | p99 |
|---|---:|---:|---:|
| predictor | 7.713 | 9.701 | 11.284 |
| planner | 12.299 | 15.820 | 18.197 |
| QDR | 0.644 | 0.826 | 1.040 |
| safety | 0.615 | 0.795 | 0.902 |
| total | 23.673 | 28.065 | 32.414 |

## 5. 结论边界

该 smoke 证明了：

1. environment 会暴露当前 queue token；
2. evaluator 会携带 token 发出恢复请求；
3. environment 会返回结构化 ACK，并记录真实的 queue override slots；
4. 单步 commit 仍只 pop 一个到期 command，token 会随队列转换推进；
5. 五类延迟日志可以在闭环中统一输出并写入 TensorBoard。

该 smoke **没有**证明 QueueToken 提高捕获率、改善 liveness、具有几何安全
保证，或支持 real-flight deployment。local CBF 仍是经验过滤器；robust
CBF-QP/R-CLBF-QP 仍为 diagnostic No-Go，不宣称安全证明。

## 6. 后续实验

- [x] 完成 opt-in QueueToken/ACK 闭环接入；
- [x] 完成 8-episode integration smoke；
- [ ] 在全新的 calibration manifest 上比较 immutable reference、token-matched
  bounded replacement，以及故意 stale-token diagnostic；
- [ ] 扩展到至少 3 个 predictor seeds，并完成 paired capture/liveness analysis；
- [ ] 统一报告 predictor/planner/QDR-or-tube/safety/total 的 p50/p95/p99；
- [ ] 只有通过预注册 timeout、exhaustion-streak 和 safe-capture
  non-inferiority gate，才允许进入 confirmation；否则冻结为负结果。

本地结果目录：
`results/phase62_queue_token_closed_loop_smoke_seed727201/`。
