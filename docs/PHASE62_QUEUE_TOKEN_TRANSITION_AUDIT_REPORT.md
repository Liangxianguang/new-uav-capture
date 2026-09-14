# Phase 62：Queue-token / ACK transition audit 报告

## 1. 阶段定位

Phase 61 表明，直接允许 `replace_nonexecuting` 或 `flush_pending` 会降低
safe capture 并拉长 QDR exhaustion streak。Phase 62 因此先把“恢复请求是否仍
针对当前队列”从闭环性能问题中分离出来，建立一个可独立复现的、可拒绝 stale
request 的 queue-token / ACK 合同。

本阶段不是捕获率实验，也不是 safety certificate。它只作为后续闭环接入的
implementation gate；没有通过该 gate 时，不开放 confirmation 或 locked-test。

## 2. 形式化的可执行合同

对 pending command queue `Q` 定义 token：

\[
\tau(Q)=(g,s,|Q|),
\]

其中 `g` 是队列版本、`s` 是 token 发布时的仿真步、`|Q|` 是待执行槽位数。
恢复请求必须携带与当前队列完全相同的 `\tau`；否则返回拒绝 ACK，且队列
不得发生任何修改。

- `immutable`：不允许修改任何已排队槽位；
- `replace_nonexecuting`：只能从索引 1 开始修改，并受
  `max_override_slots` 限制；到期槽位保持不变；
- `flush_pending`：允许从索引 0 开始修改，但同样受上限限制；
- 成功修改至少一个槽位时，`g` 增加 1；
- 每次 command commit 只 append 一个新动作并 pop 一个到期动作，不能重复
  计算 delay。

ACK 显式记录 `accepted`、`applied`、原因、覆盖槽位数以及 before/after token。
该机制是执行合同，不是 QDR 几何安全证明。

## 3. 审计设置与结果

配置：`configs/phase62_queue_token_transition_audit.yaml`。审计覆盖 queue
length `0/1/2/4/8`、三种 authority、最大覆盖槽位 `2`，共 55 个用例：

| 检查类别 | 用例数 | 结果 |
|---|---:|---:|
| 有界 authority mutation | 15 | 15/15 |
| stale-token rejection | 15 | 15/15 |
| missing-token policy | 15 | 15/15 |
| queue-length mismatch rejection | 5 | 5/5 |
| single-pop commit and token advance | 5 | 5/5 |
| 合计 | **55** | **55/55** |

25 个 mutable authority 的 stale/missing-token negative probes 全部拒绝，拒绝率
`100%`。平均覆盖槽位为：

| authority | mean applied slots |
|---|---:|
| immutable | 0.0 |
| replace_nonexecuting | 1.0 |
| flush_pending | 1.4 |

所以最终审计结果为 `overall_pass=true`。这些数字来自确定性队列合同测试，
不能解释成 safe capture、collision 或 forward invariance。

## 4. 可复现工件与记录

- 实现：`src/encirclement3d/execution_dynamics.py`；
- 审计脚本：`scripts/audit_phase62_queue_token_transition.py`；
- 测试：`tests/test_phase62_queue_token_transition.py`；
- 最终结果：`results/phase62_queue_token_transition_audit_v3/summary.json`；
- TensorBoard：`results/phase62_queue_token_transition_audit_v3/tensorboard/`，
  同时记录 YAML 配置、claim boundary、总 gate、各 authority 覆盖槽位和逐用例
  pass/fail；
- `results/` 按仓库约定被 `.gitignore` 忽略，结果目录保留在本地。

本阶段没有 predictor/planner/QDR/safety/total 的闭环延迟分位数，因为没有
运行 predictor 或 planner。下一阶段闭环接入后必须继续报告这五类的 p50/p95/p99，
且 100 ms 仍不是硬门槛。

## 5. 下一阶段 To-do

- [x] 实现 QueueToken、token freshness check、bounded override 和 ACK；
- [x] 审计 immutable 不变、mutable 上限、stale/missing rejection 和单 pop；
- [x] 运行 TensorBoard-backed deterministic audit，55/55 通过；
- [ ] 将 token 合同以 opt-in 方式接入 environment/evaluator，默认路径保持
  Phase 60/61 的 immutable 行为不变；
- [ ] 在全新 calibration manifest 上运行三 seed：immutable reference、
  token-matched bounded replacement、故意 stale-token rejection diagnostic；
- [ ] 逐步记录 prefix-recoverable → suffix-admissible 的转移、ACK 原因、
  队列版本变化、timeout、exhaustion streak，并报告五类延迟分位数；
- [ ] 只有新 calibration 同时通过 timeout、streak、safe-capture non-inferiority
  和未解释 planner failure gate，才考虑 confirmation；否则冻结为负结果；
- [ ] local CBF 继续标注为 empirical filter，robust CBF-QP/R-CLBF-QP 继续
  保持 diagnostic No-Go。

