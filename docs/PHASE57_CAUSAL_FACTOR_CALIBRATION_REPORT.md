# Phase 57：因果因素校准与运行时公平性报告

## 1. 实验范围

Phase 57 是 Phase 56 之后的全新 development-calibration 实验，用于把
QDR 的作用与 asynchronous distributed planning 的作用拆开。实验没有访问
locked-test，也没有使用 locked-test 选择参数。

- 场景：360 episodes、180 mirror groups；4 个 block，每个 block 45 个 mirror groups；
- 每个 split：120 episodes、60 mirror groups；mirror group 不跨 split；
- 预测器：冻结的 GRU `both` checkpoint，三个 predictor seeds：727201/727202/727203；
- 执行器：immutable pending-command authority，delay/noise/communication 条件由场景记录冻结；
- 安全层：`local_cbf`，仅作为经验动作过滤器；
- manifest（development-calibration selected scenes）SHA-256：
  `7484ca57990574d41a77c813267bc6cee975e4059c8017f63e3f6b8d824ebc6d`。

结果目录保留在本地 `results/phase57_causal_calibration_seed727201/`、
`seed727202/`、`seed727203/`，每个 method 均包含 JSONL、summary、配置快照和
TensorBoard event files。生成器、协议和测试分别见：

- `scripts/generate_phase57_causal_factor_scenes.py`；
- `configs/phase57_causal_factor_protocol.yaml`；
- `tests/test_generate_phase57_causal_factor_scenes.py`。

## 2. 主要闭环结果

以下为 mirror-group bootstrap 的 development-calibration 统计；括号是 95%
bootstrap 区间。它们是当前校准证据，不是 locked-test 结论。

| 方法 | Safe capture | Collision | Boundary | Timeout | Capture time (s) | Minimum clearance (m) | Total p50/p95/p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 current-state delayed-MPC | 70.56% [64.17,77.22] | 29.44% | 9.44% | 0.00% | 2.238 | 0.250 | 63.93/80.13/103.76 |
| B1 QDR-MPC | 80.00% [74.72,84.72] | 20.00% | 12.50% | 0.00% | 4.654 | 0.295 | 109.72/130.87/157.05 |
| B4 synchronous distributed | 70.83% [64.17,76.94] | 29.17% | 12.22% | 0.00% | 2.154 | 0.238 | 70.21/83.36/96.41 |
| B5 asynchronous distributed | 63.89% [57.22,70.28] | 36.11% | 7.78% | 0.00% | 2.188 | 0.163 | 68.10/80.81/93.45 |
| B6 QDR + asynchronous | 88.06% [83.33,91.67] | 6.94% | 6.11% | 5.00% | 6.216 | 0.483 | 22.04/38.01/48.65* |

`*` B6 的主矩阵是在三个 seed 进程并发执行期间采集的；因此其绝对延迟不
能直接用于宣称优于 B0/B4/B5。B6 的安全/活性结果仍然是有效的已记录闭环
结果，但效率比较必须等待隔离运行时基准。

## 3. 完整组件延迟分解

所有 method 都记录了 predictor、planner、QDR/tube、safety 和 total 五个
组件的 p50/p95/p99。主矩阵分解如下：

| 方法 | Predictor p50/p95/p99 | Planner p50/p95/p99 | QDR/tube p50/p95/p99 | Safety p50/p95/p99 | Total p50/p95/p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 33.63/43.00/55.07 | 22.03/28.43/39.13 | 0.00/0.00/0.00 | 3.62/5.18/6.97 | 63.93/80.13/103.76 |
| B1 | 32.52/40.24/49.93 | 58.40/71.03/87.95 | 3.21/5.86/7.36 | 3.50/4.83/6.22 | 109.72/130.87/157.05 |
| B4 | 32.38/38.44/45.65 | 29.79/39.42/46.92 | 0.00/0.00/0.00 | 3.51/4.67/5.98 | 70.21/83.36/96.41 |
| B5 | 31.66/38.25/45.67 | 28.32/35.74/43.64 | 0.00/0.00/0.00 | 3.40/4.52/5.60 | 68.10/80.81/93.45 |
| B6 | 5.41/11.13/15.61 | 12.41/22.72/29.65 | 0.80/1.68/2.74 | 0.64/1.71/2.29 | 22.04/38.01/48.65 |

### 运行时公平性结论

P57-A 已完成正式的多 seed process-isolated benchmark：15 个子进程顺序执行，
每个 method/seed 使用独立 Python 进程、Torch `1/1` 线程、相同 K=8、相同
sampling steps，并同时计算完整闭环和前 18 步 matched prefix。以下是跨三个
seed 的 pooled 结果：

| 方法 | 视图 | Predictor p50/p95/p99 (ms) | Planner p50/p95/p99 (ms) | QDR/tube p50/p95/p99 (ms) | Safety p50/p95/p99 (ms) | Total p50/p95/p99 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B0 | full | 4.24/6.10/9.34 | 3.23/4.41/6.90 | 0.00/0.00/0.00 | 0.50/0.71/1.45 | 8.69/11.97/17.38 |
| B0 | first 18 | 4.23/5.98/9.07 | 3.22/4.33/6.74 | 0.00/0.00/0.00 | 0.50/0.70/1.43 | 8.66/11.75/16.63 |
| B1 | full | 4.33/5.71/7.71 | 8.64/10.66/14.05 | 0.53/0.93/1.34 | 0.50/0.70/1.23 | 15.88/19.32/24.40 |
| B1 | first 18 | 4.32/5.80/7.67 | 8.64/10.80/14.20 | 0.52/0.91/1.32 | 0.50/0.71/1.19 | 15.84/19.43/24.71 |
| B4 | full | 4.31/5.41/7.27 | 4.71/6.34/7.67 | 0.00/0.00/0.00 | 0.51/0.68/1.19 | 10.31/12.57/15.37 |
| B4 | first 18 | 4.31/5.45/7.16 | 4.68/5.95/7.58 | 0.00/0.00/0.00 | 0.51/0.68/1.27 | 10.26/12.47/15.32 |
| B5 | full | 4.32/5.49/7.24 | 4.60/6.01/7.68 | 0.00/0.00/0.00 | 0.51/0.68/1.17 | 10.21/12.45/15.38 |
| B5 | first 18 | 4.32/5.49/7.27 | 4.58/5.82/7.47 | 0.00/0.00/0.00 | 0.51/0.68/1.17 | 10.17/12.31/15.31 |
| B6 | full | 4.44/5.51/7.45 | 9.55/12.08/14.74 | 0.55/0.92/1.31 | 0.52/0.67/1.13 | 16.95/20.20/24.79 |
| B6 | first 18 | 4.43/5.65/7.74 | 9.60/12.22/14.97 | 0.52/0.90/1.31 | 0.52/0.67/1.12 | 16.92/20.55/25.79 |

这组结果确认：B5 相对 B4 的 matched-prefix total p95 仅下降约 1.3%，不足以
支持异步效率优势；B6 相对 B5 的额外开销主要来自 planner 与 QDR，不能用
并发负载解释。主矩阵中的并发计时仍保留用于历史分解，但效率结论以本节的
隔离结果为准。100 ms 不是硬门槛；无论是否低于 100 ms，都公开 p50/p95/p99。

复现入口：`scripts/benchmark_phase57_isolated_runtime.py`；聚合结果保留在
`results/phase57_isolated_runtime_benchmark/aggregate.json`，TensorBoard 位于
`results/phase57_isolated_runtime_benchmark/tensorboard/`。

## 4. 因果对比

以下比较按 scene 和 predictor seed 配对，并在 mirror group 内先平均，避免把
upper/lower 镜像当作独立样本。

| Scope | B1−B0 safe/collision | B5−B4 safe/collision | B6−B5 safe/collision | B6−B4 safe/collision |
| --- | ---: | ---: | ---: | ---: |
| all | +9.44 pp [3.89,15.00] / −9.44 pp | −6.94 pp [−10.28,−3.89] / +6.94 pp | +24.17 pp [18.61,29.44] / −29.17 pp | +17.22 pp [11.94,22.50] / −22.22 pp |
| id_replication | −7.78 pp [−17.78,−1.11] / +7.78 pp | −13.33 pp [−20.00,−6.67] / +13.33 pp | +15.56 pp [7.78,23.33] / −15.56 pp | +2.22 pp [−2.22,6.67] / −2.22 pp |
| delay/noise factorial | +17.78 pp [5.56,30.00] / −17.78 pp | +2.22 pp [0.00,6.67] / −2.22 pp | +24.44 pp [8.89,38.89] / −38.89 pp | +26.67 pp [12.22,41.11] / −41.11 pp |
| interaction stress | +37.78 pp [24.44,51.11] / −37.78 pp | −5.56 pp [−11.11,−1.11] / +5.56 pp | +45.56 pp [33.33,58.89] / −50.00 pp | +40.00 pp [26.67,53.33] / −44.44 pp |

当前最可信的表述是：QDR 的收益具有明显的条件性，在 delay/noise 和
interaction-stress block 中更明显；B5 相对 B4 没有显示异步规划的稳定收益；
B6 的组合增益较强，但不能把它改写成 QDR 单模块或 asynchronous 单模块已经
通过主分支验证。

## 5. 预注册 gate

| Gate | 状态 | 证据 |
| --- | --- | --- |
| ID safe-capture non-inferiority | FAIL | B1−B0 = −7.78 pp [−16.67,−1.11] |
| Stress collision reduction | PASS | B1−B0 collision = −27.78 pp [−37.78,−18.33] |
| Liveness | PASS | timeout CI upper = 0.00%，mean max exhaustion streak = 14.66 |
| Async efficiency | FAIL | B5 对 B4 的 total p95 reduction = 1.86%，且 safe-capture CI lower = −22.22% |
| Promotion | FAIL | 主分支 gate 未全部通过，B6 仍是 conditional diagnostic |

因此当前不打开 development-confirmation 或 locked-diagnostic 的调参流程。
失败 gate 是停止选择性调参的信号，不是继续扫描阈值的理由。

## 6. QDR 形式化边界

令决策时刻为 `t`，当前不可修改的执行队列长度为 `q_t`，队列前缀为
`u^queue_{t:t+q_t-1}`。在离散动力学 `x_{k+1}=F(x_k,u_k)` 下，QDR 定义为：

1. 用真实执行队列前缀推进当前公开状态，得到第一可控状态
   `x^c_t = F^{q_t}(x_t, u^queue_{t:t+q_t-1})`；
2. 将候选目标轨迹按同一个时间索引向后取 `q_t` 个位置，得到可控 suffix；
3. 只在 `x^c_t` 和可控 suffix 上进行滚动时域优化；
4. 新命令追加到队列末端，并由执行器唯一地施加剩余 delay。

不重复计算延迟的证明是时间索引等价性：队列前缀已经产生了
`F^{q_t}`，规划 suffix 使用的是同一条轨迹的 `t+q_t` 状态；因为动力学组合
满足 `F^{a+b}=F^b∘F^a`，再对 suffix 的新命令应用一次执行器 delay 只会重复
计算已有的 `q_t` 步。实现中的独立 checker 在队列长度 `0/2/4/8` 上的
position/velocity equivalence error 均为 0，并通过 double-delay 检查。

这个证明只覆盖 QDR 的时间索引和执行合同，不证明 collision-free、forward
invariance 或闭环安全。`local_cbf` 仍然只是经验过滤器；本报告不宣称
R-CLBF-QP、CLBF certificate 或真实飞行安全证明。

## 7. 下一阶段 To-do List

### P57-A：先修复 runtime 证据链（最高优先级）

- [x] 完成可复现的 single-process / process-isolated benchmark；每个方法使用
  相同场景、相同 predictor seed、固定 Torch `1/1` 线程和顺序独立子进程；
- [x] 对每个 method 报告 predictor/planner/QDR-or-tube/safety/total 的
  p50、p95、p99，分别给出 full-episode 和 first-18-step matched-prefix；
- [x] TensorBoard 记录 `RuntimeIsolated/{method}/{view}/{component}/{p50,p95,p99}`
  以及 prefix、seed、线程和运行合同；
- [x] 以隔离 benchmark 作为 B4/B5/B6 的效率证据：B5/B4 matched-prefix total
  p95 仅下降约 1.3%，异步效率 promotion 仍为 No-Go；B6 的额外 planner/QDR
  开销已被确认，后续只报告安全--计算量 Pareto。

### P57-B：补齐强基线的可审计因果矩阵

- [ ] 在新的 calibration manifest 中加入 B2 fixed-tube、B3 queue-aware-tube
  和 B7 fixed-K=8 QDR，与 B0/B1/B4/B5/B6 保持同一 scene/seed；
- [ ] 每个新的 delay/noise/communication block 至少 45 个 mirror groups，
  三个 split 各自保持 mirror-group disjoint；总规模不少于 300 episodes；
- [ ] 不在 Phase57 locked-diagnostic split 上调参；所有预算、tube radius、
  asynchronous interval 在运行前写入协议并写入 TensorBoard。

### P57-C：验证条件性假设，而不是包装组合结果

- [ ] 预注册主假设：QDR 在高 delay/noise 下降低 collision；async 单独不保证
  capture 或效率；QDR+async 的复合收益单独作为 interaction hypothesis；
- [ ] 重新定义 promotion gate：ID 不劣性、stress collision、timeout、最大
  exhaustion streak 和隔离 runtime 必须同时通过；
- [ ] 若 ID gate 再失败，冻结 QDR 为 stress-conditional module，停止 promotion，
  不把 B6 的 interaction gain 归因给单个模块；
- [ ] 若 gate 通过，才使用全新的 confirmation holdout，最后才允许运行
  locked-diagnostic；locked-test 仍不参与任何阈值选择。

### P57-D：扩展泛化轴

- [ ] 已完成的 geometry-shift OOD 作为独立 transfer axis 保留；
- [ ] 新增 unseen target behavior/speed block，只改变目标行为参数；
- [ ] 新增 stronger communication/execution delay-noise block，只改变通信或
  执行因素，避免多个未声明因素混杂；
- [ ] 每个 OOD block 都报告结果退化曲线、failure taxonomy、五类延迟分位数和
  TensorBoard 配置快照，不把单一 OOD 轴称为“全面鲁棒”。

### P57-E：安全措辞和论文材料

- [ ] 所有结果统一写 `local CBF empirical filter`；
- [ ] 单独保留 robust CBF-QP 的 No-Go/diagnostic 结果，不将其写成安全证书；
- [ ] 论文主张限定为“延迟队列感知的条件性闭环改进”和“可复现实验/诊断
  合同”，除非正式证明和硬件/仿真审计另行完成，不使用 `guaranteed safe`、
  `forward invariant` 或 `R-CLBF-QP proof`；
- [ ] 最终补充代码版本、manifest hash、checkpoint hash、完整命令、TensorBoard
  日志索引和失败案例视频。

## 8. 当前结论

Phase 57 已经完成了“规模化场景 + 因果因素拆分 + 五类延迟统计 + QDR 时间索引
审计”的开发阶段目标，但没有完成“主方法 promotion”。当前最有价值且可以
诚实写进论文的结果是：QDR 在高延迟/噪声应激情况下有条件的安全收益，异步单独
没有稳定收益，QDR+async 的组合结果需要隔离 runtime 和新的确认实验才能升级
为主结论。local CBF 的角色严格限定为经验过滤器。
