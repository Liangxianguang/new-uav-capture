# Phase 56：强基线、规模化压力场景与 QDR 形式化实验计划

## 0. 计划定位

Phase 56 的目标不是继续堆叠 predictor、UAKR、RNIC 或 robust CBF-QP，而是
把当前最有希望的 **Queue-Aware Delayed-State Rollout（QDR）** 做成可被审稿人
复核的独立方法：强基线公平、场景规模足够、时间索引有形式化定义、统计结果
完整，并且安全表述不越界。

本阶段只允许从以下主问题出发：

> 在动作队列不可立即修改、通信与执行存在延迟时，显式按 queue prefix 和
> first-controllable state 展开候选轨迹，是否比普通 delayed-MPC 更能降低
> collision/boundary，同时保持可接受的捕获率、活性和尾延迟？

UAKR 和 RNIC 在本阶段默认关闭；它们只能作为附录负消融，不得掩盖 QDR 的
单独贡献。local CBF 继续作为经验过滤器，不命名为 R-CLBF-QP，不作为闭环
安全证明的一部分。

## 1. 当前基线与待解决问题

当前证据已经显示：QDR 在 delay/execution OOD 上有明显安全收益，但同时存在
timeout、suffix exhaustion 和延迟尾部代价；而 UAKR/RNIC 尚未通过独立主结果
gate。因此 Phase 56 要解决的是“QDR 是否在公平强基线和更大场景上仍成立”，
而不是只重复已有 QDR-on/off 比较。

必须回答的三个问题：

1. QDR 的收益来自正确的 delayed-state 时间对齐，还是来自更保守的候选筛选？
2. tube-MPC 和 asynchronous distributed MPC 是否能以更低尾延迟获得相同安全
   收益？
3. 当 delay、command noise、dropout 和 tracking dynamics 分别增大时，QDR 的
   safety--liveness Pareto 边界在哪里？

## 2. 方法定义与公平比较合同

### 2.1 固定的共同条件

所有方法必须使用同一份场景 manifest、同一 checkpoint、同一 predictor
condition、同一采样 seed、同一 candidate 数量、同一 MPC horizon、同一安全
过滤器和同一执行器。每个方法使用独立 output directory，保存：

- effective config、source hash、manifest SHA-256、随机种子；
- episode/step JSONL、failure taxonomy 和 TensorBoard event；
- predictor、planner、QDR/tube、safety、total control 的 p50/p95/p99；
- communication message count、bytes、age、queue length 和 planner fallback。

默认主比较固定为：GRU `both`、distributed delayed DN-MPC、local CBF、
immutable command queue。UAKR、RNIC、escape-gap、FC-DBF 和 robust CBF-QP
关闭。100 ms 只作为参考线，绝不是硬 gate。

### 2.2 必须实现的强基线

| 编号 | 方法 | 关键定义 | 目的 |
| --- | --- | --- | --- |
| B0 | Current-state delayed-MPC | 从当前 belief 直接展开，延迟只在执行器中体现；保持现有候选和代价 | 现有普通基线 |
| B1 | QDR-MPC | 显式拼接 immutable queue prefix，候选从 first-controllable state 对齐 | 主方法 |
| B2 | Fixed-tube MPC | 用 calibration-only 的执行误差半径收紧 obstacle/boundary/inter-agent 约束；不使用 QDR 的 prefix/suffix gate | tube-MPC 强基线 |
| B3 | Queue-aware tube MPC | B2 的 tube 约束从 queue prefix 后的 controllable state 开始，但不使用 QDR 的风险触发器 | 分离“tube”与“时间对齐”贡献 |
| B4 | Synchronous distributed MPC | 当前同步通信与固定迭代次数，作为分布式效率参考 | 通信协调基线 |
| B5 | Asynchronous distributed MPC | 每架 UAV 独立时钟，事件触发更新；允许 stale peer plan，但必须显式记录 age 和 message order | 异步规划强基线 |
| B6 | QDR + asynchronous MPC | 只叠加 QDR 和 B5，不引入 UAKR/RNIC | 检验组合是否产生额外收益 |
| B7 | Fixed-K=8 + QDR | 每步固定 K=8、每步刷新，不使用 adaptive budget | 计算预算上界参考 |

B2 的 tube 半径只能从 development-calibration 数据估计；confirmation 和
locked-test 禁止重新估计。B5 的异步调度必须使用预先固定的 event-trigger、
最大消息年龄和本地更新时间，不能根据测试结果临时改变。

### 2.3 运行时公平性

- CPU 使用 Torch intra/inter-op threads `1/1`，另行保留 CUDA 结果但不与 CPU
  混合比较；
- 所有方法报告 wall-clock latency，而不是只报告算法内部计时；
- 每种方法使用相同的最大 planner wall-time budget 和 candidate path 上限；
- 对 cached predictor 必须同时报告 refresh rate、cache age 和 predictor 非零
  调用延迟，不能用缓存 step 的 0 ms 掩盖真实推理开销；
- 统计使用 scene mirror group 为重采样单位，不能把同一镜像组的 upper/lower
  当成独立随机样本。

## 3. 新场景与延迟/噪声矩阵

### 3.1 最低规模

新建完全不复用 v4 训练、validation 或 locked-test 几何/镜像组的
`phase56_strong_baseline` 数据集：

- 最低 `360 episodes / 180 mirror groups`；每组 upper/lower 成对；
- 三个 predictor seeds：`727201/727202/727203`；
- 建议 stretch 版本为 `480 episodes / 240 mirror groups`；
- geometry、target behavior、delay、dropout 和 execution noise 的随机源分开
 记录；
- manifest 必须通过 mirror-disjoint、target-truth isolation、action-timestamp
  和 public-belief audit。

### 3.2 三个互斥场景块

每个 block 使用独立 mirror groups；同一 block 内只改变预先声明的因素。

#### A. ID 主比较块（120 episodes / 60 groups）

- geometry 和 target behavior 位于训练范围；
- delay `4`、command noise `0.08`、nominal dropout；
- 比较 B0--B7；
- 只在该块的 development split 确认实施连通性，不据此调参。

#### B. Delay--noise 网格块（120 episodes / 60 groups）

固定 geometry、target behavior 和 dropout，只改变：

| Factor | Levels |
| --- | --- |
| command delay | `0, 2, 4, 6, 8` steps |
| command noise std | `0.00, 0.04, 0.08, 0.12 m/s` |

每个 cell 至少 3 个 mirror groups。该块用于绘制 safe capture、collision、
timeout、minimum clearance 和 latency 随 delay/noise 的曲线，不允许把最难的
cell 单独称为总体性能。

#### C. Communication/execution block（120 episodes / 60 groups）

固定 geometry 和 target behavior，只改变：

| Factor | Levels |
| --- | --- |
| message dropout | `0, 0.10, 0.20, 0.30` |
| message delay | `0, 2, 4` steps |
| tracking time constant | `0, 0.10 s`，按预先固定规则分配 |

该块检验同步/异步分布式规划是否因 stale peer plans 失效。若预算允许，额外
增加 `drag coefficient` 两个固定档位，但不能把新增因素和主结果混合。

### 3.3 Split 与锁定规则

最低 360 episodes 先按完整 mirror group 划分为 development-calibration、
development-confirmation、locked-diagnostic 三份；建议每份 60 groups / 120
episodes。所有 tube 半径、event-trigger、消息年龄上限和任何阈值只能在
calibration 选择；confirmation 只做一次冻结验证；locked-diagnostic 只在
代码、配置、checkpoint、manifest 和统计脚本全部冻结后开放。

如果按 block 分层导致每份样本不足，必须扩大到 stretch 版本，不得跨 split
借用镜像组或把 locked 结果拿回来选模型。

## 4. QDR 的形式化定义

### 4.1 状态、队列与执行动力学

令离散状态为 (x_t)，控制队列为

\[
Q_t=(u_t^0,u_t^1,\ldots,u_t^{d-1}),
\]

其中 `d` 是固定执行延迟，immutable 表示在时刻 (t) 之后这些动作不能被
规划器修改。执行动力学为

\[
x_{t+i+1}=f(x_{t+i},u_t^i,w_{t+i}), \quad 0\le i<d,
\]

其中 (w) 属于预先固定的执行扰动集合。候选 suffix 为

\[
U_t=(v_t^0,v_t^1,\ldots,v_t^{H-1}).
\]

QDR 的完整 rollout 是一次拼接：

\[
\mathcal R(x_t,Q_t,U_t)=
  (x_t,x_{t+1},\ldots,x_{t+d+H}),
\]

先执行 `Q_t`，再执行 `U_t`。候选 (v_t^k) 的物理作用时刻是

\[
t+d+k,
\]

而不是 (t+k)。

### 4.2 Prefix/suffix 可行性

定义 prefix 集合 (P_t=\{x_{t+i}:0\le i\le d\})，suffix 集合

\(S_t(U)=\{x_{t+d+i}:0\le i\le H\}\)。对几何安全函数
\(g_j(x)\ge0)，分别记录：

\[
\mathrm{PrefixSafe}(t)=\mathbf1[\min_{x\in P_t,j}g_j(x)\ge0],
\]

\[
\mathrm{SuffixSafe}(t,U)=\mathbf1[\min_{x\in S_t(U),j}g_j(x)\ge0].
\]

prefix unsafe 且没有可改变动作时，任何新 suffix 都不能改变已执行的 prefix；
该状态必须记录为 `prefix_unsafe_unrecoverable`，不能用一个“成功规划出的
suffix”覆盖这个事实。

### 4.3 不重复计算 delay 的命题

**命题（delay composition equivalence）：** 若 (f) 与扰动序列固定，且
QDR 只通过一次 (mathcal R(x_t,Q_t,U_t)) 进行 queue composition，那么从
时刻 (t) 计算候选 cost 与从 first-controllable state
\(\bar x_{t+d}=\mathcal R_d(x_t,Q_t)\) 计算 suffix cost 是等价的，只需把
候选时间索引平移 `d`；不允许再向 terminal time、arrival time 或候选状态
额外加一次 `d`。

证明草案：

1. `Q_t` 的状态序列唯一确定 \(x_{t+1:t+d}\)；
2. suffix rollout 的初始状态是同一 \(\bar x_{t+d}\)，因此后续状态递推逐项相同；
3. 完整 cost 按物理状态索引求和时，prefix cost 只出现一次，suffix cost 从
   `t+d` 开始；
4. 若额外把 `d` 加到候选时间或到达时间，就会让同一执行延迟被计算两次，形成
   可检测的时间索引偏差。

需要把该命题变成独立 checker，而不是只写成文档结论。

## 5. 形式化与实现 TODO

### 5.1 数据与代码

- [ ] 新建 `scripts/generate_phase56_strong_baseline_scenes.py`，输出三 block
  manifest、mirror group 和每个 factor cell；
- [ ] 新建 `configs/phase56_strong_baseline_protocol.yaml`，冻结 split、factor
  levels、seed 和信息隔离规则；
- [ ] 新建 `configs/phase56_qdr_formalization.yaml`，只打开 QDR，不打开 UAKR/RNIC；
- [ ] 新建 `configs/phase56_tube_mpc_baseline.yaml`，只使用 calibration-only
  tube 参数；
- [ ] 新建 `configs/phase56_async_mpc_baseline.yaml`，冻结 event-trigger、消息
  age 和本地调度规则；
- [ ] 为 B0--B7 写统一 runner，确保同一场景、checkpoint、采样与线程合同；
- [ ] 每个 runner 保存 effective config、source hash、manifest hash 和
  TensorBoard log directory。

### 5.2 QDR 时间索引 checker

- [ ] 为 integrator dynamics 构造手算可验证的 toy queue；
- [ ] 对 deterministic dynamics 逐项比较 full rollout 与 shifted rollout，
  误差门限固定为 `1e-9`；
- [ ] 对随机 bounded noise 做固定 seed 的 trajectory replay，检查索引而不是
  要求随机路径逐次相等；
- [ ] 检查候选第 `k` 个动作首次影响时间严格为 `t+d+k`；
- [ ] 检查 prefix cost、suffix cost、terminal cost 各只计一次；
- [ ] 检查 `d=0` 时 QDR 退化为普通 rollout；
- [ ] 检查 immutable prefix unsafe 时输出不可恢复分类；
- [ ] 将 checker 与 planner 实现解耦，不能读取 planner 内部 residual 作为证明；
- [ ] 在 JSONL 和 TensorBoard 记录 `qdr_prefix_horizon`、
  `qdr_first_controllable_step`、`qdr_time_index_error`、
  `qdr_prefix_admissible` 和 `qdr_suffix_admissible`。

### 5.3 Tube-MPC TODO

- [ ] 只在 development-calibration 上估计执行误差的 per-step radius；
- [ ] 明确 tube 是经验 bounded-error tightening，不是概率覆盖证明；
- [ ] 将 obstacle、boundary、inter-agent 约束分别收紧，记录每类约束被收紧的
  半径；
- [ ] 记录 tube rejection、fallback、exhaustion、timeout 和 effective plan rate；
- [ ] 做 nominal / fixed-tube / queue-aware-tube 三路消融，不混入 QDR 风险触发；
- [ ] 若 tube 导致安全提高但 timeout 大幅增加，报告 Pareto，不把保守性写成
  无条件优越性。

### 5.4 异步 distributed MPC TODO

- [ ] 为每架 UAV 建立独立 local clock 和 message sequence number；
- [ ] 固定 event-trigger：本地状态变化、peer-plan age 或 action delta 超阈值
  时才发送消息；
- [ ] 显式规定 stale plan 的 hold、discard 和 fallback 语义；
- [ ] 记录 message age、out-of-order message、discard count、communication bytes
  和每架 UAV 的 planner latency；
- [ ] 确认 asynchronous 方法不能读取全局 target truth 或未来 peer action；
- [ ] 在相同 wall-time budget 下比较 synchronous、asynchronous、QDR+async；
- [ ] 将 async 的收益拆成 communication saving、planner saving 和 safety
  effect，不能只报告总耗时。

## 6. 统计指标与预注册 gate

### 6.1 必报指标

Episode-level：

- safe capture、ordinary capture、collision、boundary violation、timeout；
- capture time、minimum clearance、path length、safe abort；
- prefix admissible、suffix admissible、ever-exhausted、maximum exhaustion streak；
- planner valid/effective/converged、fallback 和 failure category。

Step-level / runtime：

- predictor、planner、QDR/tube、safety、total control p50/p95/p99；
- refresh rate、cache age、candidate count、queue length；
- communication messages/bytes、message age、stale-plan ratio；
- QDR time-index error 和每类 barrier minimum。

### 6.2 预注册 gate

1. **实现 gate：** deterministic time-index equivalence error `≤1e-9`；无 target
   truth leakage；所有方法都能由 JSONL 重算主指标。
2. **ID non-inferiority gate：** QDR 相对 B0 的 paired safe-capture CI 下界
   `≥ -2 pp`；collision/boundary 不增加，或其 paired CI 跨 0。
3. **Stress safety gate：** 在预先指定的 delay `≥6` 或 noise `≥0.08` 子集上，
   QDR collision 的 paired point estimate 至少降低 `5 pp`，且 CI 不得完全为正。
4. **Liveness gate：** timeout paired 增量 CI 上界 `≤+5 pp`，maximum exhaustion
   streak `≤24`；若安全收益依赖更长 timeout，必须报告为 safety--liveness
   trade-off，不判为 Go。
5. **Async efficiency gate：** asynchronous 相对 synchronous 的 total p95 至少
   降低 `15%`，同时 ID safe capture 不低于 `-2 pp`；否则只作为负效率消融。
6. **Tube gate：** 不预设 tube 必须优于 QDR；必须完整报告 coverage、半径、
   timeout、collision 和尾延迟，防止用过宽 tube 制造虚假安全收益。
7. **Promotion gate：** 只有 B1 在 ID non-inferiority、stress safety 和
   liveness 三项同时通过，才允许将 QDR 写成主贡献；否则保留为条件性诊断。

所有 CI 使用 paired hierarchical bootstrap，重采样单位为 mirror group；多重
比较使用预先声明的 Holm 校正。任何 gate 失败都停止在该实验块，不在 locked
结果上回调阈值。

## 7. TensorBoard 记录合同

每个 run 至少写入：

```text
Protocol/manifest_sha256
Protocol/source_hash
Protocol/method_id
Protocol/delay_steps
Protocol/command_noise_std
Protocol/dropout_probability
Outcome/safe_capture_rate
Outcome/collision_rate
Outcome/boundary_violation_rate
Outcome/timeout_rate
QDR/prefix_admissible_rate
QDR/suffix_admissible_rate
QDR/maximum_exhaustion_streak
QDR/time_index_error
Runtime/predictor_p50_ms
Runtime/predictor_p95_ms
Runtime/predictor_p99_ms
Runtime/planner_p50_ms
Runtime/planner_p95_ms
Runtime/planner_p99_ms
Runtime/qdr_or_tube_p50_ms
Runtime/qdr_or_tube_p95_ms
Runtime/qdr_or_tube_p99_ms
Runtime/safety_p50_ms
Runtime/safety_p95_ms
Runtime/safety_p99_ms
Runtime/total_p50_ms
Runtime/total_p95_ms
Runtime/total_p99_ms
Communication/messages_sent
Communication/bytes_sent
Communication/message_age_steps
Gate/id_noninferiority
Gate/stress_safety
Gate/liveness
Gate/overall
```

TensorBoard 只记录配置和结果，不作为 gate 的唯一证据；正式报告必须引用
machine-readable JSON、episode/step JSONL 和独立 checker。

## 8. 六周执行顺序

| 周次 | 任务 | 交付物 | 停止条件 |
| --- | --- | --- | --- |
| 第 1 周 | manifest、split、factor 矩阵和公平合同 | protocol、scene hash、审计报告 | split 或信息隔离失败则重建 |
| 第 2 周 | B0/B2/B4/B5 强基线实现 | configs、统一 runner、unit tests | 计时/候选/执行合同不一致则不跑大实验 |
| 第 3 周 | QDR formalization 与 independent checker | 定义、等价性 checker、toy proof tests | time-index error > `1e-9` 则暂停闭环 |
| 第 4 周 | 360-episode 三种子 development/calibration | episode/step JSONL、TensorBoard | 只在 calibration 修正实现，不看 locked |
| 第 5 周 | confirmation 和分层统计 | aggregate JSON/Markdown、bootstrap CI | 任一 gate 失败即冻结该方法 |
| 第 6 周 | locked-diagnostic、图表和论文材料 | final table、failure taxonomy、limitations | locked 只能验证，不能调参 |

若计算预算不足，优先保留 B0、B1、B2、B5 四个方法和 360 episodes；不得把
场景规模减少到只剩 20/40 episodes 后再宣称强基线结论。

## 9. 论文表述边界

可以写：

> QDR provides explicit delayed-state alignment and auditable prefix/suffix
> feasibility under the benchmark's immutable queue and kinematic execution
> contract.

不可以写：

- “QDR 给出真实飞行安全保证”；
- “local CBF 是 R-CLBF-QP 或已经证明 forward invariance”；
- “tube coverage 等于概率安全证明”；
- “UAKR/RNIC 已经在当前实验中形成有效主贡献”；
- “所有 OOD 条件都泛化”；
- “完整 QDR×UAKR×RNIC 组合已经成功”。

## 10. 最终成功定义

Phase 56 只有在以下条件全部满足时，才支持一篇以 QDR 为主贡献的完整论文：

- 至少 360 个全新场景、180 个 mirror groups 和三个匹配 seeds；
- B0、tube-MPC、synchronous/ asynchronous distributed MPC 均完成公平比较；
- QDR time-index equivalence checker 通过；
- ID safe-capture non-inferiority、stress collision 和 liveness gate 同时通过；
- 所有方法都有 predictor/planner/QDR-or-tube/safety/total p50/p95/p99；
- 失败案例、timeout、exhaustion 和 runtime Pareto 均公开记录；
- local CBF 和 QDR 的安全表述严格限制在经验/条件性合同；
- 配置、源码、manifest hash、JSONL、TensorBoard 和正式报告可独立复现。

若任一条件失败，论文应改为“delay-aware distributed encirclement 的条件性
经验研究”，而不是宣称已经完成形式化安全闭环。

