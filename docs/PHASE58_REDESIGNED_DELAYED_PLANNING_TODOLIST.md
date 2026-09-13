# Phase 58：可复现的延迟感知多无人机拦截实验计划

## 0. 阶段定位

Phase 57 已经完成了强基线雏形、QDR 时间索引检查和 process-isolated runtime
benchmark，但 promotion gate 未全部通过：QDR 在 ID block 的 safe-capture
non-inferiority 失败，asynchronous 相对 synchronous 的尾延迟优势不足。因此
Phase 58 不再沿用“继续扫描阈值、直接扩大完整组合”的路线，而是重新建立一份
独立、可审计、可复现的 delayed-planning 证据集。

本阶段的核心问题是：

> 在不可立即修改的动作队列、通信延迟、执行噪声和多机信息不同步同时存在时，
> queue-aware delayed-state rollout 是否能够相对于充分实现的 delayed-MPC 和
> tube-MPC 强基线，在高压力条件下减少碰撞，同时不牺牲 ID 捕获率、闭环活性和
> 可解释的运行时成本？

本计划不预设 QDR、tube-MPC 或 asynchronous 一定有效。每个模块都必须通过
独立的因果对照；组合结果只能作为 interaction result，不能倒推单模块有效。

## 1. 研究假设与贡献边界

### H1：延迟时间对齐

QDR 使用真实 pending queue 先推进到 first-controllable state，再展开可控
suffix。预注册目标是在高 action-delay / execution-noise 条件下降低 collision，
同时 ID safe capture 相对于 current-state delayed-MPC 的 paired 95% bootstrap
区间下界不低于 `-2 pp`。

### H2：tube tightening 的独立作用

固定 tube-MPC 和 queue-aware tube-MPC 只改变约束/拦截代价，不启用 QDR 的风险
触发、suffix gate 或 adaptive K。该对照用于区分“保守 tube 带来的变化”和“队列
时间对齐带来的变化”。tube 参数只能由 calibration split 估计。

### H3：异步规划的真实收益

asynchronous distributed MPC 只有在相同硬件、相同线程、相同候选预算和相同通信
合同下，出现预注册的 p95 改善且 safe capture 不劣，才可以称为效率收益。若只
有行为变化而没有尾延迟改善，报告为行为/通信消融；若两者都没有，冻结为负结果。

### H4：QDR 形式化边界

QDR 的正式结论只覆盖 queue composition、first-controllable state、物理时间
索引和 delay 不重复计算。它不推出 collision-free、forward invariance、概率
覆盖率或真实飞行安全。

### H5：安全层表述

`local_cbf` 仅是经验动作过滤器。除非另行完成独立的动力学假设、可行性证明、
连续时间证书和长时闭环审计，否则不得使用 `R-CLBF-QP`、`CLBF certificate`、
`guaranteed safe`、`forward invariant` 或“安全证明”等表述。robust CBF-QP
继续作为 diagnostic No-Go 分支保留。

## 2. 交付物与仓库结构

### 2.1 必须新增或更新的文件

- `configs/phase58_delayed_planning_protocol.yaml`：唯一的场景、split、seed、
  延迟/噪声等级和 gate 配置来源；
- `configs/phase58_runtime_contract.yaml`：设备、Torch 线程、wall-clock、
  warm-up、缓存和计时边界；
- `scripts/generate_phase58_delayed_planning_scenes.py`：生成全新 mirror-disjoint
  场景和 manifest；
- `scripts/check_phase58_qdr_time_index.py`：独立于 planner 的 QDR checker；
- `scripts/evaluate_phase58_matrix.py`：统一 runner，强制所有方法使用相同输入；
- `scripts/aggregate_phase58_results.py`：按 mirror group、seed 和 block 聚合；
- `tests/test_generate_phase58_delayed_planning_scenes.py`；
- `tests/test_phase58_qdr_time_index.py`；
- `tests/test_phase58_runtime_contract.py`；
- `docs/PHASE58_REDESIGNED_DELAYED_PLANNING_REPORT.md`：实验完成后的权威报告；
- TensorBoard 日志目录：`results/phase58_*/tensorboard/`，results 本身不提交 Git。

### 2.2 所有结果必须保留的审计字段

每个 run 必须保存 effective config、source hash、manifest SHA-256、checkpoint
hash、predictor seed、sampling seed、split、block、method、device、Torch
intra/inter-op threads、git commit、运行开始时间和命令行。每个 episode/step
还必须能追溯：

- queue length、prefix horizon、first-controllable index、message age、
  stale-plan 比例和 planner fallback reason；
- target/obstacle/inter-agent/boundary 的失败 taxonomy；
- safe capture、ordinary capture、collision、boundary、timeout、minimum
  clearance、capture time、最大 exhaustion streak；
- predictor、planner、QDR/tube、safety、total 五类 latency 原始样本和分位数；
- local CBF 是否修改动作、修改幅度、不可行/回退次数；
- TensorBoard scalar、JSONL 和 summary.json 三者的字段一致性。

## 3. 全新场景矩阵：480 episodes / 240 mirror groups

### 3.1 规模与 split

本阶段最低不得低于 `360 episodes / 180 mirror groups`，目标规模为
`480 episodes / 240 mirror groups`。每个 mirror group 由 upper/lower 两个
镜像 episode 组成，四个 block 各使用 60 个独立 mirror groups：

| Split | Mirror groups | Episodes | 用途 |
| --- | ---: | ---: | --- |
| development-calibration | 80 | 160 | 只用于估计 tube、冻结预算和检查实现 |
| development-confirmation | 80 | 160 | 配置冻结后的单次确认，不再调参 |
| locked-diagnostic | 80 | 160 | 仅在 confirmation gate 通过后开放 |

split 以完整 mirror group 为单位划分，任何 layout、镜像、随机场景 ID、tube
校准样本或阈值不得跨 split。若计算资源只能完成最低规模，则每个 split 至少
60 groups / 120 episodes，并在报告中明确降级。

### 3.2 四个互斥 block

每个 block 60 groups / 120 episodes；每个 block 的主要因素必须单独写入
manifest，不能把未声明的因素混在同一因果结论中。

#### A. ID replication block

- 几何和目标行为均位于训练范围，但 mirror groups 全部为新布局；
- action delay 固定为 4 steps，command noise std 固定为 0.08 m/s；
- message delay/dropout 和 tracking dynamics 使用 nominal 配置；
- 运行完整强基线矩阵，作为 ID non-inferiority 和公平性参考。

#### B. delay-noise factorial block

固定 geometry、target behavior 和 communication，分别扫描执行条件：

| Factor | Levels |
| --- | --- |
| action delay | `0, 2, 4, 6, 8, 10` steps |
| command noise std | `0.00, 0.04, 0.08, 0.12, 0.16 m/s` |
| noise bound | 固定 `3 sigma`，不在测试集重新估计 |

使用平衡的 fractional-factorial 分配；边缘风险 level 和预先指定的交互 cell
至少 4 个 mirror groups，其余 fractional cell 至少 2 个 mirror groups。报告
完整 response surface，而不是只报告最难 cell 的总体均值。

#### C. communication/execution block

固定 geometry、target behavior 和 action delay 的 nominal 值，扫描信息不同步
和执行器跟踪因素：

| Factor | Levels |
| --- | --- |
| message delay | `0, 2, 4, 6` steps |
| message dropout | `0, 0.10, 0.20, 0.30` |
| velocity time constant | `0, 0.10, 0.20 s` |
| drag coefficient | `0, 0.05` |

异步 planner 必须显式记录 stale peer plan 的 age、消息顺序和本地更新时间，
不能把丢包结果解释成纯计算加速。

#### D. joint stress / transfer block

固定目标行为和生成规则，只将已在 A--C 单独出现过的高风险因素按协议组合：

- action delay 取 `8/10` steps；
- command noise std 取 `0.12/0.16 m/s`；
- message delay 取 `4/6` steps，dropout 取 `0.20/0.30`；
- 至少一半 groups 使用训练几何之外、但经过 route-validity checker 验证的布局；
- joint block 只用于测试交互效应，不作为单因素 superiority 证据。

生成器必须在写出 manifest 前完成 route validity、public-belief isolation、
action timestamp、message timestamp、mirror disjoint 和 factor-range audit。
无效路线要记录 rejection count 和原因，不能静默重采样到易场景。

## 4. 强基线和主方法合同

所有方法使用同一个 frozen checkpoint、同一采样 seed、同一 K 上限、同一 MPC
horizon、同一 action set、同一 safety layer 和同一执行器。建议方法如下：

| ID | 方法 | 允许改变的内容 | 不允许使用的内容 |
| --- | --- | --- | --- |
| M0 | current-state delayed-MPC | 当前公开 belief 直接展开 | QDR prefix、tube tightening |
| M1 | known-delay delayed-MPC | 已知 delay 的状态推进 | queue-aware suffix 风险 gate |
| M2 | fixed-tube MPC | calibration-only 固定 tube 收紧约束/代价 | QDR 时间对齐、adaptive trigger |
| M3 | queue-aware tube MPC | prefix 后开始应用 tube | QDR 风险触发器、adaptive K |
| M4 | synchronous distributed MPC | 固定消息轮次和同步时钟 | stale-plan 事件触发 |
| M5 | asynchronous distributed MPC | 固定 event trigger、最大 age、本地时钟 | 根据结果改变 trigger |
| M6 | QDR + synchronous MPC | M1/M3 的 QDR 与同步规划组合 | async-only 归因 |
| M7 | QDR + asynchronous MPC | QDR 与 M5 组合 | 将组合增益归因单模块 |
| M8 | fixed-K=8 QDR | 固定 K=8、固定刷新 | uncertainty-triggered adaptive K |
| M9 | oracle-communication diagnostic | 无延迟 peer information | 不能作为可部署主基线 |

M0/M1 用于判断“知道 delay 但不做 queue-aware suffix”是否已足够；M2/M3 用于
判断 tube 的贡献；M4/M5 用于判断异步机制；M6/M7 只报告 interaction；M8 用于
控制 adaptive budget 的影响；M9 仅作上界诊断。若实现资源有限，M0--M8 是必须
完成的主矩阵，M9 可以放到附录。

### 4.1 Tube-MPC 的公平性约束

- tube 半径只从 development-calibration 估计，并保存估计代码、样本量、分位点、
  每步半径和 obstacle/inter-agent/boundary 分项；
- fixed tube 与 queue-aware tube 使用相同半径、相同候选预算和相同 horizon；
- tube 是经验 bounded-error tightening，不是概率覆盖证明；
- 不能只把 tube radius 写入 metadata，必须进入实际约束或 interception cost，
  并用单元测试验证 objective/constraint 确实改变；
- 记录因 tube rejection 产生的 fallback、timeout 和 exhaustion，避免只报告
  collision 降低而隐藏活性损失。

### 4.2 异步分布式合同

运行前冻结以下内容：event trigger、最大消息年龄、消息队列容量、本地更新周期、
stale peer plan 的 hold/extrapolate 规则、丢包重试规则和 planner wall-time
budget。所有同步/异步方法必须在单进程 fixed-thread 和 process-isolated 两种
视图下运行；并发进程的结果不得用于效率比较。

## 5. QDR 形式化定义与“不重复计算 delay”证明

### 5.1 离散模型

在决策时刻 `t`，状态为 `x_t`，不可修改的 pending command queue 为：

```text
Q_t = (u_t^0, u_t^1, ..., u_t^(q-1))
```

其中 `q` 是当前真实 queue length，执行动力学为：

```text
x_(k+1) = F(x_k, u_k, w_k)
```

`w_k` 可以是固定 replay 的 bounded execution disturbance。新候选 suffix 为
`V_t = (v_t^0, ..., v_t^(H-1))`。QDR 只做一次：

```text
prefix      = Rollout(x_t, Q_t)
x_first_ctl = prefix[q]
suffix      = Rollout(x_first_ctl, V_t)
```

候选动作 `v_t^j` 的物理作用时刻严格为 `t + q + j`。新命令只追加到 queue
末端，由执行器合同负责后续唯一一次 delay。

### 5.2 命题

设 `C_q(x_t,Q_t)` 是执行 queue prefix 后得到的 first-controllable state，
`S(C_q,V_t)` 是从该状态展开 suffix 的结果。对于确定性 `F`、固定 `w` 和
按物理时刻求和的 additive cost：

```text
R_full(x_t,Q_t,V_t) = Prefix(x_t,Q_t) concat S(C_q(x_t,Q_t),V_t)
```

与“先得到 `C_q`，再从 `C_q` 展开 suffix 并将 candidate time index 平移 q”
是等价的。因为：

1. queue prefix 唯一确定 `x_(t+1), ..., x_(t+q)`；
2. suffix 的初始状态与 full rollout 的 `x_(t+q)` 相同；
3. 后续递推逐项相同，prefix cost 和 suffix cost 各只出现一次；
4. 对候选 arrival time、terminal time 或执行器再额外加 `q`，会将同一段
   immutable delay 计算两次，产生可检测的时间索引偏差。

该命题只证明 time-index/queue-composition equivalence，不证明候选轨迹安全。

### 5.3 独立 checker 必须通过的性质

- `q in {0, 2, 4, 8, 10}` 时，full rollout 与 shifted rollout 的位置/速度
  误差不超过 `1e-9`；
- 候选 `j` 的首次影响时间等于 `t+q+j`，不能是 `t+j` 或 `t+2q+j`；
- prefix、suffix、terminal cost 各只计一次；
- `q=0` 时退化为普通 rollout；
- 固定 noise replay 下检查索引和组成关系，不把不同随机路径误判为错误；
- immutable prefix unsafe 时输出 `prefix_unsafe_unrecoverable`，不得被新 suffix
  的成功规划覆盖；
- checker 不读取 planner residual、solver status 或事后 safety result 作为证明；
- checker 输出 JSON、pytest 断言和 TensorBoard：
  `qdr_prefix_horizon`、`qdr_first_controllable_step`、
  `qdr_time_index_error`、`qdr_double_delay_detected`、
  `qdr_prefix_admissible`、`qdr_suffix_admissible`。

## 6. 统一指标与 latency 计时合同

### 6.1 闭环结果

每个 block、method、seed 至少报告：safe capture、ordinary capture、collision、
boundary violation、timeout、capture time、minimum clearance、最大
exhaustion streak、fallback rate、tube rejection rate 和 stale-plan rate。
统计单位是 mirror group；upper/lower 先在 group 内配对，再做 seed/bootstrap
聚合。主结果必须同时给 point estimate、per-seed 值和 95% paired bootstrap CI。

### 6.2 五段 latency 定义

每个 control decision 从统一 `decision_start` 到动作输出结束，记录以下互斥
组件：

1. `predictor`：belief 编码、候选生成/采样和 predictor 后处理；
2. `planner`：MPC/DN-MPC/分布式优化，不包含 QDR prefix 或 tube preprocessing；
3. `QDR/tube`：queue composition、first-controllable rollout、tube 计算和
   tube-aware cost/constraint preprocessing；
4. `safety`：local CBF action filtering 和 fallback；
5. `total`：从 decision_start 到最终 command 的 wall-clock，包括无法归入前四
   段的 orchestration，但不能用组件相加替代真实 total。

对每一段和每一种 method 都必须报告 `p50 / p95 / p99`，单位毫秒，至少给出：

- full-episode pooled；
- first-18-step matched-prefix；
- 每个 block；
- 每个 seed；
- process-isolated fixed-thread 主视图。

缓存 predictor 必须分开记录 refresh latency、cached decision latency、cache age
和 refresh rate，不能用大量 0 ms 缓存步掩盖真实 predictor 成本。100 ms 只作为
部署参考线，不是硬门槛；超过参考线时必须给出安全收益与尾延迟的 Pareto 解释。

### 6.3 TensorBoard 命名规范

至少写入以下 scalar：

```text
Latency/{method}/{block}/{component}/{p50_ms,p95_ms,p99_ms}
Outcome/{method}/{block}/{safe_capture,collision,boundary,timeout}
Safety/{method}/{block}/{min_clearance,cbf_modified_rate,fallback_rate}
Queue/{method}/{block}/{mean_queue,max_queue,stale_plan_rate,exhaustion_streak}
QDR/{method}/{block}/{prefix_admissible,suffix_admissible,time_index_error}
RuntimeContract/{device,torch_threads,wall_time_budget,manifest_hash}
```

TensorBoard 只是可视化和审计入口，原始 JSONL 和 summary.json 仍是统计真值。

## 7. 分阶段执行 To-do List

### P58-0：协议冻结与预注册

- [ ] 新建 Phase 58 protocol、runtime contract 和 README 入口；
- [ ] 固定方法 ID、随机种子、checkpoint、candidate K、sampling steps、MPC
  horizon、event trigger、tube 估计规则和所有 gate；
- [ ] 生成 effective config 示例和完整复现命令；
- [ ] 在代码、配置和测试提交后才生成场景，不允许先看结果再改协议。

**退出条件：** `git diff --check` 通过；配置能被单元测试读取；所有 gate 和
方法开关均有唯一来源。

### P58-1：新场景生成与数据审计

- [ ] 生成 480 episodes / 240 mirror groups 的 A--D 四个 block；
- [ ] 记录每个 factor 的 level、block、layout ID、mirror group ID 和 rejection
  reason；
- [ ] 运行 mirror-disjoint、route-validity、target-truth isolation、public
  belief、action/message timestamp 和 factor-range audit；
- [ ] 计算并保存 manifest SHA-256；
- [ ] 用固定 seed 重生成一次，验证 manifest 内容字节级一致。

**退出条件：** 所有 split 审计通过；无 mirror group 跨 split；有效场景不少于
360 episodes，目标为 480 episodes。

### P58-2：实现并验证强基线

- [ ] 完成 M0/M1 delayed-MPC，明确两者的 time alignment 差异；
- [ ] 修复并验证 M2/M3 tube-aware objective/constraint 的真实生效；
- [ ] 完成 M4/M5 同步/异步 distributed planner；
- [ ] 加入 M6/M7 composite 和 M8 fixed-K=8 control；
- [ ] 每个方法通过相同输入、相同候选、相同执行器和相同 safety layer 的 smoke；
- [ ] 单元测试检查 disabled method 不改变历史行为，enabled method 改变预期
  objective 或 queue index。

**退出条件：** M0--M8 全部完成统一 runner；每种方法都有配置快照、source
hash、TensorBoard event 和 failure taxonomy。

### P58-3：QDR checker 与形式化边界

- [ ] 先运行独立 toy dynamics checker，再接入真实执行器 replay；
- [ ] 验证 `q=0/2/4/8/10`、固定 noise、不同 horizon 和 candidate index；
- [ ] 生成 checker report，明确“无重复 delay”通过不等于安全通过；
- [ ] 对真实 planner 运行 post-hoc index audit，但不能把 post-hoc audit 当作
  形式化证明替代品。

**退出条件：** checker pytest 通过，最大 time-index error ≤ `1e-9`，
`qdr_double_delay_detected=false`，且日志字段完整。

### P58-4：calibration 全矩阵

- [ ] 在 development-calibration 上完成 M0--M8 × 三个 predictor seeds；
- [ ] 单进程 fixed-thread 运行行为结果；process-isolated 顺序运行 latency；
- [ ] 估计 tube 半径、固定 planner budget 和 async 合同，只允许使用 calibration；
- [ ] 运行 mirror-group paired bootstrap 和 component latency aggregation；
- [ ] 形成 calibration report，标记每个假设 Go / No-Go / Inconclusive。

**退出条件：** 任何主方法没有完整五类 latency 分位数、失败 taxonomy 或
TensorBoard 配置快照时，不得进入 confirmation。

### P58-5：promotion gate 与一次性 confirmation

只有 calibration 同时满足下列条件，才允许打开 development-confirmation：

- [ ] ID safe-capture paired 95% CI 下界 ≥ `-2 pp`；
- [ ] joint-stress collision 相对 M0 或预先指定 strongest baseline 的改善
  达到至少 `5 pp`，且 paired CI 上界 < `0`；
- [ ] collision/boundary 不出现统计上显著的恶化；
- [ ] timeout ≤ `5%`，timeout delta ≤ `5 pp`，最大 exhaustion streak ≤ `24`；
- [ ] 五类 latency 均完整；不以 `total < 100 ms` 作为硬条件；
- [ ] asynchronous 只有在 total p95 至少下降 `15%` 且 safe capture 非劣时，
  才能宣称效率收益；否则保留为负/中性消融；
- [ ] local CBF 的所有结果都标为 empirical filter，robust CBF-QP 不参与安全
  证明或 promotion gate。

confirmation 只运行一次，使用未参与 calibration 的 80 mirror groups；任何
失败均冻结为 negative/conditional result，不回到 calibration 扫阈值。

### P58-6：locked-diagnostic 与论文材料

- [ ] 仅当 confirmation 通过，才在 locked-diagnostic 上运行最终冻结配置；
- [ ] locked 结果不得反向选择 predictor、tube、trigger、K 或 safety 参数；
- [ ] 生成主表、component latency 表、delay/noise response surface、失败案例
  视频、QDR checker appendix 和复现命令；
- [ ] 更新 `RESULTS_INDEX.md`、`EXPERIMENT_STATUS_AND_NEXT_TODOLIST.md` 和
  Phase 58 权威报告；
- [ ] 只提交源代码、配置、测试和文档，results/ 保留本地并由 hash 指向。

## 8. 最终判定规则

### Go：可作为主论文方法

必须同时满足 ID 非劣、joint-stress safety 改善、liveness 通过、强基线公平、
QDR checker 通过、五类 latency 完整且三 seed 方向一致。此时论文主张限定为
“延迟队列感知的条件性闭环改进”，仍不能称为形式化安全保证。

### Conditional Go：可作为条件性方法/系统论文

若 QDR 只在高 delay/noise 或 joint stress 有效，但 ID 性能不劣、liveness 和
审计链通过，则以“stress-conditional delayed planning”作为主结论，明确适用
边界，不夸大为全场景提升。

### No-Go：冻结为可靠负结果

若 ID non-inferiority、liveness、强基线公平或 runtime 审计失败，则停止在当前
数据上继续调参；保留完整 negative ablation、失败 taxonomy 和 Pareto 曲线，
转向更小、更可解释的模块创新，不重新包装组合结果。

## 9. 推荐执行顺序与时间安排

```text
P58-0 protocol / runtime contract freeze       2--3 days
    -> P58-1 480-episode scene generation       3--5 days
    -> P58-2 baseline implementation + tests    5--7 days
    -> P58-3 QDR checker + formal report         2--3 days
    -> P58-4 calibration matrix                  7--14 days
    -> gate review                               1 day
    -> P58-5 confirmation, if promoted           3--5 days
    -> P58-6 locked diagnostic + paper artifacts 3--5 days
```

每天结束时提交一个可复现的小阶段：源代码/配置/测试/文档单独 commit，报告
运行中的 TensorBoard 路径、manifest hash、当前完成 episode 数和未通过的 gate。
不提交大体积 `results/`，但必须保证从 commit、命令和 hash 能恢复对应本地结果。

## 10. 本阶段不做的事情

- 不把 100 ms 作为硬门槛，也不隐藏 p95/p99 尾延迟；
- 不在 locked-test 上调参、估计 tube、选择阈值或改变通信合同；
- 不把 local CBF 写成 R-CLBF-QP 或安全证明；
- 不把 QDR+async 的组合增益改写成 QDR 或 async 单模块增益；
- 不继续扩大 learned CLBF、robust CBF-QP 或端到端 Mamba/扩散组合，除非本
  阶段先通过独立安全合同和 liveness gate；
- 不通过增加难度来“做坏”对比方法。难度必须来自预先声明、可复现、对所有
  方法相同的 delay/noise/communication 条件。
