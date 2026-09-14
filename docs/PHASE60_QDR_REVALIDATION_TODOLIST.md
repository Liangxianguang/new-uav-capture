# Phase 60：Queue-Aware Delayed Planning 重新设计与可发表性验证计划

## 0. 本阶段结论先行

本阶段不把“配置中写入了 K=4/K=8”当作多模态实验完成。Phase 59 的 GRU
repair run 已暴露一个必须修复的可复现性问题：GRU 前向只返回一条均值轨迹，
因此 R2/R3 的实际候选数仍为 1。后续所有实验都必须同时记录：

```text
candidate_budget_requested
candidate_count_realized
candidate_budget_realized_rate
```

只有使用真正支持 `sample_set` 的 diffusion/SSM checkpoint 且 realized rate 为
100% 时，才能报告 K=4/K=8 的候选预算效应。

本阶段的研究问题重新收敛为：

> 在不可取消的执行队列、延迟/噪声和多机通信不同步条件下，QDR 是否能在公平的
> delayed-MPC、tube-MPC 和同步/异步分布式基线之上，降低碰撞或延迟失效，并保留
> 可解释的时间索引、候选预算和闭环运行时证据？

local CBF 只作为经验动作过滤器；robust CBF-QP/R-CLBF-QP 继续独立记录为
诊断 No-Go 分支，不作为本阶段主方法，也不宣称安全证明。

## 1. 研究假设与不允许的结论

### 1.1 可检验假设

- H1：QDR 只将 immutable queue prefix 执行一次，再从 first-controllable
  state 展开 suffix；独立 checker 在 `q=0,2,4,8,10` 上的位置、速度和物理时间
  索引误差均不超过 `1e-9`。
- H2：在 delay-noise 与 communication-execution block 中，QDR 相对于强
  delayed-MPC/tube-MPC 基线降低 collision 或 timeout，同时 ID safe capture
  的 paired 95% CI 下界不低于 `-2 pp`。
- H3：tube-MPC 的效果来自独立校准的 reachable tube，而不是隐含的 QDR queue
  对齐；fixed-tube 与 queue-aware-tube 必须使用相同半径、horizon 和候选预算。
- H4：异步规划只有在 process-isolated、固定 Torch `1/1` 线程、相同 episode
  前缀和相同 K 下 total p95 真正下降，才报告为效率收益；否则只报告行为消融。
- H5：K=4/K=8 的任何结论必须由 raw step log 证明真实候选数达到请求值；GRU
  的单均值轨迹不能充当多模态候选预算实验。

### 1.2 明确不允许写入论文的结论

- 不把 local CBF 写成 `R-CLBF-QP`、`CLBF certificate`、`guaranteed safe`
  或 forward-invariant safety proof。
- 不把 QDR 的时间索引等价性写成 collision-free 或概率覆盖保证。
- 不使用 locked-test 选择 checkpoint、阈值、tube 半径、K、刷新周期或失败案例。
- 不把单一最难场景的提升写成全域提升；joint stress 只用于交互效应。
- 不用平均 latency 代替 p50/p95/p99，也不把低频 prediction cache 的平均收益
  掩盖 candidate age 和 refresh rate。

## 2. 场景与数据协议：新增至少 360 episodes

### 2.1 固定规模与 split

目标为 **360 个全新 episodes / 180 个 mirror groups**，全部使用新的 layout
seed 和新的 episode seed；如果资源允许，扩展至 480 episodes / 240 groups。
upper/lower 镜像必须在同一 group 内，且整个 group 不能跨 split。

| split | mirror groups | episodes | 作用 |
|---|---:|---:|---|
| development_calibration | 60 | 120 | tube/预算/阈值的唯一估计来源 |
| development_confirmation | 60 | 120 | 协议冻结后的单次确认 |
| locked_diagnostic | 60 | 120 | confirmation 全部通过后才开放 |

每个 split 都包含四个 block，各 15 个 mirror groups；扩展到 480 时改为每个
split、每个 block 20 个 groups。任何被拒绝的 route 都记录 rejection reason、
尝试次数和 generator hash，不能静默替换成容易场景。

### 2.2 四个正交 block

每个 block 只改变一个主要因素集合，所有因素写入 manifest：

1. **ID reference**：训练几何/目标速度范围内的新 layout；action delay
   `4`，command noise `0.08 m/s`，message delay `2`，dropout `0.05`。
2. **delay-noise factorial**：action delay `0/2/4/6/8/10` steps，command
   noise std `0/0.04/0.08/0.12/0.16 m/s`，bound 固定 `3 sigma`；使用平衡
   fractional-factorial 分配，边缘 level 至少 4 groups。
3. **communication-execution**：message delay `0/2/4/6`，message dropout
   `0/0.10/0.20/0.30`，velocity time constant `0/0.10/0.20 s`，drag
   `0/0.05`；记录 stale peer plan age、消息顺序和本地更新时间。
4. **joint stress / transfer**：只组合前面已经单独出现过的高风险 level：
   action delay `8/10`、noise `0.12/0.16`、message delay `4/6`、dropout
   `0.20/0.30`，并加入经过 route-validity checker 的部分 geometry-OOD layout。

可选的第五个 transfer block 用于观测噪声/检测丢失：observation noise
`0.03/0.06/0.09`，detection dropout `0.15/0.25/0.35`；若加入，必须同步
增加 group 数量，不能挤占四个主 block 的配对样本。

### 2.3 数据审计清单

- [ ] `manifest.json` 包含 split、block、mirror group、geometry ranges、
  target behavior、delay/noise overrides、seed 和 SHA-256。
- [ ] 所有窗口只使用 public belief；target truth 只能用于离线 label/metric。
- [ ] executed/planned/commanded/delayed action、observation/message/communication
  timestamp 完整存在。
- [ ] split audit、mirror-disjoint audit、factor-range audit、route validity
  audit 全部通过。
- [ ] 新 manifest 固化后再开始 calibration；任何改动都生成新 manifest hash。

## 3. 强基线合同

所有方法复用同一场景、预测 checkpoint、采样 seed、MPC horizon、action limits、
执行器和 local CBF 开关；只允许改变下表声明的因素：

| ID | 基线/方法 | 允许改变 | 禁止使用 |
|---|---|---|---|
| B0 | strong delayed-MPC | current public belief + known fixed delay rollout | pending queue、QDR suffix gate |
| B1 | fixed-tube MPC | calibration-only tube radius | QDR prefix alignment、adaptive K |
| B2 | queue-aware tube MPC | immutable prefix + same tube radius | uncertainty-triggered K |
| B3 | synchronous distributed MPC | 固定通信轮次和 stale-message contract | async event trigger |
| B4 | asynchronous distributed MPC | 固定 event trigger、最大 age、hold/extrapolate | 根据结果改 trigger |
| Q0 | QDR + synchronous DN-MPC | 一次 prefix + shifted suffix | async-only attribution |
| Q1 | QDR + asynchronous DN-MPC | QDR 与 B4 组合 | 将组合提升归因单模块 |
| K4 | fixed K=4 QDR | 真正的 4-candidate sample set | adaptive K |
| K8 | fixed K=8 QDR | 真正的 8-candidate sample set | adaptive K |
| D0 | no/local-CBF diagnostic | none 与 local empirical filter | safety proof claim |
| S0 | robust CBF-QP diagnostic | 独立 safety path | 进入主成功结论 |

B0--B4 和 Q0--Q1 是主矩阵；K4/K8 作为候选预算消融；S0 只保留失败分解和
延迟，不参与“模型成功”判定。所有 baseline 运行都输出相同的五类时延字段。

## 4. QDR 形式化定义与不重复计算证明

### 4.1 离散执行模型

决策时刻 `t` 的状态为 `x_t`，immutable pending queue 为：

```text
Q_t = (u_t^0, ..., u_t^(q-1))
```

新规划 suffix 为 `V_t=(v_t^0,...,v_t^(H-1))`，执行动力学为：

```text
x_{k+1} = F(x_k, a_k, w_k)
```

其中 `a_{t+i}=u_t^i`（`0 <= i < q`），`a_{t+q+j}=v_t^j`。QDR 定义为：

```text
x_prefix = Rollout(x_t, Q_t)
x_first_controllable = x_prefix[q]
x_suffix = Rollout(x_first_controllable, V_t)
R_QDR = concat(x_prefix, x_suffix[1:])
```

因此 `v_t^j` 的首次物理影响时刻严格为 `t+q+j`。新动作只能追加到 queue
末端，不能由 planner 再次附加 delay；执行器的 delay 也不能再次平移 suffix。

### 4.2 命题与证明要点

对确定性 `F`、固定 replay disturbance `w` 和按物理时刻求和的 additive cost，
完整 rollout 与 QDR rollout 等价：

1. prefix 唯一确定 `x_{t+1},...,x_{t+q}`；
2. suffix 的初值等于完整 rollout 的 `x_{t+q}`；
3. suffix 后续递推逐项相同；
4. prefix cost 与 suffix cost 各计一次；
5. arrival/terminal time 只应用一次 `+q`。

如果 planner 把 queue 先平移 `q`，随后 executor 又把同一 candidate 再平移
`q`，会出现 `t+2q+j`，独立 timestamp checker 必须将该 probe 判为失败。

### 4.3 独立 checker 与单元测试

- [ ] `q=0,2,4,8,10`：full/shifted position、velocity、cost error `<=1e-9`。
- [ ] candidate `j` 首次作用时间为 `t+q+j`，不允许 `t+j` 或 `t+2q+j`。
- [ ] prefix/suffix/terminal cost 计数各一次。
- [ ] fixed disturbance replay 下检查 queue composition，而不是比较不同随机轨迹。
- [ ] deliberate double-delay probe 必须失败，并把失败类别写入 JSONL/TensorBoard。
- [ ] checker 不导入 planner 内部状态，避免“自己证明自己”。

## 5. 实验阶段与停止规则

### Phase 60.0：实现与协议审计

- [x] 保留 Phase58 的 480-episode 结果，作为已完成的历史证据。
- [x] 修复 evaluator 的 candidate budget requested/realized logging。
- [ ] 对 GRU、Diagonal-SSM、Official-S4 三类 checkpoint 做 1-episode smoke，
  确认 GRU 的 realized K=1、SSM 的 K=4/K=8 realized rate=100%。
- [ ] 为 R0--R3 增加 method contract、checkpoint kind、source hash 和版本号。
- [ ] 运行 targeted tests、`git diff --check`，提交第一阶段代码和协议。

**停止规则：** budget audit 不通过时，禁止解释 K4/K8 行为结果。

### Phase 60.1：生成新场景与数据审计

- [x] 生成 360 episodes / 180 mirror groups；资源允许则生成 480/240。
- [x] 固化 calibration/confirmation/locked split 和四个 block。
- [x] 完成 route validity、public-belief isolation、factor-range、mirror-disjoint
  audit，并将 manifest hash 写入 TensorBoard。
- [x] 只提交 generator、tests、protocol 和审计报告，不提交 results/ 大文件。

**停止规则：** 任一 split 泄漏、无效 route 静默重采样或 timestamp 缺失，重建
manifest，不进入模型实验。

### Phase 60.2：强基线 calibration

- [x] 在 development_calibration 上运行 M0--M8 强基线/QDR 矩阵；本轮 M8 为真实
  fixed-K8，M0--M7 为 K=1 方法合同。
- [x] 每个 predictor seed 至少 3 个；全部方法固定 seed 和 Torch `1/1`。
- [x] B1/B2 的 tube 只从 calibration 估计，不用 confirmation/locked。
- [x] 分 block 聚合 mirror-group bootstrap，并输出 paired CI。
- [ ] 对同步/异步做 process-isolated matched-prefix benchmark。

**停止规则：** 若 B0--B4 不能复现、方法 contract 不一致或 async 的硬件/线程
不一致，停止主方法比较，只修复基线合同。

### Phase 60.3：QDR 形式化与候选预算修复

- [x] 运行独立 QDR checker 全部 q level 和 double-delay negative probes。
- [ ] 用 Diagonal-SSM 或 Official-S4 重跑 R0/R1/R2/R3 三种子 calibration；本轮
  仅完成 M0--M8 主矩阵，未把缺少真实 K4 的 R2/R3 结果写入主结论。
- [x] 记录 requested K、realized count min/max、realized rate、mismatch steps。
- [ ] 只有 R2/R3 realized rate=100% 才生成 K4/K8 paired comparison；当前仅 M8
  fixed-K8 满足该条件。
- [x] 比较 QDR prefix/suffix admissibility、exhaustion first step、maximum streak、
  recovery count 和 timeout attribution。

**停止规则：** 任何 candidate mismatch、double-delay 或 QDR checker failure，
该版本结果只归档为 implementation diagnostic，不得用于论文主表。

### Phase 60.4：独立 confirmation

- [ ] 冻结 checkpoint、K、tube、阈值、refresh interval、async trigger 和 safety
  layer；只开放 development_confirmation。
- [ ] 运行 3 seeds × 全部 block × 主方法矩阵。
- [ ] 预先执行 gate：
  - ID safe-capture paired CI lower `>= -2 pp`；
  - joint collision 不高于参考基线，且优先报告 paired CI；
  - timeout `<=5%`，相对基线 timeout delta CI upper `<=5 pp`；
  - maximum QDR exhaustion streak `<=24` steps；
  - planner fallback、候选预算 mismatch、未解释 solver failure 均 `<1%`；
  - QDR checker 和 manifest audit 100% passed。
- [ ] 若任一 gate No-Go，冻结失败结果、写 failure taxonomy，不访问 locked-test。

### Phase 60.5：locked diagnostic 与论文证据

- [ ] 仅在 confirmation 全部通过后开放 locked diagnostic。
- [ ] 不重新调参；所有结果带 checkpoint/manifest/source hash。
- [ ] 生成主结果表、按 block 表、paired CI、失败案例、QDR checker、candidate
  budget audit 和 process-isolated latency 表。
- [ ] S0 robust CBF-QP 单独放入 limitation/diagnostic appendix，明确无闭环安全证明。

### Phase 60.6：adaptive-K 一致性与失败归因（本轮已完成）

- [x] 对 fixed-K 与 adaptive-K 分开定义 candidate-budget expected value。
- [x] budget 变化时强制刷新 cached candidate set，并记录 `budget_change` 原因。
- [x] 在 validation-only smoke 中验证真实 `K={1,4,8}`、refresh、cache age 和
  realized count；v1/v2 不一致结果归档为无效诊断，v3 才可引用实现结果。
- [x] 对 M0/M5/M6/M7/M8 做 block-aware failure taxonomy，并写入 TensorBoard。
- [ ] 对 prefix-unrecoverable recovery 做新的单变量 frozen ablation；在该实验
  通过 timeout、collision、max-streak 和 total-p95 门槛前，不进入 confirmation。

**当前停止规则：** Phase 60 calibration 的 joint timeout/max-streak gate 已经
No-Go；adaptive-K smoke 不能绕过该 gate，也不能打开 locked diagnostic。

## 6. 统一指标和 TensorBoard 合同

### 6.1 结果指标

每个 episode/每个 block/总体至少报告：

- safe capture、ordinary capture、collision、boundary violation、timeout；
- capture time、minimum clearance、path length；
- QDR queue length、first-controllable step、prefix/suffix admissibility、
  first exhaustion、maximum exhaustion streak、recovery count；
- candidate budget requested/realized、candidate mismatch steps；
- planner success/valid/fallback、message age/dropout、stale-plan ratio；
- local CBF correction norm、经验过滤器修改率、fallback/不可行类别。

### 6.2 五类 latency

每个 step 保存 raw sample，所有 summary 统一输出：

```text
predictor_latency_ms: p50 / p95 / p99
planner_latency_ms:   p50 / p95 / p99
qdr_or_tube_latency_ms: p50 / p95 / p99
safety_latency_ms:    p50 / p95 / p99
total_control_latency_ms: p50 / p95 / p99
```

计时边界必须写入 config：是否包含 tensor transfer、candidate projection、QDR
shift、planner communication、local CBF 和 environment step。并行进程只用于吞吐
诊断；行为与公平延迟比较使用 process-isolated fixed-thread matched prefix。
100 ms 仅为部署参考，不是硬 gate，但任何超过参考值的分位数都必须如实报告。

TensorBoard 最低 tag：

```text
Outcome/<method>/<block>/safe_capture
Outcome/<method>/<block>/collision
QDR/<method>/<block>/prefix_admissible_rate
QDR/<method>/<block>/suffix_exhaustion_rate
CandidateBudget/<method>/<block>/realized_rate
Latency/<method>/<block>/<component>/{p50,p95,p99}_ms
Safety/<method>/<block>/empirical_filter_rate
Gate/{qdr_checker,manifest,budget,confirmation,overall}
```

JSONL、summary.json、TensorBoard 三者的 key 和数值口径必须一致；缺失字段写
`null`，不能把 unavailable 当作 0。

## 7. 最终可发表判据与论文定位

### 7.1 达到可投稿证据线的最低条件

不是要求所有方法都赢，而是要求证据链闭合：

1. B0--B4 强基线合同可复现，且行为、延迟、消息和失败原因完整；
2. QDR checker 形式化时间索引命题通过，double-delay negative control 可检出；
3. QDR 在至少一个高风险 block 上相对强 baseline 给出稳定 paired collision/
   timeout 改善，同时 ID safe capture 不劣；
4. confirmation 的 3 seeds 方向一致，locked diagnostic 不出现明显反转；
5. K4/K8 的候选预算真实执行并有成本—收益曲线；
6. 所有 predictor/planner/QDR-or-tube/safety/total 的 p50/p95/p99 可复现；
7. local CBF 的 empirical-only 边界和 robust CBF-QP No-Go 限制写入方法与讨论。

### 7.2 若结果不满足时如何收敛

- QDR 只通过时间索引 checker、没有闭环收益：论文贡献降级为 delayed rollout
  correctness / reproducible benchmark，不宣称性能提升。
- QDR 降 collision 但 timeout 增加：报告 safety--liveness Pareto，重点分析
  queue exhaustion，不继续扫描到“好看的点”。
- K4/K8 无收益：保留为负消融，主方法使用 K=1 或说明多候选只在特定 risk
  mode 有效。
- async 无 p95 收益：报告信息时效性/行为差异，不称为加速方法。
- local CBF 降 collision 但降低 capture：报告经验过滤器 Pareto，不改称安全证书。
- confirmation No-Go：不开放 locked-test，直接整理 failure taxonomy 和下一版
  方案；这仍然是可复现、可信的研究结果。

## 8. 阶段性提交清单

- [ ] Commit A：candidate budget audit + tests + TensorBoard fields。
- [ ] Commit B：360/480 新场景 generator、manifest audit、protocol。
- [ ] Commit C：B0--B4 强基线与 process-isolated latency report。
- [ ] Commit D：QDR formal definition、independent checker、negative probes。
- [ ] Commit E：SSM K4/K8 repair calibration、block aggregate、gate report。
- [ ] Commit F：confirmation report；只有 gate 全过才创建 locked diagnostic commit。
- [ ] 每次提交前运行 `python -m pytest -q`、`python -m compileall src scripts tests`、
  `git diff --check`，只提交源码、配置、测试和 docs，保留本地 results/。
