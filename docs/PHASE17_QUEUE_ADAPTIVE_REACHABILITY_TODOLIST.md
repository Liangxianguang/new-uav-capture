# Phase 17: Queue-Aware Adaptive Reachability DN-MPC TodoList

> 状态：QDR、UAKR、RNIC 的第一版适配器、TensorBoard 字段和单元测试已实现；QDR validation paired gate 为 No-Go，UAKR validation 已完成但未通过安全非劣门槛，RNIC 在 ID、target-speed OOD 和 delay/execution OOD 均未形成主结果收益且增加延迟。Phase 18 的 delay-aware split-conformal reachable-tube 已完成实现与 validation-only smoke，但确认集完整轨迹覆盖率仅 85.92%（目标 90%），因此仍为 No-Go，暂不开放完整组合主结论。
>
> 核心目标：在不重新训练主预测器、不重新打开 learned CLBF/R-CLBF-QP 路线的前提下，将三个可复现机制接入现有 distributed delayed DN-MPC：
> 1. Queue-Aware Delayed-State Rollout（QDR）；
> 2. Uncertainty-Triggered Adaptive K and Replanning（UAKR）；
> 3. Reachability-Normalized Interception Cost（RNIC）。
>
> 本阶段不把 100 ms 作为硬门槛，但必须分别报告 predictor、queue rollout、planner、safety 和 total-control 的 p50/p95/p99。

## 1. 研究问题与当前基线

Phase 16 已经给出三个直接动机：

1. ID locked-test 上，GRU K=1 + distributed delayed DN-MPC + local CBF 的 safe capture 为 `94.81% [91.85%, 97.41%]`，collision/boundary 为 `0%/0%`，total p50/p95/p99 为 `25.34/43.58/47.67 ms`。
2. Diagonal SSM 和 official S4 的 K=8 在 distributed 分支均为 `95.56%` safe capture，但相对 GRU K=1 没有统计上确立的主分支增益；K=8 的明确收益只出现在保守 worst-case 分支。
3. target-speed OOD 的 distributed safe capture 降为 `42.67%`，其中 timeout 为 `57.33%`；delay/execution OOD 的 distributed safe capture 为 `47.00%`，collision/boundary 为 `53.00%/19.00%`。

因此 Phase 17 不再把“增加更多候选”当作默认改进，而回答三个可证伪问题：

- **RQ1：** 如果规划器显式滚动不可撤销的执行队列，能否降低延迟/执行 OOD 中由过期起点导致的碰撞和越界？
- **RQ2：** 如果只在高不确定状态使用 K=8 和每步刷新，能否在不损失 safe capture 的前提下降低平均预测调用与尾延迟？
- **RQ3：** 如果拦截代价由欧氏距离改为动力学可达时间裕量，能否减少高速目标场景中的不可达追逐和 timeout？

## 2. 方法边界与论文表述

### 2.1 本阶段允许声称的内容

- QDR 是执行队列感知的规划状态与固定前缀滚动机制。
- UAKR 是基于公开观测不确定性的计算预算调度机制。
- RNIC 是基于动力学到达时间裕量的拦截代价。
- 三个模块均可独立关闭、独立测试、独立记录延迟，并在同一场景上配对比较。

### 2.2 本阶段禁止声称的内容

- QDR 不是 reachable-set 证明，也不是 CBF certificate。
- 零噪声 nominal rollout 与经验误差界不能写成闭环安全证明。
- UAKR 不能被表述为新的扩散模型；它只是推理与刷新策略。
- RNIC 的一维加速度受限到达时间是 planner heuristic，除非后续给出完整三维最短时间证明，否则不能称为精确可达集。
- robust CBF-QP 仍为独立 No-Go 诊断，不进入 Phase 17 主结果。

当前验证状态：QDR=No-Go；UAKR=compute-savings-only/closed-loop No-Go；RNIC=ID/OOD No-Go；Phase 18 conformal tube=implementation complete but confirmation-coverage No-Go。

## 3. 冻结协议与防止测试泄漏

### 3.1 主模型与主控制器

- 主控制器固定为 `distributed_delayed DN-MPC + local CBF`。
- UAKR 主预测器固定为项目内可移植的 **Diagonal SSM conditional diffusion**，因为它支持运行时 K=1/4/8，且不依赖 official S4 的 CPU slow fallback。
- GRU K=1 保留为当前误差/延迟参考，不人为生成“GRU 多样本”。
- official S4 K=8 只做一次外部实现复现检查，不用于 Phase 17 参数选择。
- target truth 不得进入预测器、UAKR 分数、QDR 或 RNIC；只允许在 episode 结束后用于评测误差。

### 3.2 数据使用顺序

- [ ] 使用 Phase 16 validation scenes 和新建的 `phase17_development_stress` 场景进行开发与阈值选择。
- [ ] `phase17_development_stress` 只能由训练/验证布局组和新的开发 seed 生成，不能复制 locked-test 或最终确认集。
- [ ] Phase 16 已公开 OOD 结果只用于故障定位，不作为 Phase 17 的全新无偏结论。
- [ ] 在方法和阈值冻结前，生成并只做结构审计、不运行结果的 Phase 17 confirmation scenes。
- [ ] 冻结源码 commit、配置 SHA-256、scene SHA-256、checkpoint SHA-256 和 sampling seed 后，才运行 confirmation。
- [ ] confirmation 结果出来后禁止调阈值；若发现代码错误，必须修复、递增协议版本并完整重跑所有方法。

### 3.3 Phase 17 confirmation 规模

- ID：200 episode / 100 镜像组。
- geometry OOD：200 episode / 100 镜像组。
- target-speed OOD：200 episode / 100 镜像组，速度档保持单轴变化。
- delay/execution OOD：200 episode / 100 镜像组，几何和目标策略保持 ID。
- target-behavior OOD：200 episode / 100 镜像组。
- 每个方法使用 3 个模型/控制 seed；论文最终版本若资源允许扩展到 5 seed。
- 混合压力测试只有在三个单模块 gate 和完整组合 gate 均通过后才开放。

## 4. 创新点一：Queue-Aware Delayed-State Rollout（QDR）

### 4.1 明确定义

在决策时刻 `t`，从 policy-safe observation 中读取：

- defender positions/velocities；
- `execution.action_queue`；
- `action_delay_steps`；
- 速度上限、加速度上限、质量缩放、阻力和 tracking time constant；
- pending-command authority。

将已有队列视为固定控制前缀：

```text
current state x_t
  -> execute immutable q_t[0]
  -> execute immutable q_t[1]
  -> ...
  -> delayed planning state x_hat_(t+d)
  -> optimize the controllable suffix
```

DN-MPC 的总 rollout 仍从 `t` 开始计算成本和约束，但前 `d` 步不得由 optimizer 修改；新动作只影响首个实际可控时刻及其后缀。目标候选轨迹同步按相同时间索引切分，禁止把 `t+1` 的目标点错误地与 `t+d+1` 的 defender 状态配对。

### 4.2 实现 TodoList

- [ ] 在 `execution_dynamics.py` 增加 `rollout_queue_prefix(...)`，复用 `advance_execution(...)`，不得复制另一套动力学。
- [ ] 新增不可变 `DelayedPlanningState`，至少记录 prefix positions、prefix velocities、delayed start state、queue length、authority mode 和 nominal execution parameters。
- [ ] 明确 `delay >= planner horizon` 的处理：配置阶段拒绝，或进入有日志的 fallback；禁止静默截断。
- [ ] 在 `DistributedMinimaxDNMPC.plan(...)` 增加可关闭的 QDR 路径，默认关闭以保证旧结果可复现。
- [ ] local candidate rollout 的前缀使用 queue command，后缀才使用候选新动作序列。
- [ ] centralized diagnostic planner 使用同一 QDR helper，避免 central/distributed 两套语义。
- [ ] target candidate、peer message 和 defender prefix 使用同一绝对 timestep。
- [ ] QDR 只使用零均值 nominal execution rollout；噪声界只进入诊断字段，不能伪装成安全保证。
- [ ] 记录 `queue_length`、`first_controllable_step`、prefix minimum clearance、delayed-state position/velocity、rollout latency。
- [ ] episode 后使用 simulator truth 计算 delayed-state endpoint position/velocity error；该 truth 只用于评测。
- [ ] 配置字段建议为 `planner.queue_aware_rollout.enabled`，并完整写入结果快照。

### 4.3 QDR 单元与集成测试

- [ ] delay=0 时 QDR 与旧 planner observation 数值一致。
- [ ] 零噪声条件下，QDR prefix 与环境真实执行的逐步位置/速度一致。
- [ ] immutable 模式下 queue 内容逐元素保持不变。
- [ ] `replace_nonexecuting`/`flush_pending` 不得由 planner 自行升级 authority。
- [ ] 不同 mass/drag/tracking-alpha/acceleration-limit 分支均与 `advance_execution` 一致。
- [ ] queue shape、NaN、负 delay 和 horizon 不足均有明确异常。
- [ ] QDR 关闭时，同 seed 的 actions、episode metrics 和 source hash 与旧实现一致。
- [ ] 静态检查 QDR 调用链不访问 `target_position`、`target_velocity` 等 truth 字段。

### 4.4 QDR 独立实验

固定预测器、K、refresh、RNIC=off，仅比较：

- `Q0`：当前 planner；
- `Q1`：QDR nominal prefix；
- `Q2`：QDR + validation-frozen empirical execution bound（只作鲁棒诊断，不称证明）。

主要测试轴：action delay `0/1/2/4`、tracking time constant、command noise 和 message delay；每次只改变一个主因子。

## 5. 创新点二：Uncertainty-Triggered Adaptive K and Replanning（UAKR）

### 5.1 不确定性只来自推理前可用量

为了避免“先生成 K=8 才决定是否生成 K=8”的循环，主触发分数只能使用廉价且公开的信息：

- belief covariance trace；
- observation/message age；
- target visibility/confidence；
- 最近一次预测的 one-step residual；
- cached candidate age；
- target estimated speed / defender reachable speed；
- S4 branch score margin 或上一轮候选 endpoint dispersion（若上一轮存在）。

所有分量先用 validation 的冻结分位数归一化到 `[0,1]`：

```text
U_t = sum_j alpha_j * normalized_feature_j
```

第一版使用可解释线性权重，不训练额外网络。权重和阈值只在 validation 选择。

### 5.2 三档调度策略

建议初始候选为：

| 档位 | 条件 | K | prediction refresh interval |
| --- | --- | ---: | ---: |
| Low | `U_t < theta_low` | 1 | 4 |
| Medium | `theta_low <= U_t < theta_high` | 4 | 2 |
| High | `U_t >= theta_high` | 8 | 1 |

增加至少 0.05 的 hysteresis 或最短驻留时间，防止 K 在阈值附近逐步振荡。若缓存轨迹已经超过最大年龄、观测重新出现、目标速度突变或预测残差越界，则强制 High refresh。

### 5.3 实现 TodoList

- [ ] 新增 `AdaptivePredictionPolicy` 和只读 `AdaptivePredictionDecision` 数据类。
- [ ] 将 `PredictionRuntime.predict(...)` 扩展为每步接受动态 `num_samples` 与动态 refresh decision。
- [ ] sampling seed 必须由 `(episode_seed, step, selected_K, refresh_count)` 确定，不能依赖调用顺序。
- [ ] 缓存复用时同步推进候选轨迹；禁止把旧绝对轨迹原样当作当前未来。
- [ ] 缓存推进后重新做 horizon、bounds 和 dynamics-status 检查。
- [ ] 明确 GRU 不支持 K>1；GRU 路径固定 K=1，不用复制均值伪造多模态。
- [ ] UAKR 主结果使用 Diagonal SSM diffusion；official S4 只做 frozen-policy transfer check。
- [ ] 记录 uncertainty components、总分、档位、K、refresh/no-refresh、cache age、forced-refresh reason。
- [ ] 分别记录 predictor sampling latency、projection latency和调度器 latency。
- [ ] 配置字段建议为 `prediction.adaptive_budget.*`，关闭时保持现有固定 K/refresh 行为。

### 5.4 UAKR 测试

- [ ] covariance、message age、speed ratio 或 prediction residual 增大时，`U_t` 不下降。
- [ ] 相同输入和 seed 必须得到相同档位、K、refresh 和候选轨迹。
- [ ] hysteresis 测试确认阈值附近不逐步抖动。
- [ ] stale candidate 达到上限后必定刷新。
- [ ] 固定 Low/Medium/High 模式分别等价于固定 K=1/4/8 的参考配置。
- [ ] 缓存 shift 后 timestamp、horizon 和第一预测点符合当前控制时刻。
- [ ] 禁止使用当前或未来 target truth 计算 uncertainty。

### 5.5 UAKR 独立实验

QDR=off、RNIC=off，比较：

- fixed K=1 / refresh=1；
- fixed K=4 / refresh=2；
- fixed K=8 / refresh=1；
- adaptive K={1,4,8} / refresh={4,2,1}；
- adaptive K 但固定 refresh=1，用于分解候选数与缓存收益。

必须报告：safe capture、失败率、K 档位占比、mean K、refresh ratio、candidate age、predictor/total p50/p95/p99。运行时间比较必须在单进程、同硬件、同线程数条件下执行，不复用此前并发 CPU 批次的延迟。

## 6. 创新点三：Reachability-Normalized Interception Cost（RNIC）

### 6.1 可复现到达时间模型

对目标候选 `k`、未来时刻 `h` 和 defender `i` 的围捕 slot `s_(i,k,h)`，计算动力学最短到达时间近似：

```text
T_i,k,h = minimum_time_1d_accel_speed_limited(
    distance_to_slot,
    velocity_projected_on_slot_direction,
    defender_max_speed,
    defender_max_acceleration
)
```

该函数使用解析分段公式或固定容差二分法，必须确定性且不依赖 optimizer。定义到达裕量：

```text
slack_i,k,h = h * dt - T_i,k,h
```

RNIC 对负裕量施加归一化惩罚：

```text
J_reach(k) = min_h sum_i softplus((margin - slack_i,k,h) / time_scale)^2
```

同时保留 obstacle、boundary、inter-agent、control 和 formation 代价。RNIC 不能删除原有安全相关惩罚。

### 6.2 实现 TodoList

- [ ] 新增纯函数 `minimum_arrival_time(...)`，覆盖加速段、达到限速后的巡航段和退向目标的初速度。
- [ ] 新增 `interception_reachability_slack(...)`，批量返回 `[sequence, scenario, horizon, defender]`。
- [ ] 在 centralized 和 distributed scenario cost 中复用同一个向量化实现。
- [ ] 用 delayed execution parameters 的实际 episode max speed/acceleration，而不是固定写死训练值。
- [ ] QDR 打开时，RNIC 从首个可控状态和剩余时间计算；禁止重复扣除 queue delay。
- [ ] 用 tetrahedron role slots 计算团队到达裕量；另记录单 interceptor 到达裕量用于诊断。
- [ ] 对所有候选时刻计算 earliest feasible intercept step、feasible defender count 和 formation-ready count。
- [ ] RNIC 权重、time margin 和 smoothness temperature 只在 validation 选择。
- [ ] 记录 selected intercept step、minimum/mean slack、unreachable-slot ratio、role switch count 和 RNIC latency。
- [ ] 配置字段建议为 `planner.reachability_interception.*`，关闭时完全恢复原距离代价。

### 6.3 RNIC 测试

- [ ] 距离增加时 arrival time 单调不减。
- [ ] max speed 或 max acceleration 增加时 arrival time 单调不增。
- [ ] 当前已经在 slot 内时 arrival time 为 0。
- [ ] 不同方向的旋转对称测试给出相同时间。
- [ ] 解析分段点附近连续且数值有限。
- [ ] QDR delay 只计算一次，无 double-counting。
- [ ] 向量化实现与逐元素参考实现一致。
- [ ] RNIC 关闭时 scenario costs 与旧实现逐元素一致。

### 6.4 RNIC 独立实验

QDR=off、UAKR=fixed K，比较：

- Euclidean distance cost；
- distance + relative-speed cost（当前实现）；
- RNIC interceptor-only；
- RNIC formation-slots（建议主版本）。

重点使用 validation speed stress 档，并按目标速度、分支、首次可行拦截时刻和 timeout 原因分层分析。

## 7. 三模块组合与完整消融

在三个单模块均通过 pilot gate 后，运行 2×2×2 全因子矩阵：

| ID | QDR | UAKR | RNIC |
| --- | --- | --- | --- |
| A000 | off | fixed K=8/R=1 | off |
| A100 | on | fixed K=8/R=1 | off |
| A010 | off | adaptive | off |
| A001 | off | fixed K=8/R=1 | on |
| A110 | on | adaptive | off |
| A101 | on | fixed K=8/R=1 | on |
| A011 | off | adaptive | on |
| A111 | on | adaptive | on |

额外保留两个锚点：

- GRU K=1 + current distributed delayed + local CBF；
- Diagonal SSM fixed K=1 + current distributed delayed + local CBF。

### 7.1 交互效应

- [ ] 检查 QDR×RNIC：delayed state 是否让到达裕量估计更准确。
- [ ] 检查 UAKR×RNIC：高速不可达性是否能触发 High 档，而不是只增加 horizon cost。
- [ ] 检查 QDR×UAKR：queue rollout latency 是否抵消减少预测调用的收益。
- [ ] 检查三阶交互：完整组合是否出现单模块都改善但组合退化。
- [ ] 若 A111 失败，保留表现最好的简单子组合，不能强行保留全部模块。

## 8. 指标与统计协议

### 8.1 闭环主指标

- safe capture；
- ordinary capture；
- collision；
- boundary violation；
- timeout；
- capture time；
- minimum clearance；
- path length、control effort 和 action jerk；
- planner success/fallback/local timeout。

### 8.2 模块专属指标

QDR：

- delayed-state endpoint position/velocity error；
- prefix minimum clearance；
- queue length 和 first controllable step；
- prefix violation count。

UAKR：

- mean K、K=1/4/8 占比；
- refresh ratio、forced-refresh ratio、candidate age；
- uncertainty 对下一步预测失败/闭环失败的 AUROC 与 reliability bins；
- 每个档位的 safe capture 和预测误差。

RNIC：

- selected intercept reachability slack；
- unreachable-slot ratio；
- earliest feasible intercept step；
- predicted-versus-realized arrival-time error；
- interceptor/role switch count。

### 8.3 延迟指标

每个 run 必须分别报告：

- predictor p50/p95/p99；
- candidate projection p50/p95/p99；
- UAKR scheduler p50/p95/p99；
- QDR p50/p95/p99；
- RNIC p50/p95/p99；
- DN-MPC p50/p95/p99；
- local CBF p50/p95/p99；
- total-control p50/p95/p99。

100 ms 只作为参考线，不是 Go/No-Go 硬门槛。不得用均值代替尾延迟，也不得把并发运行和单进程运行直接比较。

### 8.4 统计方法

- [ ] 所有方法使用同一 scene、episode seed、model seed 和 sampling seed 配对。
- [ ] 以 mirror group 为重采样单元，model seed 作为上层层级，执行 10,000 次 hierarchical paired bootstrap。
- [ ] 报告比例指标和方法差值的 95% CI。
- [ ] 三个主要假设使用 Holm 校正：QDR 对 delay collision、UAKR 对性能非劣与计算量、RNIC 对 speed timeout。
- [ ] ID safe capture 使用 `-2.0 pp` 非劣界；不能只凭点估计认定不退化。
- [ ] 同时报告各 seed 方向，禁止只汇总有利 seed。

## 9. 预注册 Go/No-Go 门槛

### 9.1 单模块 pilot gate

QDR Go：

- delay/execution development stress 中 collision 至少下降 20 pp，且 paired 95% CI 上界小于 0；
- safe capture 至少提升 10 pp，或在安全显著改善时保持非劣；
- nominal zero-noise prefix endpoint 与环境 rollout 通过一致性测试。

UAKR Go：

- 相对 fixed K=8/R=1，safe capture 的 paired 95% CI 下界不低于 `-2 pp`；
- refresh ratio 至少下降 25%，mean K 至少下降 25%；
- 单进程 predictor 或 total p95 不高于 fixed K=8/R=1；
- high-uncertainty bin 的实际预测失败率高于 low bin，证明触发分数有区分度。

RNIC Go：

- speed development stress 中 safe capture 至少提升 10 pp 或 timeout 至少下降 15 pp；
- collision/boundary 不得增加超过 1 pp；
- selected intercept 的 realized arrival-time error 相对当前距离排序显著降低。

### 9.2 完整组合强结果目标

这些是强论文目标，不是结果预设：

- ID safe capture 不低于 93%，且相对冻结参考满足 `-2 pp` 非劣；
- geometry OOD safe capture 目标不低于 88%；
- target-speed OOD safe capture 目标不低于 65%，timeout 目标低于 35%；
- delay/execution OOD safe capture 目标不低于 70%，collision 目标低于 10%，boundary 目标低于 5%；
- target-behavior OOD 不得出现超过 3 pp 的描述性退化；
- 3 个 seed 方向一致，无无法解释的 solver/fallback 失败；
- 所有组件和 total-control 的 p50/p95/p99 完整可追溯。

若 A111 未通过但某个两模块组合通过，应把论文方法简化为通过的组合；不能因为“三个创新点已经写好”而保留无收益模块。

## 10. 实验执行顺序

### M0：协议冻结与基线复现

- [ ] 建立 `configs/phase17_queue_adaptive_reachability.yaml`。
- [ ] 建立统一结果 schema 和 source-hash manifest。
- [ ] 在 8 episode smoke 上复现 QDR/UAKR/RNIC 全关闭的旧行为。
- [ ] 在 validation 子集重跑固定 GRU K=1 和 Diagonal SSM K=8 基线。
- [ ] 写明 Phase 16 OOD 已经公开，只能作为开发诊断。

**M0 gate：** 模块全关闭时行为一致，日志字段完整，未访问 target truth。

### M1：QDR

- [ ] 完成 4.2 全部实现项。
- [ ] 完成 4.3 全部测试。
- [ ] 运行 8/24/90 episode 三级 smoke→pilot→validation。
- [ ] 分解 delay、tracking、noise 和 message age。
- [ ] 给出 QDR Go/No-Go。

### M2：UAKR

- [ ] 完成 5.3 全部实现项。
- [ ] 完成 5.4 全部测试。
- [ ] 仅用 validation 冻结 normalization quantiles、weights、thresholds 和 hysteresis。
- [ ] 运行固定预算与自适应预算公平单进程 benchmark。
- [ ] 给出 UAKR Go/No-Go。

**Current M2 decision:** No-Go for promotion. The three-seed validation
achieves compute savings but the paired safe-capture CI does not establish
non-inferiority. See `PHASE17_UAKR_VALIDATION_REPORT.md`.

### M3：RNIC

- [ ] 完成 6.2 全部实现项。
- [ ] 完成 6.3 全部测试。
- [ ] 在速度 development stress 上冻结 margin、time scale 和 cost weight。
- [ ] 分解 interceptor-only 与 formation-slot RNIC。
- [ ] 给出 RNIC Go/No-Go。

**Current M3 decision:** No-Go on ID, target-speed OOD and delay/execution OOD.
The RNIC diagnostic schema is implemented and TensorBoard-verified, but no
failure axis improves robustly and latency increases; the
predicted-versus-realized arrival audit remains pending. See
`PHASE17_RNIC_VALIDATION_REPORT.md`,
`PHASE17_RNIC_TARGET_SPEED_OOD_REPORT.md` and
`PHASE17_RNIC_DELAY_EXECUTION_OOD_REPORT.md`.

### M3.1：Phase 18 delay-aware conformal reachable tube

- [x] 实现冻结的 horizon-dependent split-conformal 半径序列，并按 queue length 与 prediction age 对齐。
- [x] 校准只使用 validation 前半 episode seeds，第二半作为未参与拟合的 confirmation；locked-test 未读取。
- [x] 接入 centralized/local RNIC、UAKR tube-width diagnostic、TensorBoard 和 per-step JSONL。
- [x] 完成 8-episode online smoke，验证管半径与候选 horizon 对齐并生成完整工件。
- [x] 明确记录失败：calibration full-trajectory coverage `90.18%`，confirmation `85.92%`，半径约 `6.09--8.59 m`，online 最大约 `10.11 m`。
- [ ] 不得将该版本写成安全证明或正式 OOD 主结果；先完成按 motion mode / delay regime 的条件校准与轨迹级 coverage--volume 双门槛。

**Current M3.1 decision:** No-Go for promotion. The candidate is retained as a
reproducible negative/repair result; its confirmation artifact must remain
frozen while a new development-only calibration version is prepared. Full
QDR × UAKR × RNIC factorial evaluation remains blocked.

### M4：全因子 validation

- [ ] 运行 A000--A111 的 2×2×2 同场景配对矩阵。
- [ ] 统计三个二阶交互和三阶交互。
- [ ] 按预注册规则选择最简单的通过配置。
- [ ] 冻结最终配置、checkpoint、source commit 和 seeds。

### M5：未见 confirmation

- [ ] 审计 confirmation scene balance、镜像组、单轴范围和 SHA-256。
- [ ] 先跑 ID，再跑 geometry、speed、delay/execution、behavior OOD。
- [ ] 不根据中间结果停跑、换 checkpoint 或调阈值。
- [ ] 聚合三 seed paired bootstrap 和 Holm-corrected 结论。
- [ ] 对所有 collision、boundary、timeout 建立失败案例清单。

### M6：运行时、视频与报告

- [ ] 单进程固定 CPU 线程数重跑延迟 benchmark。
- [ ] 每个场景选择成功、timeout、collision/near-miss 各至少一个 episode 制作视频。
- [ ] 视频保留时间、K/refresh 档位、queue length、intercept slack、最近距离和捕获状态。
- [ ] 完成 Phase 17 方法报告、结果报告、README 限制和结果索引。
- [ ] 保留所有正式/失败结果，不删除 No-Go 工件。

## 11. 建议代码与工件布局

```text
src/encirclement3d/execution_dynamics.py       # rollout_queue_prefix，复用执行模型
src/encirclement3d/adaptive_prediction.py      # UAKR 纯策略与诊断
src/encirclement3d/reachability_interception.py# 到达时间与 RNIC 批量代价
src/encirclement3d/distributed_dn_mpc.py       # 三模块接入点
src/encirclement3d/minimax_mpc.py              # centralized 共享接入与 config

configs/phase17_queue_adaptive_reachability.yaml
scripts/generate_phase17_confirmation_scenes.py
scripts/evaluate_phase17_queue_adaptive_reachability.py
scripts/aggregate_phase17_results.py

tests/test_phase17_queue_rollout.py
tests/test_phase17_adaptive_prediction.py
tests/test_phase17_reachability_interception.py
tests/test_phase17_integration.py

docs/PHASE17_QUEUE_ADAPTIVE_REACHABILITY_TODOLIST.md
docs/PHASE17_QUEUE_ADAPTIVE_REACHABILITY_REPORT.md
```

结果目录继续放在 Git-ignored `results/`，正式提交只包含源码、配置、测试和报告。

## 12. 复现清单

- [ ] 每个正式 run 保存 effective config、scene file、episode/step JSONL、summary、TensorBoard、checkpoint hash、source hash 和命令行。
- [ ] 所有自适应阈值都写进配置，不允许隐藏常量。
- [ ] 所有随机选择都接受显式 seed。
- [ ] adaptive decision 和 selected action 可从 step log 独立重放。
- [ ] 运行 `python -m pytest -q`。
- [ ] 运行 Python compile check。
- [ ] 运行 `git diff --check`。
- [ ] 文档中同时报告成功与失败，不删除 No-Go 结果。
- [ ] 每个阶段形成独立 conventional commit 并推送远端。

## 13. 六周目标日程

| 周次 | 目标 | 决策点 |
| --- | --- | --- |
| Week 1 | M0 + QDR helper、测试、smoke | 旧行为是否可精确复现 |
| Week 2 | QDR validation 与失败分解 | QDR Go/No-Go |
| Week 3 | UAKR、缓存时间对齐和公平 runtime | UAKR Go/No-Go |
| Week 4 | RNIC 与速度 stress | RNIC Go/No-Go |
| Week 5 | 2×2×2 全因子 validation、冻结方法 | 选择最简单通过组合 |
| Week 6 | confirmation、统计、视频和正式报告 | Phase 17 总 Go/Conditional Go/No-Go |

## 14. 推荐提交节点

```text
docs(phase17): preregister queue adaptive reachability plan
feat(phase17): add queue-aware delayed-state rollout
feat(phase17): add uncertainty-triggered prediction budget
feat(phase17): add reachability-normalized interception cost
test(phase17): cover queue adaptive reachability contracts
docs(phase17): report validation factor ablation
docs(phase17): report frozen confirmation results
```

## 15. 最终成功定义

Phase 17 的成功不要求三个模块全部保留，也不要求 100 ms。成功必须同时满足：

1. 至少一个模块在与其对应的失败轴上取得配对统计支持的改善；
2. 最终组合在 ID 上满足 safe-capture 非劣，且不增加 collision/boundary；
3. speed 与 delay/execution OOD 至少一个主要瓶颈得到显著修复，另一个不能恶化；
4. 自适应策略确实减少平均 K 或刷新次数，并完整报告 p50/p95/p99；
5. 所有改进都能通过公开配置、固定 seed、逐步日志和独立测试复现；
6. 未通过的模块被删除或降级为消融，不包装为贡献。

如果 QDR、UAKR、RNIC 都未通过各自 pilot gate，Phase 17 应以可复现的负结果结束，并保留 Phase 16 的 GRU K=1 + distributed delayed DN-MPC + local CBF 作为主模型，不继续扩大组合实验。

## 16. 2026-09-12 diagnostic checkpoint and revised next gate

- [x] Complete RNIC post-hoc reliability analysis on the three-seed
  delay/execution-OOD block. The pooled collision AUROC of the logged nominal
  slack is `0.504`, and collision rates over ascending risk quartiles are
  `53.33%`, `65.33%`, `61.33%`, and `58.67%`; the score is not a calibrated
  failure predictor. See `PHASE17_RNIC_DIAGNOSTIC_ANALYSIS.md`.
- [x] Record the RNIC diagnostic as a No-Go subclaim. It must not be described
  as a safety certificate or used to justify a stronger closed-loop claim.
- [ ] Freeze a development-only, horizon-dependent conformal residual
  calibration for the next candidate reachable tube. The calibration input
  may use public-belief prediction residuals, queue length, observation/message
  age, and bounded execution noise, but never locked-test outcomes.
- [ ] Define a single inflation rule from the calibrated tube to the planner:
  candidate budget, interception cost, and post-hoc coverage must use the same
  radius; add a fail-open fallback when the tube is infeasible.
- [ ] Run unit tests and a 20-episode development pilot. Stop if coverage is
  below the target or if the fallback changes the baseline's collision rate in
  the wrong direction.
- [ ] Freeze thresholds and run an untouched confirmation block with paired
  scenes. Report safe capture, collision, boundary, timeout, clearance,
  coverage, fallback rate, and component/total p50/p95/p99 latency.
- [ ] Only if this candidate passes its gate may a reduced 2×2 ablation be
  opened. Otherwise retain Phase 16 as the primary model and present Phase 17
  as a reproducible failure analysis.
