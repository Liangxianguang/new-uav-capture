# 三个创新点完整 TodoList 目标计划书

> 版本：v1.2
> 更新时间：2026-09-13
> 适用仓库：[Liangxianguang/new-uav-capture](https://github.com/Liangxianguang/new-uav-capture)
> 研究对象：部分观测、通信/执行延迟和高机动目标下的多无人机围捕拦截

## 0. 先给结论

这三个点可以组成一条逻辑完整的研究主线，但不能仅凭模块名称就宣称“足够新”或“可以发表”。真正需要证明的是：

1. **Queue-Aware Delayed-State Rollout（QDR）** 是否确实解决了“规划器看到的状态不是实际将要执行的状态”这一延迟—队列错位问题；
2. **Uncertainty-Triggered Adaptive K and Replanning（UAKR）** 是否在不降低安全捕获率的前提下减少扩散候选数和预测刷新次数；
3. **Reachability-Normalized Interception Cost（RNIC）** 是否把“几何上更近”修正为“在剩余时间和动力学约束下更可达”，并改善高速/延迟 OOD；
4. 三者组合后是否仍然保持收益，而不是单模块有效、组合以后相互抵消。

本计划将“新颖性”定义为一个可验证的系统机制：

> 规划器在每一步显式使用待执行命令队列形成延迟状态滚动预测；再依据公开观测下的风险和候选分歧自适应分配预测预算，并以多无人机动力学可达裕量归一化拦截代价。

这比单独把 SSM、扩散、MPC、CBF 拼接起来更容易形成清晰贡献，也更容易复现和做消融。

---

## 1. 论文级研究问题与两句话表述

### 1.1 研究问题

在多无人机围捕中，目标状态只能通过带延迟、丢包和执行误差的公共 belief 获得，规划器还必须面对目标机动的不确定性。现有方法通常把延迟当作固定时间偏移，把多模态预测候选数固定，把拦截代价直接写成欧氏距离；这三种简化会导致规划状态、计算预算和物理可达性不一致。

### 1.2 两句话 pitch

**问题句：** 延迟和高机动目标使多无人机规划器同时面临状态错位、预测预算浪费和“近但不可达”的拦截点，单纯增加候选轨迹或优化迭代不能稳定解决这三个问题。
**方法句：** 我们提出一个可审计的 QDR–UAKR–RNIC 闭环：用命令队列生成将要执行的延迟状态，用不确定性触发候选数/重规划预算，用动力学可达时间裕量归一化拦截代价，从而在固定安全控制接口下提升延迟和 OOD 条件下的安全捕获效率。

### 1.3 不允许的表述

- 不能把 QDR 描述成一般意义上的“延迟补偿”，必须报告队列长度、命令年龄和实际执行前缀；
- 不能把 UAKR 描述成新的扩散模型，它是预测推理预算与刷新策略；
- 不能把 RNIC 的 heuristic 代价写成 reachable-set 证明，除非另行完成可达集计算和独立验证；
- 不能把 local CBF 或当前 robust CBF-QP 诊断结果写成闭环安全证明；
- 不能只报告普通 capture，主指标必须是排除 collision、boundary violation 和 timeout 污染后的 **safe capture**。

---

## 2. 统一系统定义与信息隔离

### 2.1 延迟执行系统

采用离散时间模型：

\[
x_{t+1}=f(x_t,u^{\mathrm{exec}}_t,w_t),\qquad
o_t=\mathcal{O}(x_{0:t})+\nu_t,
\]

其中 `w_t` 表示执行扰动，`o_t` 是公开观测。控制器不能直接访问目标真值，也不能用仿真器内部 target state 构造在线预测输入。

每个防御者维护：

```text
current_state
last_executed_action
pending_command_queue
command_age / observation_age / message_age
execution_noise_summary
```

QDR 的输入必须来自上述公开、可执行状态；目标真值只允许在 episode 结束后用于评测。

### 2.2 QDR 的目标接口

规划时不再从理想当前状态直接 rollout，而是从“当前实际状态 + 已进入执行队列的动作前缀”开始：

\[
\hat x_{t+j|t}=F\left(\hat x_t,
  Q_t[0:j],u^{\mathrm{new}}_{t+j},\hat w_t\right).
\]

必须同时保存：

- `queue_prefix_length`；
- 每个候选动作的计划执行时间和实际执行时间；
- prediction age、command age、observation/message age；
- rollout 使用的是 `planned`、`executed` 还是 `commanded` action；
- 队列 authority：新命令能否替换、追加、取消或 flush pending command。

### 2.3 UAKR 的目标接口

第一版只允许使用公开 belief 和预测输出计算不确定性，不使用 target truth。推荐先固定：

```text
K ∈ {1, 4, 8}
replanning interval ∈ {4, 2, 1}
```

不确定性可以由候选轨迹离散度、时间一致性、belief 年龄和模型残差组成，但每一个特征必须单独写入 step-level log。残差触发的在线干预已经出现负结果，后续不能继续无依据扫描阈值；如果重新研究，必须先校准“干预是否有效”，而不是只校准“残差是否相关”。

### 2.4 RNIC 的目标接口

对于防御者 `i` 和候选拦截槽位 `q`，定义：

\[
\tau^{\mathrm{req}}_{i,q}
 = \text{从当前延迟执行状态到槽位 }q\text{ 的最短可行到达时间},
\]

\[
\tau^{\mathrm{avail}}_{q}
 = \text{目标预测到达槽位 }q\text{ 的可用时间}.
\]

推荐第一版使用无量纲时间裕量：

\[
s_{i,q}=\frac{\tau^{\mathrm{avail}}_q-
\tau^{\mathrm{req}}_{i,q}}
{\max(\tau^{\mathrm{avail}}_q,\epsilon)},
\qquad
C_{\mathrm{RNIC}}(i,q)=
\left[\frac{\max(0,-s_{i,q})}{1+\epsilon}\right]^2.
\]

`τ_req` 必须包含队列占用、加速度/速度上限和必要的制动时间；不能用自由瞬移或只用欧氏距离估计。多无人机版本还要加入 formation-slot assignment 或 cooperative occupancy penalty，否则只能声称 interceptor-only cost。

---

## 3. 当前证据对新计划的约束

仓库现有结果不能被跳过，应当作为新计划的开发基线：

| 模块 | 已有证据 | 当前判断 | 计划含义 |
| --- | --- | --- | --- |
| QDR | 历史 validation QDR-on/off 为 `88.89%/94.44%`；Phase34 修复 planner 广播 bug 后 QDR-on 为 `85.0%` safe capture、`15.0%` collision；Phase35 加入 suffix gate 与有限 recovery candidates 后达到 `100.0%/0%`；Phase36 批量化 rollout 后 total p95 降至 `105.27 ms`，但 gate exhaustion `21.76%` 且相对 QDR-off `46.53 ms` 仍超门槛 | Safety outcome repaired, runtime improved, promotion No-Go | 继续做候选早停/缓存和增量 rollout，再做 fresh delay/authority/noise confirmation；不把当前 gate 写成安全证明 |
| UAKR | 三种子验证平均 `K≈2.68`、刷新率约 `36.6%`，但 ID safe-capture 非劣 CI 未过；残差触发 high-K 和 refresh-only 也均为负消融 | No-Go for closed-loop promotion | 保留为效率/失败分析方向；若重开，必须做 intervention-effect calibration |
| RNIC | ID 行为基本不变，target-speed/delay OOD 变差并增加延迟；slack 对 collision 的 pooled AUROC `0.504` | No-Go | 先修复 reachable-time label 和 slot-level 定义，再做 planner 消融 |
| 主参考 | GRU + distributed delayed DN-MPC + local CBF 的 locked-test safe capture `94.81%`，collision/boundary `0/0` | 当前主参考 | 所有新模块都必须与它配对比较 |
| 安全层 | robust CBF-QP 一步证书可通过，但端到端组合曾出现 `50%` safe capture、`49%` collision | 诊断 No-Go | 不能把它作为三个创新点的成功前提，也不能用失败安全层掩盖规划结果 |

因此，下面的计划是“可复现修复与重新验证计划”，不是把既有 No-Go 结果改写成成功。

### 3.1 执行回写：当前三点的真实证据

截至 2026-09-13，RNIC formation-slot 已完成 fresh ID 与 distributed-delayed
确认，结果决定了后续计划的分支：

- fresh centralized ID：RNIC-off / formation-slot 的 safe capture 为
  `87.22% / 89.44%`；paired delta `+2.22 pp [-0.56,+5.00]`，支持预注册
  `-2 pp` 非劣方向，但置信区间跨 0，不能宣称 superiority；
- fresh distributed delayed：三臂均为 `96.67%` safe capture、`3.33%`
  collision，formation 没有改变 episode outcomes；formation pooled total
  p50/p95/p99 为 `205.76/220.87/235.10 ms`，RNIC-off 为
  `75.10/82.40/89.86 ms`；
- 因此 RNIC 当前结论是“centralized ID 非劣方向、distributed 无行为增益且
  有显著计算代价”，保留为可复现机制/负消融，不晋级为主性能贡献；
- QDR 和 UAKR 的既有 No-Go 结论仍有效。除非先完成新的契约修复或
  intervention-effect calibration，否则不开放三模块 Full 组合；
- 上述结果来自 fresh validation manifest
  `5aaaa79c6dbef2d346ff57d48d238a34fe911d3534ebb576dff0252ba0593022`，不
  读取 locked-test。完整数值见 `docs/PHASE27_RNIC_FORMATION_SLOT_PILOT_REPORT.md`。

据此，计划的实际执行顺序调整为：先冻结 Phase 27 负结果并完成独立安全审计，
再决定是否投入 RNIC communication-aware redesign；只有 QDR/UAKR/RNIC 三个模块
分别在独立 confirmation 上通过对应 gate，才运行 Full factorial。若没有模块
通过，也可以以“延迟感知分布式围捕的模块化框架与组合失效边界”为论文主线，不能
把 Full 失败隐藏掉。

### 3.2 支撑性候选：freshness--covariance public-belief fusion

该模块不是第四个主创新点，而是一个可复现的支撑机制：QDR 和 RNIC 的输入若仍把
所有 public belief 当作同等可靠，队列延迟、丢包和协方差差异就会被规划器错误地
忽略。第一版采用确定性、无学习参数的融合规则：

```text
reliability_i ∝ confidence_i^p
                  × exp(-age_decay × message_age_i)
                  / trace(inflated_covariance_i)
```

并用 effective sample size 触发确定性的 self-anchor fallback。它只使用 public
belief、message age、dropout、confidence 和 covariance，不读取 target truth；不能
把该规则写成贝叶斯最优融合或安全证明。

Phase 32 的 one-seed/20-episode development pilot 给出一个值得验证但尚未成立的
信号：worst-case safe capture `85%→90%`、collision `15%→10%`，distributed
delayed episode outcome 保持 `95%/5%`。冻结参数后的 fresh 40-episode/20-group
confirmation 已完成：worst-case safe capture 为 `87.50%→86.67%`，paired delta
`-0.83 pp [-5.00,+1.67]`，collision 为 `12.50%→13.33%`；distributed delayed
两臂均为 `97.50%/2.50%`。因此该版本未通过 ID non-inferiority gate，已降级为
可复现 negative/engineering ablation，不再把它作为 QDR 或 RNIC 的支撑臂，也不
在该 holdout 上继续调权重。完整的 predictor/planner/safety/total
p50/p95/p99、TensorBoard 和 artifact 记录见
`docs/PHASE32_FRESHNESS_COVARIANCE_CONFIRMATION_REPORT.md`。若未来提出新的
public-belief fusion 假设，必须建立新的 development split 和新的预注册 gate。

完整 pilot 记录见 `docs/PHASE32_FRESHNESS_COVARIANCE_PILOT_REPORT.md`。

### 3.3 当前 QDR 契约修复状态

Phase 33 已完成一个只读的 prefix/suffix precondition audit：它沿用共享执行动力学
滚动新规划 suffix，并把已进入执行队列的 prefix 与新 suffix 分开判定。该审计明确
区分以下四种状态：

```text
prefix_safe_suffix_safe
prefix_safe_suffix_unsafe
prefix_unsafe_recoverable
prefix_unsafe_unrecoverable
```

这一步只解决“失败发生在已提交 prefix 还是新 suffix”的可观测性问题，**不改变
默认 authority、不自动 flush/replace 队列，也不提供 reachable-set 或安全证明**。
因此后续完整计划仍按 P2 QDR gate 执行：先在 fresh validation 上做
`delay ∈ {0,1,2,4}`、authority 和 bounded execution-noise 的单因素确认，再决定
是否开放 QDR×UAKR×RNIC 组合。详细 smoke 证据见
`docs/PHASE33_QDR_PRECONDITION_AUDIT_REPORT.md`。

### 3.4 Phase 34 QDR execution-aware repair

Phase 34 修复了 QDR 分支中单个队友执行轨迹的维度错误：`[H,1,3]` 被错误地
参与了与 `[C,H,3]` 的广播，导致 distributed planner 每步 fallback。修复后新增
完整动作序列执行 rollout、QDR 前视障碍范围以及 fallback 原因日志，完整回归为
`264 passed`。不过修复后的独立 development block 仍显示 QDR-on 低于 QDR-off：
queue-aware safety projection 开启时 safe capture `85.0%`、collision `15.0%`，
QDR-off 为 `100.0%/0%`；prefix/suffix admissible rate 为 `95.93%/44.63%`，
total p95 为 `118.60 ms` 对 `46.53 ms`。因此实现层已具备可复现性，但 QDR 还
没有性能晋级资格。下一步不是继续调阈值，而是在候选生成/best-response 阶段加入
suffix-feasibility gate，明确 immutable prefix 的不可修复边界，再按单因素做
delay、authority 和 bounded execution-noise confirmation。详细数值见
`docs/PHASE34_QDR_EXECUTION_AWARE_VALIDATION_REPORT.md`。

### 3.5 Phase 35 QDR suffix gate 与 recovery candidates

Phase 35 在 centralized/distributed candidate stage 加入 suffix-feasibility gate，
并增加零动作与当前速度保持 recovery candidates。相同 40-episode development
block 上，QDR-on safe capture 从未修复版本的 `87.5%` 恢复到 `100.0%`，collision
从 `12.5%` 降到 `0%`，suffix admissible rate 从 `48.03%` 提升到 `86.80%`。
但 gate exhaustion 仍为 `21.76%`，mean capture time 为 `3.915 s`，total
`p50/p95/p99=107.44/150.21/168.13 ms`，相对 QDR-off 的 `27.84/46.53/51.59 ms`
明显变慢。因此当前判断是“开发块安全 outcome 通过、效率晋级 No-Go”；当所有
候选均不可行时系统仍选择最小违反候选，gate 不构成 safety certificate，也不能
修复 immutable prefix。完整结果见
`docs/PHASE35_QDR_SUFFIX_GATE_RECOVERY_REPORT.md`。

### 3.6 Phase 36 QDR batched execution-rollout

Phase 36 将本地候选执行 dynamics rollout 改为候选维度批量计算，并用逐候选参考
实现做数值等价测试。相同 40-episode development block 上，safe capture 仍为
`100.0%`，collision/boundary/timeout 仍为 `0%`，而 planner/total p95 从
Phase35 的 `127.46/150.21 ms` 降到 `82.39/105.27 ms`。由于相对历史 QDR-off 的
`46.53 ms` 仍远超 `15%` 增幅门槛，当前只认定为 runtime optimization pass，
不认定为 QDR performance promotion。详细结果见
`docs/PHASE36_QDR_BATCHED_ROLLOUT_REPORT.md`。

### 3.7 Phase 38--40 QDR rollout cache 与候选压缩

Phase 38--40 继续保持 QDR 的信息边界和 immutable queue authority，只消除同一
控制步内的重复计算：local candidate execution rollout 使用 step-local cache，
peer action path 使用相同参数/初始状态/action bytes 的 cache；随后移除与 weighted
reference 完全相同的候选。40-episode development block 上，safe capture 仍为
`100.0%`，collision/boundary/timeout 仍为 `0%`，suffix admissible `86.80%`，
gate exhaustion `21.76%`。Phase40 的 planner/total p95 为 `40.43/63.77 ms`，
相对 Phase36 total p95 `105.27 ms` 下降约 `39.4%`，但相对 QDR-off `46.53 ms`
仍高约 `37.1%`，没有通过预注册的 `+15%` 相对延迟门槛。因此该阶段是可复现的
runtime optimization pass，不是 QDR promotion；immutable prefix 仍可能不可恢复，
也没有产生 safety proof。完整结果见
`docs/PHASE38_40_QDR_RUNTIME_COMPRESSION_REPORT.md`。

### 3.8 Phase 41 当前源码 QDR-off 配对基线

Phase 41 显式关闭 `queue_aware_rollout` 和 `queue_aware_safety_projection`，在同一
40-episode development block 上重新获得 QDR-off reference：safe capture `100%`，
collision/boundary/timeout `0%`，total p50/p95/p99 为
`24.44/43.36/47.23 ms`。与 Phase40 QDR-on 的 `43.61/63.77/72.40 ms` 配对后，
total p95 增幅约 `47.1%`，因此 QDR 的相对效率 gate 仍为 No-Go。该阶段只确认当前
源码 baseline 和 CLI 信息边界，不构成新的方法增益或安全证明。详见
`docs/PHASE41_QDR_CURRENT_SOURCE_REFERENCE_REPORT.md`。

### 3.9 Phase 42 delay4 fresh validation

在独立 80-episode/40-mirror-group fresh manifest 上，只把 command delay 从 2 提高到
4，QDR-off/on 的 safe capture 为 `77.5%/97.5%`，collision 为 `22.5%/2.5%`；配对
bootstrap delta 分别为 `+20.0 pp [11.25,30.00]` 与 `-20.0 pp [-30.00,-11.25]`。
QDR-on 的 total p50/p95/p99 为 `45.88/78.65/95.10 ms`，off 为
`29.19/54.29/61.93 ms`，说明 QDR 在延迟压力轴上有实质安全收益，但付出约 `44.9%`
的 total p95 代价。由于只有一个 checkpoint seed、仍有 `2.5%` collision 和
`19.16%` gate exhaustion，本阶段只记为 conditional fresh-axis evidence；必须先
完成另外两种子和 bounded-noise confirmation，再决定是否 promotion。详见
`docs/PHASE42_QDR_DELAY4_FRESH_VALIDATION_REPORT.md`。

### 3.10 Phase 43 delay4 三种子 confirmation

在同一独立 fresh manifest 上补齐 `727202/727203` 后，三种子、240 episodes/arm 的
QDR-off/on safe capture 为 `77.08%/96.67%`，collision 为 `22.92%/1.25%`，paired
bootstrap delta 分别为 `+19.58 pp [13.75,25.42]` 与 `-21.67 pp [-27.92,-15.83]`。
QDR-on 的 timeout 为 `2.08%`，suffix gate exhaustion 均值 `19.02%`；三种子 summary
total p95 均值为 `90.14 ms`，off 为 `64.65 ms`，相对增幅约 `39.4%`。因此 delay4
安全效果得到条件性三种子支持，但效率和 liveness gate 仍未通过；下一步需固定
delay4、单独加入 bounded execution noise，再单独测试 authority，禁止混杂调参。详见
`docs/PHASE43_QDR_DELAY4_THREE_SEED_CONFIRMATION_REPORT.md`。

### 3.11 Phase 44 delay4 + bounded execution noise confirmation

固定 delay4 和 immutable authority，只加入 command-noise std `0.08 m/s`、`3σ`
clipping 后，三种子、240 episodes/arm 的 QDR-off/on safe capture 为
`77.92%/97.92%`，collision 为 `22.08%/0.83%`；paired delta 分别为
`+20.00 pp [14.17,25.83]` 与 `-21.25 pp [-27.50,-15.00]`。QDR-on timeout 为
`1.25%`，suffix gate exhaustion 均值 `19.99%`；三种子 summary total p95 均值
`123.91 ms`，off 为 `92.49 ms`。该结果支持 QDR 在延迟和有界执行扰动下改善安全
outcome，但 efficiency/liveness promotion 仍为 No-Go；下一步只改变 authority，
不能与 noise 或 target behavior 同时改变。详见
`docs/PHASE44_QDR_DELAY4_NOISE008_CONFIRMATION_REPORT.md`。

### 3.12 Phase 45 QDR prefix recovery authority ablation

固定 delay4、noise008、checkpoint 和其他实验条件后，`replace_nonexecuting` 与
`flush_pending` 都把 collision/boundary 降为 `0%`，但 safe capture 分别降至
`81.25%/85.42%`，timeout 分别升至 `18.75%/14.58%`；相对 immutable 的 paired
safe-capture delta 为 `-16.67 pp [-22.50,-10.83]` 与 `-12.50 pp [-19.17,-5.83]`。
三种子 summary total p95 由 immutable `123.91 ms` 增至 `136.54/138.55 ms`。
因此两种可取消 pending 的 authority 均冻结为 safety--liveness negative ablation，
不能取代 immutable 主分支，也不构成安全证明。详见
`docs/PHASE45_QDR_AUTHORITY_ABLATION_REPORT.md`。

### 3.13 Phase 46 QDR prefix-failure path audit

对 Phase44 immutable 与 Phase45 两种 authority 的 9 个 distributed-delayed
`steps.jsonl` 做公开几何后验审计，所有输入行均具备九个 prefix classifier 字段。
immutable 的 pooled prefix admissible 为 `93.48%`，首次 violation 主要来自
boundary `741`、obstacle `202` 和 inter-agent `5`；replace/flush 的 prefix
admissible 降至 `60.33%/67.73%`，而 suffix gate exhaustion 仍为
`18.23%/21.28%`。在 240 个 immutable episode 中，238 个曾出现过至少一次
exhaustion，但大多数仍最终 safe capture，说明单一的“是否耗尽”指标不能代表失败。
已在 evaluator 中加入连续耗尽长度、首次耗尽位置和恢复次数日志，并通过 2-episode
schema smoke；
后续仍保持 immutable authority，
并只研究与当前候选选择语义等价的 feasibility-first/incremental rollout。已完成的
40-episode development smoke 保持 episode outcome、step-level gate 字段及测试
中的 selected action/cost 一致，但 planner/total 延迟没有下降，因此该优化暂不
晋级。该阶段是机制诊断，不改变 QDR promotion No-Go，也不构成安全证明。详见
`docs/PHASE46_QDR_PREFIX_FAILURE_AUDIT_REPORT.md`。

---

## 4. 数据集与实验协议冻结

### P0：研究协议和数据契约冻结

- [ ] 写入预注册协议：主问题、三条假设、主指标、样本量、统计检验和停止规则；
- [ ] 冻结随机种子集合，例如 `727201/727202/727203`，训练、采样和环境 seed 分开记录；
- [ ] 保留 v4 的 `600` 场景和 `420/90/90` train/validation/locked-test 划分；镜像组不能跨 split；
- [ ] validation 内再切分 development-calibration 与 confirmation，所有阈值和权重只在 calibration 上确定；
- [ ] locked-test 只在模型、阈值、代价权重、执行协议和代码 hash 全部冻结后开放一次；
- [ ] 为每个 OOD 轴单独建 manifest，不能同时改变几何、目标速度、通信和执行噪声；
- [ ] 给每个场景保存 manifest SHA-256、mirror-group id、目标策略、几何参数、delay/dropout、tracking/noise 和 episode seed。

**P0 完成标准：** 从任意一个 run 目录可以反查场景、模型、源码、配置、随机种子和信息隔离审计；任何 locked-test 指标都能由 episode/step JSONL 重算。

### P0.1：建议数据块

| 数据块 | 用途 | 建议规模 | 改变因素 |
| --- | --- | ---: | --- |
| ID train | 训练 predictor | 420 场景 | 训练分布内 |
| ID dev-calibration | 冻结阈值/权重 | 45 场景 | 只用于选择 |
| ID dev-confirmation | 确认开发结论 | 45 场景 | 不调参 |
| ID locked-test | 最终报告 | 90 场景 | 只运行一次 |
| geometry OOD | 迁移诊断 | ≥100 episode | 仅障碍几何 |
| target-speed OOD | 高速/慢速外推 | ≥100 episode | 仅目标速度 |
| delay-execution OOD | 执行感知安全 | ≥100 episode | 仅延迟/丢包/噪声 |
| behavior OOD | 未见机动策略 | ≥100 episode | 仅目标行为 |
| mixed stress | 最后压力测试 | ≥100 episode | 仅在单轴通过后启用 |

---

## 5. 基线与公平对照矩阵

先建立一个不含新模块的稳定参考，再逐个打开模块：

| 编号 | Predictor | Queue rollout | Compute policy | Interception cost | Safety interface |
| --- | --- | --- | --- | --- | --- |
| B0 | GRU / frozen main predictor | nominal current-state | fixed `K=1` | Euclidean distance | local CBF |
| B1 | same | nominal current-state | fixed `K=8` | Euclidean distance | local CBF |
| QDR | same | **QDR** | fixed `K=1` | Euclidean distance | local CBF |
| UAKR | same | chosen rollout | **UAKR** | Euclidean distance | local CBF |
| RNIC | same | chosen rollout | fixed `K=1` | **RNIC** | local CBF |
| QDR+UAKR | same | QDR | UAKR | distance | local CBF |
| QDR+RNIC | same | QDR | fixed `K=1` | RNIC | local CBF |
| UAKR+RNIC | same | chosen rollout | UAKR | RNIC | local CBF |
| Full | same | QDR | UAKR | RNIC | local CBF |

补充对照：

- `expected`、`worst-case`、`CVaR` 三种候选聚合方式；
- centralized 与 distributed DN-MPC；
- ideal delay、真实 delay、dropout/noise；
- nominal action、local CBF 和 robust CBF-QP 诊断三条安全路径；
- portable Diagonal SSM 作为主 predictor 比较，official S4 只做独立 transfer/runtime check，不因一次结果更好就替换主模型。

所有对照必须共享：场景 manifest、checkpoint、sampling seed、MPC horizon、控制周期、最大 solver iteration、硬件和线程数。

---

## 6. P1：工程接口、单元测试与可审计日志

### 6.1 QDR 实现清单

- [ ] 定义 `QueueState` 数据类，区分 commanded、planned、executed action；
- [x] 实现 pending command 的追加、替换、取消、flush 四种 authority，并在配置中显式声明；
- [x] 实现 queue-aware horizon index：候选轨迹的第 `j` 步必须对应实际执行时间；
- [x] 记录 queue prefix、queue length、action age、prediction age 和 execution error；
- [x] 实现 nominal dynamics 与 execution dynamics 两套 rollout，禁止混用；
- [x] 写出 hand-check 场景：零延迟、固定延迟、队列长度变化、flush 后重新规划；
- [x] 对比 QDR 与手工逐步执行模拟，确保同一输入得到一致的 post-state；
- [x] 在候选生成/best-response 阶段加入 suffix-feasibility gate，并验证其不会把不可修复的 immutable prefix 误标为可修复；当 gate 耗尽时显式记录 No-Go 诊断。
- [x] 增加零动作与当前速度保持 recovery candidates，并在 40-episode development block 上验证 episode-level safety outcome；
- [ ] 解决 gate exhaustion 与重复 rollout 的计算代价，不能把“选择最小违反候选”写成安全保证。

### 6.2 UAKR 实现清单

- [ ] 固定 `K={1,4,8}` 和 refresh interval `{4,2,1}`，先实现规则表，不引入额外学习器；
- [ ] 分别计算 dispersion、temporal inconsistency、belief age 和 prediction residual；
- [ ] 记录每一步 uncertainty、bucket、K、是否 refresh、cache age、候选可行率和 planner latency；
- [ ] 先离线验证 uncertainty 对 prediction miss、next-step margin violation、timeout 的排序能力；
- [ ] 只在 development-calibration 上拟合阈值/风险映射；
- [ ] 设计 intervention-effect label：同一 state 下“刷新/增加 K”相对“不干预”的局部收益，而不是直接使用 episode 终局标签；
- [ ] 若 intervention-effect 不能在 confirmation 上稳定，UAKR 只保留为 negative/efficiency ablation。

### 6.3 RNIC 实现清单

- [ ] 将 queue delay、速度上限、加速度上限、制动距离和转向约束纳入 `τ_req`；
- [ ] 先做 interceptor-only 版本，再做 formation-slot / cooperative assignment 版本；
- [ ] 输出 `tau_req`、`tau_avail`、slack、normalized cost、chosen slot 和 fallback reason；
- [ ] 用独立 reachable-time checker 检查代价输入，不复用 planner 内部近似作为“证明”；
- [ ] 对距离 cost、time-to-reach cost、RNIC cost 做相同权重预算消融；
- [ ] 做极端单元测试：目标已经不可达、刚好可达、多个防御者争夺同一槽位、队列延迟增加。

### 6.4 回归与日志完成标准

- [ ] `python -m pytest -q` 全部通过；
- [ ] 所有脚本通过 `python -m py_compile`；
- [ ] `git diff --check` 通过；
- [ ] 每次 evaluation 写入 `config.yaml`、`effective_config`、source hash、manifest hash、summary、episode/step JSONL 和 TensorBoard event；
- [ ] TensorBoard 至少包含 `SafeCapture`、`Collision`、`Boundary`、`Timeout`、`K`、`refresh`、`queue_age`、`RNIC_slack` 和四级 latency。

---

## 7. P2：QDR 单模块实验

### 7.1 研究假设

**H-QDR：** 在 command delay、observation/message age 和执行噪声增加时，使用真实待执行队列的 rollout 能减少状态错位，从而降低 collision/timeout；在 ID 条件下不应显著损害 safe capture。

### 7.2 实验顺序

- [x] Phase34 在独立 development block 上完成当前源码 QDR-off / QDR-on 配对，修复 planner fallback 的队友 rollout 维度错误，并验证 queue-aware safety projection；
- [x] Phase34 完成 TensorBoard、source hash、manifest hash、episode/step JSONL 和四级 latency 记录；
- [ ] Phase34 的性能 gate 未通过：QDR-on safe capture `85.0%` 对 QDR-off `100.0%`，collision `15.0%` 对 `0%`，suffix admissible rate `44.63%`；不得进入 confirmation；
- [x] Phase35 在 candidate stage 实现 suffix-feasibility gate、有限 recovery candidates 和 gate TensorBoard 诊断；开发块 safe capture `100.0%`、collision `0%`，但 gate exhaustion `21.76%`；
- [ ] Phase35 效率 gate 未通过：total p95 `150.21 ms` 对 QDR-off `46.53 ms`，mean capture time `3.915 s` 对 `1.963 s`；不得进入 confirmation；
- [ ] B0/B1 在 development-calibration 和 confirmation 上重跑，确认当前基线可复现；
- [ ] 只打开 QDR，固定 predictor、K、MPC cost、safety layer；
- [ ] 在零延迟、2-step、4-step、6-step、8-step 五个 delay 档测试；
- [ ] 分开测试 immutable、replace-nonexecuting、flush-pending，不把仿真器 authority 当作真实飞控能力；
- [ ] 做 queue length × prediction age 的二维分层统计；
- [ ] 做 nominal dynamics 与 noisy execution dynamics 的配对对照；
- [ ] 对失败 episode 逐步回放：规划状态、实际执行状态、队列前缀、CBF 输入和目标 belief 必须对齐。

### 7.3 QDR Go/No-Go gate

推荐在运行前冻结以下门槛：

- ID safe capture 相对 B0 的 paired 95% CI 下界不低于 `-2 pp`；
- delay-execution OOD 的 collision 至少下降 `5 pp` 或相对下降 `25%`，并且 boundary 不增加；
- timeout 不得以 collision 增加为代价下降；
- total p95 增幅不超过 `15%`，同时单独报告 predictor/planner/filter/total p50/p95/p99；
- 三个 seed 的方向一致，镜像组 cluster-bootstrap CI 可复现。

若只在理想 delay 有收益、在真实 queue execution 没收益，QDR 只能写成状态对齐实现，不能写成性能创新。

---

## 8. P3：UAKR 单模块实验

### 8.1 研究假设

**H-UAKR：** 公开不确定性可以作为计算预算调度信号，使大部分低风险 step 使用较小 K/较长刷新间隔，而在高风险 step 分配更多候选和更频繁重规划；在固定 K=8 参考下，safe capture 非劣且计算量下降。

### 8.2 必做对照

- [ ] fixed `K=1`、fixed `K=4`、fixed `K=8`；
- [ ] original UAKR threshold policy；
- [ ] risk-calibrated UAKR；
- [ ] refresh-only、K-only、joint intervention；
- [ ] uncertainty feature ablation：dispersion-only、age-only、residual-only、all-public-features；
- [ ] no target-truth leakage audit；
- [ ] 相同候选预算下比较预测质量和闭环结果，避免把更多 K 的收益误写成 adaptive 策略收益。

### 8.3 UAKR Go/No-Go gate

- 相对 fixed `K=8`，ID safe-capture paired 95% CI 下界不低于 `-2 pp`；
- collision/boundary 不得增加，timeout 不得显著增加；
- 平均 K 或 predictor call 至少下降 `25%`，并报告 cache age/stale ratio；
- 在至少一个 OOD 轴上 safe capture 或 timeout 获得预注册改善；
- 三个 seed 的预算下降方向一致，且不能靠 locked-test 重新调阈值；
- total p95 的增加必须与预算节省一起报告，不能只报告平均 latency。

当前 residual-triggered high-K 与 residual-only refresh 已经是负结果，因此不再做未经校准的 `0.1/0.2/0.3...` 阈值网格。若重开 UAKR，顺序必须是：

```text
局部 failure label
  -> intervention-effect calibration
  -> fresh development confirmation
  -> 冻结 policy
  -> locked-test 一次性确认
```

---

## 9. P4：RNIC 单模块实验

### 9.1 研究假设

**H-RNIC：** 以动力学可达时间裕量归一化后的代价，能够减少高速目标或通信/执行延迟场景中的不可达拦截选择；该收益应来自 reachable-time ordering，而不是简单放大距离 penalty。

### 9.2 必做实验

- [ ] distance-only vs time-to-reach-only vs RNIC；
- [ ] interceptor-only RNIC vs formation-slot RNIC；
- [ ] 无队列、固定队列、随机队列三种 `τ_req` 输入；
- [ ] 低速、ID 速度、高速 OOD 三档；
- [ ] perfect communication、message delay、dropout、execution noise；
- [ ] RNIC 权重 `0/0.1/0.3/1.0` 只在 calibration 上选择，confirmation 只运行冻结值；
- [ ] 记录 selected slot、slack 分布和 chosen action 的变化，证明它确实改变了可达性排序；
- [ ] 独立 checker 重算 `τ_req`，检查 planner 是否错误地把不可达候选当作低 cost。

### 9.3 RNIC Go/No-Go gate

- ID safe capture 相对 distance-only 的 paired CI 下界不低于 `-2 pp`；
- target-speed OOD timeout 至少下降 `5 pp`，或 safe capture 提升至少 `5 pp`，collision/boundary 不增加；
- RNIC slack 与 future reachability failure 的 confirmation AUROC 至少达到 `0.70`，且风险箱大致单调；
- planner p95 增幅不超过 `15%`；
- 至少一个 multi-UAV formation-slot 场景显示 assignment 发生有意义变化。

如果 RNIC slack 的风险排序接近随机，不能继续堆 planner 权重；必须先修复 label、reachable-time 计算或槽位定义。

---

## 10. P5：交互效应与完整组合

只有 QDR、UAKR、RNIC 至少各自通过单模块 gate，才开放完整组合。若只有一个模块通过，优先写模块化贡献，不强行合并。

### 10.1 交互实验

- [ ] 运行 `2^3=8` 个 factorial arms：三模块全关、单模块、两两组合和 Full；
- [ ] 每个 arm 使用相同 3 个 seed、相同 mirror-group 场景和相同 sampling seed；
- [ ] 计算主效应和二阶交互效应：QDR×UAKR、QDR×RNIC、UAKR×RNIC；
- [ ] 分析 UAKR 是否因为 QDR 产生更可靠的 age/uncertainty 信号；
- [ ] 分析 RNIC 是否因为 QDR 提供真实 queue state 而改善 `τ_req`；
- [ ] 分析 UAKR 是否因为 RNIC 改变候选风险排序而触发更多不必要预算；
- [ ] 保留每个失败 episode 的 step-level causal trace。

### 10.2 Full Go/No-Go gate

- safe capture 相对当前主参考的 paired 95% CI 下界不低于 `-2 pp`；
- collision 和 boundary 不高于主参考，且不能依赖未经认证的 safety fallback；
- 至少一个预注册 OOD 轴有改善；
- 组合后的平均预测预算或总计算量不超过固定 K=8 参考；
- 三个 seed 方向一致；
- 每个模块仍能通过开关单独复现；
- 任何“单模块通过、Full 失败”的情况必须进入论文 failure analysis，不得删除。

如果 Full 不通过，论文主线改为“模块化延迟感知围捕框架 + 组合失效边界分析”，而不是宣称三模块端到端成功。

---

## 11. P6：OOD 与压力测试

### 11.1 单轴 OOD

- [ ] geometry shift：障碍物尺寸、间距和高度全部与 train 不重叠；
- [ ] target-speed shift：低速、高速和加速度外推；
- [ ] communication shift：message delay、dropout、observation age；
- [ ] execution shift：command delay、tracking gain、velocity/acceleration scale、noise；
- [ ] behavior shift：short-lookahead、long-lookahead、突变/蛇形/加减速目标；
- [ ] 每个 block 只改变一个因素，至少 100 episode、50 mirror groups、3 evaluation seeds；
- [ ] 所有 OOD 只用于诊断和最终确认，不用于重新选择 checkpoint 或阈值。

### 11.2 混合压力场景

仅当至少两个单轴 OOD 通过后：

- [ ] delay + target-speed；
- [ ] delay + geometry；
- [ ] delay + execution-noise；
- [ ] target-speed + behavior；
- [ ] 最终 mixed stress。

每个压力场景都必须报告 transfer gap，并明确它是描述性结果，不自动等于泛化定理或安全证明。

---

## 12. 统计、延迟与 TensorBoard 规范

### 12.1 主指标

每个 episode 和汇总表必须包含：

```text
safe_capture
ordinary_capture
collision
boundary_violation
timeout
capture_time
minimum_clearance
planner_success / valid / effective
QP_infeasible / fallback / certificate_valid
```

### 12.2 机制指标

```text
QDR: queue_length, queue_prefix, command_age, prediction_age, stale_ratio
UAKR: uncertainty, bucket, K, refresh, cache_age, predictor_calls
RNIC: tau_req, tau_avail, slack, normalized_cost, selected_slot
```

### 12.3 延迟

统一使用同一硬件、同一进程、固定线程数，分别报告：

```text
predictor p50 / p95 / p99
planner p50 / p95 / p99
safety-filter p50 / p95 / p99
total-control p50 / p95 / p99
```

`100 ms` 只作为部署参考，不作为硬性淘汰线；如果超过它，应解释低频预测、高频规划/过滤、候选缓存年龄和执行安全假设。

### 12.4 统计规则

- 主要比较使用按 mirror group 聚类的 paired bootstrap 95% CI；
- 多个模块假设使用 Holm 校正；
- 结果表同时给绝对值、差值、CI 和样本量；
- 不用单次 8/20 episode smoke 作为正式显著性结论；
- 先在 validation/calibration 选择，再在 confirmation 冻结确认，最后才开 locked-test。

TensorBoard 每次运行至少写入：effective config、git/source hash、manifest hash、seed、所有主指标、机制指标、四级 latency 和 gate pass/fail。建议统一查看：

```powershell
.\scripts\start_tensorboard.ps1 -LogDir results -Port 6006
```

---

## 13. 推荐时间表与每周交付物

| 周次 | 任务 | 必须交付 |
| --- | --- | --- |
| 第 1 周 | P0 协议、manifest、信息隔离和 baseline 复现 | protocol、scene hash、baseline report |
| 第 2 周 | P1 QDR 接口、authority、队列索引和单元测试（已完成）；修复执行感知 peer rollout 维度错误 | source、tests、TensorBoard smoke、`264 passed` |
| 第 3 周 | P2 QDR suffix-feasibility gate、recovery candidates、delay/authority/noise 单因素 validation 与失败回放 | Phase35 已完成 safety-outcome development check；效率仍 No-Go，下一交付为运行时优化与 fresh confirmation |
| 第 4 周 | P3 UAKR offline reliability 和 intervention-effect calibration | calibration artifact、reliability report |
| 第 5 周 | P3 UAKR confirmation 和预算/延迟分析 | UAKR report、Go/No-Go |
| 第 6 周 | P4 RNIC reachable-time checker、slot cost 和 validation | RNIC report、slack audit |
| 第 7 周 | P5 两两交互和 Full factorial | interaction table、causal traces |
| 第 8 周 | P6 单轴 OOD 与 mixed stress | OOD matrix、failure taxonomy |
| 第 9 周 | 运行时优化、复现脚本、最终图表 | benchmark、TensorBoard dashboard、artifact index |
| 第 10 周 | 论文材料和独立审计 | method、result、ablation、limitations、reproduction appendix |

每完成一周，提交一次小 commit，推荐格式：

```text
feat(phaseX): implement <module>
test(phaseX): evaluate <hypothesis>
docs(phaseX): report <decision>
```

只提交代码、测试、配置和正式报告；大型 `results/`、checkpoint 和 TensorBoard event 保留在本地 artifact 目录，并在报告中记录路径和 hash。

---

## 14. 最小可发表路线与停止规则

### 14.1 最小可发表路线

如果资源有限，优先完成：

1. QDR 在 delay-execution OOD 上的可靠状态对齐证据；
2. RNIC 在 formation-slot 上的可达性排序证据；
3. UAKR 作为固定 K=8 非劣的计算节省消融；
4. 一个通过 gate 的单模块 + 一个清晰的 Full 组合失败分析；
5. 完整的队列、可达裕量和预算决策可视化。

这条路线比同时训练新的 Mamba、学习 CLBF、扫描大量阈值更易复现，也更容易让审稿人看懂每个贡献解决了什么问题。

### 14.2 明确停止规则

- QDR 在真实执行延迟下不改善 collision/timeout：停止扩大 QDR 超参，转入契约与失败分析；
- UAKR 不能在 fixed K=8 下保持 ID 非劣：降级为 efficiency/negative ablation；
- RNIC slack 不能在 confirmation 上可靠排序：停止 planner 权重扫描，修复 reachable-time label；
- Full 组合造成 collision 或 boundary 增加：停止端到端堆叠，保留最佳单模块结果；
- robust CBF-QP 仍不可行：保持 local CBF 主线，robust CBF 只做诊断，不把安全证明写进摘要；
- 任何结果依赖 locked-test 调参：该结果作废，重新建立 fresh confirmation。

### 14.3 最终成功定义

只有同时满足以下条件，才可以把三点写成完整方法：

- 三个模块分别有独立、可重复、与机制一致的收益；
- Full 在 ID 上相对主参考非劣，collision/boundary 不增加；
- 至少一个 delay/OOD 主轴获得统计支持的提升；
- UAKR 的预算节省与延迟尾部没有被忽略；
- QDR、UAKR、RNIC 的 step-level decision 可审计；
- 代码、配置、测试、manifest hash、TensorBoard 和结果报告齐全；
- 论文明确写出当前安全层的适用范围和失败边界。

在这些条件满足之前，最稳妥的论文标题方向应是“queue-aware delayed distributed encirclement with adaptive prediction budgeting and reachability-normalized planning”，而不是宣称已经完成形式化安全闭环。
