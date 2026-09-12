# 面向多无人机围捕的可复现创新点候选计划

> 版本：v1.0  
> 日期：2026-09-13  
> 适用仓库：[Liangxianguang/new-uav-capture](https://github.com/Liangxianguang/new-uav-capture)

## 0. 结论先行

可以增加新的创新点，但不建议继续堆叠更大的预测 backbone。当前 QDR、UAKR、RNIC
已经暴露出清晰的失败边界：QDR 在现有队列契约下为 No-Go，UAKR 有计算节省但没有
通过安全捕获率非劣门槛，RNIC 在 centralized ID 上只有非劣方向，在 distributed
delay 场景中没有行为收益且显著增加延迟。因此，下一轮应优先选择：

1. **Escape-Gap-Aware Cooperative MPC（EGC-MPC，逃逸角隙感知协同 MPC）**作为主
   候选；
2. **Freshness-Calibrated Decentralized Belief Fusion（FC-DBF，时效校准的分布式
   belief 融合）**作为第二候选；
3. **Failure-Conditioned Replay Curriculum（FCRC，失败条件重放课程）**作为训练
   和泛化方向。

三个点都可以复用现有场景生成器、DN-MPC、local CBF、TensorBoard 和审计脚本，不
   需要依赖官方 S4 的慢速 CPU fallback，也不需要先完成 R-CLBF-QP 证明。

这些是“有新颖性潜力”的候选，不应直接写成“首次提出”。投稿前仍需做一次正式文献
检索，确认“逃逸角隙代价 + 延迟分布式围捕 + 多模态目标预测”的组合没有已有同义
方法。

---

## 1. 候选 A：逃逸角隙感知协同 MPC（首选）

### 1.1 动机

普通距离代价会鼓励所有无人机同时追向目标，可能留下一个很大的逃逸方向。围捕的
关键不是每架无人机都更接近目标，而是让目标无法从最大的角隙中穿出。该机制直接针
对当前模型的核心任务，解释性强，代码改动小，适合做严格消融。

### 1.2 可复现定义

对第 `k` 条预测目标轨迹，在 MPC 预测时刻 `t+h`：

1. 以预测目标位置为圆心，将每架无人机的位置投影到水平面，计算方位角
   `theta_i(k,h)`；
2. 对角度排序，计算首尾相接的圆周角隙 `gap_i(k,h)`；
3. 取最大角隙 `gap_max(k,h)`，并找到包含目标预测速度方向的逃逸角隙
   `gap_escape(k,h)`；
4. 对所有候选轨迹按公开 belief 的置信度 `pi_k` 加权。

一个无需训练的第一版代价为：

```text
C_gap = sum_k pi_k * sum_h discount[h] *
        (w_max * relu(gap_max(k,h) - gap_safe)^2
         + w_escape * relu(gap_escape(k,h) - gap_escape_safe)^2)
```

其中 `gap_safe` 和 `gap_escape_safe` 只在 development-calibration 中选择；目标真值
不能参与在线计算。为了避免把角隙项变成隐含的无限大约束，第一版只将它作为有限
权重加入 DN-MPC objective，最后仍由现有 local CBF 过滤动作。

### 1.3 预期贡献

- 从“距离追踪”转为“封锁最大逃逸通道”的任务结构化代价；
- 对高机动目标的未来逃逸方向使用多模态轨迹加权，而不是只看单条均值轨迹；
- 不依赖新的神经网络，推理结果可由角度、角隙和候选权重逐步复算；
- 可以明确展示机制证据：最大角隙、逃逸角隙、覆盖率和捕获时间是否改善。

### 1.4 必做消融

| 对照 | 说明 |
| --- | --- |
| distance-only | 当前主参考代价 |
| max-gap-only | 只惩罚最大角隙 |
| escape-gap-only | 只惩罚目标预测速度方向上的角隙 |
| max-gap + escape-gap | 完整 EGC-MPC |
| single-mode target | 只用均值/最可能轨迹 |
| multimodal weighted | 使用全部候选及置信度 |
| centralized / distributed | 检验分布式消息延迟下是否仍有效 |

### 1.5 首选 Go/No-Go 门槛

在运行前冻结以下门槛：

- ID safe capture 相对 distance-only 的 paired 95% CI 下界不低于 `-2 pp`；
- collision 和 boundary violation 不增加；
- 至少一个预先指定的 target-behavior 或 target-speed OOD block 中，safe capture
  提升至少 `5 pp`，或 timeout/collision 下降至少 `5 pp`；
- hard-behavior block 的平均最大角隙或逃逸角隙下降至少 `10%`，且三个 seed 方向
  一致；
- planner p95 增幅不超过 `15%`。总控制延迟仍同时报告 p50/p95/p99，不把 `100 ms`
  当作硬性淘汰线；
- 角隙代价必须改变至少一部分 assignment/action，不能出现“指标不变但捕获率偶然
  变化”的不可解释结果。

若只提高角隙指标却不改善捕获，结果应作为机制负消融；若只在 centralized 有效、
distributed 无效，则不能宣称分布式主贡献。

---

## 2. 候选 B：时效校准的分布式 belief 融合

### 2.1 动机

当前分布式场景中，不同无人机收到的目标 belief 可能具有不同的 message age、
observation age 和 command age。简单平均会把旧信息当成新信息，严重延迟时还会让
多个无人机形成不一致的拦截判断。这个点比再次调 QDR 更聚焦于“信息新鲜度”。

### 2.2 可复现规则

每条消息携带：

```text
mean, covariance_or_scale, message_timestamp, source_id, dropout_flag
```

对源 `j` 到接收者 `i` 的消息年龄 `a_ij` 做确定性校准：

```text
mean_j' = public_dynamics_rollout(mean_j, a_ij)
cov_j'  = cov_j + (alpha_age * a_ij + beta_dropout * dropout_flag) * I
weight_ij = exp(-lambda_age * a_ij) / (trace(cov_j') + eps)
```

然后以 precision-weighted mean 融合 self belief 和邻居 belief。若有效样本数低于
固定阈值，回退到 self-only belief，并记录 `fusion_fallback_reason`。所有 rollout
只允许使用公开 belief 和公开动作历史，不允许读取 target truth。

### 2.3 必做消融

- naive average；
- self-only；
- age-weight-only；
- covariance inflation-only；
- age + covariance 完整版本；
- 无 dropout、message delay、observation delay 和混合延迟四个 block。

### 2.4 建议门槛

- delay/execution OOD 的 collision 至少相对下降 `5 pp`，boundary 不增加；
- ID safe capture paired CI 下界不低于 `-2 pp`；
- belief-age 分层下，融合后的下一步位置误差和最大角隙风险具有单调改善趋势；
- planner 和 predictor 的 p50/p95/p99 不能因融合层产生无法解释的长尾。

该点实现难度低于学习式通信协议，适合在 EGC-MPC 没有信号时作为替代主线。

---

## 3. 候选 C：失败条件重放课程

### 3.1 动机

现有 OOD 结果显示，target-speed 和 delay/execution shift 是主要薄弱点。与其重新
换 backbone，不如利用已生成的失败前缀，针对“什么条件导致失败”生成可审计的训练
样本。这是一条数据中心、容易复现的路线。

### 3.2 可复现流程

从 development split 的 episode/step JSONL 中提取失败前缀，构造难度向量：

```text
d = [target_speed_scale, command_age, queue_length, prediction_age,
     obstacle_clearance, uncertainty, last_margin]
```

按固定规则分桶：`nominal / delay-hard / speed-hard / clearance-hard /
compound-hard`。每轮训练只使用上一轮 development 失败样本及其 mirror-preserving
counterfactual 扩展；扩展只改变一个因素，例如只增加 command delay 或只改变目标
加速度，不能同时改变场景几何和行为。

建议训练采样比例：

```text
base ID : hard replay : counterfactual = 6 : 3 : 1
```

比例、扰动范围和停止规则必须在训练开始前冻结。validation/locked-test 失败样本
严禁回流训练。

### 3.3 必做对照

- 普通 IID training；
- 固定比例 hard replay；
- 按失败难度重加权；
- 不使用 counterfactual 的 replay；
- 完整 FCRC。

### 3.4 建议门槛

- 至少一个 OOD block safe capture 提升 `5 pp` 或 timeout 下降 `5 pp`；
- ID safe capture 下降不超过 `2 pp`；
- 不增加 collision/boundary；
- 三个训练 seed 的收益方向一致；
- 训练后 predictor 的 miss、calibration error 和闭环指标不能只改善其中一个。

该方向的算法新颖性通常弱于 EGC-MPC，但复现成本最低，适合形成可靠的泛化实验
和论文中的 data-centric ablation。

---

## 4. 可选候选 D：角隙触发的稀疏通信

如果 EGC-MPC 有效，可以进一步把最大逃逸角隙作为通信触发信号：只有当本地预测的
逃逸角隙超过阈值、或与邻居的角隙排序发生变化时，才发送高优先级协同消息。普通
状态消息继续按固定周期发送。

该点可能形成“通信预算—围捕性能”的新结果，但它不应在第一轮同时实现。原因是它
会同时改变消息年龄、队列和 planner 行为，因果归因会变得困难。只有在固定通信预算
和可控 delay block 中，EGC-MPC 单独通过后再开启。

---

## 5. 推荐的最终研究组合

不建议把四个新点全部堆叠。建议采用以下路线：

```text
主线：GRU/现有 predictor
    + distributed delayed DN-MPC
    + EGC-MPC
    + local CBF

备选增强：FC-DBF（只作为 belief 输入层）

泛化实验：FCRC（只改变训练数据，不改变闭环控制器）

保留分析：QDR / UAKR / RNIC 作为已有机制和负结果对照
```

论文中可以形成三类清楚的问题：

1. 角隙感知代价是否改善围捕拓扑，而不仅是缩短距离？
2. belief 时效校准是否降低真实延迟下的信息不一致？
3. 失败条件重放是否提升未见速度和执行延迟下的泛化？

这比把 QDR、UAKR、RNIC、S4、robust CBF-QP 同时塞入一个 Full 组合更容易复现、归因
和审稿。

---

## 6. 两周可行性筛选计划

### 第 1--2 天：协议冻结

- [ ] 冻结当前主参考 checkpoint、MPC horizon、采样 seed 和 local CBF 配置；
- [ ] 选择 validation development-calibration 与 confirmation；
- [ ] 为 EGC-MPC 预注册 `gap_safe`、`gap_escape_safe`、权重候选和停止规则；
- [ ] 为 FC-DBF 预注册 `lambda_age`、协方差膨胀系数和 fallback 阈值；
- [ ] 确认所有线上计算只使用 public belief、消息和动作历史。

### 第 3--4 天：EGC-MPC 最小实现

- [ ] 实现圆周角隙、逃逸方向和候选置信度加权；
- [ ] 增加 `max_gap`、`escape_gap`、`coverage_ratio`、`selected_assignment` 日志；
- [ ] 加入 `cost_mode=distance|gap|max_gap_escape` 开关；
- [ ] 添加零权重回归测试，确保新代码在权重为零时与旧 planner 一致；
- [ ] 添加手工几何测试：均匀四机、单角隙、目标改变逃逸方向、镜像场景。

### 第 5--6 天：20 场景 smoke

- [ ] 只在 development split 跑 20 个场景、三个 seed；
- [ ] 比较 distance-only、max-gap-only、escape-gap-only、完整版本；
- [ ] 检查角隙是否实际变化、planner latency 是否可接受、失败轨迹是否可解释；
- [ ] 若 safe capture 不改善且角隙也不改善，停止该方向，不做大规模扫描。

### 第 7--10 天：FC-DBF 小规模验证

- [ ] 单独实现 age-weight 和 covariance inflation；
- [ ] 在固定 planner 上跑 delay/execution development block；
- [ ] 检查 belief 误差、角隙风险和 collision 是否同步改善；
- [ ] 若只改善 belief 误差、不改善闭环，保留为诊断，不晋级主贡献。

### 第 11--14 天：确认与选择

- [ ] 选出最多一个主创新进入 fresh confirmation；
- [ ] 使用至少三个 evaluation seed 和 mirror-group paired bootstrap；
- [ ] 写出 predictor/planner/safety/total 的 p50/p95/p99；
- [ ] 生成一组失败回放和角隙/时效曲线；
- [ ] 依据门槛给出 Go/No-Go，不在 confirmation 上调阈值。

---

## 7. 实验矩阵与日志要求

每个新点都必须与当前主参考使用同一 scene manifest、checkpoint、sampling seed、
MPC horizon、硬件和线程数。每个 run 至少保存：

```text
effective_config
git/source hash
scene manifest hash
train/evaluation seed
safe_capture, ordinary_capture, collision, boundary, timeout
capture_time, minimum_clearance
predictor/planner/safety/total p50,p95,p99
mechanism metrics
```

TensorBoard 至少记录：

```text
EGC: max_gap, escape_gap, coverage_ratio, gap_cost, assignment_switch
FC-DBF: message_age, effective_sample_size, covariance_trace, fusion_fallback
FCRC: replay_bucket, replay_ratio, train_split_hash
```

所有正式比较仍遵守：先 development-calibration，再 fresh confirmation，最后才是
locked-test；不得把 OOD 或 locked-test 结果回流到阈值、权重或 checkpoint 选择中。

---

## 8. 如何判断“新”与“可发表”

### 8.1 新颖性判断

不能仅凭名称判断新颖性。正式检索时至少要分别检索以下组合：

- multi-UAV encirclement + angular gap / escape corridor / topological coverage；
- delayed decentralized MPC + belief freshness / covariance inflation；
- trajectory prediction or interception + failure replay / hard-negative curriculum。

如果已有工作只使用其中一个元素，而没有在“延迟、多模态目标、分布式围捕、可审计
闭环”这一组合下验证，则仍有方法组合和实验问题定义空间；如果已有同义方法，应把
贡献收窄为新的延迟协议、可复现实验基准或失败边界分析。

### 8.2 发表最低条件

建议至少达到：

- 一个主创新在 ID 非劣、至少一个 OOD block 有统计支持的收益；
- 一个机制指标与闭环收益方向一致；
- 三个 seed、mirror-group paired CI、完整失败分类；
- 所有代码、配置、测试、manifest hash、TensorBoard 和结果报告可复现；
- 明确说明 robust CBF-QP 当前仍是诊断 No-Go，不能包装成安全证明。

如果 EGC-MPC、FC-DBF 和 FCRC 都没有通过，不要继续无限加模块；可以把论文转为
“延迟分布式围捕的模块化基线、可审计协议与组合失效边界”，但不能把负结果删除。

