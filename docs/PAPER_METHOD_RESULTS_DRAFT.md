# Methods and Results Draft

## 题目建议

**面向部分观测三维高机动目标围捕的多模态状态空间预测、极小极大分布式规划与鲁棒安全过滤**

## 使用说明

本文档是一份面向顶刊或高水平机器人/智能控制期刊的 `Methods` 与 `Results`
主文草稿。当前仓库已经完成的结果直接使用实测数值；执行器不变性、完整
R-CLBF-QP 和官方 `mamba_ssm` 后端部分仍以最终实验完成后的替换段落给出。
方括号中的内容只有在对应实验真正完成并保留独立证据后才能写入投稿版本。

当前结果支持的是：在速度级三维围捕仿真和匹配安全契约下，projected
portable-SSM diffusion、worst-case DN-MPC 与 velocity-level robust CBF-QP
可以形成高安全率闭环。当前结果不支持把该系统描述为已经完成真实执行器
安全证明的 R-CLBF-QP 系统。

---

# 2. Methods

## 2.1 Problem Formulation

我们研究四架协同防守无人机围捕一架高机动逃逸目标的三维任务。环境包含
有限三维工作空间、圆柱/箱体/墙体障碍物、部分目标观测、观测延迟、通信延迟
以及通信丢包。防守无人机只能访问公开的自身状态、局部障碍物几何、延迟的
目标 belief 和允许的队友消息；目标未来真实状态仅由环境用于生成监督标签和
终止条件，不能被预测器、规划器或安全过滤器读取。

记第 `i` 架无人机在时刻 `t` 的位置和速度为
`p_i^t, v_i^t in R^3`，控制输入为速度指令 `u_i^t`。在速度级基准中，系统
动力学为

```text
p_i^{t+1} = p_i^t + Delta t u_i^t,
||u_i^t||_2 <= v_max.
```

环境使用 `Delta t = 0.1 s`、`v_max = 5.0 m/s`、无人机半径 `r_d = 0.25 m`
和目标捕获半径 `r_c = 0.80 m`。正式预测数据中的目标动力学约束为最大速度
`3.6 m/s` 和最大加速度 `4.0 m/s^2`。执行扰动审计另外引入命令延迟、命令噪声、
一阶速度跟踪、阻力以及每回合质量和动力学参数随机化。

任务在下式成立时成功：

```text
min_i ||p_i^t - p_T^t||_2 <= r_c,
```

且该回合没有碰撞、越界或其他安全失败。本文将此指标称为
`safe capture`，以区别于只统计进入捕获半径但曾发生碰撞的普通捕获。

### 部分观测接口

对每架无人机，观测包含自身和队友的状态摘要、由可见目标观测更新的 belief
位置与速度、可见性、观测置信度、协方差、观测年龄、通信消息年龄以及局部
障碍物几何。目标真值不进入策略输入。历史编码器接收长度为 `L = 16` 的
局部观测序列；每个时间步的共享策略输入为 `63` 维固定长度特征。

这一信息边界是所有非 oracle 方法的共同约束。DynamicEncirclement 只作为
不使用预测候选的行为基线，centralized worst-case MPC 作为具有完整队友状态
共享的规划上界，而正式方法仍以 belief-level target information 为输入。

## 2.2 Overall Architecture

系统由三个串联模块组成：

```text
partial observation history
        -> SSM-conditioned multimodal diffusion predictor
        -> dynamics projection and candidate audit
        -> worst-case distributed scenario MPC
        -> robust CBF safety projection
        -> environment / execution dynamics
```

预测器以低频方式刷新候选轨迹；规划器在每个控制步使用当前缓存的候选集合
执行滚动时域优化；安全层在动作下发前进行独立的一步安全投影。每个模块都
保存输入协议、配置、随机种子、源文件哈希和逐步诊断，从而可以区分预测误差、
规划失败、通信失败、QP 不可行和执行器造成的安全失败。

## 2.3 SSM-Conditioned Multimodal Diffusion Predictor

### 2.3.1 Target representation

设团队当前发布的 belief reference 为 `b_t`，未来目标轨迹为

```text
Y_t = [p_T^{t+1} - b_t, ..., p_T^{t+H} - b_t] in R^{H x 3}.
```

训练集统计量只在 train split 上拟合。对每个未来步和空间坐标执行仿射标准化：

```text
tilde Y_t[h, d] = (Y_t[h, d] - mu[h, d]) / sigma[h, d].
```

在当前正式协议中 `H = 12`，即预测窗口长度为 `1.2 s`。相对 belief reference
的表达方式避免把延迟目标绝对位置直接泄漏到标签接口之外，同时保留规划器
需要的世界坐标还原过程。

### 2.3.2 Linear-time state-space encoder

对历史输入 `x_1, ..., x_L`，编码器首先进行线性投影，然后逐层递推隐状态：

```text
z_l^k = tanh(W_l x^k + U_l h_l^{k-1}),
h_l^k = alpha_l odot h_l^{k-1} + (1 - alpha_l) odot z_l^k,
alpha_l = sigmoid(a_l).
```

每层使用 LayerNorm，最终条件向量为最后一个时间步的归一化状态
`c_t = LN(h_N^L)`。该递推沿时间维线性计算，适合长历史和频繁滚动推理。

当前可复现实验使用仓库内的 `portable_diagonal_ssm`。它是依赖无关的纯 PyTorch
对角状态空间实现，不等同于官方 `mamba_ssm` CUDA kernel。若最终实验替换为
官方 Mamba，投稿版本必须同时报告后端名称、版本、硬件、数值等价性和独立
消融结果，不能仅因结构具有 SSM 递推就统称为官方 Mamba。

### 2.3.3 Conditional diffusion

令 `Y_0` 为标准化轨迹向量，扩散前向过程为

```text
q(Y_s | Y_0) = N(sqrt(a_bar_s) Y_0,
                 (1 - a_bar_s) I),
```

其中 `a_bar_s` 是线性 beta schedule 的累积乘积。去噪器接收带噪轨迹、SSM
条件向量和扩散时间嵌入，并预测噪声：

```text
epsilon_theta(Y_s, s, c_t).
```

训练目标为

```text
L_diff = E_{Y_0,s,epsilon}
         ||epsilon - epsilon_theta(sqrt(a_bar_s)Y_0
           + sqrt(1-a_bar_s)epsilon, s, c_t)||_2^2.
```

正式模型使用两层、隐藏维度 `128` 的 portable SSM diffusion，训练 `40` 个
epoch，batch size `128`，学习率 `1e-3`；评估时生成 `K = 8` 条候选轨迹，使用
`8` 个去噪步骤和固定 sampling seed `745102`。候选接口保存
`[batch, K, H, 3]` 轨迹张量和候选 logits。

当前 logits 明确标记为 `uniform_uncalibrated`。因此这些权重只能用于候选集合
的均匀聚合，不能解释为经过校准的逐候选概率或目标 mode probability。

### 2.3.4 Dynamics projection and candidate contract

并行位置扩散本身不保证速度状态递推，因此原始 diffusion 候选可能违反加速度
约束。我们对每条候选执行四轮交替投影。设上一时刻的位置和速度为
`p_prev, v_prev`，候选希望到达的位置为 `p_des`，先计算

```text
v = (p_des - p_prev) / Delta t.
```

随后交替投影到速度球、加速度球和世界边界诱导的速度盒，最后以
`p_next = p_prev + Delta t v` 重构位置。速度和加速度约束分别为

```text
||v||_2 <= v_T,max,
||v - v_prev||_2 <= a_T,max Delta t.
```

该步骤是规划接口的动力学契约，不是障碍物避障器，也不是安全证明。规划器
拒绝状态为 `raw` 的候选，只接受显式标记为 `projected` 的候选，并在评估中同时
保留投影前后的误差、coverage 和可行率。

### 2.3.5 Coverage calibration

我们使用只依赖 validation split 前半段的 split-conformal full-trajectory
maximum-distance score：

```text
s_j = max_h ||Y_j[h] - Yhat_j[h]||_2.
```

对目标覆盖率 `1 - delta = 0.90`，取有限样本修正后的高分位数 `q_delta`。对
locked test，仅报告集合覆盖率：

```text
coverage = 1{max_h ||Y[h] - Yhat[h]||_2 <= q_delta}.
```

该量描述候选集合或预测区域的边际覆盖，不能替代逐候选概率校准。完整版本中，
若新增独立 mode scorer，应使用独立 calibration split 报告 ECE、NLL、Brier
score 和 coverage-risk 曲线，并且不能使用 locked-test 标签调节 scorer。

## 2.4 Risk-Sensitive Scenario MPC

### 2.4.1 Candidate-conditioned rollout

预测器输出的绝对候选路径记为

```text
Yhat_t^k = [y_{t+1}^k, ..., y_{t+H}^k], k = 1, ..., K.
```

给定四架无人机的动作序列
`U = [u_{t:t+H-1}^1, ..., u_{t:t+H-1}^N]`，用共享速度级动力学得到
防守位置 rollout `P(U)`。对每个候选目标路径计算场景代价

```text
C_k(U) = w_d C_distance
       + w_T C_terminal
       + w_c C_capture
       + w_f C_formation
       + w_u C_control
       + w_delta C_action_change
       + w_o C_obstacle
       + w_b C_boundary
       + w_i C_inter_agent
       + w_r C_relative_speed.
```

其中距离项鼓励团队接近目标，terminal 项约束预测窗末端距离，capture hinge
项惩罚未进入捕获半径的终端状态，formation 项将非拦截者保持在围捕周长附近。
障碍、边界和队友间距离使用 hinge-square penalty；这些规划代价不取代最后
的安全过滤器。

### 2.4.2 Minimax objective

在候选权重 `pi_k` 下，规划器可选择三类风险聚合：

```text
J_expected(U) = sum_k pi_k C_k(U),
J_worst(U)    = max_k C_k(U),
J_CVaR(U)     = CVaR_alpha({C_k(U)}, pi),
```

本文主方法采用 worst-case objective：

```text
U* = argmin_{U in U_adm} max_k C_k(U).
```

该形式显式防止控制器只追逐最乐观的单条预测路径。`expected` 和 `CVaR`
用于风险目标消融。当前 finite-shooting 实现通过参考路径、每条候选路径和
不同围捕 perimeter scale 生成有限动作序列，并在同一场景文件上比较所有方法。

### 2.4.3 Distributed sequential best response

集中式 oracle 直接选择团队动作序列；DN-MPC 则按无人机编号依次更新局部序列。
第 `i` 个局部问题固定其他无人机最近已知的动作序列：

```text
U_i^{r+1} = argmin_{U_i in U_i,adm}
            R(C_1(U_i, U_-i^r), ..., C_K(U_i, U_-i^r)).
```

每一轮更新后广播当前位置、速度和动作序列。通信模块显式记录发送、接收、
丢弃、字节数和消息年龄，并支持四种模式：ideal、固定两步延迟、10% dropout
以及 no communication。最多运行 `4` 轮，收敛判据为相邻迭代的最大动作变化
不超过 `0.15 m/s`。超过局部预算或候选契约无效时，系统依次使用预设 fallback
动作和上一轮动作序列，并将原因写入逐步日志。

正式规划参数为 horizon `8`、control horizon `3`、最大速度 `5.0 m/s`，
obstacle/boundary/inter-agent penalty 均显式配置。P7 中预测候选每 `20` 个
控制步刷新一次；预测年龄和刷新率作为独立运行时指标报告。

## 2.5 Robust Safety Projection

### 2.5.1 Safe set

安全集合由障碍、世界边界和无人机间隔共同定义。对第 `i` 架无人机和障碍 `o`，
定义带有效安全裕度的障碍 barrier：

```text
h_{i,o}(p) = d_o(p_i) - r_d - m_eff.
```

世界边界 barrier 为

```text
h_{i,a}^{lower}(p) = p_{i,a} - lower_a - r_d - m_eff,
h_{i,a}^{upper}(p) = upper_a - p_{i,a} - r_d - m_eff.
```

无人机间隔 barrier 为

```text
h_{ij}(p) = ||p_i - p_j||_2 - (2r_d + m_eff).
```

`m_eff` 是基础安全裕度与扰动、观测误差、延迟和执行误差裕度之和。只有在
这些误差界与 reset 分布和执行模型一致时，才能将其解释为鲁棒 barrier。

### 2.5.2 Velocity-level CBF projection

对当前观测位置在线性化 barrier 约束，得到

```text
h_l(p_t) + grad h_l(p_t)^T Delta t u >= (1-gamma) h_l(p_t),
```

即

```text
grad h_l(p_t)^T Delta t u >= -gamma h_l(p_t).
```

给定 DN-MPC 输出 `u_nom`，安全层求解

```text
min_{u, xi}  1/2 ||u - u_nom||_W^2 + lambda ||xi||_2^2
s.t.         A_h u + xi >= b_h,
             xi >= 0,
             ||u_i||_2 <= v_max,
             u_i in U_exec,
             xi = 0 when hard-certificate mode is enabled.
```

速度球约束与环境使用的欧氏速度约束一致。实现保留
`RobustCBFQPFilter` 接口名称，但包含显式速度球时，在线子问题严格说是小型
凸 QCQP；论文中不得将其不加说明地称为纯线性 QP。

### 2.5.3 Independent certificate

安全过滤器和独立 checker 使用不同代码路径。checker 重新计算当前状态和离散
下一状态的所有 barrier，并记录最小 barrier、约束残差、最大 slack、fallback
原因和 solver 状态。执行器扩展还检查 pending command queue、执行 rollout、
swept volume 和连续线段的 barrier。只有独立 checker 通过，动作才可计入
`certificate valid`。

### 2.5.4 Full R-CLBF-QP extension

当完整实验真正完成后，最终方法可将手工几何 barrier 扩展为鲁棒复合控制
Lyapunov-barrier function：

```text
B_theta(s) = [B_obs, B_bound, B_inter, B_capture, V_theta]_composite,
```

并要求存在扰动集合 `W`，使得对所有允许执行扰动 `w in W`：

```text
min_{w in W} [B_theta(s_{t+1}) - (1-gamma) B_theta(s_t)] >= 0.
```

最终投稿只有在以下条件全部通过后才能使用“R-CLBF-QP closed-loop safety
certificate”表述：统一的高阶动力学模型、可验证的扰动界、包含动作队列的
多步 forward-invariance 检查、连续段/扫掠体积检查、所有 fallback 分支的
证书闭合，以及至少三个独立模型种子和独立 locked-test 的端到端验证。

## 2.6 Training and Evaluation Protocol

正式预测实验使用 `v3_multimodal` 数据集，训练/验证/locked-test 样本数分别为
`15,232 / 5,712 / 7,616`，覆盖 `flee_persistence`、`random_turn`、`s_curve`、
`burst` 和 `boundary_escape` 五种目标模式。模型使用三个训练种子
`745101`、`745201` 和 `745301`。所有 split 的 episode seed 不重叠；模型选择
和 conformal calibration 不访问 locked-test 标签。

规划泛化实验在固定 S3 混合障碍 validation 集上使用 `12` 个共享场景，并在
另一个未见 `adaptive_adversarial` locked-test 上使用 `100` 个共享场景。未见
场景覆盖 `nominal` 和 `delayed_noisy` 观测层、3/4/5 个障碍和左右两种初始侧。

安全实验使用 64 个候选 reset seed，从中按初始 robust barrier 分层选出 8 个
回合，包含 tight 和 nominal 两个初始安全集 strata。执行器审计单独测试 mild
和 hard 延迟/噪声/跟踪/阻力随机化以及 immutable 与 flush-pending queue authority。

我们报告 episode-level safe capture、collision、boundary violation、timeout、
capture time、minimum clearance、planner valid/effective/convergence rate、
candidate coverage、candidate feasibility、solver failure、fallback、独立证书
有效率以及 p50/p95/p99 latency。三个训练 seed 的聚合值使用 arithmetic mean；
`+/-` 表示跨 seed 的 sample standard deviation。

---

# 3. Results

## 3.1 Main Findings

实验支持以下三个阶段性结论。

第一，projected portable-SSM diffusion 在未见 adaptive-adversarial 策略上保留
了约 90% 的全轨迹 coverage，并显著优于 projected GRU；但在原始五模式
locked-test 上，minFDE 提升未达到预注册的 10% 门槛，因此该预测优势不能被
写成无条件全面支配。

第二，候选轨迹只有经过 dynamics projection 并进入 worst-case DN-MPC 后，才
能稳定转化为未见目标策略上的围捕收益。固定 S3 和未见自适应场景均显示，
分布式规划在延迟、丢包甚至无通信条件下保持了较高的安全捕获率。

第三，velocity-level robust CBF-QP 在匹配的 robust-safe reset 和安全契约下
通过独立一步证书；但执行器队列和扰动审计暴露出当前契约不一致问题。修复
速度约束和动作变化约束后，P7 在速度级条件下达到 99% safe capture，但该
结果不等价于执行扰动下的多步闭环 R-CLBF-QP 证明。

## 3.2 Trajectory Prediction on the Five-Mode Locked Test

表 1 汇总三个训练 seed 的正式 locked-test 结果。数值越低越好的是 minADE、
minFDE 和 energy score，越高越好的是 coverage 与 feasible candidate fraction。

**表 1. 五模式 locked-test 的多模态预测结果。**

| Model | minADE (m) | minFDE (m) | Energy score | Full-trajectory coverage | Feasible candidates |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRU, raw | 0.4187 +/- 0.0122 | 0.6825 +/- 0.0163 | 1.6402 +/- 0.0461 | 90.06% +/- 1.24% | 7.82% +/- 1.12% |
| Portable SSM diffusion, raw | 0.4325 +/- 0.0030 | **0.6379 +/- 0.0078** | 1.7242 +/- 0.0243 | 89.26% +/- 0.28% | 0.00% |
| GRU, projected | **0.4128 +/- 0.0112** | 0.7075 +/- 0.0206 | 1.6456 +/- 0.0469 | 89.62% +/- 1.23% | 98.84% +/- 0.24% |
| Portable SSM diffusion, projected | 0.4161 +/- 0.0032 | **0.6802 +/- 0.0085** | 1.6918 +/- 0.0239 | 88.74% +/- 0.18% | **99.71% +/- 0.05%** |

相对 raw GRU，raw diffusion 的 minFDE 改善为 `6.54%`；相对 projected GRU，
projected diffusion 的 minFDE 改善为 `3.87%`。两者均低于预注册的 `10%`
强门槛。raw diffusion 候选的 acceleration-feasible fraction 接近零，说明
并行去噪生成的轨迹不能直接作为控制器输入。动力学投影使 diffusion 候选可行率
提高到 `99.71%`，代价是改变了预测分布并使误差略有增加。

按目标模式拆分后，diffusion 的收益主要来自 `s_curve` 和 `boundary_escape`；
在 `burst` 模式上，projected GRU 的 minFDE 为 `0.5710 +/- 0.0529 m`，优于
diffusion 的 `0.6939 +/- 0.0045 m`。因此，更准确的结论是 diffusion 对曲线型
和边界逃逸型策略更有优势，而不是在所有目标模式上全面支配 GRU。

GPU 单样本预测 p95 为 GRU `1.85 +/- 0.51 ms`、diffusion
`21.05 +/- 2.64 ms`。该数值是独立预测器延迟，不包含 DN-MPC、安全过滤和
执行器延迟。

## 3.3 Generalization to an Unseen Adaptive Target

我们进一步冻结三个 checkpoint，在未参与训练、模型选择或 calibration 的
`adaptive_adversarial` locked-test 上评估。该测试包含 `32` 个 episode 和
`7,616` 个窗口。

**表 2. 未见自适应目标策略审计。**

| Model and stage | minADE (m) | minFDE (m) | Energy score | Coverage | Feasible candidates |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRU, raw | 0.5039 +/- 0.0242 | 0.8908 +/- 0.0668 | 1.9831 +/- 0.1051 | 82.38% +/- 5.03% | 12.36% +/- 2.94% |
| Portable SSM diffusion, raw | **0.3235 +/- 0.0019** | **0.5051 +/- 0.0072** | **1.1879 +/- 0.0192** | **89.94% +/- 0.25%** | 0.0033% +/- 0.0016% |
| GRU, projected | 0.4976 +/- 0.0222 | 0.9033 +/- 0.0648 | 1.9903 +/- 0.1003 | 82.22% +/- 4.86% | **99.991% +/- 0.008%** |
| Portable SSM diffusion, projected | **0.3155 +/- 0.0016** | **0.5363 +/- 0.0071** | **1.1747 +/- 0.0152** | **90.31% +/- 0.03%** | 99.914% +/- 0.025% |

在 projected candidate 上，相对 projected GRU，diffusion 将 minFDE 降低
`40.63%`，energy score 降低 `40.98%`，coverage 提高 `8.09` 个百分点。
该结果是当前预测模块最强的正向证据，也说明 SSM 条件编码与多样本生成在
策略分布发生变化时仍能保留有效候选。然而，该结果不能覆盖五模式主实验的
10% gate 失败，也不能将 `uniform_uncalibrated` 候选权重描述为概率校准。

CPU 单样本未见策略审计 p95 为 GRU `8.08 +/- 0.96 ms`、diffusion
`39.37 +/- 8.77 ms`。预测刷新、候选缓存年龄和端到端控制延迟在规划实验中
单独报告。

## 3.4 Centralized Scenario Planning

在 `8` 个 formal-small flee-persistence 场景中，DynamicEncirclement、
expected MPC、worst-case MPC 和 CVaR MPC 均达到 `100%` safe capture、`0%`
collision。mean capture time 分别为 `0.9125 s`、`0.9000 s`、`0.9750 s` 和
`0.9000 s`；worst-case MPC planner p95 为 `11.37 ms`，total-control p95 为
`45.87 ms`。

该小规模场景较容易，不能单独证明 worst-case objective 的困难场景优势。
因此，主结论以下一节的固定场景和未见自适应目标锁定测试为依据，而不是该
诊断集的 100% 结果。

## 3.5 Distributed DN-MPC under Communication Faults

### Fixed S3 validation

在三个 checkpoint、12 个共享 S3 场景和四种通信条件下，所有分布式模式均达到
`100%` safe capture 与 `0%` collision，valid/effective/converged planning 均为
`100%`，只有 ideal communication 的 effective/converged aggregate 因极少数
数值有效性判据记录为 `99.97%`，仍高于预设阈值。DynamicEncirclement baseline
为 `91.67%` safe capture、`8.33%` collision。

**表 3. 固定 S3 场景的分布式规划结果。**

| Method | Safe capture | Collision | Valid plan | Effective plan | Convergence | Planner p95 (ms) | Total p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Centralized worst-case | 100.00% | 0.00% | 100.00% | 100.00% | n/a | 11.10 | 47.89 |
| Distributed ideal | 100.00% | 0.00% | 100.00% | 99.97% | 99.97% | 12.47 | 51.51 |
| Distributed 2-step delayed | 100.00% | 0.00% | 100.00% | 100.00% | 100.00% | 10.23 | 51.07 |
| Distributed 10% dropout | 100.00% | 0.00% | 100.00% | 100.00% | 100.00% | 10.23 | 50.51 |
| Distributed no communication | 100.00% | 0.00% | 100.00% | 100.00% | 100.00% | 8.72 | 50.38 |
| DynamicEncirclement | 91.67% | 8.33% | n/a | n/a | n/a | n/a | 1.58 |

通信审计确认延迟模式平均消息年龄约 `1.91` 步，dropout 模式平均消息年龄约
`2.06` 步，最大允许年龄为 `8` 步。每种通信模式的消息数、接收字节数和
dropout 数均被保存在 episode/step JSONL 中。

### Unseen adaptive-adversarial locked test

在 `100` 个未见自适应场景上，DynamicEncirclement baseline 的 safe capture 为
`95%`，collision 为 `2%`。集中式 worst-case 为 `97.33%` safe capture；分布式
ideal、delayed、dropout 和 no-communication 分别为 `98%`、`99%`、`99%` 和
`98%`。delayed 与 dropout 模式均达到 `0%` collision、`0%` boundary violation
和 `1%` timeout。

**表 4. 未见自适应目标的规划泛化结果。**

| Method | Safe capture | Collision | Boundary | Timeout | Valid/effective plan |
| --- | ---: | ---: | ---: | ---: | ---: |
| DynamicEncirclement | 95% | 2% | 0% | 3% | n/a |
| Centralized worst-case | 97.33% | 1% | 0% | 1.67% | 100% / 100% |
| Distributed ideal | 98% | 1% | 0% | 1% | 100% / 99.97% |
| Distributed 2-step delayed | **99%** | **0%** | **0%** | 1% | 100% / 100% |
| Distributed 10% dropout | **99%** | **0%** | **0%** | 1% | 100% / 100% |
| Distributed no communication | 98% | **0%** | **0%** | 2% | 100% / 100% |

这些结果表明，在当前仿真和候选缓存契约下，risk-sensitive DN-MPC 的未见
策略围捕效果优于 DynamicEncirclement baseline，且对有限通信故障不敏感。
但该结果仍是经验性的 finite-shooting best-response 证据，不等同于形式化
零和博弈均衡或全局最优性证明。

缓存预测每 `20` 个控制步刷新一次，最大候选年龄为 `19` 步。三个 checkpoint
的 aggregate total-control p95 为 `167.73--229.62 ms`，因此严格 10 Hz 的
`100 ms` 仅作为部署参考，没有被用作 P4 方法结果的硬门槛。当前系统不能据此
声称已经完成严格 10 Hz 部署。

## 3.6 Robust Safety Filter

在按初始 barrier 分层选出的 8 个 robust-safe reset 上，共执行 `130` 个控制
步。nominal、local CBF 与 robust CBF-QP 均达到 `100%` safe capture、`0%`
collision、`100%` independent certificate valid 和 `100%` next-state safe。
robust CBF-QP 的 solver success 为 `130/130`，QP infeasible、solver failure、
fallback 和 nonzero slack 均为零，最大约束违反为 `2.1e-15 m`，最小独立下一
状态 barrier 为 `0.1514 m`，filter p95 为 `6.27 ms`。

这证明了速度级 matched reset 条件下的可审计一步安全投影。它没有证明连续时间
扫掠体积安全、执行队列下的多步不变性或 R-CLBF-QP 闭环安全。

在执行扰动审计中，当前 execution-aware filter 尚未通过全部安全门槛：

| Variant | Safe capture | Collision | Boundary | Actual post robust state | Filter p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mild immutable | 37.5% | 0% | 25% | 90.78% | 671.36 |
| Hard immutable | 25.0% | 0% | 37.5% | 67.67% | 1945.74 |
| Mild flush-pending | 75.0% | 0% | 0% | 100% | 857.73 |
| Hard flush-pending | 50.0% | 0% | 0% | 91.70% | 2489.98 |

该反例非常关键：它说明安全层在名义下一状态通过时，仍可能因为已经进入
pending queue 的命令、跟踪误差和动力学不匹配而失去执行后的安全性。因此，
不能用分层 velocity-level 结果替代 execution-invariant 结论。

## 3.7 End-to-End P7 Integration

P7 将 projected portable-SSM diffusion、worst-case DN-MPC 和 robust safety
filter 串联，在同一份 `100` episode `adaptive_adversarial` locked scene 上
比较修复前、诊断版和 contract-repaired 版本。

**表 5. P7 联合实验及安全契约修复。**

| Version | Safe capture | Collision | Boundary | Independent certificate | Total-control p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original P7 | 50% | 49% | 16% | 64.87% | 114.80 ms |
| Matched-margin diagnostic | 30% | 70% | 3% | 93.61% | 46.95 ms |
| Contract-repaired P7 | **99%** | **0%** | **0%** | **100%** | **41.57 ms** |

contract-repaired P7 的 planner valid/effective rate 为 `100% / 100%`，
safety-filter certificate valid rate 为 `99.98%`。唯一失败为 `1%` timeout，
没有 collision 或 boundary violation。planner、predictor 和 safety p95 分别
为 `28.74 ms`、`14.12 ms` 和 `2.97 ms`。一次 SLSQP fallback 发生在 episode
79、step 8，fallback 动作通过 independent checker；该回合最终 timeout，
没有发生碰撞或越界。

该修复结果揭示了完整组合失败的主要原因并非 planner 单独失效，而是模块间
安全契约不一致：旧过滤器使用 `[-v_max/sqrt(3), v_max/sqrt(3)]` 的分量盒，
而环境和 independent checker 使用欧氏速度球；同时理想速度级环境被强制施加了
未建模的动作变化约束。修复后，过滤器、环境和 checker 共享同一速度级契约，
安全捕获率从 `50%` 提高到 `99%`。

因此，P7 的正确结论是：在速度级匹配契约下，三模块可以稳定组合；它不是
执行器扰动下的完整 R-CLBF-QP 证明，也不是已经完成真实无人机部署的结论。

## 3.8 Ablation and Sensitivity Analysis

已有 sampling ablation 在固定 40 episode validation 场景上比较了
`num_samples x diffusion_steps`：

| Configuration | Safe capture | Collision | Planner p95 (ms) | Predictor p95 (ms) | Total p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 x 8 | 92.5% | 7.5% | 13.65 | 37.61 | 51.42 |
| 4 x 8 | 92.5% | 7.5% | 9.35 | 38.96 | 48.71 |
| 2 x 8 | 92.5% | 7.5% | 6.70 | 37.62 | **44.86** |
| 8 x 4 | 92.5% | 7.5% | 27.34 | 38.76 | 53.90 |

在该 validation 样本中，减少候选数到 `2` 没有改变 episode-level outcome，且
获得最低 total p95；不过该 configuration 尚未经过新的 locked-test 预注册，
不能仅凭此 ablation 替换主实验配置。该结果支持的结论是：采样数量、去噪步数、
候选刷新周期与风险鲁棒性之间存在可测的工程折中，而不是“更大 K 始终更好”。

## 3.9 Results Required Before Claiming Full Completion

若后续实验全部顺利完成，投稿版本应在表 6 中替换占位符，并同时提供原始
JSONL、独立 checker 输出、三 seed 聚合和失败回合列表。

**表 6. 完整执行鲁棒版本的最终结果模板。**

| Metric | Current velocity-level P7 | Full execution-aware final result |
| --- | ---: | ---: |
| Safe capture | 99% | `[final value +/- seed variation]` |
| Collision | 0% | `[final value]` |
| Boundary violation | 0% | `[final value]` |
| Timeout | 1% | `[final value]` |
| Execution swept-volume certificate | not closed | `[final value]` |
| Continuous-segment certificate | not closed | `[final value]` |
| Multi-step forward-invariance | not proved | `[proved under stated assumptions]` |
| R-CLBF-QP solver success | not implemented/validated | `[final value]` |
| Fallback certificate validity | conditional | `[final value]` |
| End-to-end p95 | 41.57 ms velocity-level | `[final value]` |

只有当 `Full execution-aware final result` 一列具备独立证据后，结果段落才能
改写为：

> Under the stated dynamics, queue-authority, disturbance-bound, and continuous
> interpolation assumptions, the proposed R-CLBF-QP layer preserved the robust
> safe set over the complete evaluation horizon, including all accepted fallback
> branches. Across three independent seeds and the locked execution-perturbation
> test, the method achieved `[x]%` safe capture, `[y]%` collision, `[z]%`
> boundary violation, and `[q]%` independently certified actions.

否则应继续使用当前更严格的表述：

> The experiments demonstrate a conditional velocity-level robust CBF-QP
> composition under a matched simulation contract, while execution-invariant
> safety and the full R-CLBF-QP claim remain future work.

## 3.10 Reproducibility and Statistical Reporting

所有正式预测 checkpoint 保存在 `models/formal_phase2_v3/`，模型来源、训练 seed、
backend 和 SHA-256 记录在 `manifest.json`。主要报告和结果目录如下：

- `PHASE2_FORMAL_ANALYSIS_REPORT.md`
- `PHASE2_ADAPTIVE_GENERALIZATION_AUDIT_REPORT.md`
- `PHASE3_S3_VALIDATION_REPORT.md`
- `PHASE4_DN_MPC_VALIDATION_REPORT.md`
- `PHASE4_UNSEEN_ADAPTIVE_VALIDATION_REPORT.md`
- `PHASE5_STRATIFIED_VALIDATION_REPORT.md`
- `PHASE5_EXECUTION_STATE_AWARE_AUDIT_REPORT.md`
- `PHASE5_RECOVERABILITY_CONTRACT_AUDIT_REPORT.md`
- `PHASE7_INTEGRATION_REPAIR_REPORT.md`

当前回归测试为 `131 passed in 64.22 s`。投稿版应额外补充：硬件型号和驱动版本、
训练总时长、所有超参数搜索范围、每个独立 seed 的完整结果、置信区间或标准差
计算方式、locked scene SHA-256、预测刷新策略、失败分类和代码版本。

## 3.11 Final Interpretation

本研究的核心证据链可以概括为：

```text
SSM history encoding improves candidate diversity under policy shift;
diffusion candidates become executable only after dynamics projection;
worst-case DN-MPC converts candidate uncertainty into robust encirclement;
the safety layer prevents nominal planner actions from violating the matched set;
execution-aware certification is a separate, stricter scientific claim.
```

因此，若以当前证据投稿，最稳妥的主张是：

> 在部分观测、障碍和有限通信的三维速度级围捕仿真中，projected
> portable-SSM diffusion 与 worst-case distributed scenario MPC 能够在未见
> 自适应目标策略上获得稳定的围捕收益；在匹配安全契约下，robust CBF-QP
> 可以通过独立一步证书并实现 `99%` safe capture 与 `0%` collision 的联合结果。

不能写成：

- 官方 Mamba 已经被验证；
- raw diffusion 候选可直接执行；
- 候选 logits 已经是校准概率；
- DN-MPC 已经证明零和博弈最优性；
- 已经证明执行扰动下的多步 forward invariance；
- R-CLBF-QP 和真实无人机安全验证已经完成。

