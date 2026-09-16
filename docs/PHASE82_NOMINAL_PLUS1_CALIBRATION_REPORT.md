# Phase 82 Nominal-plus-1 困难场景标定报告

更新时间：2026-09-16
实验范围：development calibration only；未读取、修改或调参 locked-test。

本报告以与 Phase81 一致的 canonical `safety_margin=1.0 m` 重筛结果为权威结果。
此前使用环境默认 `0.35 m` 的 v4 运行保留为评估契约诊断，不用于训练门槛判断。

## 1. 目的与场景契约

本阶段验证“是否应该继续增加困难场景”，但先判断新增难度是否仍处于可学习范围。场景池共 500 个 episode、250 个完整镜像组，分为 5 个互相独立的单因素 block，每个 block 100 个 episode：

| Block | 相对 Phase81 的唯一变化 | 生成后几何检查 |
| --- | --- | --- |
| `command_noise` | command noise `0.025 → 0.030 m/s` | 其余配置保持 Nominal-plus |
| `action_delay` | action delay `1 → 2` steps | 其余配置保持 Nominal-plus |
| `maneuver_frequency` | 目标重规划间隔 `8 → 6` steps | 其余配置保持 Nominal-plus |
| `obstacle_near` | 初始侧向距离 `[6.0,6.5] → [4.5,5.0]`，且目标初始障碍物 clearance `≤2.40 m` | 实际 clearance `1.868–2.395 m` |
| `formation_tight` | defender y-scale `1.0 → 0.65` | 初始最小机间距 `1.114 m` |

目标仍然使用 Maneuvering Adversary v2 的平滑、有界机动：不要求目标主动穿越中心障碍区，不启用 intentional reverse lane change，不把目标真实状态提供给 public-belief controller。障碍物另一侧的左绕、右绕、上绕候选仍由场景与目标规则保留。所有 500 个场景的四步初始 outward-escape certificate 均通过。

场景文件 SHA-256：
`7fe8b05adcdf3a770e4dcb20c99310a3f51d6009b30e60ca5a1804705c0cb923`。

## 2. 专家筛选结果

筛选使用 local CBF，但它只是经验过滤器；本报告不作 robust CBF-QP、R-CLBF-QP 或闭环安全证明。`oracle_route` 读取目标真实状态，只用于物理可行性上界；`public_belief_route` 使用延迟、噪声、丢包后的公开 belief，是训练前真正的教师可用性门槛。训练门槛预注册为每个 block 的 acceptance rate `≥80%`。

| Block | Oracle 接受率 | Public-belief 接受率 | Public physical collision | Public target/defender boundary | Public timeout | Public route-and-safety p50/p95/p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `command_noise` | 96% | 91% | 0% | 7% / 0% | 2% | 1.401 / 35.822 / 53.343 |
| `action_delay` | 83% | 78% | 8% | 5% / 5% | 4% | 1.423 / 36.558 / 54.337 |
| `maneuver_frequency` | 98% | 88% | 0% | 9% / 0% | 3% | 1.442 / 37.335 / 56.184 |
| `obstacle_near` | 94% | 88% | 0% | 9% / 0% | 3% | 1.420 / 32.774 / 50.249 |
| `formation_tight` | 92% | 90% | 0% | 7% / 0% | 3% | 1.302 / 33.457 / 49.270 |
| **总体** | **92.6%** | **87.0%** | **1.6%** | **7.4% / 1.0%** | **3.0%** | **1.407 / 35.192 / 52.527** |

接受率的总体 95% Wilson 区间为 oracle `[89.97%, 94.58%]`、public-belief `[83.77%, 89.67%]`。public-belief 的 block 接受率为 `91%/78%/88%/88%/90%`；除 action-delay=2 外，其余四个 block 达到 80% 门槛。因此本阶段的正式决策是：

1. 只使用 command-noise、maneuver-frequency、obstacle-near、formation-tight 四个达标 block 的 357 个 accepted scenes 建立训练池；
2. action-delay=2 block 暂不进入训练，保留为未达标困难诊断；
3. 先在独立 validation 上完成三 seed 学习策略验证，不开放 Hard、Stress 或 locked-test；
4. 保留原始场景、两版 expert episodes 和失败归因数据，作为困难度诊断；
5. 当前不能把专家接受率直接解释为学习策略捕获率，学习策略结果必须在独立 validation 上重新测量。

总体 public-belief route-and-safety 延迟的 p50/p95/p99 为 `1.407/35.192/52.527 ms`。这些是专家 route controller 的诊断耗时，不是 predictor/planner/QDR/safety/total 的部署基准；正式 runtime 比较仍需受控单进程 benchmark。

## 3. 已保存的复现实验产物

- 场景生成器：`scripts/generate_phase82_nominal_plus1_scenes.py`
- 场景协议：`configs/phase82_nominal_plus1.yaml`
- 原始 500 场景：`results/phase82_nominal_plus1_calibration_500_v4/`
- oracle 筛选（默认 margin 诊断）：`results/phase82_oracle_calibration_nominal_plus1_v4/`
- public-belief 筛选（默认 margin 诊断）：`results/phase82_public_belief_calibration_nominal_plus1_v4/`
- oracle canonical margin=1.0：`results/phase82_oracle_calibration_nominal_plus1_v5_margin1/`
- public-belief canonical margin=1.0：`results/phase82_public_belief_calibration_nominal_plus1_v5_margin1/`
- block 级聚合：`scripts/aggregate_phase82_expert_calibration.py`
- 聚合结果：`results/phase82_expert_calibration_aggregate_v5_margin1.json`
- 恢复/CBF 投影采集器：`scripts/collect_phase82_recovery_data.py`

恢复数据采集只用于定位 teacher recovery、CBF projection、障碍物接近和编队紧约束触发帧，不会改变模型或锁定测试。

本次采集已经完成：500 个 episode 共保留 `43,704` 个触发帧，输入特征形状为
`[43,704, 4, 70]`。触发原因计数为 `teacher_recovery=43,704`、
`cbf_projection=43,315`、`formation_spacing=1,162`、`obstacle_near=86`；原因可以重叠。
teacher recovery 在所有保留帧上均为 true，说明当前 residual/CBF recovery 判据在这批
场景上存在饱和，不能把该布尔标签直接当成有效的稀疏 recovery supervision。下一轮应
分别保存 base-action gap、base-CBF projection 和 teacher-CBF projection 的连续值，
并在 calibration 上重新确定阈值。

该 collector 使用 public-belief DN-MPC teacher 的独立诊断 rollout，结果为 safe capture
`383/500=76.6%`、collision `104/500=20.8%`、boundary `38/500=7.6%`、timeout
`13/500=2.6%`。它不是学习策略结果，也不能替代前面的专家 acceptance gate；local CBF
仍然只是经验过滤器。

## 4. 下一步决策计划

### A. 先完成独立 validation 筛选

不继续盲目增加 Hard。当前四个 block 已达到 teacher 门槛，先在独立 seed-offset validation
池上进行同契约筛选；action-delay=2 仍不训练。若 validation pool 的可行性明显低于
calibration，则回退为更小的 ramp：

- delay：`1 → 1.5` 不适用离散步长，因此保持 1，单独诊断 `2`；
- command noise：先测试 `0.0275`，再决定是否保留 `0.030`；
- maneuver interval：先保持 8，确认 interval 6 的失败是否来自重规划频率而不是 teacher route；
- obstacle-near：保留 clearance `≤2.40 m`，优先修复绕行路线/队列恢复，而不是继续缩短初始距离；
- formation-tight：先用 y-scale `0.80`，确认 `0.65` 的失败是否由编队碰撞而不是目标机动造成。

每个新 ramp 仍需至少 100 个 calibration episode、完整镜像组和独立 manifest。

### B. 训练与三 seed validation

对每个 block 要求：oracle 与 public-belief 均 `≥80%`，且不存在明显的场景生成契约错误。当前
四个 block 已满足，训练使用 357 个 accepted scenes；validation 使用独立 seed-offset 场景，
不得使用训练场景或 locked-test。训练后要求 3 个 seed 的 safe capture `≥70%`、
collision/boundary `≤5%`，通过后才允许进入 Hard。

### C. 失败归因与 runtime

恢复数据采集完成后，按 `teacher_recovery`、`cbf_projection`、`obstacle_near`、`formation_spacing` 统计触发率，并报告 predictor/planner/QDR/safety/total 的 p50/p95/p99。local CBF 仍只作为经验过滤器，任何安全结论都必须限定为经验 outcome，不写成形式化证书。
