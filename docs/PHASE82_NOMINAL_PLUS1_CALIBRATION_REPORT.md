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

## 3. 学习策略三 seed 独立 validation

从四个通过 teacher 门槛的 block 中选择的 357 个训练场景用于 DAgger residual actor；独立
validation pool 为 362 个场景、198 个 mirror groups，训练和 validation 的场景 hash 不同。
三组实验均完成 192 个质量门控专家示范、两轮各 12 个 DAgger recovery rollout，并只在
`development_validation_only` 上评估。锁定测试没有使用。

| Seed | Episodes | Safe capture | Collision / boundary | Timeout | Mean capture time (s) | Total p50/p95/p99 (ms) | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 824601 | 362 | 92.27% | 7.18% / 7.18% | 0.55% | 8.560 | 3.246 / 77.702 / 121.743 | Fail |
| 824602 | 362 | 94.48% | 5.25% / 5.25% | 0.28% | 8.418 | 3.255 / 77.374 / 119.919 | Fail |
| 824603 | 362 | 92.82% | 6.08% / 5.80% | 1.10% | 8.642 | 3.258 / 77.429 / 120.451 | Fail |
| **Pooled** | **1,086** | **93.19%** | **6.17% / 6.08%** | **0.64%** | — | — | **Fail** |

安全捕获的 pooled 95% Wilson 区间为 `[91.53%,94.54%]`。三 seed 的安全捕获都明显高于
70% 下限，但没有一个同时满足 collision ≤5% 且 boundary ≤5%；因此预注册的
`three_seed_gate_passed=false`，Hard、Stress 和 locked-test 不开放。

对失败行进行的可观测归因显示：67 个 `safety_failure` 中有 66 个同时报告
`world_violation_steps>0` 且 `min_clearance_m≥0`，只有 1 个出现负的障碍物/机间最小
clearance。这表明当前主要问题是延迟/命令噪声与 residual action 共同造成的 world-boundary
闭环失效；不能把它描述为“障碍物绕行失败”，也不能仅凭 `collision` 字段区分越界对象
是 target 还是 defender。下一次 calibration 必须单独记录 target/defender boundary
violation counters，并测试 execution-aware boundary projection 或更小 residual scale。

当前 actor evaluator 可复现地报告 `route_intent/actor/safety/total` 的 p50/p95/p99；它没有
单独保存 predictor、planner 和 QDR 的原始 stage latency，因此本阶段不虚构
`predictor/planner/QDR` 分位数。后续若将 actor 作为完整 predictor--planner--QDR 链路的
主模型，必须补充同一控制步的 stage-level raw latency logging。

三 seed 汇总文件：`results/phase82_dagger_three_seed_validation_v1.json`；可复现汇总脚本：
`scripts/aggregate_phase82_dagger_results.py`。

## 4. 已保存的复现实验产物

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
- 合格训练池：`results/phase82_eligible_training_pool_v2/`
- 独立 validation 池：`results/phase82_eligible_validation_pool_v2/`
- 三 seed residual actor 输出：`results/phase82_dnmcp_eligible_seed824601/`、
  `results/phase82_dnmcp_eligible_seed824602/`、`results/phase82_dnmcp_eligible_seed824603/`
- 三 seed 汇总：`results/phase82_dagger_three_seed_validation_v1.json`

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

## 5. 下一步决策计划

### A. 先修复 Nominal-plus 闭环安全失败

当前不继续增加 Hard。先在 development calibration/validation 复现边界失败，并保持
locked-test 不变。action-delay=2 仍不进入训练，除非 public-belief teacher 重新达到 80%
acceptance；其余修复必须先经过单因素 calibration：

- 增加 target/defender 分离的 boundary counters 和首个 violation step；
- 固定 actor，只比较 execution-aware boundary projection、residual scale `0.25/0.35/0.50`；
- 每个候选只在 calibration 上先做 100 场景筛选，记录 safe capture、boundary、timeout、
  recovery rate 及 route/actor/safety/total p50/p95/p99；
- 只有一个候选在 calibration 上同时达到 safe capture≥70%、collision/boundary≤5% 时，
  才在同一独立 validation pool 上复跑三 seed。

每个新 ramp 仍需至少 100 个 calibration episode、完整镜像组和独立 manifest。

### B. 重新训练与三 seed validation

对每个 block 要求：oracle 与 public-belief 均 `≥80%`，且不存在明显的场景生成契约错误。当前
四个 block 已满足，但当前 residual actor 三 seed gate 未通过；validation 使用独立
seed-offset 场景，不得使用训练场景或 locked-test。修复后仍要求 3 个 seed 的 safe capture
`≥70%`、collision/boundary `≤5%`，通过后才允许进入 Hard。

### C. 失败归因与 runtime

恢复数据采集完成后，按 `teacher_recovery`、`cbf_projection`、`obstacle_near`、`formation_spacing`
统计触发率，并补齐 predictor/planner/QDR-or-tube/safety/total 的 p50、p95、p99。local CBF
仍只作为经验过滤器，任何安全结论都必须限定为经验 outcome，不写成形式化证书。
