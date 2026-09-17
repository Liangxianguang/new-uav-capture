# Phase 83 Hard-1 Calibration and Three-Seed Validation Report

更新时间：2026-09-17  
实验范围：`development_calibration` 与 `hard1_development_holdout`；未读取、修改或调参 `locked-test`。

## 1. 目的与预注册门槛

本阶段在已通过的 Nominal-plus 基础上，逐项增加目标速度、障碍物近距离和紧队形三个困难因素，验证模型是否可以从 Nominal-plus 推进到 Hard。每个因素单独构成一个 100-episode block，不把多个新增因素混在同一 calibration block 中。

学习策略 promotion gate 预注册为：每个 seed 的 safe capture `≥70%`，collision `≤5%`，boundary violation `≤5%`。所有闭环结果只使用 public-belief 可用的观测；oracle 只用于专家可行性上界与场景筛选。

`local CBF` 在本阶段仍是经验安全过滤器。本报告不宣称 R-CLBF-QP、robust CBF-QP 或任何形式化闭环安全证明。

## 2. 场景池与数据完整性

主 calibration pool 为 300 scenes、150 mirror groups；独立 holdout pool 也为 300 scenes、150 mirror groups。每个 pool 包含以下三个 100-scene 单因素 block：

| Block | 唯一新增因素 | 设置 |
| --- | --- | --- |
| `target_speed` | 目标速度 | target speed scale `0.55–0.65` |
| `obstacle_near` | 目标初始靠近障碍物 | initial side distance `4.5–5.0 m`，目标初始障碍物 clearance `约 1.84–2.40 m` |
| `formation_tight` | 防守方队形收紧 | defender y-scale `0.65`，初始最小机间距 `1.114 m` |

主 pool 场景 SHA-256：
`9a8a67328546d3ecd89fe0c8f9db8c1d164a61e7ca1fd847d3ca5afe78154f3e`。

主 pool manifest SHA-256：
`dd2a3c59e41a59b47a6083b824168a8eca549d8dc8752f46b8626631b8eb2312`。

holdout 场景 SHA-256：
`0a935043cbee3f9ba257e3abec89e3d8813dc24d02b78e6d184fa8031f7493e0`。

holdout manifest SHA-256：
`d14cdf1cb5a94d0e185f2f8c9feec781cbf2489f7d2cfb394f8ace42bad9306f`。

所有场景均为 calibration/development 数据，`locked_test_used=false`。目标仍使用 obstacle-avoidance-first 合同：不要求主动穿越中心障碍区，不启用 intentional reverse lane change，且保留左绕、右绕、上绕候选。生成时的旧 `difficulty` 字段仍有部分记录写作 `nominal_plus1`，但 `variant`、协议文件和上述哈希唯一确定本阶段 Hard-1 语义；后续生成器已支持显式写入 `hard1` 标签。

## 3. 专家筛选

专家 acceptance gate 为 `≥80%`。主 pool 和独立 holdout 的 oracle/public-belief 接受率如下：

| Pool / block | Oracle | Public-belief | 是否达到 80% |
| --- | ---: | ---: | --- |
| Main `target_speed` | 86% | 92% | 是 |
| Main `obstacle_near` | 85% | 87% | 是 |
| Main `formation_tight` | 88% | 87% | 是 |
| **Main overall** | **86.33% (259/300)** | **88.67% (266/300)** | **是** |
| Holdout `target_speed` | 90% | 88% | 是 |
| Holdout `obstacle_near` | 89% | 96% | 是 |
| Holdout `formation_tight` | 87% | 87% | 是 |
| **Holdout overall** | **88.67% (266/300)** | **90.33% (271/300)** | **是** |

public-belief holdout 的 271 个 accepted scenes 用作独立 Hard-1 闭环验证集；主 pool 中 public-belief accepted scenes 形成 266-scene training pool、144 mirror groups，三个 block 分别为 `target_speed=92`、`obstacle_near=87`、`formation_tight=87`。训练没有使用 locked-test。

专家筛选中的 target-obstacle collision 是目标风险诊断，不等价于 defender collision，也不从学习策略闭环安全率中删除或隐藏。它只用于判断场景是否满足物理可行性契约。

## 4. 学习策略与闭环结果

三个 seed 均完成 192 个质量门控初始示范、两轮各 12 个 DAgger recovery rollout，并在同一个独立 271-scene public-belief holdout 上运行完整闭环。

| Seed | Episodes | Safe capture | Collision | Boundary | Timeout | Mean capture time | Mean min clearance | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 835601 | 271 | 97.79% (265) | 0.37% (1) | 0.37% (1) | 1.85% (5) | 8.146 s | 0.678 m | Pass |
| 835602 | 271 | 98.89% (268) | 0.37% (1) | 0.37% (1) | 0.74% (2) | 7.979 s | 0.689 m | Pass |
| 835603 | 271 | 99.26% (269) | 0.37% (1) | 0.00% (0) | 0.37% (1) | 7.974 s | 0.672 m | Pass |
| **Pooled** | **813** | **98.65% (802)** | **0.37% (3)** | **0.25% (2)** | **0.98% (8)** | — | — | **Pass** |

Pooled safe-capture 95% Wilson interval 为 `[97.59%, 99.24%]`。三个 seed 均独立通过预注册 Hard-1 gate，因此 Hard 阶段可以开放 Stress calibration；该结论不代表 Stress 已经通过。

## 5. 延迟分位数

评估器保存的是 route-intent、actor、safety 和 total 的逐控制步延迟；由于本阶段没有单独运行 predictor/planner/QDR 计时器，不能虚构这些未记录 stage 的分位数。

| Seed | Route p50/p95/p99 (ms) | Actor p50/p95/p99 (ms) | Safety p50/p95/p99 (ms) | Total p50/p95/p99 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 835601 | 9.224 / 334.930 / 460.469 | 1.229 / 1.656 / 2.143 | 6.475 / 8.015 / 9.291 | **17.118 / 342.839 / 468.436** |
| 835602 | 9.244 / 334.195 / 461.127 | 1.230 / 1.659 / 2.153 | 6.493 / 7.987 / 9.194 | **17.152 / 341.872 / 468.367** |
| 835603 | 9.250 / 340.880 / 465.922 | 1.235 / 1.679 / 2.168 | 6.490 / 8.082 / 9.320 | **17.162 / 348.778 / 473.742** |

100 ms 不是本阶段的硬门槛；p95/p99 的尾延迟主要来自 route-intent 几何规划。后续 Stress 仍必须报告同样的 p50/p95/p99，并把未测量的 predictor/planner/QDR stage 明确标记为 unavailable，而不是用 total 拆分猜测。

## 6. Recovery 数据与局限

三个 Hard-1 训练输出分别保留约 16,945、17,450、17,422 帧 DAgger recovery 数据，包含 `local_observations`、`base_actions`、`teacher_actions` 与 `recovery_masks`。这些数据覆盖 obstacle-near、formation-tight 以及 CBF 投影后的教师动作修正。

当前 recovery mask 在该批 teacher rollout 上几乎饱和为 true，因此它适合作为 recovery 数据保留与加权依据，不应被解释为稀疏的 CBF 触发标签。后续 Stress 应继续保存 base-action gap、base-CBF correction、teacher-CBF correction 的连续值。

## 7. 正式结论与下一步

Hard-1 在独立 holdout 上取得稳定的经验闭环结果，满足当前预注册的三个 seed gate；因此现在可以测试 Stress。当前仍不能声称：

1. 对所有更困难组合场景具有泛化能力；
2. local CBF 提供形式化安全证明；
3. predictor、planner、QDR 的 stage latency 已被完整测量；
4. 已经通过 locked-test。

下一步只在 development calibration 上生成并筛选至少 300 个 Stress scenes，先检查 oracle/public-belief 专家接受率，再运行 Hard-1 checkpoint 的 Stress 闭环。Stress 通过后才考虑一次性 locked-test diagnostic。

可复现产物：

- Hard-1 主场景：`results/phase83_hard1_calibration_300/`
- Hard-1 holdout：`results/phase83_hard1_calibration_300_holdout/`
- 专家聚合：`results/phase83_hard1_expert_aggregate.json`、`results/phase83_hard1_expert_holdout_aggregate.json`
- 训练池：`results/phase83_hard1_eligible_training_pool/`
- 三 seed 输出：`results/phase83_dnmcp_dagger_residual_hard1_seed835601/`、`results/phase83_dnmcp_dagger_residual_hard1_seed835602/`、`results/phase83_dnmcp_dagger_residual_hard1_seed835603/`
- 三 seed 聚合：`results/phase83_hard1_three_seed_validation.json`
- 生成器：`scripts/generate_phase82_nominal_plus1_scenes.py`
- 训练/评估器：`scripts/train_phase79_dagger_residual.py`
