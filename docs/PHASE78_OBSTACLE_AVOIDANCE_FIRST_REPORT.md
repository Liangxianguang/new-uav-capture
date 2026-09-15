# Phase 78：Obstacle-Avoidance-First 目标与 Nominal 标定

## 动机

Phase 77 的 central-crossing 合同要求目标主动进入并穿越中心障碍区，导致目标
行为明显强于当前研究问题需要的“平滑、非呆板、先避障再逃逸”目标。Phase 78
保留 Phase 77 的全部结果，但建立一个独立合同：目标不需要主动变道、不需要穿过
障碍物内部，只在起始阶段优先远离最近障碍物；防守方仍需绕过中心障碍物完成拦截。

locked-test 不参与本阶段，所有场景、专家筛选和训练只属于 development calibration。

## 目标行为实现

在 `adaptive_maneuvering` 中新增 obstacle-avoidance-first 几何先验：

- 计算目标到最近障碍物中心的水平反方向，并将候选方向与该方向的一致性加入评分；
- 新配置将 `target_maneuver_obstacle_avoidance_gain` 设为 `5.0`，
  `target_maneuver_enable_reverse_lane_change=false`，因此不再主动选择反向变道；
- 仍保留 6--10 步重规划、6 步最小保持、16 步候选 rollout、转向角速度、加速度和
  jerk 限制；
- lateral jink、vertical escape 和短时 speed burst 提供有限的非平凡机动，
  但候选仍先经过边界/障碍物净空过滤；
- 初始外逃证书只检查前 4 步，避免 Hard 档目标在高速度下被错误要求持续满速冲向
  世界边界。证书不等于闭环安全证明。

## 新场景池

权威数据目录：

```text
results/phase78_obstacle_avoidance_scenes_v2/
```

该目录包含 300 个场景、150 个 mirror groups，Easy/Nominal/Hard 各 100 场。
`target_crossing_required_rate=0`，每个场景的初始外逃前缀均通过障碍物和边界净空
检查，且每档保留固定外廊 transit certificate；Easy/Nominal 首对另外执行严格
conservative grid audit。该数据池与 Phase 77 完全独立。

生成命令：

```powershell
python scripts/generate_phase78_obstacle_avoidance_scenes.py `
  --output-dir results/phase78_obstacle_avoidance_scenes_v2
```

## 当前执行状态

- [x] obstacle-avoidance-first 候选评分和关闭 intentional reverse lane change；
- [x] Easy/Nominal/Hard 各生成 100 个 calibration 场景；
- [x] 初始外逃净空证书通过率 100%；
- [x] 定向测试通过；
- [x] Nominal oracle-route 专家上界与无效场景筛选：`84/100` 场景物理可行，
  target obstacle collision `0%`，target/defender boundary violation `2%/2%`；
- [x] 固定 Nominal 难度后启动三个 seed 的学习策略训练（`782501/782502/782503`）；
- [x] Easy oracle-route 专家标定完成：`97/100` 场景物理可行，safe capture-in-pursuit
  `97%`，target obstacle collision `0%`，target boundary violation `3%`；
- [x] 三个 Nominal checkpoint 的固定筛选池闭环复核完成：每个 seed 在 84 个专家筛选
  场景上运行，共 252 个 episode；
- [ ] 只在 Nominal 学习门控通过后打开 Hard/Stress；
- [x] locked-test 未读取、未修改。

Nominal oracle route-and-safety latency（与其他 CPU 进程并发，因而仅作诊断）为
`p50/p95/p99 = 6.69/199.89/275.65 ms`。后续学习策略会继续记录 route、actor、
safety 和 total 的完整 p50/p95/p99；100 ms 不是硬门槛。

## 固定 Nominal 学习策略结果

三 seed 的 route-aware recurrent BC 均使用同一个 84 场专家有效 Nominal 集，
并固定为 1-step execution delay、`0.02 m/s` command noise、local CBF。由于本阶段
直接滚动 recurrent actor，不经过扩散 predictor、DN-MPC planner 或 QDR，这三项
延迟对该 evaluator 不适用；以下报告 route-intent、actor、safety 和 total：

| seed | safe capture | collision | boundary | timeout | total p50/p95/p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 782501 | 55.95% | 26.19% | 9.52% | 17.86% | 8.75 / 273.05 / 375.37 |
| 782502 | 67.86% | 17.86% | 7.14% | 14.29% | 8.72 / 279.57 / 389.74 |
| 782503 | 55.95% | 10.71% | 2.38% | 33.33% | 9.01 / 297.53 / 417.73 |
| **加权合计** | **59.92%** | **18.25%** | **6.35%** | **21.83%** | **8.83 / 283.38 / 394.28** |

平均捕获时间为 `16.76 s`，平均最小净空为 `0.220 m`。三个 seed 均未通过预注册
promotion gate（safe capture `≥70%`、collision `≤5%`、boundary `≤5%`、timeout
`≤10%`）。因此 Phase78 目前证明了目标合同和场景标定可执行，但没有证明当前
学习防守策略具有稳定拦截能力。固定集评估的 route-intent/actor/safety/total
均值分位数分别为 `0.374/273.175/383.081`、`1.821/3.610/8.063`、
`6.182/8.380/11.131`、`8.828/283.385/394.281 ms`；p95/p99 受到并行 CPU 负载
影响，只作诊断，不能作为部署吞吐声明。

这一结果也解释了当前捕获率下降：Phase78 不是在复现 Phase16 的 GRU+DN-MPC，
而是只用 64 条接受示范训练的直接 recurrent BC。训练器内置验证为三个 seed
`0%/25%/25%` safe capture，固定 Nominal 集聚合后为 `59.92%`，说明主要问题是
示范覆盖不足和闭环 covariate shift，而不是需要把目标再改成主动变道或钻入障碍物。

聚合结果保存在：

```text
results/phase78_nominal_policy_eval_expertaccepted_aggregate.json
```

## Promotion 规则

Nominal 学习策略必须在专家筛选后的固定场景池上由所有 seed 同时满足：

- safe capture ≥ 70%；
- collision ≤ 5%；
- boundary violation ≤ 5%；
- timeout ≤ 10%；
- 若采用 crossing 合同才报告 target crossing，本阶段不将 crossing 当作成功条件；
- 同时保存 route/predictor/planner/QDR/safety/total 的 p50、p95、p99。

local CBF 继续标注为经验过滤器，不宣称 R-CLBF-QP 或 robust CBF-QP 安全证明。

## 下一步（不改变目标合同）

1. 先在同一 84 场 Nominal 集做三项诊断：无 delay/noise、仅 delay、delay+noise，
   区分观测/执行不确定性与策略本身的失败；每项保留三 seed 和完整 termination
   reason。
2. 将当前 64 条接受示范扩展到至少 192 条，并用闭环失败状态做 2--3 轮 DAgger/
   recovery 示范；重点覆盖 safety failure 前缀、队形分裂和 timeout 前缀。
3. 采用“route-aware DN-MPC teacher + recurrent residual actor”或 teacher-action
   loss 与 recovery loss 的混合训练，避免直接 BC 在长时域内自由漂移；训练/选择仍
   只用 calibration/validation。
4. 重新运行三 seed Nominal gate。只有 safe capture、collision、boundary、timeout
   同时达标，才打开 Hard/Stress；本阶段不修改 locked-test。
