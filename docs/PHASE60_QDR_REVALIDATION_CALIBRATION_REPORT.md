# Phase 60：QDR 重新验证与强基线校准报告

## 1. 结论

Phase 60 的 calibration 已完成，但不满足进入独立 confirmation 的全部门槛，
因此本阶段判定为 **实现与诊断通过，主方法晋级 No-Go**。这不是 locked-test
结果，也不是安全证明。

本阶段得到三个可复现结论：

1. 新场景矩阵、镜像组划分、route-validity 审计和 QDR 时间索引审计均通过。
2. QDR 在 delay-noise 和 joint-stress 条件下显著降低 collision，但代价是
   timeout 和 suffix-gate exhaustion 增加；不能只用 safe capture 宣称全面提升。
3. candidate-budget 合同得到真实日志支持：M8 fixed-K8 的实际候选数为 8；
   M0--M7 本次校准请求并实现的是单候选（K=1），所以本报告不把 M6/M7 写成
   自适应 K 或 K4/K8 实验。

local CBF 在全部闭环运行中仍只是经验动作过滤器。robust CBF-QP/R-CLBF-QP
没有作为本阶段主方法，也没有宣称闭环安全证书或 forward-invariant safety
proof。

## 2. 数据、配置与可复现性

| 项目 | 结果 |
| --- | --- |
| 全量新矩阵 | 360 episodes / 180 mirror groups |
| 每个 split | 120 episodes / 60 mirror groups |
| calibration selection | 120 episodes / 60 mirror groups |
| calibration blocks | ID reference、delay-noise factorial、communication-execution factorial、joint stress/transfer |
| 校准 seed | 727201、727202、727203 |
| 方法数 | 9（M0--M8） |
| 实际闭环 episode 数 | 3,240（9 方法 × 3 seed × 120 scene） |
| 完整矩阵 manifest SHA-256 | `53688916ad4bd770cb955dda983bfd18b2e5536e59961970e3ba2c5a3fb534ab` |
| calibration selection SHA-256 | `1b178f151f5dbaee3af94542caab62584de64ccf45213ca48efb54aece0bff24` |
| local CBF | empirical filter only |
| robust CBF-QP/R-CLBF-QP | diagnostic only; proof not claimed |

每个方法固定同一场景、checkpoint、采样 seed、MPC 参数、执行器、local CBF
开关和 Torch `1/1` 线程设置。结果保存在被 Git 忽略的 `results/` 目录，聚合
JSON 和 TensorBoard 目录分别为：

```text
results/phase60_diagonal_ssm_calibration_by_block.json
results/phase60_diagonal_ssm_calibration_by_block_tensorboard/
```

## 3. QDR 形式化与时间索引审计

对 immutable pending queue

```text
Q_t = (u_t^0, ..., u_t^(q-1))
```

和新规划 suffix `V_t=(v_t^0,...,v_t^(H-1))`，本阶段采用：

```text
x_prefix = Rollout(x_t, Q_t)
x_first_controllable = x_prefix[q]
x_suffix = Rollout(x_first_controllable, V_t)
R_QDR = concat(x_prefix, x_suffix[1:])
```

因此 candidate `v_t^j` 的首次物理作用时刻是 `t+q+j`。独立 checker 在
`q={0,2,4,8,10}`、`H=8` 上得到：

- 正常 rollout 的 position/velocity 最大误差均为 `0`；terminal time-index
  error 均为 `0`；
- 故意把同一 queue delay 再加一次的五个 negative probes 均被识别并拒绝；
- `overall_pass=true`。

这只证明时间索引和 queue composition 等价，不证明 collision-free、概率覆盖
或闭环安全。artifact：`results/phase60_qdr_time_index_audit/`。

## 4. Candidate budget 审计

三个 seed 的全部 9 方法均为 `candidate_budget_realized_rate=100%`、
`candidate_budget_mismatch_steps=0`。但请求值的含义必须如实区分：

| 方法 | 请求 K | 实际 K min/max | 解释 |
| --- | ---: | ---: | --- |
| M0--M7 | 1 | 1/1 | 单候选闭环基线/QDR，不是 K4/K8 消融 |
| M8 fixed_k8_qdr | 8 | 8/8 | 真实 fixed-K8 对照 |

因此 Phase 60 证明了“请求值没有被静默改写”，但没有证明 M6/M7 已完成
uncertainty-triggered adaptive K。要研究 adaptive K，后续必须显式打开 adaptive
调度并记录每步 K 的 bucket、refresh、cache age 和 realized count。

## 5. 闭环 outcome

下表为每个 block 的 `safe capture / collision / timeout`，百分比按三个
checkpoint seed 的 episode 结果聚合。

### 5.1 ID reference

| 方法 | Safe capture | Collision | Timeout |
| --- | ---: | ---: | ---: |
| M0 current-state delayed | 97.78% | 2.22% | 0.00% |
| M1 known-delay delayed | 100.00% | 0.00% | 0.00% |
| M2 fixed-tube | 100.00% | 0.00% | 0.00% |
| M3 queue-aware tube | 97.78% | 2.22% | 0.00% |
| M4 synchronous distributed | 96.67% | 3.33% | 0.00% |
| M5 asynchronous distributed | 95.56% | 4.44% | 0.00% |
| M6 QDR synchronous | **100.00%** | **0.00%** | 0.00% |
| M7 QDR asynchronous | **100.00%** | **0.00%** | 0.00% |
| M8 fixed-K8 QDR | 96.67% | 3.33% | 0.00% |

M6 相对 M0 的 safe-capture paired delta 为 `+2.22 pp [0.00,+6.67]`；M7
相对 M5 为 `+4.44 pp [+1.11,+8.92]`。ID 非劣方向满足，但这不足以代表
高延迟压力下的整体结论。

### 5.2 Delay-noise factorial

| 方法 | Safe capture | Collision | Timeout |
| --- | ---: | ---: | ---: |
| M0 | 48.89% | 51.11% | 0.00% |
| M1 | 52.22% | 47.78% | 0.00% |
| M2 | 50.00% | 50.00% | 0.00% |
| M3 | 55.56% | 42.22% | 2.22% |
| M4 | 53.33% | 46.67% | 0.00% |
| M5 | 63.33% | 36.67% | 0.00% |
| M6 | **78.89%** | 8.89% | 12.22% |
| M7 | **78.89%** | **4.44%** | 16.67% |
| M8 fixed-K8 | 52.22% | 44.44% | 3.33% |

M6 相对 M0 的 safe-capture delta 为 `+30.00 pp [18.89,43.33]`，但 timeout
delta 为 `+12.22 pp [+3.33,+24.44]`；M7 相对 M5 的 safe-capture delta
为 `+15.56 pp [+7.78,+23.33]`，timeout delta 为 `+16.67 pp [+7.78,+26.67]`。
这属于 safety--liveness trade-off，而不是无代价改进。

### 5.3 Communication-execution factorial

| 方法 | Safe capture | Collision | Timeout |
| --- | ---: | ---: | ---: |
| M0 | 95.56% | 4.44% | 0.00% |
| M1 | 97.78% | 2.22% | 0.00% |
| M2 | 96.67% | 3.33% | 0.00% |
| M3 | 85.56% | 14.44% | 0.00% |
| M4 | **100.00%** | **0.00%** | 0.00% |
| M5 | 83.33% | 16.67% | 0.00% |
| M6 | 94.44% | 4.44% | 1.11% |
| M7 | 94.44% | 5.56% | 0.00% |
| M8 fixed-K8 | 75.56% | 24.44% | 0.00% |

该 block 中 M6 相对 M0 的 safe-capture paired CI 为
`-7.78,+5.56 pp`，不能宣称 QDR 提升；M7 相对 M5 为
`+1.11,+20.00 pp`，但不能覆盖 M6/M7 的异步单模块归因。

### 5.4 Joint stress / transfer

| 方法 | Safe capture | Collision | Timeout |
| --- | ---: | ---: | ---: |
| M0 | 2.22% | 97.78% | 0.00% |
| M1 | 11.11% | 88.89% | 0.00% |
| M2 | 2.22% | 97.78% | 0.00% |
| M3 | 32.22% | 40.00% | 27.78% |
| M4 | 3.33% | 96.67% | 0.00% |
| M5 | 3.33% | 96.67% | 0.00% |
| M6 | **51.11%** | 15.56% | 33.33% |
| M7 | **50.00%** | **13.33%** | 36.67% |
| M8 fixed-K8 | 48.89% | 48.89% | 2.22% |

M6/M7 显著减少碰撞，但 timeout 已超过预注册 `<=5%` 门槛；全量 raw log
中的最大 suffix-gate exhaustion streak 分别为 M6 `168`、M7 `112`、M8 `41`
步，均超过预注册 `24` 步上限。因此不能进入 confirmation。

## 6. 完整延迟统计

以下每个单元均为 `p50/p95/p99`，单位 ms。`QDR/tube` 对无 QDR/tube 的方法
记为 `0/0/0`；这些是 pooled step latency，不是平均 latency。

### 6.1 ID reference

| 方法 | Predictor | Planner | QDR/tube | Safety | Total |
| --- | --- | --- | --- | --- | --- |
| M0 | 6.36/10.68/16.12 | 3.37/5.87/8.75 | 0/0/0 | 0.53/0.85/1.81 | 10.97/18.32/25.09 |
| M1 | 8.30/16.27/21.02 | 4.35/8.95/11.54 | 0/0/0 | 0.67/1.87/2.18 | 14.60/26.73/32.92 |
| M2 | 7.89/14.64/18.82 | 4.24/7.94/10.52 | 0/0/0 | 0.65/1.35/2.04 | 13.95/23.97/30.40 |
| M3 | 7.99/16.40/22.08 | 11.40/20.90/25.98 | 0.67/1.58/2.11 | 0.64/1.80/2.34 | 23.77/40.68/50.20 |
| M4 | 7.62/13.26/17.61 | 5.86/10.18/13.82 | 0/0/0 | 0.63/1.16/1.96 | 15.15/24.67/31.00 |
| M5 | 7.64/13.45/17.99 | 5.75/10.29/13.60 | 0/0/0 | 0.63/1.20/2.03 | 15.09/24.39/31.23 |
| M6 | 7.70/13.14/17.84 | 12.99/20.59/26.49 | 0.64/1.02/1.78 | 0.64/1.12/2.01 | 24.81/36.53/46.56 |
| M7 | 7.75/13.83/19.14 | 12.32/20.44/25.56 | 0.64/1.21/1.90 | 0.64/1.19/2.06 | 24.14/37.53/45.83 |
| M8 | 8.19/14.65/20.24 | 39.81/59.73/73.18 | 0.66/1.17/1.91 | 0.64/1.31/2.27 | 52.00/76.74/93.64 |

### 6.2 Delay-noise factorial

| 方法 | Predictor | Planner | QDR/tube | Safety | Total |
| --- | --- | --- | --- | --- | --- |
| M0 | 7.07/13.75/18.92 | 3.82/7.41/10.63 | 0/0/0 | 0.60/1.28/2.10 | 12.43/22.96/30.06 |
| M1 | 8.42/17.27/21.78 | 4.40/9.15/11.84 | 0/0/0 | 0.68/1.99/2.34 | 15.06/27.71/33.59 |
| M2 | 7.84/15.25/19.62 | 4.19/7.99/10.67 | 0/0/0 | 0.65/1.75/2.23 | 13.87/24.22/30.66 |
| M3 | 7.88/16.11/21.06 | 11.28/20.77/26.32 | 0.85/2.16/3.37 | 0.64/1.79/2.34 | 23.83/40.38/49.94 |
| M4 | 7.67/13.94/18.88 | 5.94/10.90/14.86 | 0/0/0 | 0.64/1.41/2.14 | 15.47/26.10/32.33 |
| M5 | 7.76/14.74/18.96 | 5.87/11.04/14.31 | 0/0/0 | 0.64/1.36/2.18 | 15.51/26.71/32.62 |
| M6 | 7.72/13.40/18.40 | 12.56/20.69/26.42 | 1.03/1.75/2.99 | 0.64/1.20/2.09 | 24.60/37.62/46.31 |
| M7 | 7.77/14.91/20.03 | 11.98/21.05/27.19 | 1.04/2.13/3.38 | 0.64/1.68/2.22 | 24.24/39.56/48.71 |
| M8 | 8.12/15.04/20.19 | 40.26/62.57/77.25 | 0.70/1.69/3.07 | 0.64/1.43/2.04 | 52.88/79.51/98.75 |

### 6.3 Communication-execution factorial

| 方法 | Predictor | Planner | QDR/tube | Safety | Total |
| --- | --- | --- | --- | --- | --- |
| M0 | 7.62/15.69/20.04 | 4.09/8.57/10.88 | 0/0/0 | 0.63/1.72/2.18 | 13.49/25.52/31.73 |
| M1 | 8.61/17.75/22.62 | 4.44/9.42/11.63 | 0/0/0 | 0.68/2.01/2.32 | 15.41/28.29/35.03 |
| M2 | 7.78/15.94/20.60 | 4.17/8.69/11.18 | 0/0/0 | 0.64/1.84/2.26 | 13.88/25.85/32.22 |
| M3 | 7.71/14.59/19.66 | 11.00/19.70/24.44 | 0.65/1.55/2.05 | 0.63/1.60/2.08 | 22.89/37.79/46.12 |
| M4 | 7.85/15.37/19.85 | 6.19/11.94/14.70 | 0/0/0 | 0.64/1.72/2.20 | 16.18/27.74/34.25 |
| M5 | 7.70/14.47/20.27 | 5.82/10.76/13.86 | 0/0/0 | 0.64/1.68/2.24 | 15.42/25.67/34.91 |
| M6 | 7.77/13.60/18.29 | 13.53/21.68/27.45 | 0.64/1.12/1.86 | 0.64/1.23/2.14 | 25.33/38.43/46.81 |
| M7 | 7.75/15.17/19.93 | 13.06/22.22/28.29 | 0.64/1.50/1.98 | 0.64/1.70/2.23 | 24.92/40.51/49.77 |
| M8 | 8.02/11.98/15.56 | 37.85/54.59/71.74 | 0.65/1.05/1.34 | 0.61/1.02/1.27 | 49.37/71.00/91.02 |

### 6.4 Joint stress / transfer

| 方法 | Predictor | Planner | QDR/tube | Safety | Total |
| --- | --- | --- | --- | --- | --- |
| M0 | 7.99/16.72/20.89 | 4.23/8.65/10.84 | 0/0/0 | 0.66/1.88/2.25 | 14.08/26.47/31.61 |
| M1 | 8.30/16.95/20.84 | 4.35/9.15/11.54 | 0/0/0 | 0.67/1.91/2.33 | 14.79/27.36/32.92 |
| M2 | 7.74/16.17/20.47 | 4.10/8.39/10.96 | 0/0/0 | 0.64/1.80/2.44 | 13.69/25.42/33.03 |
| M3 | 7.76/15.00/19.87 | 11.03/19.59/24.79 | 1.16/2.59/3.48 | 0.63/1.67/2.17 | 23.61/38.71/47.53 |
| M4 | 7.70/14.40/19.40 | 6.05/11.64/14.94 | 0/0/0 | 0.63/1.73/2.23 | 15.87/26.58/33.85 |
| M5 | 7.78/15.32/20.13 | 5.87/11.33/14.43 | 0/0/0 | 0.63/1.77/2.29 | 15.81/27.34/34.92 |
| M6 | 7.76/14.76/19.51 | 12.51/21.69/27.40 | 1.16/2.58/3.51 | 0.64/1.66/2.21 | 25.01/40.33/49.25 |
| M7 | 7.73/15.04/20.31 | 11.95/21.07/26.82 | 1.12/2.59/3.43 | 0.63/1.69/2.18 | 24.45/40.31/49.28 |
| M8 | 7.98/10.15/13.14 | 37.71/45.06/52.79 | 1.23/1.67/2.35 | 0.61/0.84/1.19 | 49.82/59.23/69.38 |

## 7. 门槛判定与下一步

| 门槛 | 判定 |
| --- | --- |
| 场景、镜像组、factor-range、route-validity audit | Pass |
| QDR 正常时间索引等价 | Pass |
| 故意 double-delay probe 被拒绝 | Pass |
| candidate budget mismatch | Pass，0 mismatch；但本阶段只有 M8 是 K=8 |
| ID safe-capture non-inferiority | M6/M7 方向通过 |
| joint timeout `<=5%` | **No-Go**，M6 33.33%、M7 36.67% |
| QDR max exhaustion streak `<=24` | **No-Go**，M6 168、M7 112、M8 41 |
| robust CBF-QP/R-CLBF-QP safety proof | 未执行；不宣称 |

因此不打开 development_confirmation 或 locked diagnostic。下一阶段应按以下
顺序推进：

1. 先完成真正的 adaptive-K smoke：显式设置 K=1/4/8 对照，确认 M6/M7 的
   adaptive bucket、refresh、cache age 和 realized count 都有非零且可解释的变化；
2. 针对 joint-stress timeout 做 failure taxonomy：区分 queue-prefix 已不可修复、
   suffix gate 耗尽、planner infeasible、local CBF fallback 和真实 timeout；
3. 只在不改变场景和 checkpoint 的前提下，做一个 frozen recovery policy ablation，
   预先规定 timeout、collision、max streak 和 total p95 门槛；
4. 若 calibration 同时通过 ID、joint timeout、exhaustion 和 fallback gates，
   再在 untouched development_confirmation 做一次三 seed confirmation；否则冻结
   本报告为负/诊断结果，不访问 locked-test。

