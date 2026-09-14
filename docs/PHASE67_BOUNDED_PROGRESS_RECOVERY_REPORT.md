# Phase 67：QDR bounded public-prefix recovery 诊断

本报告只使用 fresh development-calibration manifest，不访问 confirmation 或 locked-test。
local CBF 是经验过滤器；robust CBF-QP/R-CLBF-QP 仍是 diagnostic No-Go，不宣称安全证明。
所有延迟表的格式均为 `p50/p95/p99`，单位为 ms。

## 实验契约

- scene manifest SHA-256：`7542f4a43407098ba5ec804136ce3255959b56dc93938169a1b1489a611c809f`；
- bootstrap：10000 次，seed `20260914`，按 predictor seed 与 mirror group 分层；
- calibration 场景：120 episodes，60 mirror groups；
- predictor：三个 Diagonal-SSM checkpoint seed，same scene order；
- safety：local CBF empirical filter；

## 主要观察

- M9 bounded public-prefix recovery 在 execution-tail 上相对 M6 的 safe-capture
  delta 为 `+7.78 pp [-3.33,+18.89]`，区间跨零；collision delta 为
  `-4.44 pp [-15.56,+6.67]`，timeout delta 为 `-3.33 pp [-11.11,+3.33]`，
  因此不能判定为稳定收益。
- 在 communication-tail 上 M6 已达到 `100.00%` safe capture，M9 反而为
  `96.67%`，paired safe-capture delta 为 `-3.33 pp [-8.89,0.00]`；恢复规则
  不能替代已知延迟或通信建模。
- 在 joint-tail-transfer 上 M6/M9 safe capture 为 `31.11%/30.00%`，M9 的
  timeout 由 `33.33%` 变为 `35.56%`；paired intervals 很宽，说明公开候选集
  在联合尾部压力下仍缺乏可恢复性。
- M9 的 recovery budget 为 6 步、短前缀 horizon 为 3 步；恢复触发率和最大
  持续步数均已写入 JSONL、aggregate 与 TensorBoard。该策略没有通过本阶段的
  safe-capture、timeout、exhaustion-streak 联合门槛，冻结为 diagnostic No-Go，
  停止继续堆叠同类恢复规则。

## Outcome（按 scene block）

### id_reference（30 episodes / 15 mirror groups）

| Method | Safe capture | Collision | Boundary | Timeout | Capture time (s) | Min clearance (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M6 synchronous QDR | 98.89% [95.56%, 100.00%] | 1.11% [0.00%, 4.44%] | 1.11% [0.00%, 4.44%] | 0.00% [0.00%, 0.00%] | 4.892 [4.290, 5.359] | 0.453 [0.433, 0.475] |
| M9 bounded recovery | 98.89% [95.56%, 100.00%] | 1.11% [0.00%, 4.44%] | 1.11% [0.00%, 4.44%] | 0.00% [0.00%, 0.00%] | 3.991 [3.346, 4.729] | 0.479 [0.457, 0.500] |

### execution_tail（30 episodes / 15 mirror groups）

| Method | Safe capture | Collision | Boundary | Timeout | Capture time (s) | Min clearance (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M6 synchronous QDR | 30.00% [21.11%, 38.89%] | 62.22% [53.33%, 71.11%] | 53.33% [40.00%, 65.56%] | 7.78% [2.22%, 13.33%] | 10.275 [7.156, 13.596] | 0.327 [0.272, 0.369] |
| M9 bounded recovery | 37.78% [26.67%, 48.89%] | 57.78% [45.56%, 68.89%] | 51.11% [35.56%, 65.56%] | 4.44% [1.11%, 8.89%] | 10.928 [8.354, 13.904] | 0.329 [0.274, 0.380] |

### communication_tail（30 episodes / 15 mirror groups）

| Method | Safe capture | Collision | Boundary | Timeout | Capture time (s) | Min clearance (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M6 synchronous QDR | 100.00% [100.00%, 100.00%] | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 5.986 [4.756, 7.178] | 0.491 [0.465, 0.512] |
| M9 bounded recovery | 96.67% [91.11%, 100.00%] | 2.22% [0.00%, 6.67%] | 2.22% [0.00%, 6.67%] | 1.11% [0.00%, 4.44%] | 4.571 [3.579, 5.620] | 0.490 [0.469, 0.510] |

### joint_tail_transfer（30 episodes / 15 mirror groups）

| Method | Safe capture | Collision | Boundary | Timeout | Capture time (s) | Min clearance (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M6 synchronous QDR | 31.11% [21.11%, 42.22%] | 35.56% [18.89%, 52.22%] | 5.56% [1.11%, 11.11%] | 33.33% [18.89%, 48.89%] | 13.012 [10.658, 15.456] | 0.137 [0.096, 0.175] |
| M9 bounded recovery | 30.00% [18.89%, 42.22%] | 34.44% [24.44%, 45.56%] | 7.78% [2.22%, 14.44%] | 35.56% [24.44%, 46.67%] | 12.202 [8.858, 16.583] | 0.160 [0.128, 0.195] |

## 完整组件延迟（按 scene block）

### id_reference

| Method | Predictor | Planner | QDR/tube | Safety | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| M6 synchronous QDR | 10.64/15.00/17.83 | 23.40/32.62/37.49 | 0.84/1.31/1.71 | 0.81/1.33/1.70 | 40.00/52.32/57.94 |
| M9 bounded recovery | 10.46/15.39/18.48 | 22.90/33.44/38.45 | 0.83/1.31/1.71 | 0.79/1.31/1.77 | 39.17/53.04/60.83 |

### execution_tail

| Method | Predictor | Planner | QDR/tube | Safety | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| M6 synchronous QDR | 10.52/15.12/17.78 | 22.55/32.18/37.83 | 1.70/2.60/3.43 | 0.80/1.31/1.70 | 39.94/52.29/60.40 |
| M9 bounded recovery | 10.39/15.08/18.00 | 22.85/32.18/37.02 | 1.66/2.53/3.35 | 0.79/1.28/1.67 | 40.09/52.34/60.04 |

### communication_tail

| Method | Predictor | Planner | QDR/tube | Safety | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| M6 synchronous QDR | 10.35/15.24/18.56 | 24.12/32.85/38.34 | 0.83/1.30/1.71 | 0.79/1.25/1.66 | 40.02/52.21/61.52 |
| M9 bounded recovery | 10.39/15.30/18.13 | 24.47/33.26/39.64 | 0.83/1.31/1.68 | 0.79/1.30/1.70 | 40.44/52.45/61.79 |

### joint_tail_transfer

| Method | Predictor | Planner | QDR/tube | Safety | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| M6 synchronous QDR | 10.41/15.11/17.91 | 22.48/31.65/36.78 | 1.87/2.87/3.57 | 0.79/1.29/1.65 | 39.95/52.21/59.84 |
| M9 bounded recovery | 10.13/14.93/18.18 | 22.66/32.17/37.66 | 1.80/2.74/3.36 | 0.78/1.23/1.67 | 39.22/52.69/61.58 |

## Paired comparison

### id_reference

正值 safe-capture delta 表示 candidate 优于 reference；区间为 mirror-group paired bootstrap 95% CI。

| Comparison | Safe-capture delta | Collision delta | Timeout delta |
| --- | ---: | ---: | ---: |
| M9_vs_M6 | 0.00% [-4.44%, 4.44%] | 0.00% [-4.44%, 4.44%] | 0.00% [0.00%, 0.00%] |

### execution_tail

正值 safe-capture delta 表示 candidate 优于 reference；区间为 mirror-group paired bootstrap 95% CI。

| Comparison | Safe-capture delta | Collision delta | Timeout delta |
| --- | ---: | ---: | ---: |
| M9_vs_M6 | 7.78% [-3.33%, 18.89%] | -4.44% [-15.56%, 6.67%] | -3.33% [-11.11%, 3.33%] |

### communication_tail

正值 safe-capture delta 表示 candidate 优于 reference；区间为 mirror-group paired bootstrap 95% CI。

| Comparison | Safe-capture delta | Collision delta | Timeout delta |
| --- | ---: | ---: | ---: |
| M9_vs_M6 | -3.33% [-8.89%, 0.00%] | 2.22% [0.00%, 6.67%] | 1.11% [0.00%, 4.44%] |

### joint_tail_transfer

正值 safe-capture delta 表示 candidate 优于 reference；区间为 mirror-group paired bootstrap 95% CI。

| Comparison | Safe-capture delta | Collision delta | Timeout delta |
| --- | ---: | ---: | ---: |
| M9_vs_M6 | -1.11% [-16.67%, 14.44%] | -1.11% [-18.89%, 17.78%] | 2.22% [-16.67%, 21.11%] |

## 阶段判定与边界

本阶段只完成 development calibration evidence，不是模型晋级结论。由于三个压力轴均未形成统计稳定的综合收益，本实验线按 diagnostic No-Go 冻结并停止继续堆叠同类恢复规则；不访问 confirmation/locked-test，也不据此宣称安全证明。

## 复现入口

```powershell
python scripts\aggregate_phase58_calibration.py `
  --group "phase67=results\phase67_bounded_recovery_seed*" `
  --scenes results\phase67_bounded_recovery_matrix\development_calibration\scenes.jsonl `
  --output results\phase67_bounded_recovery_aggregate.json `
  --tensorboard-dir results\phase67_bounded_recovery_aggregate_tensorboard `
  --bootstrap-seed 20260914
```

详细 episode/step JSONL、effective config 和 TensorBoard event 保留在各 seed 运行目录；本报告由 `scripts/render_phase67_bounded_recovery_report.py` 从 aggregate JSON 渲染。
