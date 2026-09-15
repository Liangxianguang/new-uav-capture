# Phase 79：DN-MPC Teacher + DAgger Recovery Residual Actor

## 目的

Phase78 已经将目标改为 obstacle-avoidance-first，但直接 recurrent BC 在固定
Nominal 集上只有 `59.92%` safe capture，并出现 `18.25%` collision。Phase79 不再
改变目标行为合同，而是修复防守方训练的闭环分布偏移：用公开 belief 上的 delayed
distributed DN-MPC 产生教师动作，用 route controller 作为低成本基准动作，recurrent
actor 学习受限 residual；随后在 actor 自己访问到的状态上执行 DAgger/recovery 标注。

## 固定合同

- 数据和训练只使用 `phase78_oracle_calibration_nominal/accepted_calibration_scenes.jsonl`；
- `target_crossing_required=false`；
- `target_maneuver_crossing_gain=0`；
- `target_maneuver_enable_reverse_lane_change=false`；
- 目标仍保留平滑 jerk/加速度/转向约束和 obstacle-avoidance-first 几何先验；
- 教师不读取 `env.target_position`，只使用 public belief、观测障碍物、队形和执行队列；
- local CBF 仅为经验过滤器，不构成 robust CBF-QP/R-CLBF-QP 安全证明；
- Hard、Stress、locked-test 在 Nominal gate 通过前不运行。

## 训练流程

1. 轮次 0：复用已保留的 Phase78 三份接受示范，合并为 `64+64+64=192` 条初始
   demonstrations；它们与当前目标合同一致，不重复消耗 CPU 生成相同轨迹。
2. 训练 recurrent residual actor：输出相对于 route base action 的有界 residual，
   residual scale 为 `2.5 m/s`，最终动作受 `5.0 m/s` 上限约束。
3. DAgger recovery round 1--2（每轮 12 条闭环 episode，共 24 条）：用当前 actor 闭环访问 Nominal 状态，同时计算
   delayed distributed DN-MPC teacher 标签；DN-MPC 每 16 步重新规划，使用 2 个
   public-belief candidates 和 1 次局部 best-response 迭代。当 teacher/base 差距、CBF 修正或 planner fallback 超过阈值
   时标记 recovery frame，并以更高 loss 权重加入聚合数据集。
4. 用聚合数据重训并保存 checkpoint、manifest、数据集和源代码 hash。
5. 在固定 84 场专家有效 Nominal 集上复核 safe capture、collision、boundary、
   timeout、termination reason，以及 route/actor/safety/total 的 p50/p95/p99。

## 预注册门控

三个 seed 都必须同时满足：safe capture `>=70%`、collision `<=5%`、boundary
violation `<=5%`、timeout `<=10%`。若未通过，只能增加 recovery 示范、做 delay/noise
分解或调整 actor residual/training，不得通过修改 locked-test 或强化目标行为来获得
结果。

## 可复现入口

```powershell
python scripts/train_phase79_dagger_residual.py `
  --config configs/phase79_dnmcp_dagger_residual.yaml `
  --output results/phase79_dnmcp_dagger_residual_seed791501 `
  --device cpu
```

smoke 可使用 `--allow-small`，但 smoke 不计入正式结果。正式生成的 `results/` 内容
保留在本地并由 manifest 记录，Git 只提交脚本、配置、测试和报告。
