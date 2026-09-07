# Phase 3 S3 集中式 Scenario MPC Validation Report

> 日期：2026-09-07
> 状态：集中式 P3 validation gate **通过**；最终 DN-MPC 和 R-CLBF-QP 仍未证明。
> 协议：`configs/phase3_s3_mpc.yaml`

## 1. 结论

在同一组 12 个冻结 S3 随机混合障碍物场景上，使用三个独立的
SSM-diffusion prediction checkpoint seed，集中式 expected、worst-case 和
CVaR Scenario MPC 均达到 100% safe capture、0% collision；当前
`DynamicEncirclement + local CBF` 基线为 91.7% safe capture、8.3% collision。
相对于基线，安全捕获率提升 8.3 个百分点，超过预注册的 5 个百分点门槛。

三次运行的 `scenes.jsonl` SHA-256 完全一致：

```text
FD883350CBF126731443EFE7796A5FA8DC6C8D1BA08BEA03A6B445B43A90CA5C
```

因此，当前证据支持进入 P4 的分布式 DN-MPC 实现；它只支持“集中式 planner
在 S3 validation 上通过门槛”，不支持把集中式结果直接称为 DN-MPC，也不支持
任何 R-CLBF-QP 形式化安全证明。

## 2. 运行协议

- S3 validation：12 episodes，episode seeds `646101--646112`，layout seeds `1646101--1646112`。
- 场景：3--5 个随机混合 cylinder/box/wall 障碍物，左右两侧、nominal/delayed_noisy 观测条件和 `flee_persistence/s_curve` 目标模式均纳入。
- 方法：`DynamicEncirclement + local CBF`、expected MPC、worst-case MPC、CVaR MPC。
- prediction：三个 checkpoint seed `745101/745201/745301`；每步 8 个候选、8 个采样步、4 次 dynamics projection。
- planner：8 步 horizon、3 步 control horizon、`dt=0.1 s`。
- 安全层：现有 local CBF；本阶段不是 robust CBF-QP。
- 真实目标状态仅用于环境终止和评估指标；planner 只接收 policy-safe observation 和 projected candidates。

复现单个 checkpoint：

```powershell
conda run --no-capture-output -n uav-encirclement-gpu python scripts/evaluate_minimax_mpc_s3.py `
  --output-dir results/phase3_s3_validation_seed745101 `
  --checkpoint results/phase2_formal_diffusion_seed745101/checkpoint.pt `
  --episodes 12 --device auto
```

## 3. 三 seed 汇总

表中 latency 是三个 checkpoint 运行中的最大 p95；candidate metric 是
规划候选 rollout 的每步 worst candidate minimum distance 的 episode 平均值。

| 方法 | Safe capture | Collision | Mean capture time (s) | Worst clearance (m) | Candidate worst distance (m) | Solver success | Planner p95 (ms) | Total p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DynamicEncirclement + CBF | 91.7% | 8.3% | 4.36 | 0.20 | -- | -- | -- | 2.91 |
| Expected MPC + CBF | 100.0% | 0.0% | 4.62 | 0.19 | 3.88 | 100.0% | 18.53 | 83.54 |
| Worst-case MPC + CBF | 100.0% | 0.0% | 4.54 | 0.19 | 3.84 | 100.0% | 14.88 | 64.05 |
| CVaR MPC + CBF | 100.0% | 0.0% | 4.49 | 0.19 | 3.85 | 100.0% | 15.37 | 65.29 |

三种 checkpoint 的方法方向一致：每个 seed 的三种 MPC 都为 100% safe
capture 和 0% collision。预测 seed 会影响平均捕获时间和候选距离，但没有
改变安全结论。worst-case/CVaR 的 planner p95 比 expected 更低，三者总控制
p95 均低于 100 ms 控制预算。

## 4. 分层观察

以 seed `745101` 的 12 个 episode 为例，基线 nominal 条件为 83.3% safe
capture、16.7% collision，delayed_noisy 条件为 100% safe capture；这说明
当前随机场景难度主要来自具体布局与目标模式，不能把单一观测退化因素写成
普遍结论。五个 `s_curve` episode 中四种方法均未发生碰撞；七个
`flee_persistence` episode 中集中式方法均安全捕获。

candidate-level 统计记录在各方法目录的 `steps.jsonl` 中，包含每步的全部
候选 minimum/terminal distances、expected/worst/CVaR 聚合值和 planner scenario
costs。该统计是预测候选 rollout 的诊断量，不是真实未来目标距离，也不能替代
真实 episode 的捕获率。

## 5. TensorBoard 和 provenance

每个 checkpoint 结果目录下的四个方法目录均包含：

```text
episodes.jsonl
steps.jsonl
summary.json
tensorboard/events.out.tfevents.*
```

TensorBoard 中保留了配置、协议文本、source hashes、episode outcome、solver
success/fallback、candidate distance、预测/规划/安全/总控制 latency 和 hparams。
每次运行的 `config.yaml` 还记录 checkpoint SHA-256、代码 SHA-256、采样参数和
场景协议，三个结果目录均保存相同的 `scenes.jsonl`。

## 6. 门槛检查和后续

| 门槛 | 结果 |
|---|---|
| S3/S5 条件 safe capture 提升至少 5 个百分点 | 通过，91.7% -> 100.0%（+8.3 pp） |
| collision rate 不恶化超过 1 个百分点 | 通过，8.3% -> 0.0% |
| solver success >= 99% | 通过，100% |
| total control p95 <= 100 ms | 通过，最大 83.54 ms |
| 三个 prediction checkpoint seed 方向一致 | 通过 |
| locked-test 结果 | 尚未执行，不得使用为最终主结果 |
| 分布式通信鲁棒性 | 尚未执行 |
| R-CLBF-QP/闭环形式化证明 | 尚未执行 |

下一阶段进入 P4：实现有限通信 sequential best-response DN-MPC，并与本报告
中的集中式结果、DynamicEncirclement fallback 和通信退化条件进行同场景比较。
