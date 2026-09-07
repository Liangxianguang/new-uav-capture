# Phase 2 Prediction Smoke Report

## 1. 结论

本轮只验证了“数据契约、训练脚本和评估链路可以运行”，没有验证 Mamba-SSM + 条件扩散预测器在预测性能上成立。

在固定的 CPU smoke validation set 上，常速度基线优于 3 epoch GRU 和 portable SSM diffusion。因而 Phase 2 的正式 go/no-go 门槛未通过，不能进入 DN-MPC 或 R-CLBF-QP 实现，也不能在论文中声称多模态预测已经优于基线。

本报告中的结果不是 locked-test 结果，也不是正式统计结论。数据量、训练轮数和目标策略覆盖都只用于发现工程和建模问题。

## 2. 实验协议

| 项目 | 设置 |
| --- | --- |
| 环境 | `capture_radius_pursuit_central_v4_flee.yaml` |
| 动力学 | 速度级运动学，`dt = 0.1 s` |
| 观测输入 | `policy_observations`，不使用目标真值；包含延迟 belief |
| 输入形状 | history `8`，4 defenders，feature dim `63` |
| 预测 horizon | `12` steps，即 `1.2 s` |
| 训练集 | 4 episodes，952 samples，seed start `745101`，`flee_persistence` |
| 验证集 | 2 episodes，476 samples，seed start `746101`，`s_curve` |
| GRU | hidden `32`，1 layer，3 epochs，CPU |
| diffusion | portable diagonal SSM，hidden `32`，1 layer，16 train steps，4 sample steps，4 candidates，3 epochs，CPU |
| 目标表示 | 未来目标位置减当前发布的团队 belief reference |

数据集显式保存 `dt_seconds`、reference velocity、episode/seed/timestep、目标模式、输入/标签协议和 source hashes。训练输出保存 metadata、config、training JSON、checkpoint 和 TensorBoard event。

## 3. Smoke 结果

指标单位为米；`minADE/minFDE` 是候选集合的 best-of-K 指标。GRU 只有一个候选，因此与 top-1 相同。

| 模型 | ADE | FDE | minADE | minFDE | candidate spread |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constant velocity | 0.6372 | 1.2782 | 0.6372 | 1.2782 | 0.0000 |
| GRU Gaussian | 2.4040 | 2.1740 | 2.4040 | 2.1740 | 0.0000 |
| Portable SSM diffusion, K=4 | 17.1724 | 17.4504 | 15.0325 | 10.0734 | 8.5821 |

训练损失和验证损失均为有限值，GRU 最终 validation loss 为 `-0.2803`，diffusion noise-prediction loss 为 `0.9710`。这些损失不能直接跨模型比较。

## 4. 阶段门槛审计

| 门槛 | 状态 | 证据或缺口 |
| --- | --- | --- |
| minFDE 比 GRU 降低至少 10% | 未通过 | 本轮 diffusion minFDE `10.0734`，GRU minFDE `2.1740` |
| 90% 置信区域覆盖率 `0.85--0.95` | 未评估 | 当前没有校准置信度或 coverage 计算 |
| 候选动力学/边界可行比例不少于 95% | 未评估 | 尚未实现候选轨迹可行性检查 |
| 目标模式切换时置信度退化可解释 | 未评估 | 当前没有轨迹分数或 mode probability |
| p95 推理时间满足控制周期 | 未完成 | 当前只记录 epoch time，尚未记录 batch/single-sample p50/p95/p99 |
| planner 接口固定 | 部分完成 | 候选 shape 已固定为 `[batch, K, horizon, 3]`；分数、协方差和校准字段尚未固定 |

因此 Phase 2 状态为 **NO-GO / 需要先修正和扩大验证**。

## 5. 当前实现边界

- `portable_diagonal_ssm` 是纯 PyTorch 的依赖无关 SSM 兼容后端，不等同于官方 `mamba_ssm` CUDA 实现。
- diffusion 当前输出候选轨迹和候选 spread，但没有经过校准的置信度；不能把 spread 或未实现的分数称为概率。
- 当前训练数据只覆盖 `flee_persistence` 与 `s_curve` 两个脚本化模式，尚未实现自适应 adversarial target、策略切换或 locked test。
- 当前还没有 DN-MPC、分布式通信、R-CLBF-QP、统一规划动力学或闭环安全证明。
- 旧的 `phase2_*_smoke` 目录保留未覆盖；本轮使用 `*_smoke_v2`，避免破坏已有实验痕迹。

## 6. 可复现产物

- 训练数据：`results/phase2_prediction_train_smoke_v2/`
- 验证数据：`results/phase2_prediction_validation_smoke_v2/`
- GRU：`results/phase2_gru_smoke_v2/`
- portable SSM diffusion：`results/phase2_diffusion_smoke_v2/`

重新运行应使用仓库根目录下的 `scripts/collect_prediction_dataset.py` 和 `scripts/train_prediction_models.py`，并保留命令中的 seed、history、horizon、model 和 sampling 参数。

## 7. 下一步顺序

1. 先做目标位移/归一化、baseline 和训练收敛诊断，增加 epoch 和独立训练 seed；在没有超过常速度与 GRU 前，不进入 planner。
2. 补齐 deterministic inference 测试、单样本延迟、候选速度/加速度/边界检查以及 trajectory score 接口。
3. 扩展 nominal、delayed-noisy、遮挡/丢包和自适应目标策略，并按 episode/scene 切分 locked test。
4. 只有预测门槛通过后，先实现集中式 scenario min-max MPC，再实现分布式 consensus/ADMM。
5. 最后实现 robust CBF-QP，明确扰动界、可行性条件和离散时间安全结论；在这些条件完成前不使用“R-CLBF-QP 已证明安全”的表述。

## 8. 回归验证

本轮完成：

```text
72 passed
```

专项预测测试为 `6 passed`。训练脚本和数据采集脚本通过 Python 编译检查。
