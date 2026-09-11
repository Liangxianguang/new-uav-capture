# 当前实验结果总览

> 更新日期：2026-09-09
> 仓库：`https://github.com/Liangxianguang/new-uav-capture`
> 总体结论：**Conditional Go**

## 1. 先看结论

当前工作已经形成了较完整的“预测 -> DN-MPC -> 安全过滤”仿真证据，
但还不是经过真实执行器验证的完整 R-CLBF-QP 方法。

## 0. 最新 Phase 15 S4-v3 进展

扩建后的 action-conditioned branching 数据已完成冻结：600 个请求场景中
566 个产生完整窗口，共 16,685 个窗口；按 300 个镜像组划分为
420/90/90 个 train/validation/locked-test 场景，审计通过。新数据包含历史
防御机动作、未来规划动作条件以及执行/通信时间戳。

在 1,539 个 locked-test 窗口上，当前单 seed、3 epoch 初版的 projected
minFDE 为：GRU `0.9286 m`，Diagonal SSM diffusion `1.3480 m`，dense
S4-DPLR `1.3551 m`，upstream official S4 `1.3492 m`。diffusion 候选投影后
可行率约 `84%`，尚未达到 `95%` 工作门槛；因此新 S4 结果目前是预测器对照，
尚未接入 DN-MPC 形成新的端到端主结果。完整表格见
`PHASE15_S4_V3_LOCKED_TEST_REPORT.md`。

| 模块/阶段 | 当前结果 | 状态 |
| --- | --- | --- |
| 三维围捕环境与数据契约 | 速度级仿真、观测/标签隔离、执行扰动接口、可复现实验协议已实现 | 已完成基础设施 |
| Portable SSM + 条件扩散预测 | 未见自适应目标上 projected minFDE `0.5363 m`，coverage `90.31%`；原五模式平均提升未达 10% 门槛 | Conditional Go |
| 集中式 Scenario MPC | 小规模和固定 S3 validation 均达到 `100%` safe capture | 条件通过 |
| 分布式 DN-MPC | 固定 S3 三种子、四种通信条件均 `100%` safe capture、`0%` collision | 模块化验证通过 |
| Velocity-level robust CBF-QP | 分层 robust-safe reset 下 `8/8` 安全捕获、一步证书全通过 | 条件通过 |
| P5 执行扰动安全 | 延迟、噪声、跟踪和队列条件下仍失败 | No-Go |
| P7 速度级完整组合 | 修复契约后在 100 个未见自适应场景达到 `99%` safe capture、`0%` collision | 条件通过 |
| R-CLBF-QP / 闭环安全证明 | 尚未实现或验证 | 未完成 |

这里的 `safe capture` 指目标进入 `0.80 m` 捕获半径，且该 episode 没有
碰撞、越界或其他安全失败。它不等同于真实物理捕获。

## 2. 历史基线

### Phase 1：基础设施

- 实现了四类执行扰动接口：命令延迟、命令噪声、一阶速度跟踪、阻力/质量/加速度限制。
- 实现了不泄漏目标真值的观测与未来轨迹标签分离。
- prediction smoke：`250` 帧、`238` 个有效窗口、`4` 个防守无人机、`63` 个局部观测特征。
- 基础测试：`69 passed`。
- 限制：该执行模型是轻量级仿真，不是飞控、PyBullet 或真实硬件模型。

### V4 locked-test 基线

保留的 BC + local CBF 三 checkpoint 在固定随机 S3 场景的结果为：

| 方法 | Cooperative safe capture | Collision | Boundary |
| --- | ---: | ---: | ---: |
| RAW actor | `2.3 +/- 1.2%` | `97.7 +/- 1.2%` | `0%` |
| BC + local CBF | `75.3 +/- 6.5%` | `4.7 +/- 1.2%` | `4.7 +/- 1.2%` |

因此 local CBF 是保留的执行基线，不能把裸策略结果当作安全结果。

### V5 exact-reactive development

该阶段不是正式 locked-test。三组开发 seed 中只有 `661606` 通过完整开发门槛：

- S3 CBF：`57/60 = 95.0%`，collision `0%`，boundary `0%`；
- seed `661602`：`53/60 = 88.3%`，collision 和 boundary 各 `3.3%`；
- seed `661604`：固定 Wall 场景失败，未进入 S3；
- 因此没有冻结 V5 locked-test 模型，`647201` seed block 保持未打开。

这说明早期高捕获率存在跨 seed 不稳定性，不能替代后续正式预测模型结果。

## 3. Phase 2：SSM + 条件扩散预测

### Formal five-mode locked-test

协议包含 `15,232 / 5,712 / 7,616` 个 train/validation/locked-test 样本，
五种目标模式，三个训练 seed：`745101`、`745201`、`745301`。

| 模型和处理阶段 | minADE | minFDE | Energy score | Full-trajectory coverage | 可行候选比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRU raw | `0.4187 +/- 0.0122` | `0.6825 +/- 0.0163` | `1.6402 +/- 0.0461` | `90.06%` | `7.82%` |
| Diffusion raw | `0.4325 +/- 0.0030` | `0.6379 +/- 0.0078` | `1.7242 +/- 0.0243` | `89.26%` | `0%` |
| GRU projected | `0.4128 +/- 0.0112` | `0.7075 +/- 0.0206` | `1.6456 +/- 0.0469` | `89.62%` | `98.84%` |
| Diffusion projected | `0.4161 +/- 0.0032` | `0.6802 +/- 0.0085` | `1.6918 +/- 0.0239` | `88.74%` | `99.71%` |

关键结论：

- 相对 raw GRU，raw diffusion minFDE 改善 `6.54%`；
- 相对 projected GRU，projected diffusion minFDE 只改善 `3.87%`；
- 预注册的 `10%` prediction gate 未通过；
- raw diffusion 几乎全部违反加速度契约，必须经过 dynamics projection；
- 单样本 GPU p95：GRU `1.85 +/- 0.51 ms`，diffusion `21.05 +/- 2.64 ms`。

按目标模式，diffusion 在 `s_curve` 和 `boundary_escape` 上更有优势，
但在 `burst` 模式上 GRU 更好。因此不能声称 diffusion 在所有目标模式上全面支配 GRU。

### Unseen adaptive-adversarial audit

该审计使用未参与训练和选择的 `32` 个自适应目标 episode，三个冻结 checkpoint。

| 模型和处理阶段 | minADE (m) | minFDE (m) | Energy score | Coverage | 可行候选比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRU raw | `0.5039 +/- 0.0242` | `0.8908 +/- 0.0668` | `1.9831 +/- 0.1051` | `82.38%` | `12.36%` |
| Diffusion raw | `0.3235 +/- 0.0019` | `0.5051 +/- 0.0072` | `1.1879 +/- 0.0192` | `89.94%` | `0.0033%` |
| GRU projected | `0.4976 +/- 0.0222` | `0.9033 +/- 0.0648` | `1.9903 +/- 0.1003` | `82.22%` | `99.991%` |
| Diffusion projected | `0.3155 +/- 0.0016` | `0.5363 +/- 0.0071` | `1.1747 +/- 0.0152` | `90.31%` | `99.914%` |

相对 projected GRU，projected diffusion 的 minFDE 改善 `40.63%`，energy score
改善 `40.98%`，coverage 提高 `8.09` 个百分点。这是当前预测模块最强的正向结果，
但 candidate weights 仍是 `uniform_uncalibrated`，不是逐候选概率校准。

该未见策略审计的 CPU 单样本 p95 为：GRU `8.08 +/- 0.96 ms`，diffusion
`39.37 +/- 8.77 ms`。这些是预测器单独延迟，不是端到端延迟。

## 4. Phase 3：集中式 Scenario MPC

在 `8` 个 formal-small、`flee_persistence` 场景上：

| 方法 | Safe capture | Collision | Mean capture time | Planner p95 | Total p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DynamicEncirclement + CBF | `100%` | `0%` | `0.9125 s` | n/a | `1.62 ms` |
| Expected MPC + CBF | `100%` | `0%` | `0.9000 s` | `11.40 ms` | `44.98 ms` |
| Worst-case MPC + CBF | `100%` | `0%` | `0.9750 s` | `11.37 ms` | `45.87 ms` |
| CVaR MPC + CBF | `100%` | `0%` | `0.9000 s` | `11.83 ms` | `52.24 ms` |

该场景过于容易，不能证明风险目标在困难场景上的优势。随后固定 S3 validation
使用 `12` 个共享场景和三个 prediction seed，集中式及四种分布式方法均达到
`100%` safe capture、`0%` collision；DynamicEncirclement 基线为 `91.7%` safe
capture、`8.3%` collision，提升 `8.3` 个百分点。

## 5. Phase 4：DN-MPC 与未见目标泛化

### 固定 S3 validation

三个 checkpoint seed、四种通信模式、每种 `12` 个相同场景：

| 方法 | Safe capture | Collision | Valid plan | Effective plan | Convergence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Centralized worst-case | `100%` | `0%` | `100%` | `100%` | n/a |
| Distributed ideal | `100%` | `0%` | `100%` | `100%` | `100%` |
| Distributed 2-step delayed | `100%` | `0%` | `100%` | `100%` | `100%` |
| Distributed 10% dropout | `100%` | `0%` | `100%` | `100%` | `100%` |
| Distributed no communication | `100%` | `0%` | `100%` | `100%` | `100%` |
| DynamicEncirclement baseline | `91.7%` | `8.3%` | n/a | n/a | n/a |

三个 seed 的平均 planner p95 为 `8.72--12.47 ms`，total-control p95 为
`50.38--51.51 ms`。该阶段固定 S3 validation gate 通过，但仍不是形式化博弈保证。

### 100 个未见 adaptive-adversarial 场景

同一份 locked scene file、三个 checkpoint、每种方法共享场景：

| 方法 | Safe capture | Collision | Boundary | Timeout | Valid/effective plan |
| --- | ---: | ---: | ---: | ---: | ---: |
| DynamicEncirclement baseline | `95%` | `2%` | `0%` | `3%` | n/a |
| Worst-case centralized | `97.33%` | `1%` | `0%` | `1.67%` | `100% / 100%` |
| Distributed ideal | `98%` | `1%` | `0%` | `1%` | `100% / 99.97%` |
| Distributed delayed | `99%` | `0%` | `0%` | `1%` | `100% / 100%` |
| Distributed dropout | `99%` | `0%` | `0%` | `1%` | `100% / 100%` |
| Distributed no communication | `98%` | `0%` | `0%` | `2%` | `100% / 100%` |

该结果支持 projected diffusion + DN-MPC 在未见目标策略上优于当前动态基线。
但预测每 `20` 个控制步刷新一次，最大候选年龄 `19` 步；三 checkpoint aggregate
的 total-control p95 为 `167.73--229.62 ms`。这只是工程折中，严格 10 Hz
部署参考尚未达到，且 100 ms 不是本项目的硬性方法门槛。

## 6. Phase 5：robust CBF-QP 安全层

### 速度级、分层 robust-safe reset

在 `64` 个候选 seed 中按几何 barrier 选择 `8` 个初始状态，保证进入预设 robust
safe set；共 `130` 个控制步：

| 方法 | Safe capture | Collision | Initial valid | Independent cert. | Next-state safe | Filter p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Nominal | `100%` | `0%` | `100%` | `100%` | `100%` | `0.002 ms` |
| Local CBF | `100%` | `0%` | `100%` | `100%` | `100%` | `10.90 ms` |
| Robust CBF-QP | `100%` | `0%` | `100%` | `100%` | `100%` | `6.27 ms` |

robust CBF-QP 的 solver success、QP infeasible、solver failure、fallback 和 slack
分别为 `130/130`、`0`、`0`、`0`、`0`。这是条件性的一步 velocity-level 结果，
不是 R-CLBF-QP 或闭环 forward-invariance 证明。

### 执行扰动和队列审计

在延迟、噪声、跟踪、阻力、质量随机化和 pending-command queue 下，当前安全层未通过：

| Variant | Safe capture | Collision | Boundary | Actual post robust state | Filter p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mild immutable | `37.5%` | `0%` | `25%` | `90.78%` | `671.36 ms` |
| Hard immutable | `25.0%` | `0%` | `37.5%` | `67.67%` | `1945.74 ms` |
| Mild flush-pending | `75.0%` | `0%` | `0%` | `100%` | `857.73 ms` |
| Hard flush-pending | `50.0%` | `0%` | `0%` | `91.70%` | `2489.98 ms` |

后续 certified-fallback 审计显示：

- mild immutable：safe capture `37.5%`，collision `50%`，certified fallback `8.33%`；
- hard immutable：safe capture `25%`，collision `75%`，certified fallback `1.44%`；
- mild flush：safe capture `75%`，collision `0%`，actual post robust state `100%`；
- hard flush：safe capture `50%`，collision `0%`，timeout `50%`，filter p95 `460.16 ms`。

因此，execution-invariant safety、连续段安全、真实执行器安全和 R-CLBF-QP 仍为 No-Go。

## 7. Phase 7：完整组合结果

完整组合使用 projected portable SSM diffusion、worst-case DN-MPC 和 robust safety
filter，在同一份 `100` episode `adaptive_adversarial` locked scene 上复测。

| 版本 | Safe capture | Collision | Boundary | Independent certificate | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| Original P7 | `50%` | `49%` | `16%` | `64.87%` | robust margin 与 P4 reset 不匹配 |
| Matched margin diagnostic | `30%` | `70%` | `3%` | `93.61%` | 去除未建模扰动 margin，但保留硬动作变化约束 |
| Contract-repaired P7 | **`99%`** | **`0%`** | **`0%`** | **`100%`** | 速度级契约一致，唯一失败为 timeout |

最终 P7 细节：

- planner valid/effective rate：`100% / 100%`；
- safety-filter certificate valid rate：`99.98%`；
- independent current/next-state certificate：`100%`；
- timeout：`1%`，无碰撞、无越界；
- planner p95：`28.74 ms`；predictor p95：`14.12 ms`；safety p95：`2.97 ms`；
- total-control p95：`41.57 ms`；
- 一次 SLSQP fallback 出现在 episode 79、step 8，回退动作仍通过 independent checker。

修复内容：

1. QP 原先使用 `[-v_max/sqrt(3), v_max/sqrt(3)]` 分量盒，与环境的欧氏速度
   限制不一致，已改为显式速度球约束。
2. `action_change_limit_mps` 和 `enforce_action_change` 改为显式可选配置。
3. 理想速度级 P7 不再强行添加未建模的加速度/动作变化约束。
4. 需要真实执行动力学时，动作变化约束仍可在独立 execution protocol 中启用。

严格说，显式欧氏速度球使在线投影子问题成为小型凸 QCQP；代码继续保留
`RobustCBFQPFilter` 名称是为了兼容现有接口，不能把它描述成未经说明的纯 QP 证明。

## 8. 模型与复现实验状态

当前正式预测模型已保存到：

```text
models/formal_phase2_v3/
```

包括：

- `ssm_diffusion_seed745101.pt`
- `ssm_diffusion_seed745201.pt`
- `ssm_diffusion_seed745301.pt`
- `gru_baseline_seed745101.pt`
- `gru_baseline_seed745201.pt`
- `gru_baseline_seed745301.pt`

模型哈希、来源路径、seed 和 backend 见
`models/formal_phase2_v3/manifest.json`。Diffusion 使用的是
`portable_diagonal_ssm`，不是官方 `mamba_ssm` CUDA kernel。

主要正式报告：

- `PHASE2_FORMAL_ANALYSIS_REPORT.md`
- `PHASE2_ADAPTIVE_GENERALIZATION_AUDIT_REPORT.md`
- `PHASE3_S3_VALIDATION_REPORT.md`
- `PHASE4_DN_MPC_VALIDATION_REPORT.md`
- `PHASE4_UNSEEN_ADAPTIVE_VALIDATION_REPORT.md`
- `PHASE5_STRATIFIED_VALIDATION_REPORT.md`
- `PHASE5_EXECUTION_STATE_AWARE_AUDIT_REPORT.md`
- `PHASE5_RECOVERABILITY_CONTRACT_AUDIT_REPORT.md`
- `PHASE7_INTEGRATION_REPAIR_REPORT.md`

本轮回归验证：

```text
python -m pytest -q
131 passed in 64.22s
```

## 9. 当前最终判断

可以成立的阶段性表述是：

> 在速度级三维围捕仿真中，projected portable-SSM diffusion 候选可以被
> worst-case DN-MPC 消费；分布式 DN-MPC 在固定 S3 和未见自适应目标场景上
> 获得稳定的模块化围捕收益；在与 P4 场景匹配的速度级安全契约下，robust
> safety filter 可以完成高安全率联合闭环。

目前不能成立的表述是：

- 官方 Mamba 已被验证；
- raw diffusion 候选可直接执行；
- 预测器候选概率已经校准；
- 已经证明形式化零和博弈最优性；
- 已经证明执行扰动下多步 forward invariance；
- 已经完成 R-CLBF-QP 或真实无人机安全验证；
- 已经完成三种 seed 的真实端到端部署级主结果。

下一步优先级应是统一真实执行器/队列契约、降低 execution-aware safety solver
成本、重新生成 robust-safe locked reset，并完成多 seed 的执行扰动端到端消融；
在此之前不应继续把完整方法写成无条件成功。
