# Phase 80：渐进 Hard 校准与 teacher 可行性诊断

## 目的与合同

Phase 80 不改变 Phase 78 的目标行为合同：目标初始优先远离最近障碍物，不要求主动
变道，不要求穿越中心障碍物，候选轨迹仍受边界、障碍物净空、速度、加速度、转向
角速度和 jerk 限制。该阶段只把延迟、噪声、目标速度和障碍物数量逐步提高，避免
把“专家无法完成的场景”直接用于学习模型评价。

所有结果均为 development calibration/validation，`locked_test_used=false`；local
CBF 仍只是经验过滤器，不构成 R-CLBF-QP 或 robust CBF-QP 安全证明。

## Hard-Lite v1

Hard-Lite v1 使用 300 场景（Easy/Nominal/Hard 各 100 场），初始外逃证书通过率为
100%。Hard 档将目标速度提高到 `0.55/0.65`，障碍物提高到 `3--4` 个，执行延迟为
2 步、命令噪声为 `0.04 m/s`。

oracle-route 在 100 个 Hard-Lite 场景中接受 71 个（71%），因此低于预设的 80%
难度晋级门槛，未进入学习训练。

## Hard-v2 独立训练/验证池

Hard-v2 保留 3 个障碍物，仅提高目标速度和延迟/噪声，避免与原始 Hard 同时增加
几何难度：

| 项目 | Hard-v2 设定 |
| --- | --- |
| target speed scale | `0.55/0.65` |
| obstacle count | `3` |
| message delay | `2` steps |
| action delay | `2` steps |
| command noise | `0.04 m/s` |
| reverse lane change / target crossing | disabled / not required |

分别生成了独立训练池和 holdout 池，每池 Hard 300 场、150 个镜像组，初始外逃证书
均为 100%，scene hash 不同：

| 池 | Hard 场景 | expert accepted | 接受率 |
| --- | ---: | ---: | ---: |
| train pool | 300 | 213 | 71.0% |
| holdout pool | 300 | 210 | 70.0% |

训练池的 213 个接受场景数量超过 192 条示范要求，holdout 的 210 个接受场景也
保持了独立性。但两池专家接受率都低于 80% gate，因此该档只能作为 teacher/难度
诊断，不能作为已通过的 Hard promotion 结果。

## 训练决策

曾启动 Hard-v2 的 192 条 DN-MPC teacher 示范收集。前 48 条 rollout 中出现较多
早期 `safety_failure`，说明在当前 Hard-v2 延迟/噪声条件下，public-belief teacher
本身尚未提供足够稳定的高质量示范。该进程在训练开始前主动停止，没有生成完整
checkpoint，也不计入模型性能结果；这样避免把 teacher 失败轨迹误当作正向示范
污染 residual actor。

## 当前结论

1. Phase79 conservative Nominal 仍是当前有效 reference：三 seed、252 episodes
   的 safe capture 为 `94.05%`，collision 为 `3.97%`。
2. 原始 Hard 的专家可行率只有 `11/100`；Hard 可行子集上的学习模型 safe capture
   为 `15.15%`，因此原始 Hard 明显过难。
3. Hard-v2 将几何难度降回 3 个障碍物后，专家可行率稳定提高到 `70--71%`，并
   形成了超过 192 条可用场景的独立训练/holdout 池，但尚未达到预设晋级门槛。
4. Delay-only 档将目标速度和障碍物数量保持为 Nominal，只增加 2-step 延迟和
   `0.04 m/s` 噪声；300 场中接受 `227` 场（75.67%），仍低于 80% gate，但已
   证明延迟/噪声本身就是主要失败轴。
5. 下一步应做更温和的 1-step/低噪声过渡档并加入 teacher 质量门控；当前任何
   Hard-v2/Hard-Lite/Delay-only 档都未取得 Hard promotion，Stress 或 locked-test
   继续关闭。

## 可复现实验配置

- `configs/phase80_progressive_hard_lite.yaml`
- `configs/phase80_progressive_hard_v2.yaml`
- `configs/phase80_progressive_delay_only.yaml`
- `configs/phase80_dnmcp_dagger_residual_hard_v2.yaml`
- `scripts/generate_phase78_obstacle_avoidance_scenes.py --seed-offset`

生成结果目录保留在本地 `results/`，不纳入 Git；其中 Hard-v2 holdout 使用
`seed_offset=1000000`，与训练池隔离。
