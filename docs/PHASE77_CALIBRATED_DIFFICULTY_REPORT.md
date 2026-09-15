# Phase 77：候选轨迹可行性过滤与分级难度标定

## 目的与实验边界

本阶段执行新的 calibration-only 流程：先约束目标候选轨迹的物理可行性，
再用专家上界标定 Easy/Nominal/Hard，最后只在 Nominal 上训练和比较学习策略。
所有场景均来自 `calibration/validation`，没有读取、修改或重跑 locked-test。

本阶段不把 local CBF 当作形式化安全证明。它仍是经验安全过滤器；
R-CLBF-QP/robust CBF-QP 仍保持 diagnostic No-Go。

## 已实现的候选轨迹过滤

`CaptureRadiusPursuit3DEnv` 的 adaptive maneuvering adversary 现在对每条短期候选
轨迹执行有限时域可行性检查：

- 每次模式重选间隔仍为 6--10 个控制步，候选 rollout horizon 为 16 步；
- 对 rollout 的起点、中点和终点检查世界边界以及所有障碍物的净空；
- 净空阈值为 `max(target_maneuver_feasibility_margin_m, drone_radius + safety_margin)`，
  当前标定值为 `0.60 m`；
- 有可行候选时只在可行集合中按最难拦截分数选择；不可行候选会记录拒绝原因、
  首次失败步和可行前缀长度；
- 所有候选都不可行时使用 boundary-recovery/即时安全优先的受限 fallback，避免
  目标被迫瞬时穿墙或瞬间改变方向；
- 候选集合包含直逃、横向转向、左/右/上绕行、反向变道、爬升/下降、短时加速
  和 boundary recovery；绕行分支可在接近障碍物前选择路线。

该过滤器约束的是目标模拟器的候选动作，不泄露目标真实状态给防守方。
`oracle_route` 只用于 calibration 上界诊断，不能作为部署方法。

## 分级场景池

当前权威场景池为：

```text
results/phase77_calibrated_difficulty_scenes_v4/
```

共 300 个场景、150 个 mirror groups，每档 100 个场景、50 个 mirror groups；
Easy/Nominal/Hard 均具有 central crossing contract、直接路径被障碍物阻挡、
至少两条旁路。原始场景 SHA-256 为：

```text
85de3ca8c2bef5b23182d1600bd712a6aa5c36b071fa1e2b2ec8b2b5912fad6d
```

分档只改变预先声明的难度因素，不改变 locked-test。当前 v4 profile 为：

| 档位 | 目标速度倍率 | 障碍物数 | 感知/通信 | 执行延迟 |
|---|---:|---:|---|---:|
| Easy | 0.45--0.55 | 3 | 无 dropout、低噪声 | 0 步 |
| Nominal | 0.50--0.60 | 3 | dropout 0.12、观测噪声 0.025、消息延迟 1 步 | 1 步 |
| Hard | 0.65--0.75 | 4--5 | 更高 dropout/噪声、私有机动延迟 2 步 | 4 步 |

Hard 已生成但在 Nominal 学习策略门控通过前不运行闭环评价，避免先看 Hard
结果再调整 Nominal。

## 专家上界与有效场景筛选

专家 calibration 使用仅限场景标定的 `oracle_route`：它读取模拟器目标状态，
因此是物理可完成性的上界，不是可部署 controller。有效场景定义为：捕获或安全
终止、无物理碰撞、无目标/防守方越界、无目标障碍物碰撞。另行报告 route-exercise
指标（目标实际进入障碍物另一侧 zone），不把两者混为一谈。

| 难度 | 物理可行专家场景 | 物理可行率 | route-exercise 场景 | safe capture in pursuit | capture event |
|---|---:|---:|---:|---:|---:|
| Easy | 100/100 | 100% | 39/100 | 39% | 100% |
| Nominal | 90/100 | 90% | 42/100 | 39% | 90% |
| Hard | 暂不运行 | — | — | — | — |

Nominal 的物理可行率超过预设的 80% calibration gate，因此允许进入 Nominal
学习策略阶段；其中 10 个场景被专家物理失败筛除，不应作为学习策略的负标签。
Nominal 的 39% safe-capture-in-pursuit 不是 39% 物理可行率，而是更严格的
“成功捕获且完成中央穿越路线”的 route exercise 指标。

专家运行的 pooled route-and-safety latency（仅作诊断）为：

| 难度 | p50 | p95 | p99 |
|---|---:|---:|---:|
| Easy | 1.59 ms | 46.23 ms | 57.82 ms |
| Nominal | 1.61 ms | 48.17 ms | 61.83 ms |

100 ms 不是硬门槛；后续学习策略仍统一保存 predictor/planner/QDR/safety/total
的 p50/p95/p99。

## Nominal 学习策略训练现状

固定 Nominal profile 后，已经用三个独立 seed 收集 route-aware public-belief
expert demonstrations 并训练 recurrent behavior-cloning actor：

| seed | 接受示范 | 拒绝示范 | trainer 内置随机验证 safe capture | collision |
|---:|---:|---:|---:|---:|
| 771501 | 48 | 5 | 0% | 50% |
| 771502 | 48 | 6 | 0% | 75% |
| 771503 | 48 | 10 | 50% | 50% |

三份 checkpoint 都已保留在：

```text
results/phase77_nominal_training_seed771501/checkpoint.pt
results/phase77_nominal_training_seed771502/checkpoint.pt
results/phase77_nominal_training_seed771503/checkpoint.pt
```

上表是训练脚本内部的小规模随机验证，不能替代固定 100 场 Nominal block 的
多 seed 复核。固定 Nominal block 的结果完成后，只有同时满足预注册的安全、
捕获和 mirror-group 稳定性门槛，才会把某个 checkpoint 晋级为 Nominal reference。

## 当前门控结论

- [x] 候选轨迹边界/障碍物可行性过滤已实现，并有单元测试；
- [x] Easy/Nominal/Hard 各至少 100 个 calibration 场景已生成；
- [x] 专家上界已完成 Easy/Nominal，Nominal 物理可行率通过 80% gate；
- [ ] 三 seed 学习策略在固定 Nominal 100 场景上的闭环复核尚未完成；
- [ ] Nominal 学习策略尚未证明优于已有 reference；
- [ ] Hard/Stress 尚未开放；
- [x] locked-test 保持不变、未访问。

因此当前不能宣称“新 adversary 下学习模型已经有效拦截”。最稳妥的状态是：
场景与专家标定已通过，候选学习模型仍处于 Nominal validation gate，且已有训练
信号显示明显的 covariate-shift/数据覆盖不足风险。

## 下一步

1. 完成三 seed、固定 `blocks/nominal.jsonl` 的闭环评估，报告 safe capture、
   capture event、collision、boundary、timeout、capture time、clearance，
   以及 route/predictor/planner/QDR/safety/total 的 p50/p95/p99。
2. 若 Nominal 未通过，先增加安全 route-aware expert 数据覆盖和 scheduled
   perturbation/DAgger 式再训练；不降低目标可行性过滤器，也不把专家失败场景
   混入成功示范。
3. 只有 Nominal 通过后，锁定 checkpoint、阈值和随机种子，再评估 Hard，之后
   才能打开 Stress；任何难度标定仍只使用 calibration/validation。
4. 最后才考虑一次性 locked-test 诊断；在此之前不允许调参或重采样 locked-test。
