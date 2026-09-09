# P8 Queue-Aware Reachable-Tube Robust CBF-QP 实验报告

> 本报告记录 `Queue-Aware Reachable-Tube Robust CBF-QP for Delayed Multi-UAV Encirclement` 的 P8 工程实验。结果来自仓库内可复现的锁定种子矩阵，不等同于真实飞行验证，也不构成完整的闭环形式化安全证明。

## 1. 实验目标

P8 的目标是把执行延迟、命令队列、执行噪声和动力学不确定性纳入 CBF-QP 的安全合同：

1. 用执行动力学的多步位置不确定性构造 reachable tube；
2. 在 QP 线性化和独立证书中使用相同的 tube 半径；
3. 分别审计命令、执行 rollout、扫掠体积和连续执行段；
4. 比较 immutable pending queue 与允许清空待执行命令的 `flush_pending` 策略。

本实验没有训练 Learned CLBF，也没有把经验 tube 倍率包装成形式化 reachability 证明。

## 2. 协议与实现

配置文件：`configs/phase8_queue_aware_reachable_tube.yaml`

执行命令：

```powershell
$env:PYTHONPATH="src"
python scripts/evaluate_safety_execution.py `
  --config configs/phase8_queue_aware_reachable_tube.yaml `
  --output-dir results/phase8_queue_aware_reachable_tube_locked_v1
```

实验使用 4 个执行 variant、8 个既有 locked reset、每个 episode 最多 250 步，并同时保留 `nominal`、`local_cbf` 和 `robust_cbf_qp` 三种方法。所有 variant 的安全配置、episode JSONL、step JSONL、TensorBoard 和源文件哈希均写入结果目录。

### 2.1 Tube 倍率

| 场景 | 延迟 | 队列权限 | tube 倍率 |
| --- | ---: | --- | ---: |
| `mild_delay_noise` | 1 步 | immutable | 1.0 |
| `hard_randomized_execution` | 2 步 | immutable | 2.1 |
| `mild_flush_pending_brake` | 1 步 | flush pending | 1.0 |
| `hard_flush_pending_brake` | 2 步 | flush pending | 2.1 |

`2.1` 来自独立的 hard execution 校准结果，并采用向上取整的保守工程倍率。它不是由可达集定理、概率覆盖定理或闭环不变性证明推出的，因此报告中只称为 **empirical calibration multiplier**。

### 2.2 安全合同

P8 同时记录以下指标：

- command certificate：当前命令在预测执行下是否满足 barrier；
- execution rollout certificate：离散执行 rollout 是否满足 tube barrier；
- swept-volume certificate：每个离散动作段的扫掠体积是否满足 barrier；
- continuous-segment certificate：连续段内部的保守安全检查；
- actual post-state：真实执行后状态的 barrier 检查；
- prefix admissibility / abort-required：不可修改队列前缀是否仍可安全执行；
- QP fallback、emergency brake、queue override 与 p50/p95 latency。

## 3. 锁定矩阵结果

### 3.1 Robust CBF-QP 主结果

| Variant | 捕获率 | 安全捕获率 | 碰撞率 | 超时率 | 实际后状态安全 | rollout cert | swept cert | continuous cert | abort-required | QP fallback | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mild + immutable | 37.5% | 37.5% | 0% | 62.5% | 55.5% | 47.9% | 47.9% | 16.8% | 61.2% | 1007 | 110.24 |
| hard + immutable | 12.5% | 12.5% | 0% | 87.5% | 27.4% | 17.7% | 17.7% | 6.8% | 82.0% | 1598 | 108.11 |
| mild + flush | 75.0% | 75.0% | 0% | 25.0% | 100.0% | 96.6% | 99.4% | 35.5% | 0% | 22 | 159.97 |
| hard + flush | 25.0% | 25.0% | 0% | 75.0% | 91.7% | 31.3% | 31.4% | 5.3% | 12.8% | 1291 | 255.39 |

补充现象：

- mild + flush 触发 `370` 次 emergency brake 和 `370` 个 queue override slot；它是本矩阵中安全性最好的 robust 配置，但 p95 已高于 100 ms，且 continuous certificate 仍只有 35.5%。
- hard + flush 触发 `1111` 次 emergency brake、`2222` 个 queue override slot，仍有 `12.8%` 的 abort-required；它把实际后状态安全率提高到 91.7%，但没有解决 hard 场景的捕获率问题。
- immutable 两组分别有 61.2% 和 82.0% 的 abort-required。由于 pending command 不可修改，QP 不能对已经进入执行队列的前缀做事后安全投影。
- robust QP 的碰撞率在四个 variant 上均为 0%，但这不能替代连续段证书；连续段 barrier 仍出现负值，说明“没有观测到碰撞”和“已证明连续时间安全”不是同一件事。

### 3.2 方法对照

在同一锁定矩阵中，nominal/local CBF 的捕获率较高，但不能作为安全模型结论：

| Variant | nominal 捕获 / 碰撞 | local CBF 捕获 / 碰撞 | robust CBF-QP 捕获 / 碰撞 |
| --- | ---: | ---: | ---: |
| mild + immutable | 87.5% / 12.5% | 87.5% / 12.5% | 37.5% / 0% |
| hard + immutable | 100% / 0% | 100% / 0% | 12.5% / 0% |
| mild + flush | 87.5% / 12.5% | 87.5% / 12.5% | 75.0% / 0% |
| hard + flush | 100% / 0% | 100% / 0% | 25.0% / 0% |

nominal/local CBF 的高捕获率伴随较低的执行证书有效率，不能直接与 robust CBF-QP 的安全捕获率比较。P8 的主要代价是：当执行合同变得保守且不可行时，robust QP 选择 fallback、brake 或 abort，而不是继续执行未经证书支持的追捕命令。

## 4. 结论

### 4.1 已经完成的部分

- reachable tube 半径已经统一接入执行 barrier、动作 Jacobian、rollout、swept-volume、continuous-segment 和 recoverability 路径；
- variant 级倍率会被记录到 QP assumption、TensorBoard hparams 和结果配置快照；
- analytic rollout Jacobian 在 robust QP 的 execution-aware 步骤中保持可用；
- 完整矩阵在 4 个 variant、8 个 locked seed、3 种方法下完成并落盘；
- robust CBF-QP 在本矩阵中未出现碰撞或边界越界，但 hard 场景仍未达到可接受的捕获率和证书水平。

### 4.2 还不能宣称的部分

P8 不能支持以下表述：

- 不能称为 Learned CLBF 或 R-CLBF-QP 已完成；
- 不能称为 reachable tube 已被形式化证明覆盖；
- 不能称为整个 queue-aware 闭环系统已经保持不变性；
- 不能用 `0% collision` 代替 continuous-time、continuous-segment 或 swept-volume 的完整证明；
- 不能把 8 个 locked seed 的结果当作充分的统计泛化结论；
- 不能把仿真捕获率外推到真实无人机拦截成功率。

## 5. 当前主要缺陷

1. **队列可控性不足**：immutable queue 在高机动和大延迟下产生大量不可逆的 unsafe prefix，导致 abort-required 和 fallback 激增。
2. **证书层次不一致**：实际后状态安全率明显高于 continuous-segment certificate，说明当前连续段检查仍偏保守或与离散执行模型没有完全对齐。
3. **fallback 只保证保守性，不保证任务进展**：brake/abort 能减少风险，却会显著降低捕获率；需要一个带进展目标的可证书恢复控制器。
4. **tube 倍率仍是经验参数**：尚未基于独立 holdout 给出覆盖率随 horizon、queue length、mass/drag randomization 的统计置信区间。
5. **实时性尚未稳定**：hard flush 的 p95 为 255.39 ms，mild flush 的 p95 为 159.97 ms；即使不把 100 ms 作为硬门槛，也应继续报告尾延迟。
6. **统计规模偏小**：8 个 locked reset 适合审计协议，不足以支撑泛化声明；还需要未参与校准的多种子 holdout 和置信区间。

## 6. P9 建议计划

### P9-A：队列权限消融

在相同随机流下比较 `immutable`、`replace_nonexecuting` 和 `flush_pending`，明确每种权限对应的执行器假设。主要指标是安全捕获率、abort-required、queue override、实际后状态安全率和连续段证书。任何允许改写队列的结果都必须单独标注为不同执行器合同。

### P9-B：按队列状态自适应的 tube

把固定倍率改为 `multiplier(queue_length, delay, uncertainty, speed)`，并在独立校准集上报告每个 horizon 的覆盖率、最小覆盖率和 95% 置信区间。倍率校准集不得与最终 locked holdout 重叠。

### P9-C：把连续段约束前移到规划层

当前 continuous-segment 主要是事后审计。下一版应在候选动作生成和 QP 约束阶段加入分段中间点/速度界约束，并验证“QP 输出通过连续段证书”的比例，而不是只在执行后发现负 barrier。

### P9-D：安全恢复与任务进展联合优化

设计带优先级的恢复问题：先满足队列前缀安全，再在可行集合内最大化捕获进展或缩短到目标的距离。应分别报告安全不可行、任务不可行和超时，避免把所有情况合并成 fallback。

### P9-E：扩大审计规模

保留当前 8 个 locked seed 作为回归集，新增至少 30--50 个未参与倍率校准的 holdout seed，并对安全捕获率、碰撞率、abort-required 和证书覆盖率报告 Wilson 或 bootstrap 区间。

## 7. 可复现证据

- P8 汇总：`results/phase8_queue_aware_reachable_tube_locked_v1/summary.json`
- variant 级结果：`results/phase8_queue_aware_reachable_tube_locked_v1/<variant>/<method>/summary.json`
- 配置快照：`results/phase8_queue_aware_reachable_tube_locked_v1/<variant>/<method>/config.yaml`
- 实验脚本：`scripts/evaluate_safety_execution.py`
- reachable tube：`src/encirclement3d/execution_dynamics.py`
- 执行证书：`src/encirclement3d/safety_certificate.py`
- robust QP：`src/encirclement3d/safety_qp.py`

本轮测试结果：全量测试 `133 passed`；P8 pilot 和完整锁定矩阵均运行成功；`git diff --check` 通过。
