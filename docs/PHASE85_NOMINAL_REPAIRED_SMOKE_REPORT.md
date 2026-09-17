# Phase 85：Nominal Repaired Smoke 报告

> 日期：2026-09-17
> split：development calibration only
> locked-test：未读取、未修改

## 1. 配置和数据

- 环境配置：`configs/phase85_target_contract_repaired_environment.yaml`
- smoke protocol：`configs/phase85_nominal_repaired_smoke.yaml`
- 场景数：4 episodes / 2 mirror groups
- 场景 manifest hash：`20e5a0458cda362ec61429aed2b8451f53269e135499041d7c87c7c1ca804d9f`
- 场景文件 hash：`a0ad981d8a982b9d978c915b4851c952e624f5ae7d0bad73c8e24302107f35eb`
- 目标安全集：`target_boundary_margin=2.0 m`，`target_radius=0.0 m`
- execution：零延迟、零命令噪声

## 2. 闭环结果

| 指标 | Oracle route | Public-belief route |
|---|---:|---:|
| safe capture | 100% | 100% |
| target invalid episode | 0% | 0% |
| target boundary violation | 0% | 0% |
| target obstacle violation | 0% | 0% |
| defender physical collision | 0% | 0% |
| defender boundary violation | 0% | 0% |
| timeout | 0% | 0% |
| expert acceptance | 100% | 100% |

## 3. 压力因素诊断

另一个 10 episode 的单因素 smoke 同时包含 action delay、command noise、obstacle-near 和 formation-tight。该池的目标非法率仍为 `0%`，但专家接受率为 `70%`，失败集中在两个 action-delay 镜像和一个 obstacle-near 场景，防守方首个边界失败在第 `24/24/25` 步。

失败时防守方 y 位置达到 `+10/-10`，目标仍在有效安全集内；这归因于执行延迟下的防守方追击过冲，不能作为目标合同失败，也不能作为已通过 Nominal promotion 的证据。该压力池暂不进入训练。

## 4. 门控结论

Nominal repaired 的目标合同 smoke gate 已通过，但样本规模远小于正式门槛，当前不训练新策略。下一步是按相同安全集生成 development calibration、development validation 和 external holdout，分别达到预注册规模后重新运行 oracle/public-belief 专家筛选；压力因素保持单独 block，不能混入 nominal promotion。
