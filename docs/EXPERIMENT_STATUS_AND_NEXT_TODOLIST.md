# 当前实验状态与后续 TodoList

> Phase 53 QDR confirmation 更新：在新的 `validation_confirmation` 100 episode / 50 mirror-group manifest 上，三种子 nominal/windowed 的 safe capture 均为 `99.33% [98.00,100.00]`，paired delta 为 `0.00 pp [-1.67,+1.67]`；windowed minimum clearance 增加 `0.044 m [0.030,0.056]`，但最大 exhaustion streak 为 `30`，超过预注册 `24` 步上限，liveness gate 判定 **No-Go**。windowed predictor/planner/QDR/safety/total p50/p95/p99 为 `5.39/7.40/10.71`、`12.51/17.24/22.03`、`0.63/0.96/1.28`、`0.62/0.85/1.72`、`21.54/28.24/36.03 ms`；100 ms 仍不是硬门槛。nominal QDR 保留为当前固定 delay4/noise008 合同下的参考，windowed tube 冻结为负消融；不访问 locked-test，不混入 UAKR/RNIC。详见 `docs/PHASE53_QDR_RECOVERABILITY_CONFIRMATION_REPORT.md`。

> Phase 54 UAKR fresh-development 更新：在新的 `validation_development` 100 episode / 50 mirror-group manifest 上，固定-K=8 safe capture 为 `98.33% [95.67,100.00]`，original UAKR 为 `95.00% [92.00,97.67]`，paired delta `-3.33 pp [-7.67,+1.00]`，未通过预注册 `-2 pp` 非劣界；timeout 从 `1.00%` 增至 `4.33%`，paired delta `+3.33 pp [+0.33,+7.00]`。UAKR 的平均 K 为 `2.145`、刷新率 `30.98%`，预算节省可复现，但平均 cache age 为 `1.264` steps、suffix admissible rate 下降 `5.68 pp`。uncertainty/residual 对 next-state violation 的 confirmation AUROC 仅 `0.520/0.513`，接近随机。该阶段判定 UAKR **No-Go**，冻结为 efficiency/negative ablation；不扫描阈值、不访问 locked-test、不开放 Full 组合。详细结果见 `docs/PHASE54_UAKR_DEVELOPMENT_REPORT.md`。

> Phase 51 single-process runtime 更新：在 Phase48 同一验证前缀的 40 个 episode 上，QDR-off/on 按顺序单进程运行并固定 Torch intra/inter-op 线程为 `1/1`。QDR-on safe capture 为 `100%`，off 为 `85%`；collision 为 `0%/15%`，paired safe-capture delta 为 `+15.00 pp [+5.00,+27.50]`。匹配前 18 个控制步的 total p50/p95/p99 为 `21.22/26.67/29.39 ms` 对 `12.77/15.15/16.90 ms`，planner p95 增加 `8.66 ms`，QDR 自身 p95 仅 `0.84 ms`。这消除了 Phase49 的并发 CPU 负载混杂，但仍是一种子 development runtime diagnostic；QDR safety-axis 保留，efficiency promotion 仍 No-Go。详见 `docs/PHASE51_QDR_SINGLE_PROCESS_RUNTIME_REPORT.md`。

> Phase 50 QDR×UAKR development pilot 更新：在 Phase48 新鲜 manifest 前 40 个 episode 上，固定 QDR、delay4、bounded noise、immutable authority 和 local CBF，比较 fixed-K=8 与冻结 UAKR。fixed-K=8 safe capture 为 `100%`，QDR+UAKR 为 `97.5%`，paired delta 为 `-2.50 pp [-7.50,0.00]`；UAKR mean K 为 `2.156`、refresh ratio 为 `30.54%`，total p95 从 `64.28` 降到 `57.15 ms`，但 timeout 从 `0%` 增至 `2.5%`。该结果支持预算节省但未通过 `-2 pp` safe-capture 非劣界，冻结为效率/负消融，不继续阈值扫描或开放 Full 矩阵。详见 `docs/PHASE50_QDR_UAKR_DEVELOPMENT_REPORT.md`。

> Phase 49 runtime benchmark 更新：对 Phase48 三种子 off/on 日志按相同 episode index 截取前 18 个控制步，双方各保留 300 个 episode、5,400 个 step。QDR-on total p50/p95/p99 为 `84.81/103.14/117.92 ms`，off 为 `30.59/59.51/68.32 ms`；planner p95 增加 `35.92 ms`，QDR 自身 p95 为 `2.90 ms`。该实验控制了终止长度混杂，但原始日志来自独立 evaluator 进程，因此只作为负 runtime ablation，不作为 process-isolated 部署结论。详见 `docs/PHASE49_QDR_RUNTIME_BENCHMARK_REPORT.md`。

> Phase 48 QDR liveness confirmation 更新：在新的 `validation_confirmation` 100 episode / 50 mirror-group manifest 上，固定 nominal sensing、delay4、bounded command noise `0.08`、immutable authority 和 local CBF，完成 QDR-off/on 三种子配对。QDR-on safe capture 为 `98.33% [95.67,100.00]`，off 为 `83.33% [79.00,87.33]`；paired safe-capture delta 为 `+15.00 pp [10.33,19.67]`，collision delta 为 `-15.33 pp [-20.00,-11.00]`，timeout delta 为 `+0.33 pp [0,1.33]`。QDR-on timeout `0.33%`、最大 exhaustion streak `23` 步，均通过预注册 `5%/5 pp/24 步` gate；prefix/suffix admissible 为 `95.36%/84.44%`，QDR queue mean/max 为 `4/4`。total p50/p95/p99 为 `84.60/103.99/120.67 ms`，因此 liveness confirmation 通过但效率不作正面结论，且不构成安全证明。gate 结果及 evaluator 指标已写入 TensorBoard；下一步为同 episode-length runtime benchmark，再考虑 QDR×UAKR development block。详见 `docs/PHASE48_QDR_LIVENESS_CONFIRMATION_REPORT.md`。

> 更新时间：2026-09-13
> 仓库：[Liangxianguang/new-uav-capture](https://github.com/Liangxianguang/new-uav-capture)
> 当前总判定：**Conditional Go**。预测与 DN-MPC 已形成可复现的模块化证据；robust CBF-QP 只通过了冻结条件下的一步安全 gate；三者直接端到端组合失败，完整方法尚未完成。

> 新的三创新点总计划已整理在 `docs/THREE_INNOVATIONS_MASTER_TODOLIST.md`。该计划以现有 QDR/UAKR/RNIC 的 No-Go 证据为约束，要求先做单模块修复、独立 confirmation 和预注册 gate，再决定是否重新开放 Full 组合。

> Phase 16 v4 更新：扩建的 600 场景 S4 原始档案已按 300 个镜像组冻结为 420/90/90 场景的 train/validation/locked-test；由公开 team belief 重新构建预测原点后得到 13,224/1,922/1,539 个窗口。预测的 validation-only 选型和 locked-test 预测确认均已完成：GRU 是误差主参考，Diagonal SSM、dense S4 与 official S4 保留为 K=8 多模态比较，不能声称 S4 优于 GRU。三种子 validation 闭环选择冻结为每步刷新、`both` 因果动作条件和 local CBF；GRU + distributed delayed DN-MPC 的 validation safe capture 为 `93.70% [90.74%, 96.30%]`，collision 为 `0%`、total p95 为 `44.78 ms`。locked-test 三种子矩阵现已完成：GRU K=1 的 distributed safe capture 为 `94.81% [91.85%, 97.41%]`、total p95 为 `43.58 ms`；Diagonal SSM K=8 与 official S4 K=8 均为 `95.56% [92.96%, 97.78%]`、0 collision，但 total p95 分别为 `90.81/90.02 ms`。相对 GRU 的 distributed gain 均为 `+0.74` 点且 CI 下界为 0，不能声称主分支提升。新增严格 K=1/K=8 配对消融显示：主 distributed 分支均为 95.56% safe capture，候选数无可见收益；worst-case 中 K=8 对 Diagonal SSM 增益 `+7.78` 点 `[+3.70,+12.22]`，对 official S4 增益 `+5.56` 点 `[+1.85,+9.26]`，且 K=1 出现少量 collision/boundary failures、K=8 为 0。因此多候选的当前证据限于保守 worst-case 规划，不可泛化为所有场景收益。新增 GRU none/local-CBF 三种子安全消融显示：distributed 中 local CBF 将 collision 从 `5.56%` 降到 `0%`（paired `-5.56` 点 `[-8.89,-2.96]`），safe capture 差异不显著；worst-case 中 local CBF 同样消除 collision，但 safe capture 下降 `8.15` 点 `[-14.44,-1.48]`，表现为安全--捕获 Pareto，非无代价优势。新增 action-conditioning 闭环消融显示 `none/history/future` 相对 `both` 的 safe-capture paired CI 均跨 0；当前只支持动作条件接口已因果实现和审计，不能支持“动作条件提升闭环捕获率”。并行运行的 221--223 ms p95 不可与 earlier `both` 批次作因果延迟比较。robust CBF-QP diagnostic 仍为 No-Go。完整记录见 `docs/PHASE16_LOCKED_TEST_CLOSED_LOOP_REPORT.md`。

> Phase 16 几何 OOD 更新：新增并冻结 100 个 episode / 50 个上下镜像组的几何外推诊断；wall half-x、half-y、height 均与训练范围不重叠。三种子 GRU `both` + distributed delayed DN-MPC + local CBF 的 safe capture 为 `84.00% [79.33%, 88.33%]`，collision / boundary 均为 `0%`，timeout 为 `16.00% [11.67%, 20.67%]`，total p50/p95/p99 为 `72.93/93.60/111.26 ms`。同一 OOD 块上 worst-case 仅为 `46.33% [39.67%, 53.33%]`，且 collision / boundary 各为 `1.67%`，不可作为 OOD 几何主方案。该块只改变几何，且与 locked-test 场景文件不同，因此相对 ID `94.81%` 的下降只能作为描述性 transfer gap，不能写成配对显著性结论，更不能写成全面鲁棒性或安全证明。详见 `docs/PHASE16_OOD_GEOMETRY_REPORT.md`。

> Phase 16 目标速度 OOD 更新：新增并冻结 `0.82/0.90` 两个训练外速度档，各 50 个 episode，共 50 个上下镜像组，几何仍在训练范围内。三种子 GRU `both` + distributed delayed DN-MPC + local CBF 的 safe capture 为 `42.67% [37.00%, 48.33%]`，collision / boundary 均为 `0%`，timeout 为 `57.33% [51.67%, 63.00%]`，total p50/p95/p99 为 `83.94/106.72/130.87 ms`。worst-case safe capture 为 `55.00% [47.33%, 63.00%]`，timeout 为 `45.00%`，collision / boundary 均为 `0%`。相对 ID `94.81%` 的差异因场景文件不同仅作描述性 transfer gap；该结果表明速度外推是当前主要性能瓶颈，不代表安全证明。详见 `docs/PHASE16_OOD_TARGET_SPEED_REPORT.md`。

> Phase 16 通信/执行 OOD 更新：新增并冻结 `delay6_dropout20` 与 `delay8_dropout25` 两个感知条件，各 50 个 episode，并真正激活 2-step command delay、执行噪声、速度/加速度随机缩放和一阶跟踪 dynamics。三种子 distributed delayed + local CBF 的 safe capture 为 `47.00% [41.00%, 53.00%]`，collision / boundary 为 `53.00% / 19.00%`，total p50/p95/p99 为 `86.09/122.82/178.46 ms`；worst-case safe capture 为 `53.67%`，collision / boundary 为 `46.33% / 24.00%`。这是当前执行感知安全契约的 No-Go 诊断，不是模型成功结果或安全证明。详见 `docs/PHASE16_OOD_DELAY_EXECUTION_REPORT.md`。

> Phase 16 目标行为 OOD 更新：新增并冻结 `short_lookahead`（0.20 s）与 `long_lookahead_margin`（1.20 s / 0.20 s）两种训练外的目标分支决策规则，各 50 个 episode，保持速度、几何和执行配置在 ID 支持内。三种子 distributed delayed + local CBF 的 safe capture 为 `94.33% [91.33%, 97.33%]`，collision / boundary 均为 `0%`，timeout 为 `5.67%`，说明该两种行为参数外推没有可见退化；这只是单轴描述性证据，不能外推到任意对抗策略。详见 `docs/PHASE16_OOD_TARGET_BEHAVIOR_REPORT.md`。

> Phase 27 RNIC formation-slot 更新：新增了有界、确定性的目标相对 3-D 槽位分配代价，并接入 centralized planner、distributed team score、CLI override、单元测试和 TensorBoard。历史 20 场景 pilot 的 formation-slot safe capture 为 `100.0%`，但不作统计主张。随后在全新 `60` episode / `30` mirror-group / `3` checkpoint seed 的 centralized ID confirmation 上，RNIC-off / interceptor-RNIC / formation-slot-RNIC 的 safe capture 分别为 `87.22% [81.11%,92.78%]`、`88.89% [82.22%,95.00%]`、`89.44% [83.33%,94.44%]`；formation 相对 off 的 paired delta 为 `+2.22 pp [-0.56,+5.00]`，collision delta 为 `-2.22 pp [-5.00,+0.56]`，支持预注册 `-2 pp` 非劣方向但不能宣称 superiority。随后在同 manifest 的 distributed delayed confirmation 中三臂均为 `96.67%` safe capture / `3.33%` collision，formation 未改变 episode outcomes，但 pooled total p95 为 `220.87 ms`，高于 off 的 `82.40 ms`；因此当前 distributed promotion No-Go。新 manifest hash 为 `5aaaa79c6dbef2d346ff57d48d238a34fe911d3534ebb576dff0252ba0593022`。详见 `docs/PHASE27_RNIC_FORMATION_SLOT_PILOT_REPORT.md`。

> Phase 30 EGC-MPC 更新：实现了不改预测 backbone 的 geometry-only Escape-Gap-Aware Cooperative MPC，包括 centralized / delayed-distributed 接入、信息边界审计、单元测试、配置快照和 TensorBoard 指标。validation-selection 模式下的一种子/20 场景 development smoke 中，centralized `weight=0.30` 的 safe capture 为 `90%`，较 paired off 的 `85%` 仅作描述性 `+5 pp`，但 mean max gap 从 `5.4929` 增至 `5.4972 rad`，机制指标没有改善；`weight=1.0` 将 mean max gap 降至 `5.3971 rad`、coverage 提至 `0.1887`，但 safe capture 降到 `75%`、collision 升到 `25%`。distributed 分支 off/on 均为 `95%/5%` safe capture/collision，而 total p95 从 `82.07` 增至 `98.93 ms`、p99 为 `108.07 ms`。因此当前 EGC 判为 promotion No-Go，保留为可复现负结果；下一候选转向具有离散可行性门控的 FC-DBF，不开放 locked test。详见 `docs/PHASE30_EGC_PILOT_REPORT.md`。

> Phase 31 FC-DBF 更新：实现了 Feasible-Consensus Distributed Barrier Formation，包括有限 formation-slot 分配、延迟 peer 信息边界、progress/slack/switch 可行性门控、previous-slot hold、consensus-token 优化、配置/测试/源哈希/TensorBoard 记录。受控 validation-selection 一种子/20 场景 smoke 中，FC-DBF 相对 paired off 未改变 worst-case 的 `85%` 或 distributed delayed 的 `95%` safe capture；centralized total p95 为 `68.80→88.92 ms`，distributed total p95 为 `81.37→244.79 ms`。当前结论为工程实现通过但 promotion No-Go，gate exhaustion 为 centralized `0%`、distributed `0.806%`；该结果不开放 locked test，也不构成安全证明。详见 `docs/PHASE31_FC_DBF_PILOT_REPORT.md`。

> Phase 31 confirmation 更新：从 Phase 27 validation manifest 中按完整 mirror group 冻结后 20 组（40 episodes）作为 holdout，完成 checkpoint seeds `727201/727202/727203` 的 off/on 配对确认。worst-case safe capture 为 `87.50% [80.83%,93.33%]` vs `86.67% [80.00%,92.50%]`，paired delta `-0.83 pp [-3.33,0.00]`，collision delta `+0.83 pp [0.00,3.33]`；distributed delayed 两臂均为 `97.50%` safe capture、`2.50%` collision。FC-DBF total p95 为 worst-case `71.06→87.48 ms`、distributed `83.33→243.68 ms`。因此未通过预注册 non-inferiority/latency gate，FC-DBF 冻结为可复现负结果，不开放 OOD 或 locked test。详见 `docs/PHASE31_FC_DBF_CONFIRMATION_REPORT.md`。

> Phase 32 freshness--covariance public-belief fusion 更新：已实现只使用公开 belief 的确定性融合，按 confidence、message age、dropout 和 covariance trace 加权，并提供有效样本数、年龄、协方差迹和 deterministic self-anchor fallback 诊断；centralized、distributed 和闭环评估器均已接入，相关 focused tests 为 `34 passed`。在一组 one-seed/20-episode validation development prefix 上，worst-case safe capture 为 `85.0%→90.0%`、collision 为 `15.0%→10.0%`，distributed delayed 两臂均为 `95.0%/5.0%` safe capture/collision；distributed total p95 为 `89.61→100.28 ms`。这只是描述性 pilot，不能作为统计提升或 locked-test 结果；参数已冻结，下一步必须用 fresh holdout 和三种子做 confirmation。完整数值和 TensorBoard 标签见 `docs/PHASE32_FRESHNESS_COVARIANCE_PILOT_REPORT.md`。

> Phase 32b confirmation 更新：在冻结参数、40-episode/20-mirror-group fresh holdout 和 checkpoint seeds `727201/727202/727203` 上，freshness--covariance fusion 的 centralized worst-case safe capture 为 `86.67%`，baseline 为 `87.50%`，paired delta 为 `-0.83 pp [-5.00,+1.67]`；collision 为 `13.33%` vs `12.50%`。distributed delayed 两臂均为 `97.50%` safe capture、`2.50%` collision。worst-case 的 non-inferiority CI 下界低于预设 `-2 pp`，因此该版本 promotion No-Go 并冻结为可复现负消融；不开放 locked-test，也不继续在该 holdout 上调参。6 个 run 均保存了 effective config、hash、episode/step JSONL 和 TensorBoard。详见 `docs/PHASE32_FRESHNESS_COVARIANCE_CONFIRMATION_REPORT.md`。

> Phase 33 QDR 前缀/后缀审计更新：新增共享执行动力学的 nominal suffix rollout，并把每个控制步划分为 `prefix_safe_suffix_safe`、`prefix_safe_suffix_unsafe`、`prefix_unsafe_recoverable` 和 `prefix_unsafe_unrecoverable` 四类；同时记录 authority、recovery recommendation、suffix barrier/clearance 和 TensorBoard 标签。2 episode schema smoke 中出现 `13/29/3` 个前三类（第四类为 0），safe capture `50.0%` 仅用于 schema/因果审计，不是性能结论。该阶段确认了“不可变队列前缀已经不安全时，新 suffix 无法事后修复”的契约边界；下一步必须在 fresh validation 上按 delay、authority、bounded execution noise 逐轴验证，不能把 precondition classifier 写成安全证明。详见 `docs/PHASE33_QDR_PRECONDITION_AUDIT_REPORT.md`。

> Phase 34 QDR 执行感知修复更新：修复了 QDR 分支单个队友 rollout 的 `[H,1,3]`/`[H,3]` 维度错误，该错误此前使 distributed planner 每步 fallback；新增完整动作序列执行 rollout、QDR 前视障碍范围和 planner fallback 原因日志，完整测试为 `264 passed`。在独立 40-episode/20-mirror-group development block、固定 2-step immutable delay 上，当前源码 QDR-off 为 `100.0%` safe capture、`0%` collision，total p50/p95/p99 `27.84/46.53/51.59 ms`；QDR-on + queue-aware safety projection 为 `85.0%` safe capture、`15.0%` collision，QDR prefix/suffix admissible rate 为 `95.93%/44.63%`，total `87.83/118.60/135.02 ms`。因此实现修复通过，但 QDR 性能 promotion 仍为 No-Go；queue-aware safety projection 仅带来描述性改善，不能掩盖大量 unsafe suffix 和 immutable prefix 不可修复问题。完整记录见 `docs/PHASE34_QDR_EXECUTION_AWARE_VALIDATION_REPORT.md`。未通过 QDR gate 前不进入 confirmation、locked-test 或 Full 组合。

> Phase 35 QDR suffix gate + recovery candidates 更新：在相同 40-episode development block 上加入候选可行性 gate、零动作 suffix 和当前速度保持 suffix 后，QDR-on 达到 `100.0%` safe capture、`0%` collision/boundary/timeout；suffix admissible rate 从未修复版本的 `48.03%` 提升到 `86.80%`。但 gate exhaustion 仍为 `21.76%`，mean capture time 为 `3.915 s`，predictor/planner/QDR/safety/total p50/p95/p99 分别为 `10.94/21.90/23.77`、`87.56/127.46/146.08`、`0.77/1.42/1.68`、`1.18/2.53/2.89`、`107.44/150.21/168.13 ms`。相对 QDR-off total p95 `46.53 ms` 明显恶化，因此本阶段是“安全 outcome 修复通过、效率晋级 No-Go”；不进入 QDR confirmation、locked-test 或 Full 组合。完整记录见 `docs/PHASE35_QDR_SUFFIX_GATE_RECOVERY_REPORT.md`。

> Phase 36 QDR batched rollout 更新：将候选执行 dynamics rollout 改为候选维度批量计算，在相同 40-episode development block 上保持 `100.0%` safe capture、`0%` collision/boundary/timeout，suffix admissible `86.80%`、gate exhaustion `21.76%` 不变；planner p50/p95/p99 降为 `57.10/82.39/93.80 ms`，total 降为 `78.06/105.27/120.09 ms`。相对 Phase35 的 total p95 `150.21 ms` 有明显改善，但相对 QDR-off `46.53 ms` 仍超出 `15%` 增幅门槛，因此效率晋级仍 No-Go。完整记录见 `docs/PHASE36_QDR_BATCHED_ROLLOUT_REPORT.md`。

> Phase 38--40 QDR rollout 压缩更新：在不改变候选排序、执行 dynamics、immutable authority、QDR gate 或 safety layer 的前提下，加入同一控制步内的 local/peer rollout cache，并移除与 weighted reference 完全相同的重复候选。40-episode development block 仍为 `100.0%` safe capture、`0%` collision/boundary/timeout，suffix admissible `86.80%`、gate exhaustion `21.76%`；Phase40 planner/total p50/p95/p99 为 `24.51/40.43/46.77` 与 `43.61/63.77/72.40 ms`。相对 Phase36，total p95 再降约 `39.4%`；但相对 QDR-off total p95 `46.53 ms` 仍高约 `37.1%`，超过 `15%` 相对延迟门槛。运行时实现通过，QDR promotion 仍 No-Go；当前不开放 fresh confirmation 或 Full 组合。详见 `docs/PHASE38_40_QDR_RUNTIME_COMPRESSION_REPORT.md`。

> Phase 41 当前源码 QDR-off 配对基线更新：由于 protocol 文件默认 `queue_aware_rollout=true`，本阶段显式使用 `--no-queue-aware-rollout --no-queue-aware-safety-projection` 重跑相同 40 集 development block。结果为 `100.0%` safe capture、`0%` collision/boundary/timeout，total p50/p95/p99 为 `24.44/43.36/47.23 ms`；与 Phase40 QDR-on 的 `43.61/63.77/72.40 ms` 配对后，QDR-on total p95 增幅约 `47.1%`。基线审计通过，但 QDR efficiency promotion 仍 No-Go；该结果不开放三种子确认或 Full 组合。详见 `docs/PHASE41_QDR_CURRENT_SOURCE_REFERENCE_REPORT.md`。

> Phase 42 delay4 fresh validation 更新：在独立 `80` episode / `40` mirror-group manifest 上，只把 command delay 从 2 提高到 4，保持 immutable authority、GRU `both`、采样 seed、MPC 和 local CBF 不变。单 checkpoint seed 下，QDR-off/on safe capture 为 `77.5%/97.5%`，collision 为 `22.5%/2.5%`，按 episode 配对 bootstrap delta 分别为 `+20.0 pp [11.25,30.00]` 和 `-20.0 pp [-30.00,-11.25]`；total p50/p95/p99 为 `29.19/54.29/61.93` 对 `45.88/78.65/95.10 ms`。这是目前 QDR 在延迟压力轴上最强的条件性证据，但仍有 `2.5%` collision、`19.16%` suffix gate exhaustion 和 `44.9%` total p95 增幅；需完成另外两种子及 bounded-noise confirmation 后再判断 promotion。详见 `docs/PHASE42_QDR_DELAY4_FRESH_VALIDATION_REPORT.md`。

> Phase 43 delay4 三种子 confirmation 更新：在同一独立 fresh manifest 上完成 `727201/727202/727203`，off/on 各 `240` episodes。QDR-off/on safe capture 为 `77.08%/96.67%`，collision 为 `22.92%/1.25%`，boundary 为 `5.00%/1.25%`；paired bootstrap delta 为 `+19.58 pp [13.75,25.42]` safe capture、`-21.67 pp [-27.92,-15.83]` collision。三种子 summary total p50/p95/p99 均值为 `45.65/64.65/80.66` 对 `47.68/90.14/107.40 ms`，QDR-on timeout 为 `2.08%`，suffix gate exhaustion 均值 `19.02%`。因此 delay4 安全效果 confirmation 有条件通过，但效率/liveness promotion 仍 No-Go；下一步只加入 bounded execution noise 或单独改变 authority，不进入 Full 组合。详见 `docs/PHASE43_QDR_DELAY4_THREE_SEED_CONFIRMATION_REPORT.md`。

> Phase 44 delay4 + bounded noise 三种子 confirmation 更新：固定 delay4 和 immutable authority，只加入 command-noise std `0.08 m/s`、`3σ` clipping。三种子、240 episodes/arm 的 QDR-off/on safe capture 为 `77.92%/97.92%`，collision 为 `22.08%/0.83%`，paired bootstrap delta 为 `+20.00 pp [14.17,25.83]` 和 `-21.25 pp [-27.50,-15.00]`；boundary 为 `4.58%/0.83%`，timeout 为 `0%/1.25%`。三种子 summary total p50/p95/p99 均值为 `61.75/92.49/117.87` 对 `74.92/123.91/154.77 ms`，QDR-on suffix gate exhaustion 均值 `19.99%`。因此 QDR 在延迟+有界执行扰动轴上保持安全收益，但效率/liveness promotion 仍 No-Go；下一步单独测试 authority，不混入其他变量。详见 `docs/PHASE44_QDR_DELAY4_NOISE008_CONFIRMATION_REPORT.md`。

> Phase 45 authority 消融更新：固定 Phase44 的 delay4、noise008、immutable 参考条件后，分别测试 `replace_nonexecuting` 和 `flush_pending`。两者均将 collision/boundary 降为 `0%`，但 safe capture 下降到 `81.25%/85.42%`，timeout 升到 `18.75%/14.58%`；相对 immutable 的 paired safe-capture delta 为 `-16.67 pp [-22.50,-10.83]` 和 `-12.50 pp [-19.17,-5.83]`。三种子 summary total p50/p95/p99 均值分别为 immutable `74.92/123.91/154.77`、replace `96.71/136.54/168.29`、flush `97.92/138.55/169.45 ms`。因此取消/清空队列不是无代价安全修复，冻结为 safety--liveness negative ablation，主分支继续采用 immutable。详见 `docs/PHASE45_QDR_AUTHORITY_ABLATION_REPORT.md`。

> Phase 46 QDR 前缀失败路径诊断更新：对 Phase44 immutable 与 Phase45 两种 authority 的 9 个 `steps.jsonl` 做公开几何审计，所有 19,951--21,378 行均具备九个 classifier 字段。immutable 的 pooled prefix admissible 为 `93.48%`，首次 violation 主要为 boundary `741`、obstacle `202`、inter-agent `5`；replace/flush 分别降至 `60.33%/67.73%`，但 gate-exhaustion 仍为 `18.23%/21.28%`。240 个 immutable episode 中有 238 个曾出现过至少一次 exhaustion，说明“ever exhausted”不能直接作为失败标签。evaluator 已加入连续耗尽长度、首次耗尽步、恢复次数和 episode-level exhausted 标志，并通过 schema smoke；可选 feasibility-first 在 40-episode development smoke 中保持 outcome、step gate 字段及测试中的 selected action/cost 一致，但 planner/total 延迟没有下降。该阶段是后验机制诊断，不改变 QDR promotion No-Go，也不构成安全证明。详见 `docs/PHASE46_QDR_PREFIX_FAILURE_AUDIT_REPORT.md`。

> Phase 47 QDR OOD 延迟/执行压力更新：在同一冻结的 100 episode / 50 mirror-group 通信+执行 OOD manifest 上，使用 GRU `both` 三种子、local CBF、同一采样和 planner 条件完成 QDR-off/on 配对。QDR-on 的 distributed safe capture 为 `93.67% [90.33,96.67]`，QDR-off 为 `47.00% [41.00,53.00]`；collision 为 `2.33%` 对 `53.00%`，paired delta 为 `-50.67 pp [-57.00,-44.00]`，safe-capture delta 为 `+46.67 pp [40.00,53.00]`。boundary 为 `2.33%` 对 `19.00%`，但 timeout 从 `0%` 增至 `4%`，mean capture time 从 `3.874` 增至 `5.681 s`，minimum clearance 从 `0.136` 增至 `0.522 m`。QDR-on pooled total p50/p95/p99 为 `50.35/97.74/111.21 ms`；QDR queue mean/max 为 `2/2`，prefix/suffix admissible 为 `95.84%/86.14%`，ever-exhausted episode rate 为 `96.67%`，说明安全收益伴随 liveness/feasibility 代价。该阶段是 OOD 条件性安全证据，不是形式化安全证明或无条件 promotion；下一步固定 QDR，单独预注册 timeout/exhaustion gate 和同长度 runtime benchmark。详见 `docs/PHASE47_OOD_DELAY_EXECUTION_QDR_REPORT.md`。

> Phase 18 delay-aware conformal reachable-tube pilot 更新：已实现公开 belief 边界内的 horizon-dependent split-conformal 半径校准、queue/prediction-age 对齐、RNIC 接入、UAKR 管宽诊断、TensorBoard 和在线 smoke。835 个 calibration windows 的完整轨迹覆盖率为 `90.18%`，1,087 个 untouched confirmation windows 为 `85.92%`，低于预设 `90%`；半径约为 `6.09--8.59 m`，在线 smoke 最大约 `10.11 m`，说明第一版管过宽且跨 split 泛化失败。8 场景 smoke 的 safe capture 为 `87.50%`、collision 为 `12.50%`，total p50/p95/p99 为 `36.81/57.88/66.22 ms`，仅用于连通性和日志验证。该候选判定 No-Go，不是安全证明，也不开放完整 QDR×UAKR×RNIC 组合。详见 `docs/PHASE18_DELAY_AWARE_CONFORMAL_TUBE_PILOT_REPORT.md`。

> Phase 18b 平衡 mirror/context 修复更新：按完整 mirror group 和公开 context score 重划 validation，固定半径 confirmation 完整轨迹 coverage 为 `83.32%`；引入预设 `gain=0.25` 的公开 context 自适应缩放后升至 `99.79%`，但有效半径均值/最大值为 `8.416/10.404 m`，仍偏宽，且紧致性门槛未在探索性确认前预冻结。在线 8 场景 paired smoke 与无管对照的 episode 结局 `8/8` 一致，safe capture 均 `87.50%`、collision 均 `12.50%`，total p95 为 `56.27/55.99 ms`。该阶段是 coverage-repair diagnostic，不是捕获率提升或安全证明。详见 `docs/PHASE18B_BALANCED_CONTEXT_TUBE_REPORT.md`。

> Phase 18c tube-width UAKR smoke 更新：将冻结 tube width 以 `weight=0.15`、`scale=10 m` 真正加入 UAKR 调度；相对无管，mean K `3.364→3.952`、prediction refresh ratio `45.64%→52.02%`，但 8 场景 safe capture、collision、boundary 仍为 `87.50%/12.50%/0%`，total p95 `55.99→57.87 ms`。这是负消融：增加了预算但没有闭环收益，不晋级为 UAKR 主贡献。详见 `docs/PHASE18C_TUBE_UAKR_SMOKE_REPORT.md`。

> Phase 18d coverage--compactness gate 更新：将 confirmation coverage、mean effective radius 和 maximum effective radius 做成可执行 gate。当前 artifact 的 coverage `99.79%` 通过，但 mean/max radius `8.416/10.404 m` 超过预冻结 `8/10 m` 限制，脚本输出 `development_confirmation_no_go` 并将 `Gate/coverage_pass`、`Gate/compactness_pass`、`Gate/overall_pass` 写入 TensorBoard。该阶段进一步确认当前管只能以过宽为代价获得高覆盖率。详见 `docs/PHASE18D_COMPACTNESS_GATE_REPORT.md`。

> Phase 15 v3 正式更新：600 场景的数据划分和动作/时间戳契约审计已通过；30 epoch、3 seed 的冻结离线测试中，GRU 的 projected minFDE 为 `0.7244 +/- 0.0352 m`，官方 S4 为 `1.3373 +/- 0.0275 m`。每步刷新且因果动作条件可用率约 `92%--95%` 时，GRU + distributed delayed DN-MPC 为 `95.56% [92.96%, 97.78%]` safe capture，官方 S4 为 `94.44% [91.48%, 97.04%]`。不能声称 S4 优于 GRU；下一步是 validation-only 的 `none/history/future/both`、单/多模态和风险/刷新消融。详见 `docs/PHASE15_S4_V3_FORMAL_MULTISEED_REPORT.md`。

> Phase 15 action-conditioning 消融已完成：GRU 在相同 30 epoch、3 个匹配 seed 下，`both` 的 validation minFDE 为 `0.8000 +/- 0.0064 m`，优于 `future` 的 `0.8388 +/- 0.0099 m`、`history` 的 `0.9994 +/- 0.0076 m` 和 `none` 的 `1.0288 +/- 0.0101 m`；相对 `both` 的配对 minFDE delta 区间均为正。后续冻结测试确认了这一排序，`both` 的 projected minFDE 为 `0.7000 +/- 0.0057 m`、coverage 为 `94.59%`，但 candidate feasibility 只有 `77.60%`。`both` 已锁定为后续主配置，不能继续用 locked test 选模型；下一步是 no/local/robust safety 三路闭环和单/多模态消融。详见 `docs/PHASE15_S4_V3_ACTION_CONDITION_ABLATION_REPORT.md`。

> 最新 Phase 14 authority matrix：hard `immutable` / `replace_nonexecuting` / `flush_pending` 的 safe capture 分别为 `12.5% / 0% / 25.0%`，actual post-state safety 为 `29.55% / 90.05% / 99.75%`，collision/boundary 均为 `0% / 0%`。`flush_pending` 仍只是仿真执行器能力，不能直接外推到真实飞控；完整执行感知安全控制仍为 No-Go。详见 `docs/PHASE14_AUTHORITY_MATRIX_REPORT.md`。

## 1. 先给结论

目前不能把项目描述为“实验已经完善”或“R-CLBF-QP 闭环安全证明已经成立”。更准确的表述是：

1. Mamba/portable SSM + 条件扩散预测器在未见 `adaptive_adversarial` 目标策略上有效，候选轨迹必须经过动力学投影。
2. 投影候选驱动的 worst-case / distributed DN-MPC 在 100 个 locked-test 场景上达到 97.33%--99.00% safe capture，优于 95.00% 的 DynamicEncirclement 基线。
3. velocity-level robust CBF-QP 在预先筛选的 robust-safe reset 上通过了一步独立证书检查，但这不是执行扰动下的闭环安全证明。
4. 将 robust CBF-QP 直接接入 P4 流程后，safe capture 只有 50.00%，collision 为 49.00%，boundary violation 为 16.00%；因此当前端到端组合路径为 **No-Go**。
5. P7.2 新复核目录 `results/phase7_p7_2_independent_certificate_reaudit_v5/` 已具备完整配置、summary、episode/step 日志和 TensorBoard；当前 source 下独立 current/next-state certificate 均为 `100%`，但与历史 P7 v2 仅有 `50/100` episode 结局一致，safe capture 从 `50%` 变为 `99%`，因此 exact behavior-parity gate 为 No-Go，不能替换历史 P7 结论。详见 `docs/PHASE28_P7_2_INDEPENDENT_CERTIFICATE_REAUDIT_REPORT.md`。
6. P8 viability-guard full development audit 已完成：hard `flush_pending`、8 seeds 下 safe capture `12.50%`，collision/boundary `0%/0%`，timeout `37.50%`，safety abort `50.00%`，actual post-state safety `100%`，但 continuous certificate 仅 `18.31%`，safety p95 `1207.35 ms`。这是 safety--liveness Pareto No-Go，不能写成 R-CLBF-QP 证明。详见 `docs/PHASE29_VIABILITY_GUARD_AUDIT_REPORT.md`。

7. 最新 Phase 14 八种子 authority matrix 已完成，但三种权限均未达到 hard `80%` safe-capture 工作门槛；下一步必须先做证书/队列失败分解，再扩大 unseen seed。
8. 最新 continuous-QP progress/recovery audit 仍未通过：2-seed/250-step hard `flush_pending` 的 safe capture 为 `0%`，actual post robust safety 为 `71.4%`，continuous certificate 为 `12.4%`；重复 emergency brake 消除了该小样本中的物理碰撞/越界，但没有恢复 robust tube 不变性。详见 `docs/PHASE14_PROGRESS_RECOVERY_REPORT.md`。

## 2. 当前捕获率

这里的主指标是 `safe capture`：目标进入捕获半径且没有被碰撞、越界或其他安全失败污染。普通 `capture` 不等于安全捕获，不能只报告普通捕获率。

### 2.1 P4：预测 + DN-MPC，100 个未见策略 locked-test

所有方法复用同一份 `adaptive_adversarial` 场景文件；下表为三种 prediction checkpoint 的聚合结果，DynamicEncirclement 是共享场景基线。

| 方法 | Safe capture | Collision | Boundary | Timeout | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| DynamicEncirclement baseline | 95.00% | 2.00% | 0.00% | 3.00% | 基线 |
| Worst-case DN-MPC | 97.33% | 1.00% | 0.00% | 1.67% | 通过 |
| Distributed ideal | 98.00% | 1.00% | 0.00% | 1.00% | 通过 |
| Distributed delayed | **99.00%** | **0.00%** | 0.00% | 1.00% | 通过 |
| Distributed dropout | **99.00%** | **0.00%** | 0.00% | 1.00% | 通过 |
| Distributed none | 98.00% | 0.00% | 0.00% | 2.00% | 通过 |

P4 的 planner solver/valid/effective rate 为 99.97%--100%，没有出现正式结果级别的 planner fallback。这个结果支持“预测候选能够转化为围捕收益”的模块化结论，但不等于真实飞行安全保证。

### 2.2 P5：robust CBF-QP 独立安全过滤

在冻结的 8 个 robust-safe episode、130 个控制步上：

| 指标 | 结果 |
| --- | ---: |
| Safe capture | 100%（8/8） |
| Collision / boundary violation | 0% / 0% |
| Solver success | 130/130 |
| Independent one-step certificate | 100% |
| Next-state safety | 100% |
| Fallback / QP infeasible | 0 / 0 |
| Filter p95 latency | 6.27 ms |

这是一个**有前置条件的 velocity-level 一步安全结果**。执行延迟、跟踪误差、动作队列、连续段 swept-volume 和多步 forward invariance 审计均未通过，因此不能命名为 R-CLBF-QP，也不能写成闭环安全证明。

### 2.3 P7：三模块直接组合

同一份 P4 100-episode locked-test 上，使用 worst-case DN-MPC + robust CBF-QP：

| 指标 | Local CBF baseline | Joint robust CBF-QP |
| --- | ---: | ---: |
| Safe capture | 97% | **50%** |
| Collision | 1% | **49%** |
| Boundary violation | 0% | **16%** |
| Safety certificate valid | n/a | 64.87%（原始诊断） |
| Safety fallback | n/a | 35.13% |
| Planner success / valid / effective | n/a | 100% / 100% / 100% |

5,289 个控制步中的 1,858 个 fallback 已分类为：`precondition_invalid=1,225`、`qp_infeasible=350`、`inconsistent_action_bounds=283`。因此主要问题在安全层的 reset/margin/action-bound/fallback 契约，不是 DN-MPC planner。

## 3. 预测模块目前能声称什么

在未见 `adaptive_adversarial` 审计中，三种冻结训练 seed 的均值为：

| 模型 | Projected minFDE | Full-trajectory coverage |
| --- | ---: | ---: |
| GRU | 0.9033 ± 0.0648 m | 82.22% ± 4.86% |
| Projected SSM diffusion | **0.5363 ± 0.0071 m** | **90.31% ± 0.03%** |

相对 projected GRU，diffusion 的 minFDE 改善 40.63%，coverage 提高 8.09 个百分点。但原五模式正式 locked-test 的 minFDE 仅改善 3.87%，没有达到预注册的 10% 门槛；候选权重仍是 `uniform_uncalibrated`，不能解释为每条轨迹的概率。

因此预测模块是 **Conditional Go**，不是无条件通过。

## 4. 延迟结论

P4 使用每 20 个控制步刷新预测并缓存候选。三 checkpoint 聚合的 total-control p95 约为 167.73--229.62 ms，严格 100 ms/10 Hz 参考尚未达到；但用户已明确 100 ms 不作为硬门槛，因此它属于部署架构和工程优化项，不推翻 P4 方法结果。

后续应分别报告 predictor、planner、safety 和 total-control 延迟，并验证 batch inference、异步预测、少步扩散或蒸馏方案。不能用低频缓存的平均收益掩盖候选年龄和执行安全问题。

## 5. 实验完成度判断

| 层级 | 当前状态 | 是否完成 |
| --- | --- | --- |
| 工程实现、日志、基础回归 | 代码和主要审计链已具备 | 基本完成 |
| SSM/扩散预测贡献 | 未见策略结果成立，但原始 10% 门槛未过 | 条件完成 |
| Scenario MPC / DN-MPC | S3 与 adaptive locked-test 已通过 | 完成当前模块 gate |
| 一步 velocity-level robust CBF-QP | 冻结 robust-safe reset 上通过 | 条件完成 |
| 执行扰动下安全 | 多轮 audit 为 No-Go | 未完成 |
| R-CLBF-QP 形式化证明 | 尚未成立 | 未完成 |
| 三者端到端完整方法 | 50% safe capture / 49% collision | 未完成，No-Go |
| P7.2 独立证书复核 | 工件完整；与历史 P7 行为 parity No-Go | 条件完成 |

结论：**P4 模块化实验已经完善到可以写阶段性结果；P5/P7 还没有完善到可以宣称完整方法成功。**

## 6. 接下来的 TodoList

### P7.2：完成独立证书审计

- [x] 复核历史 v4 目录的缺失工件，并另建完整的 P7.2 re-audit 输出目录。
- [x] 使用原 P7 的 100 个场景、checkpoint、MPC 配置和 20-step refresh，重新运行独立 `check_one_step_safety` 审计。
- [x] 验证 `summary.json`、`episodes.jsonl`、`steps.jsonl`、TensorBoard event、配置快照和 source hash 全部落盘。
- [x] 汇总 independent certificate valid、current-state safety、next-state safety、barrier 和 violation counts；当前 source 下独立 current/next-state certificate 均为 `100%`。
- [x] 对比历史 v2 与当前 re-audit：仅 `50/100` episode 结局一致，safe capture `50%→99%`，因此 exact behavior-parity gate 判定 No-Go。
- [x] 新增 `PHASE28_P7_2_INDEPENDENT_CERTIFICATE_REAUDIT_REPORT.md`，明确工件完整但不能替换历史 P7 结论。

**P7.2 gate：** 工件完整和逐步证书追溯通过；控制结果与原始 v2 的 exact parity 失败，因此 P7.2 仅为条件完成，不能自动把 P7 的 No-Go 改成 Go。

### P8：修复安全契约，而不是继续堆叠模型

- [x] 完成最新 hard `immutable`、`replace_nonexecuting`、`flush_pending` 八种子 authority matrix；结果为执行审计证据，仍未通过 hard 捕获与连续证书 gate。
- [ ] 按队列前缀、horizon index、barrier family、execution error 和 fallback candidate 分解 Phase 14 失败。
- [ ] 在同一批 step states 上对比 horizon `1/2/3/5`，区分证书保守性与真实状态离开安全集。

- [ ] 冻结 P4 的同一组场景，分别复现 local CBF、当前 robust CBF-QP 和 nominal action。
- [ ] 将 `precondition_invalid`、`qp_infeasible`、solver numerical failure、fallback 后证书失败严格分开统计。
- [ ] 重新定义 reset protocol：初始状态必须明确属于 robust safe set，并分别测试 tight / nominal / out-of-contract 三类状态。
- [ ] 统一 safety margin、速度盒约束、加速度约束、动作变化约束与环境实际执行模型。
- [ ] 明确 pending command 的 authority；不可修改的 queue prefix 必须进入 certificate，而不是只检查当前动作。
- [ ] 设计经过独立 checker 的 certified fallback，并记录 fallback 前后 current/next/prefix/swept safety。
- [ ] 在 mild / hard delay-noise-tracking 扰动下执行多步 post-step 与 continuous-segment audit。
- [x] 修正 delayed fallback 的进展评分：在 `queue length + 1` 的首个可控时刻评价新动作，而不是只评价旧队列前缀。
- [x] 对 prefix 仍不可行的状态重复保持 emergency brake；只有 prefix 恢复可行后才允许 resume。
- [x] 修正 `actual_post_robust_state_safe` 的统计口径，使其真正包含 reachable-tube robust barrier。
- [x] 完成 braking/viability-aware recovery guard 的 8-seed development audit；actual post-state safety `100%`，但 safe capture `12.50%`、abort `50.00%`、continuous certificate `18.31%`，判定为 safety--liveness Pareto No-Go。
- [ ] 将 safe abort、robust-contract violation 和 physical collision 分开建模，并在 episode 终止协议中明确其优先级。

**P8 gate（建议预注册）：** 初始契约有效率 100%；post-step independent safety、fallback 后安全率和关键场景证书覆盖率至少 99%；QP infeasible、未解释 solver failure 和未认证 fallback 均低于 1%；collision/boundary 不得劣于 local CBF baseline；否则停止扩大模型，报告安全--捕获 Pareto。

### P9：运行时工程优化

- [ ] 先保留当前 8×8 配置作为主结果，不因一次 40-episode 消融直接替换。
- [ ] 在同一 locked-test 上比较 batch inference、异步 predictor、少步 diffusion、蒸馏和候选缓存年龄。
- [ ] 报告 predictor/planner/filter/total p50、p95、p99 及 stale-candidate 比例。
- [ ] 100 ms 只作为部署参考；若仍超过参考值，给出明确的低频预测 + 高频 planner/filter 架构和安全假设。

### P9.1：Phase 16 OOD 泛化矩阵

- [x] 完成 geometry-only OOD：100 episode / 50 镜像组，三种子 distributed 分支为 `84.00% [79.33%, 88.33%]` safe capture，0 collision/boundary；worst-case 为 `46.33% [39.67%, 53.33%]` 且各有 `1.67%` collision/boundary。该结果仅是单轴诊断，详见 `docs/PHASE16_OOD_GEOMETRY_REPORT.md`。
- [x] 完成 target-speed OOD：速度 `0.82/0.90` 各 50 episode，几何保持 ID；三种子 distributed 分支为 `42.67% [37.00%, 48.33%]` safe capture、`57.33%` timeout、0 collision/boundary，total p50/p95/p99 为 `83.94/106.72/130.87 ms`。该结果说明速度外推是主要性能瓶颈，详见 `docs/PHASE16_OOD_TARGET_SPEED_REPORT.md`。
- [x] 完成 delay/execution OOD：消息延迟 6/8 步、dropout 20/25%，并激活 2-step command delay 和 tracking/noise dynamics；三种子 distributed 分支为 `47.00% [41.00%, 53.00%]` safe capture、`53.00%` collision、`19.00%` boundary。该结果是当前 local-CBF execution-safe contract 的 No-Go，详见 `docs/PHASE16_OOD_DELAY_EXECUTION_REPORT.md`。
- [x] 完成 unseen target behavior OOD：两种未见 lookahead/commit 规则，各 50 episode，速度/几何/执行保持 ID；三种子 distributed 为 `94.33% [91.33%, 97.33%]` safe capture、0 collision/boundary。该结论限于两个冻结行为规则，详见 `docs/PHASE16_OOD_TARGET_BEHAVIOR_REPORT.md`。
- [ ] 冻结 stronger communication / execution delay / tracking-noise OOD 块，保持几何和目标策略在 ID 范围内。
- [ ] 每个 OOD 块固定三种子、镜像配对、source hash 和完整 episode/step 工件；报告 safe/ordinary capture、collision、boundary、timeout、clearance、动作条件可用率以及 p50/p95/p99。
- [ ] 在各单轴诊断完成前，不运行混合压力场景，也不基于 OOD 结果重新选 checkpoint、调整模型或改写 locked-test 主结论。

### P10：重新开放端到端三种子实验

- [x] 已完成 GRU 三种子 × 90 locked-test 场景的 validation-selected local-CBF 闭环确认；该结果是 locked-test diagnostic，不替代 P8 安全 gate。
- [ ] 只有 P8 通过后，才重开 robust CBF-QP/certified-fallback 端到端 locked-test。
- [ ] 固定同一 scenes、target-truth 隔离、seed、source hash 和日志格式。
- [ ] 至少保留 local CBF、robust CBF-QP、certified fallback 三条路径；不要只报告成功率。
- [ ] 同时报告 safe/ordinary capture、collision、boundary、timeout、capture time、minimum clearance、certificate、fallback、solver failure 和完整延迟。
- [ ] 端到端 gate：aggregate safe capture 不低于 local CBF baseline，collision/boundary 不增加，三种 seed 方向一致，且每个失败可追溯。

### P11：R-CLBF-QP 只在 P8/P10 通过后开展

- [ ] 明确定义 CLBF/robust CLBF、扰动集合、执行动力学和离散时间安全条件。
- [ ] 训练集、验证集、locked-test 完全分离，禁止用 locked-test 调 margin 或选模型。
- [ ] 独立 certificate checker 必须与训练损失和 QP 内部 residual 解耦。
- [ ] 证明或验证 multi-step forward invariance、fallback、solver tolerance、饱和、延迟和连续段安全。
- [ ] 若无法完成形式化闭环证明，贡献名称降级为“经验型/条件性 robust CBF-QP”，不使用 R-CLBF-QP 安全证明表述。

### P12：最终整理与论文证据

- [ ] 固化最终实验协议、配置、checkpoint SHA-256、场景 SHA-256 和源码 hash。
- [ ] 生成完整 ablation 表：GRU/SSM diffusion、raw/projected、expected/worst-case/CVaR、centralized/distributed、local/robust safety。
- [ ] 单独整理失败案例和 No-Go 结果，不删除失败运行目录。
- [ ] 更新 `RESULTS_INDEX.md`、阶段报告和 README 限制说明。
- [ ] 完整回归：`python -m pytest -q`、Python 编译检查、`git diff --check`。
- [ ] 仅提交代码、测试和正式报告；保留用户已有的 `README.md` 与 `docs/EXPERIMENTAL_STUDY_REPORT.md`，不得误暂存。

### P13：Phase 17 可复现三创新点路线

- [x] 按 `PHASE17_QUEUE_ADAPTIVE_REACHABILITY_TODOLIST.md` 完成第一版 Queue-Aware Delayed-State Rollout 适配器、单元测试、配置快照和 TensorBoard smoke 记录；它修正规划状态和候选时间索引，不重新打开 robust CBF-QP。
- [x] 完成 QDR 的 nominal immutable-prefix 几何诊断、post-hoc endpoint audit、TensorBoard 字段和 90 场景 validation paired comparison；QDR-on safe capture 为 `88.89%`，QDR-off 为 `94.44%`，collision 为 `11.11%/5.56%`，当前 gate No-Go。详见 `docs/PHASE17_QDR_VALIDATION_REPORT.md`。
- [x] Phase 17 geometry-OOD Diagonal-SSM 20-episode pilot 已完成；结果显示 no-QDR 100%/0% collision、QDR fixed-K 85%/15% collision、UAKR+QDR 90%/10% collision，仍不足以作为正式统计结论。
- [x] 完成 UAKR 的可解释策略、动态 K/refresh、缓存轨迹推进、残差触发与 TensorBoard 记录；三种子 validation 显示平均 K 约 2.68、刷新率约 36.6%，但 safe-capture 非劣 CI 为 `[-2.59,+2.96] pp`，未通过 `-2 pp` gate。详见 `docs/PHASE17_UAKR_VALIDATION_REPORT.md`。
- [x] 完成 RNIC 的受限到达时间代价、central/local DN-MPC 接入、单元测试与 TensorBoard 记录；三种子 ID validation 行为与固定距离代价完全相同，target-speed OOD safe capture 由 `95.00%` 降至 `94.33%`，delay/execution OOD safe capture 由 `41.33%` 降至 `40.33%`，并增加 planner/total latency，判定 No-Go。进一步的 300-episode 后验诊断显示 RNIC 最小到达裕量对 collision 的 pooled AUROC 为 `0.504`，四个风险箱的 collision rate 不单调，因此当前 slack 也不能作为可靠风险预警。详见 `docs/PHASE17_RNIC_VALIDATION_REPORT.md`、`docs/PHASE17_RNIC_TARGET_SPEED_OOD_REPORT.md`、`docs/PHASE17_RNIC_DELAY_EXECUTION_OOD_REPORT.md` 与 `docs/PHASE17_RNIC_DIAGNOSTIC_ANALYSIS.md`。
- [x] 实现只使用 policy-safe uncertainty 的 K={1,4,8} 与 refresh={4,2,1} 自适应调度；主结果使用可移植 Diagonal SSM diffusion，GRU 保持 K=1 参考。当前仍需 validation 冻结阈值。
- [x] 实现基于加速度/限速到达时间裕量的 Reachability-Normalized Interception Cost；当前为 heuristic 版本，interceptor-only 与 formation-slot 消融仍未完成。
- [x] 完成 RNIC delay/execution-OOD 后验可靠性审计；RNIC slack pooled AUROC `0.504`，诊断子结论 No-Go。
- [ ] QDR、UAKR、RNIC 尚未形成可推广的单模块通过证据；不直接运行完整 2×2×2 主结果矩阵，先完成“delay-aware conformal reachable tube”候选方案的开发集校准与 confirmation gate。
- [ ] 方法冻结后才运行新建且未见的 ID/OOD confirmation；Phase 16 已公开 OOD 只作开发诊断。
- [ ] 不设 100 ms 硬门槛，但必须逐组件报告 p50/p95/p99，并在同硬件单进程条件下进行公平比较。

**P13 gate：** ID safe capture 满足 `-2 pp` 非劣；QDR/UAKR/RNIC 至少一个对应失败轴获得配对统计支持的改善；最终组合不增加 collision/boundary；所有阈值、seed、hash 和 step-level decision 可复现。详细 gate、指标、测试和六周日程见 `docs/PHASE17_QUEUE_ADAPTIVE_REACHABILITY_TODOLIST.md`。

### P14：Phase 18 可达管修复与下一轮实验

- [x] 完成 `DelayAwareConformalReachableTube`：horizon 逐步半径、同时覆盖倍率、有限样本上分位数和 provenance。
- [x] 校准脚本严格分离 validation 前半 calibration 与后半 confirmation；禁止读取 locked-test，并保存 score JSONL、summary、config、source hash 和 TensorBoard。
- [x] 将管半径按 `queue_length + prediction_age` 对齐，短 horizon 截断、长 horizon 按目标速度尾部外推，并在 `ScenarioTrajectorySet` 中做长度校验。
- [x] 在 central/local RNIC 使用 `distance + tube_radius` 的保守规划 surrogate；代码注释明确其不是动态可达集证明。
- [x] 完成 8-episode online smoke；确认闭环日志和 TensorBoard 写入 `enabled_rate`、mean/max radius、budget score 以及 p50/p95/p99。
- [x] 固化失败结果：confirmation full-trajectory coverage `85.92%` 低于目标 `90%`，且管半径约 `6.09--8.59 m`，因此不晋级为正式方法。
- [x] 完成 Phase 18b：完整 mirror group + public-context 分层；固定管 confirmation `83.32%`，context-adaptive confirmation `99.79%`，paired online smoke 结局 `8/8` 与无管对照一致。
- [ ] 冻结 Phase 18 失败 artifact，不用 confirmation 或 locked-test 调半径、coverage、uncertainty gain 或 RNIC 权重。
- [ ] 新建 development-only split，按 target motion mode、observation/message age 和 execution-delay regime 做条件校准；保留全新 confirmation split。
- [ ] 并行比较 horizon schedule 与 trajectory-level scalar conformal score，预注册 coverage、tube volume/radius 和 interception performance 的双门槛。
- [ ] 运行单进程、固定线程数的 disabled / RNIC-only / UAKR-diagnostic 三路对照，分别报告 safe capture、collision、timeout、clearance、predictor/planner/RNIC/total p50/p95/p99。
- [ ] 只有新 confirmation 同时满足 trajectory coverage 与紧致性门槛后，才开放一个小规模未见 OOD block；在此之前不运行完整 2×2×2 组合矩阵。
- [ ] Phase 18b 的 `99.79%` 不得直接晋级：先在新 confirmation 前冻结 radius/volume 上限，再验证 coverage--compactness 双门槛。
- [x] 完成 Phase 18c tube-width→UAKR 受控 smoke；结果判定为 negative ablation，不能把更多 K/refresh 解释为性能提升。
- [x] 完成 Phase 18d 可执行 coverage--compactness gate；当前 artifact coverage pass、compactness fail，自动 No-Go。
- [ ] 若继续使用 tube 触发 UAKR，必须在新 confirmation 前冻结计算预算上限，并证明 ID 非劣与 OOD 受益同时成立；否则仅保留 RNIC/coverage 诊断用途。

**P14 gate：** confirmation full-trajectory coverage 不低于预注册目标（当前目标 90%），同时满足预注册 tube-radius/volume 上限；RNIC 或 UAKR 接入不得使 ID safe capture 低于固定参考的 `-2 pp` 非劣界，且不得增加 collision/boundary。任一条件失败则保留负结果并降级为诊断工具。

### P15：Phase 19 gated-RNIC repair pilot

- [x] 增加显式 `reachability_activation_slack_s`，只在严重负到达裕量时激活 RNIC 代价；默认 `None` 保留原始 RNIC 语义。
- [x] 在 Diagonal-SSM、distributed delayed DN-MPC、local CBF 和固定 20 场景 validation 前缀上完成 RNIC-off 与 `-0.75/-0.50/-0.25 s` 配对 pilot。
- [x] 三个阈值均为 safe capture `95.0%`、collision `5.0%`、timeout `0%`，与 RNIC-off 的 `20/20` episode outcomes 完全一致；total p95 为 `67.97/67.07/66.15 ms`，高于 RNIC-off 的 `51.43 ms`。
- [x] 记录每个 pilot 的 config、source hash、episode/step JSONL、summary 和 TensorBoard；未读取 locked-test。
- [x] 将 gated RNIC 判为 No-Go for promotion；不得把阈值化解释为风险预测器或安全证明。
- [ ] 若重新研究 RNIC，先构造独立 reachable-set/risk label 并在 fresh validation block 检验 risk ranking；在信号层修复前不再扩大 planner 超参扫描。

详见 `docs/PHASE19_GATED_RNIC_PILOT_REPORT.md`。Phase 16 的 GRU `both` + distributed delayed DN-MPC + local CBF 仍是主参考模型。

### P16：Phase 20 UAKR threshold-calibration pilot

- [x] 增加 `--adaptive-low-threshold` 与 `--adaptive-high-threshold`，使 UAKR 阈值可以显式覆盖并写入 effective config。
- [x] 在固定 20 场景 validation 前缀上测试 `low=0.30, high=0.55`；RNIC、QDR 关闭，采样协议和 checkpoint 固定。
- [x] 结果为 safe capture `90.0%`、collision `10.0%`、boundary `0%`，mean K `3.19`，refresh ratio `42.0%`，total p50/p95/p99 `70.64/138.04/170.73 ms`；RNIC-off paired reference 为 `95.0%/5.0%/0%` 和 `36.58/51.43/59.87 ms`。
- [x] 将“仅降低 UAKR 阈值”判为 No-Go；不继续进行无校准阈值扫描。
- [ ] 下一轮仅在 fresh development calibration split 上构造独立 failure/risk label，先验证 uncertainty reliability，再设计 risk-calibrated budget policy。

详见 `docs/PHASE20_UAKR_THRESHOLD_CALIBRATION_PILOT_REPORT.md`。该结果不能改写 Phase 17 的三种子结论，也未访问 locked-test。

### P17：Phase 21 UAKR reliability audit

- [x] 增加离线 `scripts/analyze_uakr_reliability.py`，从 episode/step 日志重建 episode-level mean/max uncertainty、AUROC 和分位箱，并写入 TensorBoard。
- [x] 重跑 Phase 17 三种子 validation 日志，共 270 episodes；mean uncertainty 对 failure/collision 的 AUROC 为 `0.740`，maximum uncertainty 为 `0.703`。
- [x] 记录最高 uncertainty 四分位 failure rate `16.42%`，最低四分位 `0%`；同时记录中间分位不严格单调，避免把该分数误写成 calibrated probability。
- [ ] 在 fresh development calibration split 上针对 prediction miss、next-step safety margin violation、timeout 分别建立风险映射；不把 episode 终局标签直接泄漏给在线策略。
- [ ] 冻结映射后再做 untouched confirmation，并同时检查 safe capture、collision、timeout、mean K、refresh ratio 以及 p50/p95/p99。

详见 `docs/PHASE21_UAKR_RELIABILITY_AUDIT_REPORT.md`。当前 UAKR 是有希望但尚未晋级的方向。

### P18：Phase 22 UAKR mirror-group confirmation audit

- [x] 将 reliability audit 扩展为 `mirror_group_half`，按完整 mirror group 分配 calibration/confirmation，避免 episode 顺序切分导致的分布偏移。
- [x] 在三种子、135 个 per-run mirror-group 单元上完成固定 seed `20260912` 的审计：calibration/confirmation 为 67/68 组、134/136 episodes。
- [x] mean-U AUROC 为 `0.771`（calibration）和 `0.687`（confirmation），max-U 为 `0.720/0.667`；confirmation mean-U 最高/最低四分位 failure rate 为 `8.82%/0%`。
- [x] 判定为“弱 ranking signal、未校准”：不得直接接入风险门控或把 uncertainty 写成 failure probability。
- [ ] 在 calibration groups 上拟合预注册标签的 calibration map，并只在 confirmation groups 上评估冻结映射；分别记录 prediction miss、next-step safety violation 和 timeout。

详见 `docs/PHASE22_UAKR_MIRROR_GROUP_CONFIRMATION_REPORT.md`。只有映射在 confirmation 上稳定，才允许进入新的闭环 pilot。

### P19：Phase 23 risk-calibrated UAKR confirmation

- [x] 实现无第三方依赖的单调 PAVA risk map，并输出可冻结 JSON artifact、derived UAKR thresholds 和 TensorBoard 指标。
- [x] 增加 `--adaptive-risk-calibration`，评估器自动读取 artifact、写入 effective config 并记录 artifact hash；同时拒绝与显式阈值覆盖混用。
- [x] 在 canonical mirror-group confirmation 上完成 3 seed × 46 episodes；risk-calibrated UAKR safe capture `95.65% [92.03%, 98.55%]`、collision `4.35%`、boundary/timeout `0%`，mean K `4.05`，refresh `51.77%`，total p50/p95/p99 `79.97/141.98/153.67 ms`。
- [x] 相对原始 UAKR 的 safe-capture delta 为 `+1.45 pp [0.00, +4.35]`，但相对 fixed K=8 为 `-2.17 pp [-5.07, 0.00]`，未通过固定预算非劣和效率 gate。
- [x] 将 risk-calibrated UAKR 判为“有改善但不晋级”的可复现实验消融；Phase 16 继续作为主参考。
- [ ] 若重新尝试，改用 prediction miss 或 next-step safety-margin violation 等局部标签，先证明 calibration 稳定且预算下降，再开放新的闭环 confirmation。

详见 `docs/PHASE23_UAKR_RISK_CALIBRATED_CONFIRMATION_REPORT.md`。

### P20：Phase 24 UAKR residual-triggered budget pilot

- [x] 增加显式 `residual_high_trigger_m`：当 public prediction-to-belief residual 超过阈值时，升到 high-K 并强制刷新；默认 `None` 保持旧行为。
- [x] 在固定 20 场景 validation 前缀、Diagonal-SSM seed `727201`、distributed delayed DN-MPC、local CBF 上测试 `0.40 m`。
- [x] 结果 safe capture `85.0%`、collision `15.0%`、boundary/timeout `0%`、mean K `3.02`、refresh `37.27%`、total p50/p95/p99 `65.30/141.61/151.57 ms`；原始 UAKR 为 `90.0%/10.0%` 与 `60.15/133.44/139.84 ms`。
- [x] 判定 No-Go；不继续扫描 residual threshold，也不把局部 AUROC 解读为干预保证。
- [ ] 若重开该方向，先分别评估 refresh-only 与 K-escalation 的 intervention effect，再在独立 development split 上冻结策略。

详见 `docs/PHASE24_UAKR_RESIDUAL_TRIGGER_REPORT.md`。

### P21：Phase 25 UAKR residual-only refresh pilot

- [x] 增加 `residual_refresh_trigger_m`，在不改变 K bucket 的条件下，只强制 predictor refresh。
- [x] 在固定 20 场景 validation 前缀、Diagonal-SSM seed `727201`、distributed delayed DN-MPC、local CBF 上测试 `0.40 m`。
- [x] 结果 safe capture `85.0%`、collision `15.0%`、boundary/timeout `0%`、mean K `2.67`、refresh `36.96%`、total p50/p95/p99 `67.25/142.34/154.52 ms`；原始 UAKR 为 `90.0%/10.0%` 与 `60.15/133.44/139.84 ms`。
- [x] 判定 residual-only refresh 与 residual-triggered high-K 均 No-Go；不再扫描 residual threshold。
- [ ] 若继续 UAKR，只能研究 intervention-effect calibration，并在 fresh development split 上比较 refresh-only、K-only 和 joint intervention。

详见 `docs/PHASE25_UAKR_RESIDUAL_REFRESH_REPORT.md`。

### P22：Phase 26 QDR delayed-state safety projection pilot

- [x] 增加显式 `queue_aware_safety_projection` 开关；仅在 QDR + local CBF 时将安全投影锚定到首个可控的延迟状态，默认关闭以保持旧行为。
- [x] 完成 20 场景、同 seed 的 QDR-off、旧 QDR-on 和新投影三臂配对 pilot。
- [x] 新投影将 nominal prefix 最小 clearance 从 `-4.844 m` 改善到 `-0.229 m`，但 safe capture 仍为 `80.0%`、collision 仍为 `20.0%`，total p95 为 `149.90 ms`，高于 QDR-off 的 `141.40 ms`。
- [x] 判定为“契约语义改善、闭环性能 No-Go”；不继续扫描 safety-projection margin。
- [x] 发布 prefix-risk classifier：新 schema 的 465 个 QDR 控制步中 `92.47%` 前缀可行，`7.53%` 存在 public-geometry 违规；首个违规原因计数为 obstacle `30`、inter-agent `3`、boundary `2`。旧日志缺字段时分析器明确返回 incomplete，不做事后猜测。
- [x] 完成显式 authority 诊断：replace-nonexecuting 为 `75%` safe capture / `20%` collision / `5%` timeout，flush-pending 为 `70%` / `5%` / `25%`；两者均未通过 safe-capture 非劣，保留为 safety--capture Pareto 失败边界。
- [ ] 下一轮只研究经过预注册的 prefix precondition 与 recovery policy，不再扫描 safety-projection margin；任何 authority 变更必须作为 plant contract 单独报告。

详见 `docs/PHASE26_QDR_SAFETY_PROJECTION_PILOT_REPORT.md`。

### P23：Phase 27 RNIC formation-slot pilot

- [x] 增加 `formation_slot` RNIC：在每个 horizon step 对目标相对 3-D 槽位进行有界精确排列搜索，使用归一化到达时间短缺代价选择 assignment；最多支持 6 架机，明确标注为 heuristic 而非 reachable-set proof。
- [x] 将 formation-slot 代价接入 centralized finite-shooting planner；修复 planner 传递 conformal tube 半径时的潜在局部变量错误。
- [x] 将 formation-slot 代价接入 distributed DN-MPC：局部 best response 在获得所有 peer message 时使用 cooperative proxy；最终 team score 对完整 rollout 只计一次 formation cost，缺失/延迟 peer 时不伪造全局信息。
- [x] 增加 `--rnic-cost-mode` 与 `--rnic-slot-radius-m` validation-only CLI override；记录 `rnic_cost_mode` 和 `assignment_switch_rate` 到 episode/summary/TensorBoard。
- [x] 完成 20 场景三臂 centralized validation pilot：formation-slot 相对 RNIC-off 的描述性变化为 safe capture `+5.0 pp`、collision `-5.0 pp`、mean clearance `+0.037 m`，但 total p95 增加 `17.61 ms`；相对 interceptor-only 同样只观察到 1 个 paired collision 被消除。
- [x] 增加 formation-slot 形状、assignment、边界条件、distributed team-score 测试；相关 focused tests 共 `24 passed`。
- [x] 不把单 seed pilot 晋级为主结果；在 fresh validation confirmation 前冻结 `slot_radius_m`、`weight_reachability` 和 assignment policy。
- [x] 用 3 个 checkpoint seed、同一 fresh 镜像组协议完成 off / interceptor / formation 三臂 confirmation，并报告 paired CI、planner/RNIC/total p50/p95/p99；ID safe-capture 非劣方向通过，但 superiority 未建立。
- [x] 在 distributed delayed DN-MPC 下复现 off / interceptor / formation 三臂 confirmation；三臂均为 `96.67%` safe capture / `3.33%` collision，formation 未改变 episode outcomes，但 pooled total p95 增至 `220.87 ms`，因此当前 distributed promotion No-Go。
- [ ] 不再为当前 formation-slot 版本开放 OOD promotion gate；若保留诊断，geometry、target-speed、delay/execution 只做逐轴 transfer profiling，不能调参或改写主结果。
- [ ] 若重新研究 RNIC，先设计 communication-aware local assignment surrogate 或低秩 consensus，再在新的 development-calibration 上冻结计算预算、switch-rate 和 paired non-inferiority gate。
- [x] 在 distributed confirmation 前不重新开放 QDR×UAKR×RNIC full matrix；当前 formation-slot 结果应作为 centralized ID 非劣、distributed 无行为增益且有显著延迟代价的可复现消融。

详见 `docs/PHASE27_RNIC_FORMATION_SLOT_PILOT_REPORT.md`。

### P24：Phase 30 EGC-MPC 可复现创新候选

- [x] 实现纯 NumPy 的 `escape_gap_metrics`，输入只包含预测候选、defender rollout 和可公开获得的几何信息；不改变预测 backbone，不引入目标真值泄漏。
- [x] 将有限软 escape-gap cost 接入 centralized worst-case 与 delayed distributed DN-MPC；缺失 peer 信息时不伪造全局状态，team score 只计一次完整 EGC 项。
- [x] 增加配置、CLI 开关、逐步/逐 episode 指标、source hash、TensorBoard 标量和 focused tests；相关 focused tests 共 `22 passed`，并完成 Python 编译和 `git diff --check`。
- [x] 完成一组 paired development smoke：20 场景、checkpoint seed `727201`、centralized/distributed 两分支、EGC off/on `weight=0.30`，另做 centralized `weight=1.00` 方向性探针。
- [x] 记录组件延迟 p50/p95/p99。centralized off/on(0.30) total 为 `65.23/71.69/83.13` 与 `65.18/71.98/78.14 ms`；distributed off/on(0.30) total 为 `75.50/82.07/85.81` 与 `88.94/98.93/108.07 ms`。
- [x] 将结果判定为当前 EGC promotion No-Go：centralized `weight=0.30` 只有描述性 `+5 pp` safe capture，mean max gap 未改善；`weight=1.00` 虽略微改善几何指标但 safe capture 下降 `10 pp`、collision 上升 `10 pp`；distributed episode outcome 不变而 planner/total latency 增加。
- [ ] 不再扫描更多 EGC 权重，也不在 locked-test 上调参；如保留 EGC，只作为 terminal tie-breaker 或诊断指标，等待可控的离散 topology action。
- [ ] 下一候选转向 FC-DBF：有限 formation slot、局部可行性门控、保持旧 slot 的 fail-safe 规则；先做同场景/同 seed validation confirmation，再决定是否进入 OOD。

详见 `docs/PHASE30_EGC_PILOT_REPORT.md`。

### P25：Phase 31 FC-DBF feasible-consensus formation gate

- [x] 实现纯 NumPy 的有限 formation-slot gate，输入仅为 defender rollout、预测候选、公开几何和最新 delayed peer message；不读取目标真值，不改变 predictor backbone。
- [x] 增加 slot tracking error、arrival slack、slot-error progress、assignment switch rate 四类诊断，并对 previous assignment 提供 hold/penalty 语义。
- [x] 当 peer 信息不完整时显式跳过全局 formation gate；当所有候选均不可行时回退 base objective 并记录 `gate_exhausted`，不伪造 feasibility。
- [x] 用 consensus token 将一次 nominal candidate 的精确有限排列分配复用于局部候选，增加固定 assignment evaluator、形状/有限性测试和 distributed integration。
- [x] 增加 centralized/distributed CLI 选项、配置快照、source hash、step/episode 指标与 TensorBoard `Summary/FCDBF/*` 标量；focused tests 为 `22 passed`。
- [x] 在同一 validation-selection 20 场景、checkpoint seed `727201`、采样 seed `745102`、CPU 和 local CBF 条件下完成 FC-DBF off/on 受控 paired smoke。
- [x] 结果：worst-case off/on 均为 `85%` safe capture、`15%` collision；distributed delayed off/on 均为 `95%` safe capture、`5%` collision；centralized total p95 `68.80→88.92 ms`，distributed total p95 `81.37→244.79 ms`。
- [x] 结果：centralized gate exhaustion `0%`、distributed `0.806%`；mean slot progress 分别为 `0.584/0.577 m`，但未观察到闭环捕获收益。
- [x] 判定当前 FC-DBF 为“可复现工程实现但 promotion No-Go”；不在 locked-test 调参，不把负 slack development contract 写成安全裕量或 reachability proof。
- [x] 用 `scripts/select_mirror_group_scenes.py` 按完整 mirror group 从 60 场景源 manifest 中去除前 10 个 smoke groups，冻结后 20 groups/40 episodes 为独立 confirmation holdout；selected scene hash 为 `67d2de08013a9dd61b160cdeffda489fc8a75d4c4625dc2d0c13f435f829ad52`。
- [x] 在 checkpoint seeds `727201/727202/727203` 上完成 off/on 配对 confirmation，使用 10,000 次 hierarchical bootstrap，并记录 aggregate JSON/Markdown、源 hash、配置、step/episode JSONL 和 TensorBoard。
- [x] confirmation 结果：worst-case safe capture `87.50%→86.67%`，paired delta `-0.83 pp [-3.33,0.00]`，collision `12.50%→13.33%`；distributed delayed 两臂均 `97.50%` safe capture / `2.50%` collision。
- [x] 预注册 gate 判定：worst-case safe-capture CI 下界低于 `-2 pp` 且 collision CI 上界为正；distributed outcome 虽不变，但 total p95 `83.33→243.68 ms`，因此整体 confirmation No-Go。
- [x] 冻结 FC-DBF 为可复现负/工程结果，不开放 FC-DBF OOD 或 locked-test，不继续扫描 weight、slot tolerance 或 gate threshold。
- [ ] 转回 QDR/UAKR/RNIC 契约修复或论文材料整理；任何新的 formation 方案须先通过独立 planner-cost benchmark，再进入同一 paired non-inferiority gate。

详见 `docs/PHASE31_FC_DBF_PILOT_REPORT.md`。

### P26：Phase 16b OOD geometry-shift transfer diagnostic

- [x] 新增确定性 OOD 几何生成器，保持完整 mirror group 不跨 split，并将 wall half-x、half-y 和 height 同时移出训练范围；manifest 为 100 episodes / 50 groups，hash 为 `47c393d1d678f25aad14c978734aae5d89105d826ea3e7b32b16a46cc1654501`。
- [x] 在 GRU `both` checkpoint seeds `727201/727202/727203` 上完成 centralized worst-case 与 distributed delayed 三种子评估；QDR、RNIC、EGC 均关闭，使用 local CBF，未使用 locked-test 调参。
- [x] centralized worst-case safe capture 为 `46.33% [39.67%,53.33%]`，collision/boundary 为 `1.67%/1.67%`，timeout 为 `52.00%`；distributed delayed safe capture 为 `84.00% [79.33%,88.33%]`，collision/boundary 为 `0%/0%`，timeout 为 `16.00%`。
- [x] 记录 predictor/planner/safety/total p50/p95/p99：centralized total `61.77/77.13/97.40 ms`，distributed total `72.93/93.60/111.26 ms`；100 ms 仅作参考，distributed p99 超过该值。
- [x] 完成 aggregate JSON、源/配置快照、episode/step JSONL、TensorBoard 保留和报告；判定为“可复现的单轴 transfer diagnostic”，不晋级为泛化保证，不开放新的 locked-test。
- [ ] 后续 OOD 只允许逐轴新增 target behavior/speed 或 delay/execution block；每个 block 先做开发 smoke，再做 fresh three-seed confirmation，不把几何 OOD 结果回流为调参依据。

详见 `docs/PHASE16_OOD_GEOMETRY_REPORT.md`。

### P27：Phase 52 QDR empirical execution tube

- [x] 使用独立 `2048` 样本标定 delay4、bounded-noise 和随机执行参数压力块；基础管覆盖率为 `100.00%/43.80%`，校准后均为 `99.02%`。
- [x] 冻结 multiplier `2.0343` 和逐步 radius vector，将其接入 QDR suffix obstacle/boundary/inter-agent gate；默认关闭，记录配置、hash、逐步诊断和 TensorBoard。
- [x] 在同一 40-episode validation development prefix 完成 nominal/empirical 配对；safe capture `100.00%→15.00%`，timeout `0.00%→85.00%`，collision/boundary 均为 `0%`。
- [x] 判定当前版本为 safety--liveness negative ablation：suffix gate exhaustion `89.25%`，不进入 locked-test，不构成安全证明。
- [ ] 不继续在当前 prefix 扫描 multiplier；若重开，只研究与剩余可恢复时间、局部障碍距离耦合的 gated tube，并新建 development split 与 liveness gate。

详见 `docs/PHASE52_QDR_EMPIRICAL_EXECUTION_TUBE_REPORT.md`。

### P28：Phase 53 QDR recoverability-window repair

- [x] 将经验管从全 horizon 改为显式 `qdr_execution_tube_active_steps=2` 的 immediate replanning window；配置、source hash、step/episode 诊断和 TensorBoard 已接入。
- [x] 在 40-episode development prefix 上完成 nominal/windowed 配对；两臂 safe capture `100%`，timeout/collision/boundary `0%`，paired safe-capture delta `0 pp [0,0]`。
- [x] 记录 windowed predictor/planner/QDR/safety/total p50/p95/p99：`5.47/6.93/7.77`、`12.44/15.93/17.50`、`0.63/0.84/1.05`、`0.62/0.81/0.97`、`21.43/25.88/28.04 ms`。
- [x] 新建独立 `validation_confirmation` manifest：100 episodes、50 mirror groups，manifest hash 为 `b2ec4092bd92c81ac4462cfd72b6c7bc0f08cd68f89526a9acc37517decd38a4`。
- [x] 使用 `727201/727202/727203` 三种子完成 nominal/windowed 配对、aggregate、TensorBoard 和 machine-readable gate。
- [x] timeout rate `0.33%` 与 timeout delta `0 pp` 通过 gate；safe-capture paired delta 为 `0.00 pp [-1.67,+1.67]`，不支持性能 superiority。
- [x] 确认最大 exhaustion streak 为 `30` 步，超过 `24` 步上限；windowed tube 判定 liveness **No-Go**，冻结为负消融。
- [ ] 不在 confirmation 结果上继续扫描 `active_steps`、multiplier 或 planner 权重；若重开，必须新建 development-calibration block 并预注册新 gate。

详见 `docs/PHASE53_QDR_RECOVERABILITY_WINDOW_REPORT.md`。

### P29：Phase 54 UAKR fresh development 与可靠性审计

- [x] 新建并冻结 `validation_development` manifest：100 episodes、50 mirror groups，
  hash 为 `10521ad12ee30071b7e946c7e83889bd348b0d1e9fbd357bc48e1c4bb9c5764b`；
- [x] 在完全相同的 QDR、delay4、bounded noise、GRU checkpoint、planner 和 local
  CBF 合同下，完成 fixed-K=8 与 original UAKR 的三种子配对；
- [x] 完成 episode bootstrap、预算/缓存/suffix 指标、predictor/planner/QDR/
  safety/total p50/p95/p99 和 TensorBoard 记录；
- [x] 完成按 mirror group 划分的 intervention 前置可靠性审计；
- [x] fixed-K=8 / UAKR safe capture `98.33%/95.00%`，paired delta
  `-3.33 pp [-7.67,+1.00]`；UAKR timeout 增量 `+3.33 pp [+0.33,+7.00]`；
- [x] 判定当前 original UAKR 未通过安全/活性 gate，冻结为 efficiency/negative
  ablation，不访问 locked-test；
- [ ] 不在当前开发结果上继续扫描阈值或选择性报告平均延迟；
- [ ] 若重开 UAKR，加入 public queue/QDR prefix feasibility 或 recoverable-margin
  特征，先做 intervention-effect calibration，再新建 confirmation；否则停止
  UAKR 主线并转向 RNIC 或 QDR 模块化论文材料。

详见 `docs/PHASE54_UAKR_DEVELOPMENT_REPORT.md`。

### P30：Phase 55 queue-prefix-risk UAKR 修复候选（仅实现阶段）

- [x] 增加只使用公开 immutable queue prefix geometry 的有界风险特征；
- [x] 将 `queue_prefix_risk` 接入 UAKR 的显式加权分数，默认权重保持为 `0`，
  不改变已有实验行为；
- [x] 记录 step-level 风险、aggregate summary 和 TensorBoard 指标；
- [x] 新增 `configs/phase55_uakr_queue_risk_development.yaml`，固定
  buffer=`0.15 m`、scale=`0.50 m`、weight=`0.30`，不做网格调参；
- [x] 新增风险公式、实验矩阵、预注册 gate 和复现产物说明；
- [x] 通过全量回归测试；
- [ ] 新建独立 fresh development manifest 并完成 fixed-K=8、original UAKR、
  queue-risk UAKR 三种子配对；
- [ ] 若 confirmation AUROC 不能达到 `0.60` 或 safe-capture/timeout gate 失败，
  停止 UAKR 主线并保留为负消融；
- [ ] 在 Phase 55 通过独立 gate 前，不访问 locked-test、不开放 Full 组合。

详见 `docs/PHASE55_UAKR_QUEUE_RISK_PLAN.md`。

## 7. 当前推荐执行顺序

```text
P24 EGC-MPC No-Go freeze
  -> P25 FC-DBF pilot + P26 confirmation (completed; promotion No-Go)
  -> P26b Phase 16b geometry OOD diagnostic (completed; transfer boundary exposed)
  -> P32/P32b freshness--covariance fusion pilot + confirmation (completed; promotion No-Go)
  -> P33 QDR prefix/suffix precondition audit (completed; diagnostic only)
  -> P34 QDR execution-aware contract repair (completed; implementation pass, promotion No-Go)
  -> P8 QDR suffix-feasibility gate + recovery candidates (development safety pass, efficiency No-Go)
  -> P8 runtime optimization and fresh delay/authority/noise confirmation (completed; efficiency/liveness No-Go)
  -> Phase 46 prefix-failure audit + persistence logging (completed; diagnostic only)
  -> Phase 47 delay/execution OOD QDR confirmation (completed; conditional safety-axis Go)
  -> P8 liveness confirmation: timeout/exhaustion-streak gate on a new manifest (completed; Go)
  -> P9 controlled same-length runtime benchmark (completed; negative runtime ablation)
  -> Phase 50 QDR × UAKR development pilot (completed; efficiency/negative ablation)
   -> P9 single-process fixed-thread benchmark (completed; safety direction reproduced, efficiency No-Go)
   -> Phase 52 empirical QDR tube calibration + closed-loop diagnostic (completed; safety--liveness No-Go)
  -> P9 delayed execution contract repair / recoverability-aware tube redesign (Phase53 windowed confirmation No-Go; nominal QDR retained)
  -> P10 Phase54 UAKR fresh development + reliability audit (completed; original UAKR No-Go)
  -> P10b Phase55 queue-prefix-risk UAKR development calibration (implementation complete; closed-loop pending; no threshold scan)
  -> P10c RNIC independent repair/confirmation, or freeze UAKR as negative ablation
  -> Full confirmation remains closed until each promoted module passes its independent gate
  -> P11 可选 R-CLBF-QP 与形式化证明
  -> P12 最终统计、复现和论文材料
```

在 P8 通过之前，不继续训练 learned CLBF，不继续扩大端到端矩阵，也不把 50% safe capture 的联合结果包装成成功结果。即使 P8 最终未通过，P2/P4 已完成的预测和 DN-MPC 模块化结果仍然可以独立形成阶段性贡献。

最新 continuous-QP 结论同样不改变该顺序：短 pilot 的 100% post-state safety 不能替代 250-step 闭环审计；在 recovery gate 通过前，不扩大到 30--50 unseen seeds，也不重新接入 Mamba/扩散/DN-MPC 端到端组合。
