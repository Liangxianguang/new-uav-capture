# Phase 55：Queue-Prefix Risk Augmented UAKR 开发计划

## 目标

Phase 54 表明原始 uncertainty/residual 对下一步违例的排序接近随机。Phase 55
不重新训练 predictor，也不改变 QDR 的 immutable authority，而是在 UAKR 中加入
一个可审计的 public queue-prefix risk feature，测试它能否把“当前待执行队列
已经逼近几何安全边界”的状态转化为更及时的高预算/刷新干预。

这不是第四个主创新点，而是对“Uncertainty-Triggered Adaptive K and Replanning”
的安全耦合修复候选。若失败，UAKR 仍冻结为负消融，QDR 和 RNIC 的实验不受影响。

## 固定定义

从 `prefix_geometry_diagnostics` 的公开输出计算：

\[
r_q=\operatorname{clip}\left(
\frac{\max\{s_m+b-c_{\min},\;v_m\}}{\sigma_m},0,1\right),
\]

其中 `c_min` 是 immutable queue prefix 的最小公开 clearance，`v_m` 是公开的
safety-margin violation，`s_m=0.35 m`，buffer `b=0.15 m`，scale
`σ_m=0.50 m`。空队列的 clearance 为 `+∞`，因此 `r_q=0`。这个分数：

- 只使用 defender geometry、obstacle geometry、world bounds 和公开 action queue；
- 不读取 target truth、终局标签或未来 episode outcome；
- 不取消队列、不改变 command authority，不是安全证书；
- 以 `queue_prefix_risk_weight=0.30` 作为原始 UAKR 分数上的有界加性触发项，
  风险为 0 时严格退化为原始 UAKR；阈值仍固定为 `0.35/0.65`，不在
  locked-test 上选择。

## 实验矩阵

固定 Phase 54 的场景生成方式和执行合同，新建独立 manifest，至少 100 episodes、
50 mirror groups、三个 predictor seeds `727201/727202/727203`。运行：

1. fixed-K=8 reference；
2. original UAKR；
3. queue-prefix-risk UAKR；
4. queue-prefix-risk-only（只保留该 feature，作为机制诊断）；
5. optional no-risk-feature ablation（相同平均预算约束时比较）；

所有臂保持 GRU `both`、QDR、delay4、bounded command noise、distributed delayed
DN-MPC、local CBF、采样 seed 和 episode 配对不变。

## 预注册 gate

- queue-risk UAKR 相对 fixed-K=8 的 paired safe-capture CI 下界不低于 `-2 pp`；
- collision、boundary、timeout 不增加，或 timeout 的 paired CI 必须跨 0；
- 相对 original UAKR，至少一个 next-state violation / suffix exhaustion 指标
  获得明确改善；
- 平均 K 或 predictor refresh 至少下降 `25%`；
- queue-risk feature 在 calibration 和 confirmation 的 violation AUROC 均高于
  `0.60`，否则不称为可靠风险触发器；
- 三种子方向一致，报告 predictor/planner/QDR/safety/total p50/p95/p99；
- 任一 gate 失败都记录为 negative ablation，不扫描 buffer/scale/weight 网格。

## 复现产物

- 配置：`configs/phase55_uakr_queue_risk_development.yaml`；
- 风险计算：`queue_prefix_risk_score`；
- 每步日志：`adaptive_queue_prefix_risk`；决策对象内部同时保留
  `components.queue_prefix_risk`；
- TensorBoard：`Summary/UAKR/mean_queue_prefix_risk`；
- 每个 run 必须保存 manifest/source/config hash、episode/step JSONL 和五级延迟；
- 该计划文档只记录实验设计；在 fresh development 实验完成前不写性能结论。

## 当前状态

实现、单元测试和冻结配置已完成；Phase 55 闭环实验尚未运行，因此不能把该
候选称为有效改进。
