# V30：真实原周期计时检查通过，完整独立重放运行中

GitHub连接恢复后，研究工作树与远端研究分支精确核对到
`d184d0576fbe4dbcaaff7b683c3bbbdd0d63f760`，此前本地完整原基线结果、
V28第一完整失败summary和固定配对/流式核验代码都已成功发布。
没有通过修改网络/Git配置或改变实验协议跳过预注册。

真实计时检查会话50479已terminal exit0，输出
`D:\uav-capture\cwm30-cycle-smoke-20261011`。固定原归档920000640和
10038两条，plain/off/refusal各完整执行；每次轨迹数值与原归档一致，
两个观察模式的计划及CBF命令与本轮plain逐字节一致，所有原生非计时
诊断一致，确实读取原`run_episode`的`control_started`并保留完整周期
计时。没有可选模型/私有标签/主动响应或新场景输入。

原始完整summary包含逐步原生诊断和计时观察，按原字节复制到
`experiments/cwm_v30/reports/actual_original_cycle_measurement_smoke_20261011.json`，
SHA256 `9828a25acbd748acb1df2059e64e30f81198a530b804ed4b3ce5ff076853d1ca`。
它是实际两条原入口的计时仪器检查，仍明确full_population_exercised、
active_model_faults_exercised、latency_qualified及artifact_packaged_and_replayed
全部false。不能用两条检查证明完整场景、主动已训练模型故障/延迟或
因果围捕收益。原始轨迹/命令及源快照仍在上述完整本地输出目录。

在既有干净本地Git检出
`results/cwm_v27/fresh_checkout_20261010`核对无修改及同一精确源提交后，
已实际启动完整独立重放会话65259、PID32992，输出新短路径
`D:\uav-capture\cwm30-audit-20261011`。前一取消的只是未运行子任务的
排队shell，当前是新发布流式代码的首次完整原生重放，不是重复训练或
因为观察超时重启。两种模式均固定全部1280条/原数值轨迹/计划/CBF及
所有非计时原生诊断。最近观察off已至少496/1280，refusal尚未执行，
不能将部分结果当完整核验。此检出不是全新网络克隆或跨平台验证。

`package_full_baseline.py`实现后续全部基线+两模式审计归档：只接受完整
报告且复核所有保存场景/步骤/轨迹/计划/CBF、源字节和原胶囊；归档
包括原胶囊/源快照，不夹带restored树、用户无关文件或缓存。流式校验
ZIP全部CRC/长度/SHA/路径和精确成员库存，归档后从短新目录提取完整
证据，再独立子进程实际重跑off/refusal各1280条；全部比较及结束源
哈希通过后才产生证书。45MiB无损分片仅在真实重放成功后生成/回拼。
匹配Git历史和原Python环境必需，不执行任意提取出的Python，不宣称
bare-ZIP/跨平台、主动故障、holdout/捕获收益或形式化CBF证明。
19项归档完整性/拒绝测试通过（3.43秒）；整合归档工具后的V21–V30
完整测试436项通过（127.06秒）。归档实际执行仍待完整审计。

V28同一会话27155/PID18724仍进行第二完整训练：第一种子运动和普通
响应最终80epoch已完成，排序响应也已完成80epoch并刚保存第一种子结果；
第二种子尚未开始输出。第一轮失败不变，不能
选择另一个seed/配置，未启用主动研究、新几何或holdout。原五项训练
不改，原DN-MPC+local CBF仍为默认模型，整体目标active。
