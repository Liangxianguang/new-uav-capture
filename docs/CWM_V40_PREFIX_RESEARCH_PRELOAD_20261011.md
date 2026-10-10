# V40新版模型独立预加载入口；真实权重/归档尚未产生

2026-10-11。新增V40，不改运行中采集源码、旧训练/重载/归档实现或
原GRU252/DN-MPC/CBF。默认off/refused直接返回未授权空bundle，不解析
路径、读协议/模型/归档或创建cache。仅显式research_prepare可预加载。

新加载器验证V38完整归档与原生提取科学重放证书的身份、全部证据/
源码/依赖/Git预注册和严格收益门。固定前缀主模型失败时，在任何
神经checkpoint/提案器重建之前拒绝，不以更好的对照或种子替代。
通过者按原final80 BASE ADE中位规则独立核选定种子，使用未改V37
恢复模型/Adam/采样器/Torch RNG但不更新，核数据/审计/归一化/缓存/
原冻结公共运动阶段；调用者RNG保持。全部加载在规划计时前完成。

推理只接收原六个公开数组，原TRAIN-only归一化和神经float32/
目标路径float64精度；至多24个BASE候选、原速度/形状/有限性与
prefix-anchor浮点身份必须满足。已有OptionalResponseAdapter核
公共运动独立/锚响应零/预测分解并拒绝坏预测；目标路径不是UAV动作。
长加载结束后再次核归档/证书/源码/提取证据，最后才赋予隔离研究
预加载资格；deploy始终false，不执行环境、动作、训练或holdout。

bundle为新类型，旧V29主动selector不会接受它；旧L2未加载且
full_v39_runner_ready=false。完整8方法闭环runner、分别绑定旧L2
的基准加载、原生故障回退、结果独立重放/归档及一次密封holdout仍
待实现与验证。本阶段不会把预加载资格当作完整研究/部署资格。

V40最终31项合成测试exit0，5.06秒；V28–V40/API联合回归exit0，
701项通过，99.02秒。初次单测存在cache拒绝顺序错误，已修复并
保留失败记录。测试使用玩具2call80epochcheckpoint和模拟归档/
证书权威，真实玩具Adam/RNG恢复不等于原生训练/资格。44个运行中
采集源码快照再核无差异。报告见
`experiments/cwm_v40/reports/implementation_regression_not_native_preload_20261011.json`。

完整采集与原生独立审计仍未结束，没有新一轮科学权重/V38归档，
未真实加载因果增强。旧V28仍失败，增强关闭，原模型完整回退归档
保持可用，目标继续进行。
