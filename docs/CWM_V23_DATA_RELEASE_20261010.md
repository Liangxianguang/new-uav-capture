# V23新数据完整复现包：原控制器证据，不是因果增强成绩

新32镜像组64回合（训练24组/开发8组）完成采集、独立公开重放、完整数据
审计、归档前/后审计，以及全新检出269b5b2中的独立全量归档复算。

1,536次调用：训练1,152/开发384；完整UNION成本支持训练818/开发270。
原控制器64/64安全捕获，零碰撞、越界、超时和目标无效；observer/plain/
public轨迹及原252特征历史、GRU预测、延迟peer上下文、候选、合法mask/
配对噪声标签、完整原成本与选择通过独立核对。这是该批新场景的原控制器
成绩，不是因果世界模型成绩，不是原8Level增强闭环覆盖结论。

## 完整证据包与验证

ZIP共39,212,316bytes，manifest包含3,427个成员，SHA256：
`b3794bdc99d424ecddbb9232fb2998db344e46d5f92e0a5334b8096631aeee64`。

发布使用单个无损part001及parts.json，见experiments/cwm_v23/artifacts/
fresh_cost_data_release_20261010.parts.json。原完整ZIP留在results，避免
重复Git存储。协议、所有数据/public源码快照、原轨迹、公开全帧、所有
候选未来/终止mask/训练监督、原成本/排序证据均保留；不存在只给summary
却无法重新计算的“成功包”。未包含任何增强模型训练成绩。

```powershell
python experiments/cwm_v23/cost_data_release.py --verify experiments/cwm_v23/artifacts/fresh_cost_data_release_20261010.parts.json --restored-output results/cwm_v23/data_verify_fresh_restored
```

独立验证在全新Git检出、显式core.autocrlf=true检出后的新进程进行，
重新执行完整audit_data，不只是核对checksum。原胶囊hash和核心源码hash
也与此前新检出一致；原dirty目录和原训练作业未改动/停止。详细记录见
experiments/cwm_v23/reports/data_release_manifest.json和data_independent_validation.json。
这不构成普适跨平台、GPU或其他依赖版本的完全一致保证。

## 模型工作仍在继续

两轮18模型/轮的完整训练还未结束，主实验仍为cv_cost_l2，所有最终80epoch/
三seed保留。新数据及原引擎审计不是最终资格；六项真实完成证据语义检查
也仍待完整训练后执行。V21/V22失败记录保持，不能用单seed指标覆盖它们。

完整模型/优化器/RNG/历史、预测、部署参考成本缓存/梯度、actual/union原
成本和门槛要在两轮结束后独立重推理/复算、归档、再独立验证并发布。
原入口严格关闭/拒绝/异常回退、顺序候选、未触碰holdout、原8Level及新
场景增强捕获/安全/延迟仍待完成。因果增强关闭，原DN-MPC+经验local CBF
继续保留，未新增形式化安全保证，整体目标未完成。
