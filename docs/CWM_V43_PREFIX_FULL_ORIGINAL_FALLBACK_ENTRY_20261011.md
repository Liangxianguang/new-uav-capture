# V43新版入口的完整原模型关闭/拒绝回退重放

2026-10-11。新增独立`experiments/cwm_v43/replay_prefix_fallback.py`，
为V42入口验证全部原1280个开发场景、off/refusal各1280个episode。
不修改正在运行的V34采集、V36训练、V42入口、旧V30完整审计源码或
原GRU252/DN-MPC/CBF。该实验只证明原模型复现，不证明因果收益。

首先核V30完整发布ZIP及原生提取重放证书精确SHA，验证全部提取文件
字节/长度、原清单、完整库存和原1280场景/审计证据。相应Git历史和
Python环境仍必需；不声称只靠ZIP、跨平台或形式安全复现。
使用旧`audit_full_baseline.run`不变的完整审计循环，进程内临时替换
待测entry类为V42子类；它只允许off/refusal，不接受bundle、评分配置
或活动增强。结束/失败均还原旧模块entry绑定；磁盘旧源码保持不变。

原审计在每个episode检查精确原父planner/runtime类，并原生运行
完整原评估器，核全部轨迹/计划/CBF指令的dtype/shape/字节，以及
全部非计时episode/step诊断。计时数据保留，排除原非确定性计时字段
的数值等价要求；不声称时延资格。所有场景、原L0诊断、L7两变体、
失败/碰撞/越界保留，不抽样或用少量检查替代完整人群。

新输出必须不存在，且不能覆盖输入或源码。精确已发布HEAD和所有
实际使用入口源码在原生实验前核验。只有旧完整审计exit0、源闭包确实
含V42/V43、全部保存证据语义核验及原输入复核成功才写新终态报告。
旧原生audit报告放在`native/summary.json`，新入口专用证据与独立
preregistration保留在父目录，避免把旧入口证书冒充新版证书。

回归16项是证书/配置/入口替换还原/失败拒绝测试；模拟完整审计的
成功只检查编排，不能证明原生1280重放。实现回归与CLI导入结果见
`experiments/cwm_v43/reports/implementation_regression_before_native_fallback_20261011.json`。
原生执行完成前不得声称新版完整回退已验证；完整新证据归档和发布
也仍需后续执行。原DN-MPC+CBF继续可用，因果增强保持关闭。

原生运行示例（reference是已验证原完整归档的提取根目录，输出全新）：

```powershell
python experiments/cwm_v43/replay_prefix_fallback.py --archive D:/uav-capture/cwm30-full-20261011.zip --certificate D:/uav-capture/cwm30-ar-20261011/summary.json --reference D:/uav-capture/cwm30-ar-20261011/e --output D:/uav-capture/cwm43-off-20261011
```

这项实验不加载真实或假合格因果权重，不运行新场景或holdout，不做
optimizer更新、活动模型故障、时延/捕获增益确认或门槛放宽。
完整V34数据和独立审计、两轮9模型训练、18模型冷重载、V38完整
重放、固定前缀收益门以及实际增强闭环/故障/重放仍是后续任务。
