# V44：新版回退入口完整归档和独立原生重放入口，尚无归档验收结果

2026-10-11。此阶段实现V43全部证据的发布入口，不是因果模型训练或
围捕收益。V34采集和V43原生关闭/拒绝重放仍在运行；不修改这些任务
的实际源码闭包、不重启任务、不加载因果权重、不放宽旧门槛。

`experiments/cwm_v44/package_prefix_fallback.py`必须先确认V43父目录
及`native/summary.json`均已完成，再核固定V30完整ZIP/原生证书、
完整提取参考和V43保存证据。off与refusal各1280个原开发episode，
逐条检查轨迹/计划/CBF数组和全部非计时episode/step诊断，不能用
部分进度或终态标签替代完整内容检查。原L0诊断、L7两变体、碰撞和
越界结果均保留；原8Level宏平均保持原值。

归档包含V43两模式全部轨迹/计划/CBF、episode/step诊断、原生源码
快照、父目录预注册/绑定/报告，以及未改写的完整V30 ZIP和原生证书。
不重复打包运行时临时目录、部分进度或用户无关文件。V30 ZIP本身包含
完整旧参考和原胶囊依赖；对应Git历史与原Python环境仍是复现条件，
不保证只靠ZIP、任意平台或形式安全复现。

归档先流式核验并提取到全新短路径，再提取其中的完整原参考。
独立子进程运行实际`cwm_v43/replay_prefix_fallback.py`，通过实际
V42入口重新原生执行off/refusal各1280个episode。不能改用旧V30
入口、模拟审计器或存档比对冒充这次独立原生执行。所有原始证据、
提取证据、当前源码与保留发布commit在出具证书前复核。

计时和压缩文件散列保留，但原非确定性计时不要求逐位相等；每份
完整证据先独立核验，再比较科学数组和全部非计时字段。新独立重放
可以使用更新的已发布commit，但实际V42/V43源码必须一致。预运行
`reference_binding.json`的完成字段保持原来的false，不篡改为终态；
实际完成资格由完整原生证据和专用终态报告确定。

只有实际子进程exit0、两模式完整证据比较和全部输入复核后，才写
`complete_prefix_entry_original_fallback_archive_independently_replayed`
证书。失败日志保留；未完成不能生成该证书。可选GitHub无损分片只在
独立原生重放成功后生成并重组检查，传输校验不是另一次科学重放。
输出必须全新，禁止互相包含、覆盖输入或已有分片。

执行入口（原生V43完成且精确源码HEAD发布后；所有输出路径不存在）：

```powershell
python experiments/cwm_v44/package_prefix_fallback.py --archive D:/uav-capture/cwm30-full-20261011.zip --certificate D:/uav-capture/cwm30-ar-20261011/summary.json --reference D:/uav-capture/cwm30-ar-20261011/e --proof D:/uav-capture/cwm43-off-20261011 --artifact D:/uav-capture/cwm43-full-20261011.zip --replay-output D:/uav-capture/cwm43-ar-20261011 --parts-output experiments/cwm_v44/artifacts/prefix_full_original_fallback_20261011.parts.json
```

已有完整归档的复现入口（不会重新训练）：

```powershell
python experiments/cwm_v44/package_prefix_fallback.py --verify <ZIP_or_parts_manifest> --replay-output <NEW_short_absolute_directory>
```

回归测试只验证完整证据资格拒绝、源身份、归档完整性、输入不覆盖、
子进程编排及失败时不出具通过证书。合成成功不证明原生1280重放。
本阶段未执行真实V44归档、提取重放或分片发布，不新增因果收益结论。

科学主线仍为完整V34采集与独立审计→固定两轮9模型V36训练→全部
18模型冷重载→V38完整科学归档重放→固定前缀收益门→实际增强闭环
比较、故障回退和未使用holdout确认。当前增强关闭，原DN-MPC+CBF
继续保留；旧失败结论和安全合同均不改变。
