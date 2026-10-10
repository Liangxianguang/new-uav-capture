# 原模型完整回退归档已独立重放，增强仍关闭

2026-10-11。原生打包/独立提取重放驱动实际退出0；证书状态为
`complete_original_population_archive_independently_replayed`。
全部原1280开发episode，以及关闭增强/拒绝增强各1280条原生轨迹、
计划、CBF指令、非计时episode/step诊断均一致。计时保留但不要求
逐位一致，本成果不构成时延资格、跨平台或形式安全保证。

原L1–L8按家族/变体/Level等权宏平均安全捕获率99.12109375%，
碰撞率0.87890625%，越界率0.29296875%，无超时/目标无效。
L0只作诊断并排除在8Level宏平均外；该百分比不是1280条简单合并率，
更不是因果模型、新几何或未见场景效果。碰撞/越界仍存在，不能称
原系统100%安全，也没有放宽旧安全门以启用新模型。

发布`experiments/cwm_v30/artifacts/original_full_population_20261011.parts.json`
和3个无损分片。整体ZIP94931869字节、7764证据成员（另有清单），SHA256
`a6995520623e2fe311f06777ca97bae0d7a763eec6fd0cd1a821d863a28173e3`。
分片清单SHA256：`02ad9bab2e70889b309de0977a65b19831081219b58b681c606524aefa5b0769`。
原生证书SHA256：`8179de04eae22b3a7d8e0c5ad290e2f4c8a15dfba7205d56281b7cd340728983`。
严格类型副本见`experiments/cwm_v30/reports/completed_native_original_archive_replayed_20261011.json`。
分片在完整科学重放成功后生成，整包重组/散列检查另行通过；传输
重组不是第二次科学重放。旧pending报告保留为当时状态，不覆盖历史。

归档含原模型胶囊/运行环境记录/配置/场景和完整基线/两模式审计证据。
必须使用对应Git检出及预注册历史、原Python依赖环境；不保证只有
ZIP或任意平台即可重现。用原V21`artifact_transport.resolve_artifact`
逐片校验并重组，再运行（输出目录必须全新）：

```powershell
python experiments/cwm_v30/package_full_baseline.py --verify <joined.zip> --replay-output <NEW_replay_directory>
```

原GRU745101/252、distributed-delayed DN-MPC、CBF margin0.35、speed5、
horizon8/dt0.1/max250及目标规则均未改变。此证据只覆盖关闭/拒绝增强；
活跃因果模型故障回退、增强收益、原holdout和时延仍待后续验证。
