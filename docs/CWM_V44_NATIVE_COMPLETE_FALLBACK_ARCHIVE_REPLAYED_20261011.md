# 原模型完整回退归档：独立原生重放完成

2026-10-11观察到既有V44会话51487实际退出0。最终状态为 `complete_prefix_entry_original_fallback_archive_independently_replayed`。这项验收证明原模型关闭/拒绝可复现，不证明因果增强有效。

完整归档含5306个证据成员（不计归档自描述manifest），151655152 bytes，SHA256 `15a2554c51c0eea442e6ef44d1aeb6e32f20de54fab7b222ff8d7b5ebecaa91d`。四个无损传输分片及清单位于 `experiments/cwm_v44/artifacts/`，不是精选轨迹或仅权重包。归档包含完整V43旧证明、其原生关闭/拒绝记录及源码、完整原模型V30依赖ZIP和证书、验证源码与release contract。

既有任务实际独立解包归档及原模型依赖后，用原Python环境重新运行实际V43/V42入口，off/refusal各1280个原开发episode。全部数值轨迹、规划序列、CBF命令以及非计时episode/step诊断与原模型和归档证明相等；时间测量保留，不声明确定性或时延合格。完整输入、科学源码、归档成员及解包成员在原生重放前后核验，分片拼回ZIP也核对相同长度/散列。

退出成功后另做只读发布核验：检查全部归档成员长度和SHA、四分片、已存在的无损重组ZIP、旧/新wrapper报告及原V30归档/证书绑定。实际退出0，未额外重新运行物理episode。原生证书SHA为 `d850e5b42415ebd353ccde112e54571bb546caec0762c8d6ecf7165430fa7ac4`，传输报告SHA为 `c0a8b7b7cd0cf91fddbbac58bc2dd1f0b2b3f4e913c4a0d665d86d48418a2276`；发布回执在 `experiments/cwm_v44/reports/native_full_fallback_archive_replayed_20261011.json`，不把回执当另一次原生实验。

提交前还逐字节核对Git暂存的四分片及清单，发现默认换行规范化会改变清单散列；因此为V44 artifacts增加 `-text` 精确字节规则后重新暂存和核对。不修改原生清单内容或散列来迁就Git转换，科学源码及运行任务保持原状。

L1–L8及Level内变体/组等权宏平均：安全捕获99.12109375%，碰撞0.87890625%，越界0.29296875%，超时/目标无效0，restricted safe capture time 2.954150390625s。L0仅诊断，不混入主宏平均；碰撞和越界不一定互斥。此结果不是100%安全、形式CBF保证、跨平台复现或新场景泛化。

复现入口（要求匹配Git历史/科学源码及原Python依赖，输出必须全新短绝对路径；尚未执行以下第三次重放）：

```powershell
& 'D:\miniconda3\envs\uav-encirclement-gpu\python.exe' experiments/cwm_v44/package_prefix_fallback.py --verify experiments/cwm_v44/artifacts/prefix_full_original_fallback_20261011.parts.json --replay-output <NEW_short_absolute_directory>
```

ZIP并非无需环境的独立可执行程序。核验失败应保留输出，不覆盖重跑，也不改原安全合同。

因果增强保持默认关闭。V34采集/独立审计、两批固定V36训练、全部18模型V37重载、V38科学归档，以及合格模型的全量配对闭环、真实主动故障、独立活动重放和一次封存holdout仍必须依次完成。当前报告不授权这些步骤的资格跳过或部署。原GRU权重、252编码、DN-MPC与CBF未改动；未停止其他训练。
