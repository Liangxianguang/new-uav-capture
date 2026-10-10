# V28完整归档重放完成；主模型仍未通过收益门槛

2026-10-11。原生归档驱动实际退出0，证书状态为
`complete_fixed_ranking_archive_independently_replayed`。
全部64条原始轨迹/计划/指令、3072次公开调用及其全部物理分支、
两次完整训练的18个最终模型预测/原始成本/固定门槛已从归档重放。
两次训练按逐文件身份加内容逐位相等合同验证；旧序列化失败记录保留。

这不是围捕增益确认。`primary_research_eligible=false`、
`enhanced_control_enabled=false`、`holdout_used=false`。
固定主模型`cv_rank_l2`/994101对自身运动组件的开发集原成本收益
0.02105845115469825，描述性95%区间
[-0.04692820272562559, 0.11698807169227458]；对CV、独立运动组件、
普通响应模型的必要收益门也未通过。相对同一扩展候选库中的原GRU
正收益不能单独证明因果响应收益，更不是原8Level闭环捕获率提升。

## 已发布证据

- 原生证书的严格类型保留副本：
  `experiments/cwm_v28/reports/completed_native_archive_replay_failed_primary_20261011.json`。
- 原生证书SHA256：
  `58f759987122f1753d6b8fe32b95de05992f8cc8edfd8bceb2fb65787e088668`。
- 完整ZIP：347970680字节，6998个证据成员（另有清单）；SHA256：
  `8580199311357b3fad2f829fcb390bfa2686c9c78cfd9333f766c3e746f9b7b1`。
- 8份无损分片及清单：
  `experiments/cwm_v28/artifacts/complete_fixed_ranking_v2_20261011.parts.json`；
  清单SHA256：`dbcc249b13ddd00e48f439c3f51469c93734cdc4ebb12a8083281d76cf1a0793`。

打包器在实际语义重放成功后才生成分片。另一次独立重组检查通过
全部分片的长度/散列以及整体ZIP长度/散列；这是传输检查，不是第二次
科学重放。报告副本通过V31的原生文件SHA及全部JSON类型/值检查。

## 复现入口与限制

必须使用对应Git检出及预注册提交历史、原Python依赖环境；不保证仅凭
ZIP或任意平台重现。`experiments/cwm_v21/artifact_transport.py`中的
`resolve_artifact(parts_manifest, NEW_transport_directory)`逐片验证并重组ZIP。
随后可执行（所有输出目录必须不存在）：

```powershell
python experiments/cwm_v28/package_ranking_release_v2.py --verify <joined.zip> --replay-output <NEW_replay_directory>
```

归档含完整数据、两次训练权重/优化器/RNG/历史、独立审计、原模型与
固定失败提案器依赖、验证源码。旧V29部署加载器不接受v2归档；本主模型
本来就未达标，不应以修改加载器来启用。原GRU252编码、DN-MPC、CBF
参数和关闭/拒绝增强回退合同均未改变。下一步使用独立预注册的新数据
和匹配前缀/全序列对照，不改变或覆盖本次失败结论。
