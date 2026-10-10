# V28 两轮训练完成：检查点内容一致，因果决策门仍失败

两轮独立训练各完成三配置、三种子、固定最终80epoch的9份模型；没有
增加epoch、改seed、选最佳决策种子或调整任何科学门槛。两轮固定主模型
cv_rank_l2的research_eligible均为false。原GRU、DN-MPC、local CBF及
holdout保持原合同，增强默认关闭。

## 实际只读内容核验

`experiments/cwm_v28/reports/two_run_content_identity_pending_full_reload_20261011.json`
记录了实际24份检查点（每轮9份最终模型及3份运动阶段）的完整读取比较。
所有检查点字段及所有最终/阶段历史文件一致。张量比较覆盖dtype、shape
和原始数值位，标量区分类型及有符号零；字典只忽略插入顺序。

原始summary身份分别为：

- primary：`87254fad514a7ee45da3f325324660b443ea638c52a166668e1833a603cfd1f7`
- repeated：`f0f20a9a5e16fbaae2792fdbe929317964bc544f941f4c4eece382c63b576e44`

两轮12对检查点的文件SHA全部不同，但完整内容一致。训练source_hashes
从set迭代形成，实际检查显示两轮字典插入顺序不同；这会改变torch保存
文件的字节。没有重写检查点，也没有把第二轮摘要的SHA换成第一轮SHA。
每份最终检查点仍须匹配其所属轮次摘要内的完整文件SHA。

原审计`release_ranking_training.py`在完整重推理后用整份summary相等
作为前提，包含各轮checkpoint_sha256，因而会拒绝上述序列化差异。
原打包器也有相同问题。保留原入口及正在执行的原审计证据，不就地修改。
V30原模型全场景审计导入原打包器，因此也保持其字节和冻结源证据不变。

## 新的完整审计与打包入口

- `repeatability_identity.py`：首先核对各轮自己的最终文件SHA，随后
  比较完整检查点内容、阶段内容及全部历史字节。摘要仅排除
  `models[*].checkpoint_sha256`这一项，其他科学及来源字段必须严格相同。
- `release_ranking_training_v2.py`：保留原入口的全部18模型重推理、全部
  train/development调用预测字节、原引擎完整成本、候选支持、固定选择和
  门槛重算。跨轮核验调用上面的完整内容比较，并记录新审计源身份。
- `package_ranking_release_v2.py`：保留全部数据、64回合、两轮训练和
  来源快照；使用新的v2归档schema，核对新审计源和内容证据。归档后仍
  需实际子进程重新执行完整数据审计及18模型原成本重载，成功后才签发
  归档复现证书、生成无损分片。哈希或单元测试不能替代实际重放。

训练/资格代码、v1审计/打包代码、任何正在运行的V30源均未改动。
新的完整审计必须等原审计实际终态后使用新的独立输出路径。
原v1 V29主动加载器不接受v2归档；本次入口用于失败实验的完整离线复现。
未来主动接入需要单独匹配v2读入契约和完整接入核验，不通过既有失败门。

## 效果边界

两轮主模型均得到相同固定开发评价：相对自身冻结运动的平均原成本增益
0.021058，95%开发描述区间[-0.046928,0.116988]；相对独立运动对照
-0.064358，区间[-0.178499,0.059740]；相对普通响应L2的增益0.008593，
区间[-0.022151,0.047437]。均未满足固定严格正下界。对原GRU在同一
扩展候选库重评分的改善不能替代这些因果贡献对照。

只读内容核验已完成；全部模型原引擎重载、完整归档、独立归档实际重放
尚待运行。没有因果围捕捕获率、时间、安全或时延提升结论，没有新场景
主动控制或holdout使用。完成复现工程不等于整体研究目标完成。

新完整入口（使用绝对短路径，且输出必须不存在）：

```powershell
python experiments/cwm_v28/release_ranking_training_v2.py --data results/cwm_v28/fresh_sequential_ranking_20261010 --audit results/cwm_v28/independent_sequential_ranking_audit_20261010 --primary results/cwm_v28/primary_fixed_ranking_20261010 --repeated results/cwm_v28/repeated_fixed_ranking_20261010 --output D:/uav-capture/cwm28-reload-v2-20261011
```

只有上一步实际exit0且完整summary存在，才可执行：

```powershell
python experiments/cwm_v28/package_ranking_release_v2.py --data results/cwm_v28/fresh_sequential_ranking_20261010 --audit results/cwm_v28/independent_sequential_ranking_audit_20261010 --primary results/cwm_v28/primary_fixed_ranking_20261010 --repeated results/cwm_v28/repeated_fixed_ranking_20261010 --reload D:/uav-capture/cwm28-reload-v2-20261011 --archive D:/uav-capture/cwm28-v2-20261011.zip --replay-output D:/uav-capture/cwm28-ar2-20261011 --parts-output experiments/cwm_v28/artifacts/complete_fixed_ranking_v2_20261011.parts.json
```

新增40项测试覆盖真实torch序列化差异不改文件、最终/阶段权重与优化器、
两个随机状态、归一化、数据/源/协议、历史、指标、门、seed选择及支持
变化的拒绝；还覆盖v2完整归档清单和实际子进程顺序/失败不签发证书。
这些是实现测试；归档测试中的科学子进程是mock，不能称为科学重放。
