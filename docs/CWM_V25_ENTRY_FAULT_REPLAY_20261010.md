# V25：代表性真实入口故障注入回放，不是增强模型成功

本阶段在独立研究目录运行，原checkout、训练进程、GRU权重、252维编码、
DN-MPC和local CBF参数均不编辑。全部V23响应配置仍不合格，增强关闭。

## 已运行证据

两个独立Python进程，每个回放两个固定历史记录：level5 episode920000640，
level6 episode10038。每个进程包含plain、off、资格拒绝、加载失败、
推理失败、合成shadow六条路径，共24次代表性episode运行。
plain直接使用原planner入口；其它路径在原plan返回后调用可选适配器。

- 每次defender/target位置数组精确匹配原胶囊历史轨迹。
- DN-MPC完整计划和CBF后的实际命令精确匹配plain。
- 两独立进程上述数组逐字节一致；latency不要求相等。
- off/拒绝不读取预测输入、不加载或推理；拒绝读取的是V23真实失败状态。
- 加载失败恰好尝试一次，推理异常恰好尝试一次后停用；加载/推理中的
  Python/NumPy/Torch CPU随机数消耗被恢复，推理输入被复制。
- 合成shadow在每次计划返回后调用预测，始终无控制资格。
- 原胶囊所有成员哈希在运行后再次验证。

归档前/归档后复算通过，并由另外一个进程执行归档复核。
归档路径：`experiments/cwm_v25/artifacts/entry_fault_replay_20261010.zip`。
大小429176bytes；SHA256：
`9f5db7c2a2ec524594a4113a751f001041e6fc3994094b544b8f9b0569ab85d7`。
报告：`experiments/cwm_v25/reports/entry_fault_replay_20261010.json`。
归档含两进程完整记录、轨迹/命令数组、运行源码、manifest和验收器源码。

## 明确局限与下一步

故障注入采用形状合法的零history/relative/CV夹具，不是真实因果checkpoint
的公开输入。调用发生在原plan完成之后，不修改顺序局部选择器。这不能
证明实际学习模型接入顺序求解器后的完整回退、推理延迟或捕获率改善。
不能用合成shadow预测或两个历史场景冒充原全Level增强闭环验收。

接下来必须实现真实公开历史/延迟peer context和真实checkpoint的shadow
接入，再验证全部顺序调用与模型生成候选，保持原动作路径可随时回退。
只有在候选排序的独立新数据资格通过后，才开展保留holdout、原Level与
新场景的配对增强闭环捕获/安全/延迟验收。原local CBF仍是经验安全过滤，
本阶段没有新增形式化保证，也没有训练或启用新的合格围捕策略。

## 同期V23自包含检出复核

之前独立进程session66127已正常结束（exit0），命令：

```powershell
python experiments/cwm_v23/cost_training_release.py --verify experiments/cwm_v23/artifacts/deployed_cost_contrast_20261010.parts.json --restored-output results/cwm_v23/self_contained_full_verify_restored_20261010
```

工作目录是`results/cwm_v23/fresh_checkout_20261010`。进程输出
`passed_independent_fresh_deployed_cost_contrast_audit`，完整归档SHA256为
`2ec8d4165e2b56c99f1781c38e54122a481ef63dc3f8aacf1069d53ace11ca78`，
195963891bytes、3809members。分片读取自该独立本地检出；不是网络新克隆
或跨平台复现保证。这确认失败的V23训练产物可复算，并不推翻资格失败。

## 普通测试

V21–V25完整实现测试169项通过（20.56s）。其中8项V25真实归档语义测试：
完整正例复核，加上启用标志、支持集、控制权限、未执行故障路径、源记录、
目标轨迹、CBF后命令七种伪造拒绝。数组伪造同时修改两轮结果，因此不能
仅靠两轮一致性过关；仍必须匹配历史轨迹或未包裹入口命令。普通测试数量
不是捕获率提升、真实checkpoint集成或全场景资格的证据。
