# P0：原 DN-MPC + local CBF 复现保护验收

日期：2026-10-09。状态：**PASS，256/256**。增强控制仍关闭。

## 保护对象与结果

保护的是2026-10-05能力审计所用的Phase2 GRU seed745101 +
distributed-delayed DN-MPC + local CBF。不是缺失权重的Phase16复测，
也不是旧目录中的strict Joint CBF。

- 原provenance的9项SHA256全部吻合：权重、配置、场景及核心源码。
- 独立封装原源码、原GRU权重、验证场景、运行版本、原指标和256条参考轨迹。
- Level5的128场、Level6的128场全部重放。
- 每场捕获/碰撞/越界/超时/目标合同指标及捕获时间、最小净空吻合。
- 无人机及目标完整轨迹数值逐元素一致；发布打包器另检验dtype、shape和原始字节。
- 最大轨迹误差0。原始6个Level6安全失败也如实复现，不把“复现通过”写成256场全成功。
- 包内源码和参考材料在重放后再次逐项校验。

原性能仍为Level5 128/128=100%，Level6 122/128=95.3125%。
捕获0.8m、物理边界margin0.3m、CBF margin0.35m、dt0.1s、250步均未改变。
独立目录不修改原工作目录，不触碰现有RL训练进程。

复现ZIP SHA256：
`c83a27a4360ad055b69dbced4c28370cb31a32ae89ea9fbab13523fa6218a329`

## 独立复现

完整运行环境信息见`experiments/cwm_v1/baseline/capsule.json`的runtime字段；
使用对应Python/PyTorch/Numpy/Scipy等版本。下面命令不依赖原D盘工作目录：

```powershell
python experiments/cwm_v1/replay_baseline.py --capsule experiments/cwm_v1/baseline/capsule.zip --output results/cwm_v1/replay_new --all
```

脚本校验包、恢复独立源码，再重放全部场景。失败或中断后可使用相同参数加
`--resume`：先重新检查已保存参考轨迹，再补完未完成项，不删除既有结果。
未捕获场景的capture_time为null；比较器已显式处理这种合法结果。

机器可读审计：`experiments/cwm_v1/reports/baseline_replay_all.json`及
`baseline_release_manifest.json`。冻结包是唯一基线来源，新训练不得写入该包。

## 保证边界

已验证的是本机记录CPU运行环境中的完整轨迹与物理结果。墙钟耗时不能逐位
复现；其他平台/依赖版本必须重新验收，不能宣称任意机器上绝对一致。
local CBF仍是经验过滤器，不据此宣称形式化安全证书或真实飞行安全。
