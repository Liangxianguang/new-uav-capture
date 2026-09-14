# Phase 68：多场景拦截视频展示

本组视频使用已经冻结的 locked-test Official S4 checkpoint、distributed
delayed DN-MPC 和 local CBF，采用当前 v3 publication-style 3-D renderer。
视频是 replay-only visual evidence，不会修改正式 JSONL 统计，也不是新的
调参或测试集结果。

## 视频清单

| 场景 | 结果 | 捕获时间 | 最小安全距离 | 视频 |
| --- | --- | ---: | ---: | --- |
| nominal，episode 0 | safe capture | 1.3 s | 0.503 m | [MP4](media/phase68_interception_showcase/official_s4_episode_000.mp4) / [GIF](media/phase68_interception_showcase/official_s4_episode_000.gif) |
| partial dropout，episode 2 | safe capture | 1.3 s | 0.480 m | [MP4](media/phase68_interception_showcase/official_s4_episode_002.mp4) / [GIF](media/phase68_interception_showcase/official_s4_episode_002.gif) |
| delayed noisy，episode 4 | safe capture | 1.2 s | 0.486 m | [MP4](media/phase68_interception_showcase/official_s4_episode_004.mp4) / [GIF](media/phase68_interception_showcase/official_s4_episode_004.gif) |
| nominal，episode 6 | safe capture | 1.1 s | 0.569 m | [MP4](media/phase68_interception_showcase/official_s4_episode_006.mp4) / [GIF](media/phase68_interception_showcase/official_s4_episode_006.gif) |

每个视频同时保存一张最终帧 PNG。末帧的 `CAPTURE CONFIRMED` 只表示该条
replay 进入了 `0.80 m` capture radius，并不等于整套场景的捕获率。

## 复现

```powershell
python scripts\render_dn_mpc_episode.py `
  --evaluation-dir results\phase16_locked_closed_loop_official_s4_seed727201_local_refresh1 `
  --method distributed_delayed `
  --episode-indices 0 2 4 6 `
  --output-dir results\phase68_interception_showcase_official_s4_seed727201 `
  --device cpu --fps 12 --frame-stride 1 --tail-length 0 --freeze-seconds 1.5
```

回放脚本支持 S4 frozen scene 的扁平记录格式，并保留原始场景索引、seed、
mirror group 和观测条件。运行目录位于被 Git 忽略的 `results/` 下；已审阅的
媒体副本位于 `docs/media/phase68_interception_showcase/`。

## 解释边界

- locked-test 主结果中，Official S4 K=8 + distributed delayed + local CBF 的
  safe capture 为 `95.56% [92.96%, 97.78%]`，collision 为 `0%`；该结果支持
  在当前仿真契约下进行有效拦截。
- 这不是所有延迟、噪声、目标机动和几何 OOD 条件下都可靠。Phase 67 的
  joint-tail-transfer calibration 中 safe capture 仅约 `30%`，timeout 约
  `35%`，说明联合尾部压力下仍不能稳定拦截。
- 该环境是运动学仿真，capture-radius entry 不是物理接触、真实飞控、真实
  视觉或实机拦截验证；local CBF 是经验过滤器，不提供 R-CLBF-QP 安全证明。
