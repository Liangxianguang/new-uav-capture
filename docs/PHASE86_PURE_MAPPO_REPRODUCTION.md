# Phase86 MAPPO / recurrent IPPO 复现实验手册

本文记录当前 Phase86 四防御者协同捕获任务中的 MAPPO 与 recurrent IPPO 实验：模型结构、环境与数据契约、训练参数、BC 初始化、CBF 使用位置、断点续训、TensorBoard、独立验证、已有结果与限制。所有结果均为开发阶段证据，不是锁定测试或正式多种子结论。

## 1. 先区分正在比较的算法

| 实验 | 网络 / critic | rollout 中的 CBF | seed | 训练目标 | 本机输出目录 |
| --- | --- | --- | ---: | ---: | --- |
| 纯 MAPPO | 参数共享的前馈 actor；集中式 public-state critic | 关闭 | 101 | 5000 updates | `models/phase86_pure_mappo_route_nocbf_long_seed101_20260921_v2` |
| 修正续跑的 recurrent IPPO | 参数共享的 GRU actor；每个防御者的 local-observation critic | 开启，1 次投影；动作尺度从 0.6 线性升至 1.0 | 207 | 5000 updates | `models/phase86_ippo_recurrent_cbftrain_curriculum_seed207_20260924_v3_recovery_from_u000001` |

“纯 recurrent IPPO”是旧实验标签。**当前 seed 207 续跑不是纯 IPPO rollout**：训练时启用了 `--use-cbf-train`，因此应准确称为“CBF-assisted recurrent IPPO curriculum”。CBF 在两种评估模式中的含义则不同：`raw` 不加 CBF，`eval-CBF` 在相同 checkpoint 的确定性策略动作后加外部 CBF。不要把两者混为一谈，也不要把 MAPPO 的 eval-CBF 结果称为纯 MAPPO 安全性能。

### 本机状态快照（2026-09-24）

- MAPPO 的 `progress.json` 记录至 update 2798/5000，但完整 checkpoint 仅保存至 update 2500；当时没有 MAPPO 训练进程。因此 2798 不能当作可恢复 checkpoint，也没有 update 2798 的独立评估结果。
- recurrent IPPO 旧目录 `..._v3` 的 `progress.json` 记录 update 206，但 `checkpoint_manifest.json` 只有完整 update 1 checkpoint。续跑目录 `..._recovery_from_u000001` 因而从 update 1 恢复，而非从 206 恢复；BC 权重已在 update 1 checkpoint 内，不应在 resume 时再次 BC。
- 本文审计快照时，续跑目录 `progress.json` 为 update 31/5000，但完整权重仍为 update 1；该恢复目录尚未生成自己的 `checkpoint_manifest.json` / `checkpoints/` 归档，第一次新的完整保存应在 update 100。训练中 `progress.json` 每 update 更新、权重只按 checkpoint 间隔更新，因此二者短暂不一致是预期行为，不能把 progress 计数当作可恢复权重进度。该状态属于本机运行产物，不随仓库发布；复现者应以 checkpoint payload 的 `updates_completed`、匹配的 checkpoint manifest（生成后）和 SHA-256 为准。

## 2. 环境、观测与任务契约

主配置为 [`configs/phase86_pure_mappo_repro.yaml`](../configs/phase86_pure_mappo_repro.yaml)，环境基础配置为 [`configs/phase85_target_contract_repaired_environment.yaml`](../configs/phase85_target_contract_repaired_environment.yaml)。训练与验证场景按记录逐条重建环境；不要改用随机重置场景来替代冻结场景集。

| 项目 | Phase86 设置 |
| --- | --- |
| 仿真 | 运动学后端；执行延迟、命令噪声、drag / actuator lag 关闭；`dt=0.1 s`，每集最多 250 步（25 秒） |
| 场地 | x/y 半宽 10 m，高 10 m，最低高度 0.5 m；3 个 mixed-profile 障碍物 |
| 智能体 | 4 个防御者 + 1 个机动目标；防御者速度/加速度上限 5.0 m/s、6.0 m/s²，半径 0.25 m；目标 3.6 m/s、5.0 m/s² |
| 目标 | `adaptive_maneuvering`；目标观测 delay 1、noise 0.08、dropout 0.08；主要观测条件还包含检测 dropout 0.12、noise 0.025、队友消息 delay 1、dropout 0.03 |
| 策略输入 | 每个防御者仅使用延迟/带噪的 public belief、观测几何、role-slot 与 route-intent 特征；本配置 local observation 为 77 维，route-intent 子特征为 7 维 |
| 禁止输入 | 目标未来真值、专家动作标签、专家 route 标签不得输入 actor |
| safety / shaping | safety margin 1.0 m；防御者边界 margin 1.25 m；队友间 clearance target 1.25 m；边界 proximity / progress 权重 1.50 / 0.80；队友 clearance reward 权重 4.0 |
| 成功定义 | capture radius 为 0.80 m；在时限内发生 capture event，且该集没有目标契约无效、防御者碰撞或防御者越界，才计为 safe capture |

### 奖励函数

奖励由 [`src/encirclement3d/pursuit_env.py`](../src/encirclement3d/pursuit_env.py) 的 `reward_components` 计算，不是另一个外部 reward model。当前配置实际使用：progress `+3.0 × clipped normalized distance progress`；距离惩罚 `-0.12 × nearest-target distance`；coverage `+0.15 × coverage score`；队友 clearance shaping 权重 `4.0`、目标 clearance `1.25 m`；每步 time penalty `0`；安全捕获 bonus `+25`；防御者物理碰撞或边界 safety failure penalty `-15`；timeout penalty `0`；边界 proximity / progress 权重 `1.50 / 0.80`，按边界 margin `1.25 m` 归一化裁剪。目标自身 target-invalid 终止单独记录，不施加防御者 safety penalty。任何 reward/config/source 改动都构成新实验，不能续写同一 run 名称。

这里是运动学仿真，不代表真实飞行器动力学、机载感知、SITL/HITL 或飞行安全证明。目标边界/障碍违规单独记录为 target-invalid contract fault，不能记作防御者碰撞，也不能把该集算成有效安全捕获。

### 场景与专家数据

- Calibration / training 场景：`results/phase86_nominal_repaired_development_calibration/scenes.jsonl`，300 条、150 个镜像组；manifest seed block `861201`；scenes SHA-256 `a631c960873be35dafd025eaf575a3f5798514e70775468212cdf6d885c58b71`，manifest SHA-256 `af22742aaadb5026a7515201eef8fff7ba9a2c9180f00a94e1207e305c64b6d0`。
- Development validation 场景：`results/phase86_nominal_repaired_development_validation/scenes.jsonl`，300 条、150 个镜像组；seed block `961201`；scenes SHA-256 `b903cf5fb6819a1ad970ce035b9cc9b25236e316c30fcf2b4e8e19401758869b`，manifest SHA-256 `86cf8e451f06f2d96bb961c66cabb9d2da24ae8302b35250ce7d75292deb689c`。
- 两个 split 没有 layout-seed 或 episode-seed 重叠；每个镜像组有两条记录。仍属 development calibration/validation，不是 holdout locked test。
- BC 数据：`models/phase86_fix_route_teacher_formal_contract_20260921/expert_dataset.npz`，SHA-256 `84151c7421107fa0df74a35fdd18cc4ab4f96ab18770753002ee08521d0d30a7`；从 episode seeds 861201–861220 生成的 20 条教师 rollout 中选取 19 条、1612 frames，sequence length 206。教师为基于 public belief 的 route-intent controller，数据经 local CBF 过滤并通过 safe-capture / target-valid 筛选。数据仅用于 actor warm-start，不是正式测试数据。

### 精确的数据准备与完整性检查

训练时应直接使用仓库中冻结并带 hash 的两个 `scenes.jsonl` 与 expert `.npz`；这可避免不同机器重新采样造成的场景漂移。先验证工作目录中的数据确实与本次实验输入相同：

```powershell
(Get-FileHash results/phase86_nominal_repaired_development_calibration/scenes.jsonl -Algorithm SHA256).Hash.ToLower()
(Get-FileHash results/phase86_nominal_repaired_development_validation/scenes.jsonl -Algorithm SHA256).Hash.ToLower()
(Get-FileHash models/phase86_fix_route_teacher_formal_contract_20260921/expert_dataset.npz -Algorithm SHA256).Hash.ToLower()
```

预期值依次为上表 calibration hash、validation hash，以及 expert hash `84151c7421107fa0df74a35fdd18cc4ab4f96ab18770753002ee08521d0d30a7`。协议审计脚本会校验配置/环境/manifest/scenes 的 hash、每 split 300 条记录、每个 split 150 个双成员 mirror groups、target-invalid 场景字段为 0，以及两个 split 的 layout / episode seeds 无重叠：

```powershell
python scripts/freeze_phase86_protocol.py `
  --config configs/current_rl_comparison_phase86_v1.yaml `
  --output "results/phase86_reproduction_audit/protocol_lock_$(Get-Date -Format yyyyMMdd_HHmmss).json"
```

`protocol_lock.json` 会记录源 commit、各输入文件 hash、镜像组和 split overlap；在新的 clean clone 上 `formal_promotion_allowed` 才可能为 true。此检查只验证输入合同，不代表策略通过测试。

BC 训练样本可从冻结 calibration scenes 用 `scripts/collect_frozen_policy_dataset.py` 重建到新的、空的输出目录。以下参数对应存档的 expert metadata：public-belief route-intent teacher、local CBF、20 个 scenes、最多 250 步、仅保留 safe-capture 且 target-valid 的 episode：

```powershell
python scripts/collect_frozen_policy_dataset.py `
  --config configs/phase86_pure_mappo_repro.yaml `
  --scenes results/phase86_nominal_repaired_development_calibration/scenes.jsonl `
  --output models/phase86_fix_route_teacher_formal_contract_20260921_rebuilt `
  --episodes 20 --max-steps 250 --use-cbf `
  --controller route --interceptor-id 0 --accepted-only
```

应得到 19 条 accepted episodes、1612 valid frames、77 维 local observations（其中 7 维 route-intent features）、每步 4×3 actions，selection policy 为 `safe_capture_and_target_valid`。读取新生成目录的 `metadata.json`，检查 `scenes_sha256`、`config_sha256`、selected episodes / frames / observation shape，并把新文件 SHA 与 run manifest 一起归档。ZIP/NPZ 容器字节可能受 Python/NumPy 版本影响；固定输入做精确训练复现时，优先使用仓库里已冻结且 hash 已知的 `expert_dataset.npz`，而不是用重新压缩后的文件替代。

NPZ 的键分别为 `local_observations`、`actions`、`valid_masks`；存档数据张量形状为 `[19, 206, 4, 77]`、`[19, 206, 4, 3]`、`[19, 206]`。每帧包含 4 个防御者各自的 77 维局部观测，以及每个防御者 3 维连续速度命令（单位 m/s；教师/BC 动作尺度为 5 m/s）。不足 206 帧的 episode 用 0 padding，`valid_masks` 只把实际有效时间步标为 1；BC loss 应使用 mask，不得把 padding 当训练样本。

存档 expert 的历史 `metadata.json` 指向 `phase86_fix_route_mappo_development_v1.yaml`（其中含原工作站绝对路径）；上面的 portable config 含相同的有效 environment/task overrides，但 provenance YAML 的字节 hash 不同。因此，若要求与存档 run 完全相同的 BC 输入文件，直接使用已提交 `.npz`；若重建数据，则记录新的 metadata/config hash，并对比场景、有效配置、19/1612/77 等结构检查，不能声称两个 metadata hash 相同。

当前仓库保存了确切场景 records、protocol/config 与最终 expert archive，因此按相同训练输入可复现训练。原始 calibration/validation scene pool 的一次性生成入口没有作为受版本控制的专用 Phase86 generator 保存；manifest 中的 seed block 能审计来源，但**仅靠 seed 不能保证跨版本逐字节重建场景文件**。如需生成新的场景池，应新建版本和 hash，不得覆盖这里的冻结 records 或冒称同一实验。

机器可读的 comparison contract 在 [`configs/current_rl_comparison_phase86_v1.yaml`](../configs/current_rl_comparison_phase86_v1.yaml)。该 protocol 计划 seeds 101/202/303 和 300 集验证；本文所列 MAPPO 与 IPPO 是 seed 101、207 的单种子开发运行，不能据此宣称多种子优势。配置中标明 `not_a_locked_test: true`、`locked_test_tuning_forbidden: true`。

## 3. 策略与 PPO 更新细节

- 四个同构防御者共享 actor 参数；actor 各自产生 3 维速度命令。连续动作来自对角 Gaussian，经过 `tanh` 并乘以当前 `action_scale`。
- 前馈网络使用两层 128 单元 Tanh MLP。Recurrent actor 另含 128 单元 GRUCell，并为四个防御者分别维护 hidden state；每个完整 episode sequence 在 recurrent PPO minibatch 中保持连续，不跨 episode 延续 hidden state。
- MAPPO critic 读取 centralized public simulator state（只用于训练）；IPPO critic 仅读取每个防御者的 local observation。执行时 actor 不读取 centralized critic 状态。
- 使用 PPO clipped objective、GAE；共享参数用 Adam。共同参数：8 episodes/update、4 PPO epochs、minibatch 512、gamma 0.99、GAE lambda 0.95、clip range 0.2、hidden dim 128、最多 250 steps/episode、PyTorch CPU threads 1。
- Recurrent IPPO 每次 update 的 recurrent batch 维是 8 条完整 episode，minibatch size 512 大于该 batch 数，因此每个 PPO epoch 是一个包含全部 8 条 episode sequence 的 minibatch（每 update 4 次 optimizer passes）；MAPPO 前馈分支按扁平 transition minibatch 切分。
- MAPPO seed 101：learning rate `3e-4`、entropy coefficient `0.01`、action scale `1.0`；BC 10 epochs、batch 1024、prefix weighting 关闭；训练 rollout 不使用 CBF。
- recurrent IPPO seed 207：learning rate `5e-5`、entropy coefficient `0.002`；动作尺度 `0.6 -> 1.0`，在前 1500 updates 线性 ramp；训练 rollout 开启 local CBF，投影 1 次。BC 60 epochs、action scale 1.0、prefix weight 3、prefix steps 64；BC 权重已进入 update 1 checkpoint。
- CBF-training 是近似 likelihood：策略采样动作先经过 CBF 投影，PPO 保存/重算的是基础策略在“已执行投影动作”处的密度（`projected_action_base_policy_density`），不是投影分布的精确概率密度。这是该训练方法的实现细节和局限，不能把它描述成普通无 CBF 的 IPPO。
- RNG：训练入口调用 `torch.manual_seed(seed)`，CUDA 运行时调用 `torch.cuda.manual_seed_all(seed)`；有冻结 training scenes 时，每 update 用 `numpy.default_rng(seed + update_index)` **有放回**抽取 8 个 scene records，环境初始 RNG 使用每条 record 固定的 `episode_seed`；没有 scene records 时 episode seeds 为 `seed + update_index*8 + slot_index`。Recurrent BC minibatch generator 使用固定 seed 17；resume checkpoint 恢复 Torch CPU/CUDA RNG。代码没有启用 `torch.use_deterministic_algorithms`，因此固定 seed 保证协议输入/随机流可追踪，但不同 GPU、驱动、CUDA/PyTorch 版本之间不承诺 bitwise identical weights。

## 4. RTX 5050 工作站环境

仓库的 `environment.yml` 钉住 Python 3.11、PyTorch `2.7.1+cu126`、NumPy 2.1.3、TensorBoard 2.19.0、PyYAML 6.0.2、SciPy 1.15.3 等。安装 NVIDIA 驱动后，从仓库根目录：

```powershell
conda env create -f environment.yml
conda activate uav-encirclement-gpu
python -c "import torch; print('torch=',torch.__version__,'compiled_cuda=',torch.version.cuda,'available=',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

先做硬件/序列化 smoke，不要把 smoke 输出当作策略成绩；输出名使用新目录，避免覆盖其他 run：

```powershell
python scripts/smoke_cuda_phase86.py `
  --output results/phase86_rtx5050/cuda_smoke.json `
  --model-output models/_phase86_cuda_smoke_rtx5050 `
  --updates 2
```

RTX 5050 的驱动、GPU compute capability 与 PyTorch wheel 必须相互兼容。若 CUDA smoke 报 no kernel image / CUDA unavailable，先按 NVIDIA 与 PyTorch 官方兼容矩阵换用支持该显卡的 CUDA wheel，再重跑 smoke；训练 launcher 指定 `-Device cuda` 时不会静默退回 CPU。不要因为 GPU 利用率低就认定 CUDA 未生效：当前环境 rollout 是 Python 单进程顺序执行，环境步数和逐步 CBF 投影可能比小型网络更新更耗时；尚无 profiler 数据能把耗时精确分摊到某一函数。

## 5. 从头训练

每个 run 使用唯一、空的输出目录；不要覆盖已有 artifacts。下例将 checkpoint 间隔设为每 100 updates，适合断连后恢复。MAPPO 原有 seed 101 运行的 checkpoint 间隔为 500；将它改为 100 只改变落盘频率，不应改变策略超参数。

### A. 纯 MAPPO seed 101

```powershell
.\scripts\run_phase86_pure_mappo.ps1 `
  -Algorithm mappo -Seed 101 -Device cuda -Python python `
  -Updates 5000 -EpisodesPerUpdate 8 -PpoEpochs 4 -MinibatchSize 512 `
  -HiddenDim 128 -LearningRate 3e-4 -Gamma 0.99 -GaeLambda 0.95 `
  -ClipRange 0.2 -EntropyCoef 0.01 -MaxSteps 250 -TorchThreads 1 `
  -TrainingEpisodes 300 -BcEpochs 10 -BcBatchSize 1024 `
  -CheckpointIntervalUpdates 100 `
  -Config configs/phase86_pure_mappo_repro.yaml `
  -Output models/phase86_pure_mappo_seed101_rtx5050
```

该 launcher 默认读取 development calibration scenes 与 versioned expert dataset；不加 `-UseCbfTrain`。TensorBoard 事件写入 run 下的 `tensorboard/`。

### B. CBF-assisted recurrent IPPO seed 207

由于当前 PowerShell launcher 未暴露 `--recurrent`，按原实验用 trainer CLI 直接启动：

```powershell
python scripts/train_mappo_ippo_baseline.py `
  --config configs/phase86_pure_mappo_repro.yaml `
  --output models/phase86_ippo_recurrent_cbftrain_curriculum_seed207_rtx5050 `
  --algorithm ippo --recurrent --seed 207 --device cuda --updates 5000 `
  --episodes-per-update 8 --ppo-epochs 4 --minibatch-size 512 --hidden-dim 128 `
  --learning-rate 5e-5 --gamma 0.99 --gae-lambda 0.95 --clip-range 0.2 `
  --entropy-coef 0.002 --max-steps 250 --torch-threads 1 `
  --checkpoint-interval-updates 100 --log-interval 1 `
  --action-scale-factor 0.6 --action-scale-end-factor 1.0 `
  --action-scale-ramp-updates 1500 `
  --use-cbf-train --cbf-train-iterations 1 `
  --training-scenes results/phase86_nominal_repaired_development_calibration/scenes.jsonl `
  --training-episodes 300 `
  --expert-dataset models/phase86_fix_route_teacher_formal_contract_20260921/expert_dataset.npz `
  --bc-epochs 60 --bc-batch-size 8 --bc-action-scale-factor 1.0 `
  --bc-prefix-weight 3 --bc-prefix-steps 64 `
  --tensorboard
```

BC metadata currently records dataset hash, epochs, action scale and prefix weights, but does not serialize `bc_batch_size`; preserve this command with the run notes. A fresh training run writes an update 1 checkpoint, then interval checkpoints at 100, 200, …; a resumed/copy-recovery directory may initially contain only `checkpoint_latest.pt`, so the manifest/archive are not guaranteed to exist until its next save boundary. `progress.json` is updated every update and can be ahead of the latest complete checkpoint.

### 每个 run 的外部 provenance manifest

训练器不会在进程刚启动时就把所有 CLI 参数复制进 output（`config.yaml` / `training.json` 在完成时才写出），所以长跑开始前要在**独立的 manifest 目录**记录输入和运行时版本。不要把这些文件预先放进模型 output 目录，因为训练器会拒绝使用非空 output 作为新 run。

```powershell
$Meta = 'results/phase86_run_manifests/ippo_seed207_rtx5050'
New-Item -ItemType Directory -Force $Meta | Out-Null
git rev-parse HEAD | Set-Content "$Meta/source_commit.txt"
git status --short | Set-Content "$Meta/source_worktree_status.txt"
python --version 2>&1 | Set-Content "$Meta/python_version.txt"
python -m pip freeze | Set-Content "$Meta/pip_freeze.txt"
conda list --explicit | Set-Content "$Meta/conda_explicit.txt"
nvidia-smi -q | Set-Content "$Meta/nvidia_smi.txt"
Get-FileHash configs/phase86_pure_mappo_repro.yaml, configs/phase85_target_contract_repaired_environment.yaml, results/phase86_nominal_repaired_development_calibration/scenes.jsonl, results/phase86_nominal_repaired_development_validation/scenes.jsonl, models/phase86_fix_route_teacher_formal_contract_20260921/expert_dataset.npz -Algorithm SHA256 | Format-List | Out-File "$Meta/input_sha256.txt"
```

另外把完整 CLI 命令（包括输出目录与 checkpoint 间隔）、开始/结束时间、GPU 型号/显存、NVIDIA driver、PyTorch 与 `torch.version.cuda`、退出原因一并保存。`source_worktree_status.txt` 非空就说明源码不是纯 commit 状态；正式比较应在 clean commit 上重跑。模型 output 本身保留 `config.yaml`、BC metadata、`progress.json`、`checkpoint_manifest.json`、checkpoint SHA、TensorBoard events 和 stdout/stderr。

## 6. 断点续训

只有当训练进程已退出后才从 checkpoint 启动一个新进程；禁止两个进程同时写同一 output。每个 update 会更新 `progress.json`，而 checkpoint 与 manifest 只在保存边界更新，所以训练中二者落后于 progress 是正常的。恢复时以 checkpoint payload 的 `updates_completed` 和其 SHA-256 为权重事实；若 `checkpoint_manifest.json` 已存在，必须确认 manifest 的路径、update 与 hash 对应该 checkpoint。迁移/恢复目录若没有 manifest 或 archive，先单独核验 checkpoint 来源及 payload，不得仅凭 progress 数字推断权重进度。命令中的 config、算法、recurrent 标志、training scenes 和 BC/CBF/action-scale/优化参数应保持一致。`--updates` 是总目标 update 数，不是额外追加数。

IPPO resume 示例（保持第 5B 节中的其余训练参数完全相同）：

```powershell
$Run = 'models/phase86_ippo_recurrent_cbftrain_curriculum_seed207_rtx5050'
python scripts/train_mappo_ippo_baseline.py `
  --config configs/phase86_pure_mappo_repro.yaml `
  --output $Run --resume "$Run/checkpoint_latest.pt" `
  --algorithm ippo --recurrent --seed 207 --device cuda --updates 5000 `
  --episodes-per-update 8 --ppo-epochs 4 --minibatch-size 512 --hidden-dim 128 `
  --learning-rate 5e-5 --gamma 0.99 --gae-lambda 0.95 --clip-range 0.2 `
  --entropy-coef 0.002 --max-steps 250 --torch-threads 1 `
  --checkpoint-interval-updates 100 --log-interval 1 `
  --action-scale-factor 0.6 --action-scale-end-factor 1.0 `
  --action-scale-ramp-updates 1500 `
  --use-cbf-train --cbf-train-iterations 1 `
  --training-scenes results/phase86_nominal_repaired_development_calibration/scenes.jsonl `
  --training-episodes 300 --tensorboard
```

resume 会恢复模型、optimizer 和 Torch CPU/CUDA RNG 状态，并校验 algorithm、recurrent 标志、观测维度、config hash 和训练 scenes hash；不要同时传 `--expert-dataset`、`--bc-epochs` 或重新执行 BC。若只有旧 `progress.json` 的较高 update、却没有对应 checkpoint，以 `checkpoint_manifest.json` 标记的最高完整 checkpoint 为准；手工复制了不匹配的 `progress.json` 时，先恢复同一来源的配套状态，不要把历史 update 数冒充权重进度。

## 7. 每个 checkpoint 的独立验证

对同一 checkpoint 分别运行 300 条 validation scenes。每次必须用新的空输出目录；验证建议用 CPU，避免和训练抢 GPU。策略按 actor mean 确定性执行；`--use-cbf` 只在 eval-CBF 命令中出现。当前评估器逐集输出并聚合 `defender_physical_collision_rate`，同时保留兼容的 `collision_rate` 复合指标。

```powershell
$Run = 'models/phase86_ippo_recurrent_cbftrain_curriculum_seed207_rtx5050'
$Checkpoint = "$Run/checkpoints/checkpoint_update_001500.pt"
$Scenes = 'results/phase86_nominal_repaired_development_validation/scenes.jsonl'
$EvalRoot = "$Run/independent_validation/update_001500"

# raw policy
python scripts/evaluate_mappo_ippo_phase87.py `
  --config configs/phase86_pure_mappo_repro.yaml --checkpoint $Checkpoint `
  --scenes $Scenes --output "$EvalRoot/raw" `
  --algorithm ippo --recurrent --episodes 300 --start-index 0 --max-steps 250 --device cpu

# 同一 checkpoint + external evaluation CBF
python scripts/evaluate_mappo_ippo_phase87.py `
  --config configs/phase86_pure_mappo_repro.yaml --checkpoint $Checkpoint `
  --scenes $Scenes --output "$EvalRoot/eval_cbf" `
  --algorithm ippo --recurrent --episodes 300 --start-index 0 --max-steps 250 --device cpu --use-cbf
```

MAPPO 验证时把 `--algorithm ippo --recurrent` 替换为 `--algorithm mappo`，且不要添加 `--recurrent`。每次都保存 `evaluation.json`；报告必须列出 safe capture、defender physical collision、defender boundary violation、target invalid、timeout，并同时保留 `capture_event` 与 clearance 等诊断项。300 集是 validation selection evidence，不是 holdout claim；checkpoint 选择只能按预先约定的 validation gate 做，不能据它反复改环境/reward 后再称为独立泛化结果。正式优越性结论至少需要 seeds 101/202/303、镜像组配对 bootstrap 区间和锁定 holdout，当前工作未完成这些门槛。

### 指标兼容性

环境 `info` 中 `defender_physical_collision` 与 `defender_boundary_violation` 是两个独立布尔量。当前版本 `scripts/evaluate_mappo_ippo_phase87.py` 已逐集保存 `defender_physical_collision` 并输出 `defender_physical_collision_rate`；同时保留 `collision_rate`，它来源于 `info["collision"]`，含义是 **defender safety failure（physical collision 或 defender boundary violation 的并集）**，不是物理碰撞单项。旧版 evaluator 生成的 JSON 没有物理碰撞单项：若旧 JSON 的 boundary rate 严格为 0，则 composite `collision_rate` 与物理碰撞率相同；boundary 非零时不能只凭两个 aggregate rate 反推。旧 checkpoint 应用新版 evaluator 重新独立评估后，才能得到物理碰撞单项；不要把复合 `collision_rate` 改名为 physical collision。

## 8. TensorBoard 与日志

训练器每 update 写 `train/safe_capture_rate`、`train/defender_physical_collision_rate`、`train/timeout_rate`、loss、episode length、CBF correction/intervention 等 scalar；step 为累计环境步数。日志位于各自 run 的 `tensorboard/`，不要把不同算法写进同一个目录。

```powershell
python -m tensorboard.main --logdir models/phase86_ippo_recurrent_cbftrain_curriculum_seed207_rtx5050/tensorboard --host 127.0.0.1 --port 6006
```

浏览器打开 `http://127.0.0.1:6006`。曲线展示的是每 update 8 个训练 episode 的 rollout 统计，方差很大；TensorBoard 用于看趋势和定位训练过程，不代替固定 300 集 raw/eval-CBF 验证。另应归档终端命令、`config.yaml`、`progress.json`、`checkpoint_manifest.json` 和 environment/runtime 版本。

## 9. 已有开发结果（只做描述，不作正式结论）

### MAPPO seed 101

下表取本机已有 300 集验证 JSON。此处的“碰撞/防御者安全失败”是上一节定义的 composite collision 字段；update 1500–2500 的 boundary rate 为 0，因此这些行中该值等于物理碰撞事件率。数值以百分比显示。

| Checkpoint | 模式 | Safe capture | Collision union | Boundary | Target invalid | Timeout | 平均最小 clearance |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1500 | raw | 86.67% | 13.00% | 0% | 0% | 0.33% | 0.525 m |
| 1500 | eval-CBF | 99.67% | 0% | 0% | 0% | 0.33% | 1.034 m |
| 2000 | raw | 90.67% | 9.33% | 0% | 0% | 0% | 0.479 m |
| 2000 | eval-CBF | 99.67% | 0% | 0% | 0% | 0.33% | 1.009 m |
| 2500 | raw | 89.00% | 11.00% | 0% | 0% | 0% | 0.520 m |
| 2500 | eval-CBF | 99.67% | 0% | 0% | 0% | 0.33% | 1.038 m |

MAPPO raw 尚有约 9–13% safety failure；eval-CBF 显著降低该失败，但仍有少量 timeout。CBF 后处理不是 MAPPO actor 自身的能力，且这些数据仅来自 seed 101。

### 已停止的 recurrent IPPO conservative seed 207

旧目录 `models/phase86_ippo_recurrent_bcconservative_seed207_20260923_v2` 在 update 1500 的历史评估：raw safe capture 14.67%、collision union 85.33%、boundary 11.67%、target invalid 0%、timeout 0%；eval-CBF 由四个 75 集 chunk 汇总，safe capture 69.67%、collision union 0%、boundary 0%、target invalid 0%、timeout 30.33%。该旧 checkpoint/run 已停止，不是目前 CBF-training curriculum 续跑的表现，且 raw collision union 与 boundary 存在重叠语义，不能用相减方式估算 physical collision。

当前 CBF-assisted recurrent IPPO recovery 在本文审计快照时仅到 update 31，尚无 update 1500 的 300 集 raw/eval-CBF 结果；因此不对当前 run 宣称已收敛或达到可接受效果。

## 10. 保存到仓库与复现边界

仓库提供源代码、配置、场景清单、教师数据和本复现说明；训练 checkpoint、TensorBoard event files、progress/logs、验证逐集输出应保留在本机 run 目录或另行归档，不应把活跃训练目录/大体积 checkpoint 提交进 Git。迁移到 RTX 5050 时，克隆仓库后检查输入文件存在与上述 SHA-256 一致，再新建独立 run 目录。单 seed、单环境的仿真结果不可表述为真实无人机系统安全性或一般化结论。

核心代码入口：[`scripts/train_mappo_ippo_baseline.py`](../scripts/train_mappo_ippo_baseline.py)、[`scripts/evaluate_mappo_ippo_phase87.py`](../scripts/evaluate_mappo_ippo_phase87.py)、[`scripts/collect_frozen_policy_dataset.py`](../scripts/collect_frozen_policy_dataset.py)、[`scripts/freeze_phase86_protocol.py`](../scripts/freeze_phase86_protocol.py)、[`src/encirclement3d/learning.py`](../src/encirclement3d/learning.py)、[`src/encirclement3d/pursuit_env.py`](../src/encirclement3d/pursuit_env.py)。
