# Flow ≠ Affordance — 运行日志

> **计划文档（artifact，会重新发布到同一 URL）：**
> https://claude.ai/code/artifact/2034f363-21b2-4dff-acef-4ca6199064a8
>
> 分工：**artifact 放稳定的计划**（motivation / 定位 / Phase 结构 / 风险），
> **本文件放每天的日志和数字**（跑了什么、结果是多少、什么坏了）。

---

## 当前状态

| | |
|---|---|
| Phase | **2 · 按阶段 ablation（ATM 侧）** —— Phase 0 已全部通过，Phase 1 的阶段切分已内置进探针 |
| 阻塞项 | 无（PPI 侧的 CoppeliaSim 是并行任务，不阻塞 ATM） |
| 正在跑 | F 组 sanity check（`FLOW_PROBE_MODE=all`），对照 baseline 0.62 |
| 上次更新 | 2026-07-27 |

### 探针实现要点（`atm/utils/flow_probe.py`）

**拦截点选在 `model.track.reconstruct`。** 原因：ATM 的 track 有**两条**路进 action ——
① `track_encode()` → tr tokens → spatial/temporal transformer；
② `act()` 里 `rec_tracks` **被直接 concat 进 policy head 的输入**
（`feat = torch.cat([x[:, -1], rearrange(rec_tracks[...])], dim=-1)`）。
两条都源自 `track.reconstruct`，所以那是唯一能一次切断两条的单点。

**替代品用「同一时刻、同一任务、另一个并行 env 的真实 track」**（batch 上的 derangement），
而不是 zero —— 边缘分布完全不变，只破坏配对关系。

**阶段判据与 PPI 一致**：gripper 一旦闭合就永久进入 transport（`action[:, -1] > 0`）。
探针包了 `act()` 用上一步的动作更新阶段，包了 `reset()` 清状态 —— 因果上不泄漏。

**视角索引**：`atm/utils/env_utils.py` 里 `cameras.sort()` → **0 = `agentview`，1 = `robot0_eye_in_hand`**
（与 `engine/utils.py` 里 base/wrist 的用法一致，双向确认）。

**开关**：`FLOW_PROBE_MODE ∈ {off, all, approach, transport}`、`FLOW_PROBE_VIEWS`、`FLOW_PROBE_SEED`。
默认 `off`，正常 eval 路径完全不受影响。单元测试（derangement 无不动点、视角选择性、阶段门控取反）已全过。

---

## Gate 结果

| ID | 问题 | 结果 | 日期 |
|---|---|---|---|
| G0.1 | ATM 的 query point 采不采在机器人本体上？ | ✅ **采。全图无 mask，且 variance filter 主动偏向运动物体（= 机械臂）。见下方证据** | 2026-07-26 |
| G0.2 | PPI 有 checkpoint 吗？eval 跑得通吗？keyframe 怎么定义？ | ✅ checkpoint 有、ablation 代码有、keyframe 可直接复用；⚠️ eval 栈重（CoppeliaSim+PyRep+pytorch3d+SAM+GDINO，本机全无），见下方 | 2026-07-26 |
| G0.3 | 能否复现 PPI 在 ≥1 个 RLBench2 task 上报的数字（差 < 5 点）？ | ⏸ PPI 待办；**ATM 侧 eval 已端到端跑通**：libero_spatial 10 tasks × 5 rollouts，`success_env_avg = 0.62` | 2026-07-26 |

---

## 实验结果

### Phase 2 — 按阶段 ablation（Figure 1）

标本：*待填* · 每组 ≥50 episodes × 3 seeds

标本 **ATM**（`atm-policy_libero-spatial_demo10`）· libero_spatial 10 tasks × 5 rollouts × seed 0
（注：ATM 没有 keypose interface，D/E 两组只适用于 PPI）

| 组 | 条件 | 成功率 | 相对 A | 备注 |
|---|---|---|---|---|
| A | 完整 | **0.62** | — | baseline，已复现 |
| B | approach 掐 flow | **0.52** | **−0.10** | z = 1.02，**未达显著** |
| C | transport 掐 flow | **0.32** | **−0.30** | z = 3.15，显著 |
| D | approach 掐 keypose | — | | 仅 PPI |
| E | transport 掐 keypose | — | | 仅 PPI |
| **F** | **全程掐掉（sanity）** | **0.28** | **−0.34** | ✅ **探针生效**，`corrupted_view_slots = 60000 / reconstruct_calls = 6000` |

**F 组通过意味着两件事：**① hook 挂对了位置（两条路径都被切断）；
② ATM 的 track conditioning **不是装饰品**，掉 0.34 说明它承重 —— 于是「按阶段拆」这个问题才有意义。

#### 关键统计（n = 50 / 组）

```
A vs B   diff +0.10   z = 1.02    未达显著
A vs C   diff +0.30   z = 3.15    显著
B vs C   diff +0.20   z = 2.07    ← 核心对比，p ≈ 0.04
A vs F   diff +0.34   z = 3.64    显著
可加性   B+C = 0.40   vs  F = 0.34   → 大致可加，略次可加
```

**首个真结果：flow 的贡献集中在 transport 阶段，transport 侧约为 approach 侧的 3 倍。**
图：`figures/fig_phase_ablation.png`

**必须注意：** pilot 规模（每组 50 episode）只够支撑 B vs C 的边缘显著。
正式实验要 20 rollouts × 3 seeds（每组约 200 episode，SE ≈ 0.05）才能把 0.10 量级测实。
同一 task 内 episode 相关，真实不确定度只会更大。

#### PPI 侧（2026-07-27）

CoppeliaSim 4.1 在 **Ubuntu 24.04** 上无头启动成功（官方只发 20.04 构建，这是最大风险项，已过）。
PyRep / RLBench 1.2.0（双臂 fork）/ YARR / CUDA 12.8 toolkit / **pytorch3d（sm_120）全部装好**。
踩坑：PyRep 要 `--no-build-isolation` + `cffi`；RLBench 要 `poetry-core`；
YARR 会把 numpy 升到 2.x，需回钉 `numpy<2` 并把 opencv 降到 4.11；
pytorch3d 编译要 `CPATH=$CONDA_PREFIX/targets/x86_64-linux/include`（conda 的 CUDA 头不在 `$CONDA_PREFIX/include`）。
**PPI 栈已补完（2026-07-27）**，全部 import 通过：
`pyrep` / `rlbench` / `yarr` / `pytorch3d`(CUDA op on sm_120) / `ppi` / `segment_anything` /
`groundingdino`(含 `_C` CUDA 扩展) / `agents.ppi.ppi_agent`。
权重已下：`pretrained_models/{sam_vit_b_01ec64.pth (358M), groundingdino_swinb_cogcoor.pth (895M)}`。

再踩两个坑：
- **GroundingDINO 的 CUDA 扩展编译不过** —— `ms_deform_attn_cuda.cu:65,135` 用了新 torch 已删除的
  `tensor.type()`：`error: no suitable conversion function from "const at::DeprecatedTypeProperties"
  to "c10::ScalarType"`。已 sed 成 `.scalar_type()` / `.is_cuda()`，编译通过。
  注意 `groundingdino._C` **必须在 `import torch` 之后**才能加载（否则找不到 `libc10.so`）。
- **`pip install -e .` 装不了 PPI 本体和 inference 包**：`ppi/` 缺 `__init__.py`（`find_packages()`
  找不到）；`inference-for-rlbench2/pyproject.toml` 声明了 `packages=[..., "voxel"]` 但
  **仓库里根本没有 voxel 目录**（上游打包 bug）。
  → 改用 `PYTHONPATH`，与 PPI 自己脚本从仓库根目录直接跑的方式一致。
  环境脚本：`source /workspace/code/PPI/run_env.sh`。

#### RLBench2 测试数据 + PPI checkpoint（2026-07-27）

**数据源分工搞清楚了：**
- HF `yuyinyang3y/Open-PPI` 只有 **训练数据 + checkpoint**，**没有 test split**。
- test split 在官方 PerAct² 服务器：
  `https://dataset.cs.washington.edu/fox/bimanual/image_size_256/bimanual_<TASK>.test.squashfs`
  （`eval_ppi.yaml` 里 `camera_resolution: [256, 256]`，所以必须要 256 那版，不是 128。）

**只下 `bimanual_push_box`（9.93 GB），因为 Table VII 那四组 ablation 就是在这个 task 上做的**
（`scripts/ablation/ablation_box_*.sh` → `task='box'`，`eval_ppi.yaml` 默认也是 `bimanual_push_box`）。
七个 task 全下要 ~83 GB，没必要。已装 `squashfs-tools`（`unsquashfs 4.6.1`）解包。

**checkpoint 已下**：`exp_logs/ckpt/bimanual_push_box/epoch{250,300,350}_model.pth.tar`，各 216 MB。

**重要发现：官方只放了「完整 PPI」这一个模型，没有放 ablation 变体的 checkpoint。**
→ Table VII 的 +6.0 / +6.5 **无法直接复现**，除非自己重训四个变体。
→ **但我们不需要。** 我们的方法就是在**完整模型上做 inference 时的 interface ablation**
（和 ATM 侧完全一样的做法），Table VII 只作为 motivation 被引用。**这反而省掉了重训成本。**

**PPI 的 hook 挂点已在 checkpoint 里确认**（`model_state_dict`，618 个参数）：

```
model.cross_attn_pointflow.*       32 个参数   ← pointflow interface
model.self_attn_keyframe.*         64 个参数   ← keypose interface
obs_encoder.point_flow_mlp.*      132 个参数   ← pointflow encoder
```

两个 interface 在结构上是分开的模块，可以像 ATM 那样各挂一个 forward hook 分别打乱。

剩余：解包数据 + 改 `eval_ppi.yaml` / inference 脚本里的路径。

#### 正式规模实验已排队（2026-07-27 05:41 启动）

`run_formal_batch.sh` —— 6 条件 × 2 seeds = 12 次，每次 10 tasks × **20 rollouts**，
`vec_env_num=10`（pilot 是 5）。每条件 400 episodes → **SE_diff ≈ 0.035**，足以把 0.10 量级测实。

条件：`A_full` / `B_approach` / `C_transport` / **`Bw_approach_wrist`** / **`Ba_approach_agent`** / `F_all`
（后两个是组内对照：approach 阶段只掐 `eye_in_hand` vs 只掐 `agentview`，用来验证
「贡献 ∝ 追踪点里机器人本体占比」这条律 —— 不依赖 PPI）。

脚本带断点续跑（已有 summary 的条件自动 skip）和逐条耗时打印。

组内对照（ATM 专属，不依赖 PPI）：

| 组 | 条件 | 成功率 | 备注 |
|---|---|---|---|
| B-wrist | approach 只掐 `eye_in_hand` | | 预期比 B-agent 更疼（该视角几乎全是机器人本体） |
| B-agent | approach 只掐 `agentview` | | |

### Phase 3 — 跨方法定量律（Figure 2）

| 方法 | 追踪点在机器人本体上的比例 | approach 阶段贡献 |
|---|---|---|
| PPI | | |
| ATM | | |

---

## 日志

### G0.1 证据（ATM 的 query point 采样）

代码路径 `/workspace/code/ATM`。

1. **随机点：全图无 mask。** `scripts/preprocess_libero.py:127`
   ```python
   points = sample_from_mask(np.ones((H, W, 1)) * 255, num_samples=1000)
   ```
   mask 是全 255，即整幅图。1000 个随机点落在任何地方 —— 机械臂、夹爪、桌面、背景。

2. **网格点：也是全图。** `preprocess_libero.py:132`，`sample_double_grid(7)` 在
   `[0.05,0.85]` 与 `[0.15,0.95]` 上各铺一个 7×7 网格 = 98 点，均匀覆盖全帧。

3. **关键：variance filter 主动偏向「会动的东西」。** `preprocess_libero.py:96-110`
   ```python
   var = torch.var(pred_tracks, dim=1)          # 轨迹在时间上的方差
   idx = torch.where(var > var_threshold)[0]    # 保留高方差 = 会动的点
   ```
   （注释写的是 "low variance"，**代码保留的是高方差，以代码为准**。）
   随机点 `var_threshold=10.`，网格点 `var_threshold=0.`（全保留）。
   → LIBERO 场景里会动的只有 **机械臂/夹爪（全程动）** 和 **被操作物体（抓取后才动）**，
   静态桌面背景被丢掉。**所以保留下来的点是明显偏向机器人本体的。**

4. **两个视角，其中一个几乎全是机器人。** `agentview` + `eye_in_hand`，后者是腕部相机，
   画面中很大一块恒定是夹爪。

5. **policy 端也没有语义筛选。** `atm/dataloader/bc_dataloader.py:71` 用
   `sample_tracks_nearest_to_grids(..., num_samples=32)`，按「首帧位置离 4×4 双网格最近」
   取 32 条轨迹（`conf/train_bc/libero_vilt.yaml`: `num_track_ids: 32`, `num_track_ts: 16`）
   —— 依然是全帧几何采样，与物体/机器人无关。

**结论：ATM 是 x 轴「追全图、且偏向机器人本体」那一端的理想标本，比预期更极端。**
配合 PPI 的「只追物体」，定量律的两端到位。

**额外发现（免费的第三个数据点）：** ATM 的两个视角本身就构成一组组内对照 ——
`eye_in_hand`（几乎全是机器人）vs `agentview`（机器人只占一部分）。
若定量律成立，**approach 阶段掐掉 `eye_in_hand` 的 track，掉分应显著大于掐掉 `agentview` 的**。
这是 ATM 内部的对照实验，比跨方法对比更干净、更便宜。**写进 Phase 2。**

### ⚠️ 框架修正（2026-07-27）：pointflow 不只是 flow

原来的论证跳得太快。「object flow 对 grasp 不变」→「所以 flow 的收益必然来自 transport」
中间漏了一步 —— **PPI 输入的所谓 pointflow 至少捆绑了三个不同的信号：**

1. **哪个物体**（GroundingDINO 用任务文本 prompt）→ 语言 grounding
2. **它现在在哪**（SAM mask）→ 定位
3. **它接下来往哪走**（预测的位移）→ **只有这一个才是不变性论证所指的 flow**

在一个七物体的场景里，(1)+(2) 本身就解决了任务的一大块，**而它们跟 grasp 姿态无关，也不受
不变性论证约束**。所以 PPI 那 +26.7 有多少来自运动、多少来自 grounding，现在是未知的。

**对定量律的威胁：** PPI 和 ATM 都可能在 approach 阶段掉分，但机制不同 ——
ATM 是因为追了机械臂本体的点，PPI 是因为 pointflow 里带 grounding。
**同一个观测量，两种机制 → 定量律被 confound。**

**修正方案：新增 G 组「freeze-flow」**

| 干预 | 破坏了什么 | 保留了什么 |
|---|---|---|
| shuffle（B/C/F） | grounding + 定位 + 运动 | — |
| **freeze（G）** | **只破坏运动**（把预测轨迹压成 t=0 的静止点重复） | grounding + 定位 |

**G 与 shuffle 的差值 = 纯运动分量的贡献。** 这是把「flow」从「pointflow 这个输入」里剥出来的
唯一干净办法。一个条件，每次 26 分钟，必须加。

### 另一个更锋利的 claim（由 mentor-facing 讨论中发现）

追踪机械臂本体的点**确实**能让 flow 决定 action —— 但那等于放弃了 flow 最初的卖点：

| 追踪什么 | 能决定 grasp？ | embodiment-agnostic？ |
|---|---|---|
| 只追物体点 | ❌ | ✅（所以能从人类视频学） |
| 加上机械臂点 | ✅ | ❌（Franka 的手臂 flow 迁不到别的本体） |

**两者不可兼得，而没有一篇论文把这条 trade-off 写出来。**
这比原来的「flow 不提供 affordance」更强：**flow 的 affordance 含量与它的可迁移性直接冲突。**
→ 定量律的 x 轴（追踪点中机器人本体占比）因此不是干扰变量，**它就是这条 trade-off 曲线本身。**

### G0.2 证据（PPI，`/workspace/code/PPI`）

**① Checkpoint / 数据 / ablation 代码：全部已放出**
- ckpt: https://huggingface.co/datasets/yuyinyang3y/Open-PPI/tree/main/ckpt
- 数据集: https://huggingface.co/datasets/yuyinyang3y/Open-PPI
- README 的 TODO 四项全打勾（训练码、数据、checkpoint、RLBench2 eval 码）。

**② Table VII 的四个条件就是四个 shell 脚本**（`scripts/ablation/`），一行开关的事：

| 条件 | `what_condition` | `predict_point_flow` | `horizon_keyframe/continuous` | `n_action_steps` |
|---|---|---|---|---|
| continuous only | `'continuous'` | false | 0 / 50 | 50 |
| keyframe only | `'keyframe'` | false | 4 / 0 | 4 |
| + keypose | `'keypose_continuous'` | false | 4 / 50 | 54 |
| + pointflow | `'pointflow_continuous'` | **true** | 4 / 50 | 54 |

对应四个模型文件：`ppi/model/diffusion/diffuser_actor_{pure, keypose_continuous, pointflow_continuous, ppi}.py`
→ **hook 挂点清晰，且 +6.0 / +6.5 那两个数可复现。**

**③ Keyframe 定义 —— 可以直接当阶段边界用**
`inference-for-rlbench2/helpers/demo_loading_utils.py::_keypoint_discovery_bimanual`：
```python
right_state_changed = obs.right.gripper_open != right_prev_gripper_open
left_state_changed  = obs.left.gripper_open  != left_prev_gripper_open
state_changed = right_state_changed or left_state_changed
if i != 0 and (state_changed or last or stopped):
    episode_keypoints.append(i)
```
即 keypoint = **gripper 开合状态变化** ∨ 停止（`joint_velocities ≈ 0`，`stopped_buffer=4`）∨ 末帧。
**gripper 开合变化正是我们要的 approach→transport 边界，而且它按左右手分别追踪。**
→ **Phase 1 不用自己定义切分，直接复用 `_keypoint_discovery_bimanual`，取每只手第一个
`state_changed` 的 keypoint 作为该手的 grasp 时刻。双臂按手分别统计也免费拿到。**

**④ PPI 的 query point 确认为「纯物体、零机器人」**（`scripts/data_generation/save_point_flow.py`）
GroundingDINO（text prompt → box）→ SAM（box → mask）→ `sample_farthest_points` **在 mask 内部**采点。
→ 与 ATM 的「全图 + 偏机器人」构成 x 轴两个极端，**两端都由代码证实，不是推测。**

**⑤ 风险：eval 栈比 ATM 重得多**
需要 CoppeliaSim 4.1 + PyRep + RLBench2 + Xvfb + Qt5 + pytorch3d + SAM ckpt + GroundingDINO ckpt。
本机现状：**CoppeliaSim / PyRep / RLBench 全都没有**；pytorch3d 只在 `forehoi5090` 里有（torch 版本不同）。
README 钉 `python=3.8` + `cuda 11.8` → 与 ATM 同样的 sm_120 问题，必须升级。
**估计 2–4 天**，CoppeliaSim 在无显示容器里是典型时间黑洞。

---

### 2026-07-26
- 确定方向：从 GD-4D（已停止）转到 Flow ≠ Affordance。
- 完成 prior-art check：Track2Act / Im2Flow2Act 的 limitation 里有未量化的提及；PPI (RSS 2025)
  已有 keypose+pointflow 架构但未解释原因。**按阶段归因、零交互解释、跨方法律三项无人做过。**
- 产出 workplan（见顶部链接）。
- **ATM 环境配好并冒烟通过。** conda env `atm5090`，配方见 `/workspace/code/ATM/setup_atm5090.sh`。
  - **偏离 README**：`environment.yml` 钉 `python=3.8.18` + `torch==2.0.1`，两者都早于 sm_120，
    在 RTX 5090 上会死于 `no kernel image is available for execution on the device`。
    基座改为克隆 `gd4d5090`（py3.10 + torch 2.11+cu128，`sm_120` 在 `get_arch_list()` 里）。
  - `gym==0.21.0` 在现代 setuptools 上构建失败 → 换 `gym==0.26.2`（`libero/envs/venv.py` 只用
    `gym.Env` / `gym.spaces` 做类型标注和 isinstance，无 API 影响）。
  - `pybullet-svl` 构建失败（`ModuleNotFoundError: pkg_resources`）→ 跳过，LIBERO 用 mujoco/robosuite。
  - `tensorflow` / `tfds` / `deepspeed` / `keras` → 跳过，eval 用不到。
  - 冒烟测试：`OffScreenRenderEnv` + EGL 在 GPU0 上建环境、reset、step 全通过；
    obs 里两个视角确认为 `agentview_image` 与 `robot0_eye_in_hand_image`。
- **G0.1 完成，结论见上（ATM 是「追全图 + 偏向机器人本体」那一端，比预期更极端）。**
- **G0.2 完成**（PPI ckpt / ablation 码 / keyframe 全 OK，eval 栈重）。
- **ATM eval 端到端跑通** —— `libero_spatial` 10 tasks × 5 rollouts，`success_env_avg = 0.62`。

#### ATM eval 跑通的六个坑（都已修，固化在 `run_eval_gpu0.sh`）

1. **数据直接复用** —— `ln -sfn /workspace/datasets/libero/hdf5 data/libero`。
2. **eval 不需要 CoTracker 预处理**。`eval_libero_policy.py` 只用 `data/atm_libero/` 枚举 task 和读
   `env_meta.json`；真正的 track 是 policy 内部的 track transformer 在 rollout 时预测的。
   → 写了 `scripts/make_env_meta_only.py`，只抽 `env_args` 生成 40 个 `env_meta.json`，
   **省掉 1000 点 × 2 视角 × 40 task 的 CoTracker 预处理。**
3. **GPU 硬编码** —— 原脚本 `train_gpus=[0,1,2,3]`、`env_gpu_ids=[4,5,6,7]`；
   新 launcher `run_eval_gpu0.sh` 全部改成 `[0]`。
4. **`~/.libero/config.yaml` 是全局共享的**，当前指向 `Geometric-Action-Model/LIBERO-plus`，
   且缺 `task_embeddings` 键。**没有覆盖它** —— 用 `LIBERO_CONFIG_PATH=/workspace/code/ATM/.libero`
   建项目本地 config；另用 `get_task_bert_embs` 生成 40 条
   `libero/task_embedding_caches/task_emb_bert.npy`（只跑 BERT，不跑 CoTracker）。
5. **torch ≥ 2.6 的 `weights_only=True`** 打死 LIBERO 的 init_states（numpy pickle）。
   已给 `libero/benchmark/__init__.py:170`、`atm/model/track_transformer.py:333`、
   `atm/policy/vilt.py:519` 三处补 `weights_only=False`。
6. **lightning ≥ 2.3 要求登记非 forward 方法**，否则 rollout 里 `policy.act()` 抛
   `RuntimeError: ... from outside the model`。已在 `engine/eval_mv_bc.py` 的 `fabric.setup` 后加
   `model.mark_forward_method("act")`。

#### 时间成本（影响 Phase 2 排期）

10 tasks × 5 rollouts × horizon 600、`vec_env_num=5`、单张 5090 ≈ **25–30 分钟**。
按论文默认 20 rollouts 外推：**一个 suite 一个条件 ≈ 2 小时**。
Phase 2 有 6 组条件 × 3 seeds = 18 次 → **单 suite 约 36 GPU 小时**。
→ 排期时要么减 task 数、要么调大 `vec_env_num`、要么主实验只做一个 suite。

- **未做：** G0.3 的 PPI 部分（卡在 CoppeliaSim）；ATM 侧按 20 rollouts 正式复现。

---

## 踩坑记录

- rollout 时 `batch=1`，**不能在 batch 维打乱** flow —— 必须用时间维的 buffer 取替代品。
- ablation **用 shuffle 不用 zero**：zero 会改激活统计量，掉点可能是分布 shift 的假阳性。
- 所有 GPU 任务只用 GPU0（`CUDA_VISIBLE_DEVICES=0`），GPU1 被别的 job 占着。

---

## 2026-07-27 正式实验 + grasp-feasibility pilot

### ATM 正式规模结果（10 tasks × 20 rollouts = 200 episodes/条件）

```
                        seed0     seed1     Δ vs A_full
A_full                  0.710     0.710       —
B_approach (双视角)        —       0.650     −0.060
C_transport (双视角)       —      running
Bw_approach_wrist       0.645       —        −0.065     腕部，机器人采样点 21%
Ba_approach_agent       0.720       —        +0.010     第三人称，机器人采样点 9%
F_all                   0.275       —        −0.435
```

**两个 seed 的 A_full 完全一致（0.710 / 0.710）—— 仪器稳定。**

**核心结构：approach 阶段的效应全部来自腕部相机，第三人称贡献为 0。**
这直接回答了「approach 时物体没动，flow 当然没用」的质疑：该质疑在**物体通道**上成立，
但它预测「两个视角都是 0」，而腕部不是 0。approach 阶段唯一起作用的 flow 是**机械臂自身的
flow** —— 因为夹爪的未来轨迹就是 action 本身。

统计功效：单条对比 n=200，SE_diff ≈ 0.046，z ≈ 1.3–1.6，**尚未显著**。
pass2 补齐 `B_approach_seed0` / `Bw,Ba_seed1` 后 n=400，预期 z ≈ 2.3。

### grasp-feasibility pilot —— 判定「transport-conditioned 打分」这个 module 不成立

**要测的**：知道物体轨迹 T(t)，能从「语义上都合法」的候选抓法里排除掉多少个。
若排除不掉，pipeline 里那个 scoring 步骤就没价值。
**预先定死的判据**（看数据之前）：headroom = oracle − blind+reach，< 5 点 → 死。

代码：`tools/grasp_pilot.py`（候选生成 + 3D 可视化）、`tools/grasp_feasibility.py`（可行性矩阵）。
候选族最终版：8 方位角 × 3 倾角 × 3 高度 × 3 半径 = 216，再按「贴着物体表面」过滤
（穿透 ≤ 3 mm 且 间隙 ≤ 8 mm）。可视化已人眼确认（夹爪到碗 0.9–3.6 mm，无穿模）。

**结论：pipeline 那个 module 死了。所有 headroom 数字（+8.2 / +4.5 / +22 / +18 / +12）全部作废。**

死因是自检暴露的 —— 加了一条「demo 自己真实执行过的抓法必须判为可行」的自检后：

```
task                                          自检   C漂移mean   C漂移max
put_the_bowl_on_the_plate                     1.00      3.1mm      9.5mm
open_the_top_drawer_and_put_the_bowl_inside   0.40     22.4mm     92.4mm   (两段式 task，t_g 模型不适配)
KITCHEN_SCENE4 碗放进底层抽屉                  0.30      2.3mm      6.1mm
KITCHEN_SCENE6 杯子放进微波炉                  0.40      5.2mm     17.4mm
```

三个原因，前两个是 bug，第三个不是：

1. transport 段跑过了松手点 —— `..._and_close_it` 的 task 在放下之后还要回去关抽屉，
   强行令 E(t)=T(t)·C 把夹爪拽回抽屉里 → 全 0。**已修**：`t_end` = 夹爪首次张开处。
2. geom 级接触白名单被 box 分解击穿 —— 碗 41 个 box、柜子 40+ 个，IK 解差几毫米就从
   `white_cabinet_1_g35` 换成 `g37`，白名单失效。**改成 body 对 + 穿透深度判定，没救回来。**
3. **紧公差场景下可行性由毫米级间隙 + 7-DoF 冗余解算决定。** SCENE4 漂移仅 2.3 mm
   （刚性抓取假设成立）而自检仍只有 0.30。加了零空间偏置（把姿态拉向 demo 真实关节角）
   也没改善。**修不了。**

**第三条同时判死 pipeline 本身**：在唯一值得打分的 regime（紧公差放置）里，feasibility
不是 (抓法, 物体轨迹) 的函数 —— 它还取决于毫米级间隙和 planner 的冗余解算。
我算不出来，一个学出来的打分器同样算不出来，因为**输入里根本没有这个信息**。
这不是实现问题，是这个 module 的输入输出定义本身不成立。

**可信的那部分数字说「没效应」**：自检 = 1.00 的桌面自由放置 task，每条轨迹 25–39/197 个
候选都可行，headroom ≈ 0–3 点。挑不挑无所谓。

**若要救**：只能把「几何可行性」换成「从执行结果学出来的 critic」——
不算 feasibility，直接学「这个抓法配这条轨迹最后成没成」。是个大得多的项目，
就凭现在的证据不建议开。

### 新增踩坑

- **写测量脚本必须先写自检。** 今天两次全 0 翻车（夹爪-物体接触被当成碰撞；transport 段
  跑过松手点）都能被「demo 自己的抓法必须判可行」这一条当场抓住。第一版就该有。
- LIBERO 的 collision mesh 是 box 分解的，**任何 geom 级的白名单/黑名单都不可靠**，
  必须在 body 级做。
- PPI eval 慢到不可用：`num_inference_steps=1000` × `query_freq=10` → **37 分钟/episode**
  （07:50:54 → 08:28:08）。正式跑前必须降步数或换 DDIM。

---

## 待办：PPI（2026-07-27 09:40 挂起）

**状态：栈已跑通并验证成功，但暂停使用。**

```
Evaluating bimanual_push_box | Episode 0 | Score: 100.0
Evaluating bimanual_push_box | Episode 1 | Score: 100.0
```

2/2 成功 —— CoppeliaSim + PyRep + RLBench2 + GroundingDINO + SAM + DINOv2 整条栈正确。
（2 个 episode 不足以复现论文成功率，只证明 pipeline 没搭错。）

**挂起的两个理由：**

1. **太慢**：episode 0 = 37 min，episode 1 = 52 min，约 **45 min/episode**。
   1 条件 × 20 episodes = 15 小时；6 条件 × 2 seed = 7.5 天。而且它跟 ATM batch
   抢 GPU0（`B_approach_seed1` 因此跑了 44 分钟，均值 28）。已 kill 释放 GPU。
2. **上游复现性存疑**：GitHub issue #7 有人用官方 checkpoint + 官方 pipeline 得到
   `bimanual_lift_ball` ~40%、`bimanual_handover_item_easy` ~0%，**作者未回复**。
   说明 PPI 的复现是 task-dependent 的。我们选的 `bimanual_push_box` 属于能跑通的那类。

**重启它之前必须先做的事（按顺序）：**

1. **量时间构成** —— `ppi_agent.py:481,489` 已有 `ptc_time` / `dino_time` 变量但从不打印。
   把三段耗时打出来：点云预处理 / DINOv2+GDINO+SAM / diffusion 采样。
   45 min ÷ 25 次 policy call ≈ **108 s/次**，先搞清楚这 108 秒的构成。
2. **别凭猜降 `num_inference_steps`。** 如果瓶颈是 CoppeliaSim 的 Xvfb 软件渲染
   （我们用 `headless=True` + Xvfb，作者脚本用 `headless=False`），砍 diffusion 白砍。
3. **若确认瓶颈是 diffusion**：切 `DDIMScheduler` + `num_inference_steps: 20`
   —— 这是**作者自己真机 config（`ppi/config/ppi_real.yaml:45`）的设置**，
   不是我们瞎改。仿真 config 用 1000 只是为了 benchmark 数字。
   改完必须先重跑 baseline 确认成功率没掉，然后所有条件统一。

**什么时候需要它：** ATM 那套结论要做「跨方法定量律」时 —— 需要第二个方法点来确认
「approach 阶段贡献 ∝ 被追踪点里机器人本体的占比」不是 ATM 独有的。
在此之前 PPI 不在关键路径上。

**复跑命令：** `cd /workspace/code/PPI && bash run_eval_box.sh <EPISODES> <SUFFIX>`

---

## 2026-07-27 pass1 完成 —— 主结论已定

**统计口径：按 10 个 task 配对做 t 检验（df=9），不是把 episode 当独立 Bernoulli。**
同 task 内的 episode 相关，朴素 z 会高估显著性；而且两种口径的差异方向**不一致**
（`C_transport` 朴素 z=-6.34 但配对 t=-4.52；`Bw` 朴素 -2.86 但配对 -3.58），
不能统一打折。**论文必须报配对检验。**

```
条件                       n    mean       Δ     配对t        p          95%CI
A_full (基线)             400   0.710       —       —        —
Bw_approach_wrist        400   0.615   -0.095   -3.58   0.0060   [-0.155,-0.035]
Ba_approach_agent        400   0.713   +0.003    0.12   0.9043   [-0.043,+0.048]
B_approach               200   0.650   -0.060   -1.17   0.2731   [-0.176,+0.056]
C_transport              200   0.410   -0.300   -4.52   0.0014   [-0.450,-0.150]
F_all                    400   0.338   -0.372   -9.69   0.0000   [-0.459,-0.286]
────────────────────────────────────────────────────────────────────────────
Bw vs Ba (同阶段直接对比)  400           -0.098   -4.17   0.0024   [-0.150,-0.045]
```

### 三句核心结论

1. **approach 阶段掐掉腕部视角的 flow：−0.095，p=0.006。**
2. **同一阶段掐掉第三人称视角的 flow：+0.003，p=0.90，95% CI = [−0.043, +0.048]。**
   这是**等价检验**，不是"没测出来"：n=400 能排除任何大于 4.8 个点的效应。
   物体区域 flow 对 approach 的贡献，上界 4.8 点，点估计 +0.3。
3. **两者直接对比：−0.098，p=0.0024。**

同阶段、同样静止的物体，**只换掐哪个相机**，效应从 0 变成 −0.095。
腕部里夹爪占 21%，第三人称里机器人占 9%。

**这就是对「approach 时物体没动，flow 当然没用」的最终回答**：该质疑预测
两个视角都是 0，实测第三人称是 0 而腕部不是。质疑在**物体通道上完全成立**
（`Ba` 就是它的确认），但它推不出「恰好是零，而剩下那点全在机械臂像素上」。

### seed 稳定性

```
A_full              0.710 / 0.710   |差|=0.000
Ba_approach_agent   0.720 / 0.705   |差|=0.015
Bw_approach_wrist   0.645 / 0.585   |差|=0.060
F_all               0.275 / 0.400   |差|=0.125   ← 其他条件的 2–8 倍
```

**`F_all` 的 seed 方差异常大。** 假设（未验）：它是唯一全程双视角都被 shuffle 的条件，
policy 完全失去 flow、行为近乎崩溃，此时成功率对初始条件更敏感。
**免费的检验：pass2 的 `G_freeze_all` 保留了 grounding 和定位（只抹运动），
若上述解释成立，它的 seed 方差应显著小于 `F_all`。**

不打算给 `F_all` 加第三个 seed —— CI 已够窄且不沾零，它只是上界参照。
预算花在 `B_approach` 上更值（现在 n=200，CI 跨零）。

---

## 事故：pass2 静默死亡，空转 14.5 小时（2026-07-27 10:56 → 07-28 01:34）

`chain_pass2.sh` 在 pass1 结束时正确触发，`G_freeze_all_seed0` 于 10:38:51 开跑，
**10:56 进程静默消失**，日志停在一个 0% 的 tqdm 进度条，无 traceback、无 OOM 记录。
结果目录 `eval_results_G_freeze_all_seed0/` 存在但无 summary CSV。
**8 个 pass2 条件一个都没完成。**

**注意：该 batch 当时已经用 `setsid` 脱离了会话。** 死因未定位。

**修法 —— 不赌"这次能活"，加看门狗** `/workspace/code/ATM/watchdog_pass2.sh`：

```bash
for i in $(seq 1 40); do
  n=$(for d in $EXP/eval_results_*_seed*/; do [ -f "$d/summary_libero_spatial.csv" ] && echo x; done | wc -l)
  [ "$n" -ge 18 ] && break        # 9 条件 × 2 seed
  bash run_formal_batch2.sh       # batch 自身有 skip 逻辑，重跑幂等
  sleep 30
done
```

用**结果文件计数**判定完成，不用 `pgrep`（`pgrep -f` 会匹配到调用它的 shell 自己，
本 session 已经因此吃过两次 exit 144）。

**教训：长跑任务只 `setsid` 不够，必须配一个基于产物计数的重启循环。**

---

## 新产出：`FLOW_TO_ACTION_SURVEY.md`

20 篇 flow→action 方法**逐字读过正文**（ATM/PPI 另有源码级证据；
ToolFlowNet/AVDC 因早于 arXiv HTML 渲染，读的 PMLR/arXiv PDF 原文）。
三张表：主表（追什么/怎么变 action/contact 从哪来）、按「追什么」排的光谱、contact 来源汇总。

**核心发现：二十个方法，没有一个从 flow 里得到 contact。**
前十二个明码标价地从外部搬进来，后八个没讨论，**只有 ATM/Tra-MoE 那一支是隐性的**。

最有用的四条原句：

- **AVDC**："if the object is graspable, **we randomly sample a grasp on the object**"
  —— object flow 推不出抓哪儿，所以随机采。最直白的承认。
- **General Flow**：标题叫 "Foundation **Affordance**"，论证靠 "remaining neutral to
  specific manipulators"，实验里 "we **manually position the robotic arm**"。
  **用来论证它是 affordance 的那条性质，正是让它不可能是 affordance 的那条性质。**
- **FOFPred**：独立作者组做出了同样的诊断 —— 只追物体会 "**overlooking crucial global
  information, such as the overall movement of a manipulator**" —— 于是改用整幅图 dense
  光流故意把机器人放进来，**但从未量过这改动买到了多少**。诊断被独立验证，测量的空位还留着。
- **ToolFlowNet**：对机器人控制的刚体预测 flow，作者点破它 "**reflects the 'intended'
  action from the robot**"。追机器人刚体，flow 就是 action 的投影；追物体，就不是。

**已排除：`FlowPolicy` (AAAI'25) 的 "flow" 是 flow matching，同名不同义。**
同类需警惕 `PointFlowMatch` / `CoLA-Flow` / `Trajectory-Consistent Flow Matching`。

---

## 2026-07-28 · freeze 干预失效 —— 修正方案 anchored

pass2 一开跑就把 G 组自己判死了。

```
条件                        每 task 成功率                                    avg
A_full (基线)                                                              0.710
F_all      (shuffle 全程)   0.20 0.35 0.35 0.30 0.15 0.45 0.15 0.35 0.25 0.20  0.275 (seed0)
G_freeze_all                0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.000
Gb_freeze_approach          0.05 0.0  0.0  0.0  0.0  0.15 0.0  0.0  0.0  0.0   0.020
```

### 判死的是干预，不是假设

freeze 破坏的东西是 shuffle 的**真子集**：

| 干预 | 哪些点 | 在哪 | 怎么动 |
|---|---|---|---|
| shuffle | ❌ | ❌ | ❌ |
| freeze | ✅ 保留 | ✅ 保留 | ❌ |

破坏得更少 ⇒ 成功率**必须 ≥** shuffle。实测 **0.000 < 0.275**，严格更差。
**这在逻辑上不可能，除非 freeze 引入了计划外的第三条破坏通道。** 两条，都找到了：

**① 分布外取值。** 恒定 track 在训练数据里从不出现（相机相对运动至少让它动一点），
而 ATM 把 `rec_tracks` **直接 concat 进 policy head**
（`feat = torch.cat([x[:, -1], rearrange(rec_tracks[...])], dim=-1)`），
恒定值是 policy 从没见过的取值。10 个 task **精确为 0**（不是「低」）—— 手臂基本没动。

**② 阶段状态机被锁死。** 阶段判据是 gripper 闭合（`_update_phase`: `a[:, -1] > thresh`，
一旦闭合永久进 transport）。freeze 让手臂不动 → 夹爪永不闭合 → **永远停在 approach**
⇒ **`freeze_approach` 退化成 `freeze_all`**。0.020 ≈ 0.000 不是巧合，是同一个条件。

对照：shuffle 没有 ② 这个陷阱 —— 它给的是另一个 env 的**真实** track，手臂照动，
`B_approach = 0.650` 说明 episode 正常越过了 approach。

**`freeze_transport` 不受 ② 影响**（approach 段无干预，状态机能正常翻转），
所以 `Gc_freeze_transport_seed0` 是 G 组唯一没被污染的数据点，留着跑完。
但它仍受 ① 影响，只能当定性参考。

### 修正方案：anchored shuffle

取**另一个 env 的位移场**，从**本 env 的 t=0 位置**发出：

```python
donor = x[perm] - x[perm][:, :, :1] + x[:, :, :1]
```

| | 哪些点 | 在哪 | 怎么动 | 在分布内？ |
|---|---|---|---|---|
| freeze | ✅ | ✅ | ❌ | **❌ 恒定值** |
| **anchored** | ✅ 本 env | ✅ 本 env 的 t=0 | ❌ 另一 env 的位移 | **✅ 真实位移幅度** |

这才是原本想要的「只剥掉运动分量」，且不制造 OOD 输入、不锁死状态机。
**anchored 与 shuffle 之差 = 纯运动分量的贡献。**

已实现：`atm/utils/flow_probe.py` 新增 `anchored` / `anchored_approach` / `anchored_transport`
三个 mode（现有 shuffle / freeze 行为逐元素不变，已回归验证）。
单元测试 `tests/test_flow_probe_anchored.py` 六条全过：

```
PASS  test_anchored_is_not_constant            与 freeze 的关键区别，且位移幅度未塌
PASS  test_anchored_motion_comes_from_another_env
PASS  test_anchored_preserves_t0_exactly       t=0 逐元素不变
PASS  test_off_is_identity
PASS  test_phase_gating_is_complementary       approach/transport 门控严格互补
PASS  test_view_selectivity
```

### 教训

**「破坏得更少却掉得更多」是一个可以事先写下来的自检。**
`freeze ⊂ shuffle ⇒ SR(freeze) ≥ SR(shuffle)` 这条单调性不需要任何领域知识就能写，
而它一条就抓住了两个独立的 bug。**和 grasp-feasibility 那次是同一个教训的第二个实例：
测量脚本的第一行代码应该是自检。**

另注：**当干预会改变阶段判据本身所依赖的信号时，「只在某阶段干预」是不成立的。**
gripper 状态既是阶段判据又受干预影响 → 反馈回路。设计分阶段 ablation 时必须检查这条。

### Gc_freeze_transport（02:32）—— 把两条病因分离开，并量出 OOD 惩罚 = 24 点

```
                              每个 task 的成功率                                avg
A_full   (基线)          0.45 1.0  0.9  0.8  0.8  1.0  0.65 0.65 0.55 0.3   0.710
C_transport  (shuffle)   0.0  0.8  0.75 0.15 0.45 1.0  0.25 0.3  0.15 0.25  0.410
Gc_freeze_transport      0.15 0.0  0.45 0.0  0.0  0.2  0.4  0.15 0.0  0.35  0.170
G_freeze_all             0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.000
Gb_freeze_approach       0.05 0.0  0.0  0.0  0.0  0.15 0.0  0.0  0.0  0.0   0.020
```

**Gc 不是零，且 task 间散布正常（0.0–0.45）—— 这是一个活着的 policy。**
因为它的 approach 段完全不受干预，手臂正常伸过去、正常闭合夹爪，状态机正常翻转
⇒ **陷阱 ②（阶段锁死）在 Gc 上不存在**。G/Gb 的 ~0 由 ② 主导，Gc 只剩 ①。

**于是 Gc 单独量出了陷阱 ①（分布外取值）的大小：**

| 同为 transport 段、同为双视角，只换手法 | 成功率 |
|---|---|
| shuffle（`C_transport`） | 0.410 |
| freeze（`Gc`） | 0.170 |
| **差 = 纯 OOD 惩罚** | **−0.240** |

单调性第三次被违反（freeze ⊂ shuffle 却掉得更多），这次幅度 24 点。

**这个数字是判死 freeze 的定量依据：OOD 通道本身值 24 点，
而我们要测的 approach 效应只有 9.5 点 —— 噪声是信号的 2.5 倍。**
freeze 在任何阶段都不可能量出想量的东西。anchored 必须替换它。

**副产品：G 组三个条件的 ~0 / 0.02 / 0.17 现在有了完整的机制解释**，
不是「policy 崩了」这种含糊说法：

| 条件 | 陷阱 ① OOD | 陷阱 ② 阶段锁死 | 结果 |
|---|---|---|---|
| `G_freeze_all` | ✅ 全程 | ✅（本来就是全程，无所谓） | 0.000 |
| `Gb_freeze_approach` | ✅ | ✅ **退化成 all** | 0.020 ≈ G |
| `Gc_freeze_transport` | ✅ | ❌ 不触发 | **0.170** |

**队列状态**：`B_approach_seed0` 02:32 开跑（补 n=400 的两条之一），接着 `C_transport_seed0`。
之后会进 seed1 的 3 个 freeze 条件（约 75 分钟，已知无效）——
**那是需要动手跳过的点**。现在不改 `run_formal_batch2.sh`：它正在运行，
bash 按字节偏移续读脚本，改运行中的脚本可能读到半截命令。

---

## 2026-07-28 03:30 · pass2 收工 —— 12 个非-freeze 条件全部 n=400

`C_transport_seed0 = 0.43`（28 分钟）出完，队列的有效工作结束。
随后 batch 自动进了 `G_freeze_all_seed1`（已知无效），**手动停掉**：
按 PID 依次 kill 看门狗 6726 → batch 6733 → python 127171，
删除未完成的 `eval_results_G_freeze_all_seed1/`。GPU0 已释放（2 MiB）。
（日志里的 `_pickle.UnpicklingError: pickle data was truncated` 是被 kill 的
vec env 子进程的正常退出噪声，不是新问题。）

### 最终结果表（配对 t 检验，10 个 task 配对，df=9）

```
条件                       n    mean       Δ     配对t        p          95%CI
A_full (基线)             400   0.710       —       —        —
Bw_approach_wrist        400   0.615   -0.095   -3.58   0.0060   [-0.155,-0.035]  ✅
Ba_approach_agent        400   0.712   +0.003    0.12   0.9043   [-0.043,+0.048]  等价检验
B_approach               400   0.637   -0.072   -1.95   0.0829   [-0.157,+0.012]  ✗ 跨零
C_transport              400   0.420   -0.290   -4.64   0.0012   [-0.431,-0.149]  ✅
F_all                    400   0.337   -0.372   -9.69   0.0000   [-0.459,-0.286]  ✅
─────────────────────────────────────────────────────────────────────────────
Bw vs Ba (同阶段)         400           -0.098   -4.17   0.0024   [-0.150,-0.045]  ✅ 头条数
```

**加倍样本没有动摇任何一个显著结论**，`C_transport` 从 −0.300 微调到 −0.290（p 0.0014→0.0012）。

### `B_approach` 补到 n=400 仍不显著 —— 这与主结论一致，不是矛盾

p 从 0.273 → 0.083，CI 从 [−0.176,+0.056] 收到 [−0.157,+0.012]，**仍跨零**。

原因：`B_approach` 是**两个相机都掐**，而这两路一实一虚：

| | Δ | 每-task 标准误 |
|---|---|---|
| `Bw`（只掐腕部） | −0.095 | 0.027 |
| `Ba`（只掐第三人称） | +0.003 | — |
| `B`（两个都掐） | −0.072 | **0.037**（+40%） |

**掐第三人称不带来效应，只带来方差。** 效应量没增加而 task 间标准误高 40%，
所以显著性反而更差。**这正是「approach 段的效应全在机械臂像素上」的另一个侧面证据：
往里加一个纯噪声通道，只稀释信噪比。**

→ **论文的头条数必须是 `Bw` vs `Ba`（p=0.0024），不是 `B_approach`。**
B 是个更粗的条件，把一个真效应和一个零混在一起。

### seed 稳定性（全部 n=400 后）

```
A_full              0.710 / 0.710   |差|=0.000
Ba_approach_agent   0.720 / 0.705   |差|=0.015
C_transport         0.430 / 0.410   |差|=0.020
B_approach          0.625 / 0.650   |差|=0.025
Bw_approach_wrist   0.645 / 0.585   |差|=0.060
F_all               0.275 / 0.400   |差|=0.125   ← 仍是次大值的 2 倍
```

**`F_all` 的方差异常在补满数据后依然存在。**
原本登记的检验（`G_freeze_all` 的 seed 方差应更小）随 freeze 被判死而作废。
**改由 anchored 组来验**：anchored 保留 grounding 和定位、只抹运动，
若「policy 完全失去 flow 才崩溃、崩溃状态对初始条件更敏感」这个解释成立，
它的 seed 方差应显著小于 0.125。

### pass2 完整产出清单

| 条件 | seed0 | seed1 | n | 用途 |
|---|---|---|---|---|
| `A_full` | 0.710 | 0.710 | 400 | 基线 |
| `Bw_approach_wrist` | 0.645 | 0.585 | 400 | **头条对照的一半** |
| `Ba_approach_agent` | 0.720 | 0.705 | 400 | **头条对照的另一半 + 等价检验** |
| `B_approach` | 0.625 | 0.650 | 400 | 粗条件，不显著（见上） |
| `C_transport` | 0.430 | 0.410 | 400 | transport 段（平凡效应的确认） |
| `F_all` | 0.275 | 0.400 | 400 | 破坏上限参照 |
| `G_freeze_all` | 0.000 | — | 200 | **干预失效，仅作记录** |
| `Gb_freeze_approach` | 0.020 | — | 200 | **干预失效，仅作记录** |
| `Gc_freeze_transport` | 0.170 | — | 200 | **干预失效；但量出 OOD 惩罚 = 24 点** |

**下一轮**：anchored 组（`anchored` / `anchored_approach` / `anchored_transport` × 2 seed，
约 2.5 小时）。代码和单元测试已就绪，未启动。

---

## 2026-07-28 深夜 · PPI 条件化机制的代码级澄清（影响挂起中的 PPI 实验设计）

**问题**：PPI 到底 condition 在 object flow 上吗，还是 flow 只是个 regression 输出？

**答案（`ppi/model/diffusion/diffuser_actor_ppi.py` 实测）：两者都对一半 ——
被 condition 的是 pointflow 分支的 latent 特征，不是解码出的 flow 坐标。**

```python
# L353  keyframe token 序列直接拼入 pointflow 特征
features_keyframe = torch.cat([..., features_point_flow], 0)
# L389  作者注释：detach keypose and pointflow features,
#       which act as conditions for continuous action
# L396  continuous 分支拼 features_keyframe_detach（内含 pointflow 特征）
```

- `features_point_flow`（latent）→ 经两级 attention 进 action：**条件化为真**
- `position_point_flow`（解码坐标）→ 只进 regression loss，**从不回流给 action**
- `.detach()`：action 梯度不回传 flow 分支 —— flow 只由自身 regression loss 训练

**⚠️ 对 PPI 重启检查单的追加（第 4 条）**：
将来对 PPI 做 flow 干预，**hook 必须挂在 `features_point_flow` 上（L353 拼接之前）**。
挂在解码输出 `position_point_flow` 上是无效的 —— 那是 output-only 的死胡同。

**论文可用的一句话**：PPI 已经把 $T$ 的原材料解码出来了（`position_point_flow` 就是
物体点的未来轨迹），却不做任何解析利用，而是把 flow 信息重新编码成 latent 喂给黑箱 ——
$E=T\cdot C$ 摆在它的输出端口上，它没用。

## 2026-07-29 · 进入实现阶段：分阶段实现计划落档

`PIPELINE_IMPL_PLAN.md` 创建，作为实现期的进度追踪主文档。阶段划分：〇守门实验（BC 因果测试）→ 一 C 分布头 → 二 approach robot-flow → 三 transport object-flow → 四整合。关键设计注入：(1) 阶段二保留 IK-to-C baseline 作为对照（B2 = 避障优势是主路线存在理由）；(2) v1 flow backbone 统一复用 ATM track transformer，只换 mask 内查询点采样；(3) 阶段三化简 E(t)=T_rel(t)·E(t_g)，无需显式 C、无需绝对物体位姿；(4) A2 latent-probe 验收把一票否决问题定量化。下一步：测 BC 单 epoch 墙钟。

## 2026-07-30 · 阶段〇执行完毕（判定暂缓）+ 阶段一设计定向

**阶段〇结果**（n=400×3，全记录在 `PIPELINE_IMPL_PLAN.md` §1.7 + `RoutedFlow/experiments/stage0_routing_causal_test/`）：
object_only 0.630 / robot_only 0.6525 / phase_switched 0.1975（配对 t p=5.1e-6）。表面 kill 触发；
**用户指示暂缓判定**（单模块不足以下全局结论）。机制线索：iii latch 中位 160 步 vs demo 45、27% 从不闭合、
val loss 反而最低 → 疑似 phase 信号自指的 exposure bias（freeze-lockup 的深化版），候选诊断 = 强制 44 步锁存重评（未跑，入 D6）。

**阶段一设计定向**（讨论记录在 `PIPELINE_IMPL_PLAN.md` §2）：两因子分解 P(C|I,L)=P(p|I,L)×P(R,w|p,I)；
坐标系约定定稿（C=物体系相对量，执行层统一 robot base 系）；文献 sweep 更新：AFUN(2606.02551,
VLM+SAM3+MetaQuery，权重公开) 取代 AffordanceLLM 成前端首选，D1 改三选一（倾向冻结 AFUN，spike 定稿）；
撞车确认：离散候选+VLM 过滤胶水全被占，可微分布+下游 latent 接口未被占；AFUN 的 post-contact 曲线
挤压阶段三叙事 → 叙事上移到 routing/合成/可微集成。

## 2026-07-30 · AFUN spike 完成，D1 修订为 C′

环境：`third_party/AFUN` + conda env `afun`（cu130）；SAM3 gated-repo 曾阻塞，token 和缓存在
`/workspace/huggingface_cache`（用户指出），symlink 接入解决。5 个 libero_spatial 任务、512² 真值深度、
`--no-refine`。结果：**空间指代 3/5**（between/drawer/next-to-plate 对；table-center、next-to-cookie-box
错——都拿显眼碗当答案，grounding 捷径），选对时 mask 距 GT ≤9px。预注册线压线过，但失败项正中 LIBERO
命门 → 冻结单用否决，**D1=C′：自训前端（仿真特权标签）+ AFUN mask 作辅助输入通道**；AFUN 转任
zero-shot baseline + v2 scale-up 故事。详情 `RoutedFlow/experiments/stage1_afun_spike/README.md`。

## 2026-07-30 · 统一标签提取完成（阶段一数据环节 ✅）

- 链路 v0.1 定稿（§2.7：4 修正 2 补全，flow 模块 text condition 移除防因果旁路）+ 学习模块清单 L1–L5（§2.8）+ 冷启动分析（PPI 同构，§0.3）。
- `RoutedFlow/run_stage1.py extract`：500/500 demos，一次零漂移重放 = C 标签 + 全帧位姿轨迹（flow 标签此后纯投影可得）+ t=0 三渲染 512²。
- QC 全过：montage 20/20 接触点落碗沿；被抓物体运动判定 10/10 正确；lift 系统偏差实测（悬空 11.6cm→robust 5.0cm，= §3.5 B2 的证据来源）；libero_spatial 示教单峰（§2.3 #2 关闭）。
- 下一步：L1 模型选型细化（C′ 的 backbone + AFUN prior 通道接法）→ L1+L2 开训 → A1/A2 验收。

## 2026-07-30 · 标签可视化 → mask 定向 bug（阶段〇作废）

- 四联板可视化（`run_stage1.py viz`）Panel C 颠倒 → 顺藤摸瓜：`get_camera_segmentation` 内置 `[::-1]`（与 raw render 相反），convert/eval 各多翻一次 → 阶段〇两视角路由标签全颠倒（IoU 0.004/0.000），train/eval 自洽但语义全错，§1.7 三变体结果作废。
- 修复：3 处代码 + 650 个 h5 离线翻回（IoU 复验 0.971/0.923）。阶段一 C 标签不受影响（不依赖 seg）。
- 暂缓判定的价值实证：如果当时下了结论，现在就是错误结论入库。
- 待决定：阶段〇重训（12.6h）vs 阶段一 L1 开训的 GPU0 排期。

## 2026-07-30 · pipeline 架构图 v0.1

- 论文级双分支训练图（SVG 矢量 + 真实标签数据嵌图）：`RoutedFlow/doc/pipeline_fig_v01.svg`；
  artifact https://claude.ai/code/artifact/12bf11ca-7294-40c6-acb1-69a18af26e5d
- 图内固化的设计要点：text 不进 L3（防旁路）、transport 无 action loss、C 头辅助监督、D7 课程、D6 未解项标注。

## 2026-07-30 · 阶段一全量代码落地 + 首轮训练（grill 会话拍板后执行）

- 拍板：范围=L1+L2 全量+L3/L5 骨架；L3 GT=提取器补 link 位姿（✅ 17×500）；D8=a 定稿；D10 新开（L4 吃 L3 预测+加噪）；A1 5-fold；A2 加 z_random 对照。
- 离线作业：DINO 特征 1G ✅；AFUN prior 500 张实测 **3 分钟**（原估 2–3h 是模型重复加载错觉）；**prior 正确率 in-mask 26% / ≤20px 58%**（C′ 判定全量坐实）。
- 代码：`stage1/` 七个模块 + `flow_models.py` 骨架 + 10 单测全过（轴约定数据锚定）。7.5M 参数，2s/epoch。
- fold0 首轮（未调参，勿下结论）：A1 val_id spike 任务 0.8；val_ood 0.33（between 0.66 / **table_center 0.0**——空间指代之墙，fallback 阶梯的观察点）；A2 contact：z 18.3px < dino 21.4 < random 28.9（序正确），yaw 尚弱；train 0.009 vs val best 1.76 = 强过拟合（360 样本 vs 7.5M，正则/增广是下一杠杆）。奇观察：a1_prior_wrong > a1_prior_correct（分层样本少，待 5-fold 再看）。
- 夜链启动：阶段〇修复标签重训 ×3 + n=400 eval ×3（旧作废 runs → `runs_invalid_maskbug/`）。

## 2026-07-31 · approach 分支全链联合训练启动

- 用户指令：写完整个 robot-flow 阶段（全可微）再端到端训练验证。grill 拍板 4 项；D3 定稿（z 占 text 槽）、D11 新增（FK 链点，mask 退役，QA 100% in-mask）、D12（双图像单 flow）。
- 全链代码落地 + 三层 smoke（单测/训练/rollout）当日全通；联合训练（L1 warm + L3 warm + L4 scratch，三 loss 可微贯通）运行中，~4h 收。
- 顺手发现的工程教训：atm 循环 import 陷阱、vilt train() 依赖 self.track、eval obs 键名映射。

## 2026-07-31 · 首次端到端 rollout：approach 分支成立

- **train-8 approach-SR 0.2875，ood-2 0.10**（joint ckpt ep~20，训练被 OOM 早停于 24/60）。全可微链路第一次驱动机器人完成 approach+抓取。
- 三轮诊断：渲染视口坑 → 过早闭合（D6 脆弱性实证：阶段〇迟疑 vs 这里过早触发，同一信号源的两种失败极性，论文动机素材）→ 鲁棒锁存后从 0 → 0.29。
- 下一轮杠杆：补完训练、λ_action 调度、latch 细化、C-VLM ood 弱项（与 stage-1 A1 ood 0.33 同源）。
