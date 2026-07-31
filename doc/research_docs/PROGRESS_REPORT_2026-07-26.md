# 进展汇报：从 GD-4D 到 Flow ≠ Affordance

**日期** 2026-07-26 · **作者** yd3007@nyu.edu
**在线版** https://claude.ai/code/artifact/fd736a95-fe5c-45d2-8d9b-8da984348b4f
**同目录 HTML（可直接双击打开 / 打印成 PDF）** `PROGRESS_REPORT_2026-07-26.html`

---

## 摘要

一天之内提出并检验了 **8 个 architecture / pipeline 方向，全部被 prior art 或机制问题否决**；
**2 个 measurement 方向通过检验**。最终收敛到一个由我自己提出、通过全部 prior-art 检查的方向，
环境已配好、baseline 已跑通。

| 指标 | 数值 |
|---|---|
| architecture 类 idea：提出 / 存活 | **8 / 0** |
| measurement 类 idea：提出 / 存活 | **2 / 2** |
| 实测 OpenReview 评审意见 | 785 条 / 200 篇论文 |
| ATM baseline 已复现（libero_spatial） | `success_env_avg = 0.62` |

---

## 00 · 起点：为什么放弃 GD-4D

GD-4D（goal-image 条件的 policy + 冻结 4D backbone 提供 disagreement 信号）在 2026-07-25 停止，
三个独立原因：

1. **核心前提被自己的实验证伪** —— 冻结 backbone 的几何一致性检测不出「梦到的目标是错的」，
   confound-free 的 delta = 0.51（等于随机）。
2. **pipeline 已是他人工作** —— GD-4D ≈ SuSIE（arXiv 2310.10639）+ AVDC（arXiv 2310.08576）。
3. **剩下的空隙分量太轻** —— 两篇都不验证生成的 subgoal，但「加一个 verifier」是 reviewer 会拒的
   间接方案。

> **代价：** 整个项目从建到停只用了一天（23 个 commit 全在 2026-07-24），但那一天是**先写代码、
> 后查 prior art**。今天的全部工作反过来做：**先查，再决定要不要写。**

---

## 01 · 先量化：reviewer 到底因为什么拒 VLA 论文

为了不再凭感觉判断「这个 idea 够不够」，接入 OpenReview API，抓了
**ICLR 2026 / NeurIPS 2025 / ICLR 2025 共 200 篇 VLA 相关论文的 785 条评审意见**
（含 132 篇被拒 / 撤稿的，那里信息量最大），做关键词归类并统计每类意见对应的评分偏移。

| 评审意见 | % 论文被提到 | 评分偏移 Δ |
|---|---|---|
| **novelty ——「这是已知方法的组合」** | 55% | **−0.89** |
| 没有 error bar / trial 数太少 | 27% | −0.56 |
| benchmark 太简单 / 提升在噪声内 | 17% | −0.46 |
| baseline 缺失或太弱 | 75% | −0.43 |
| 只有仿真 / 真机太薄 | 56% | −0.34 |
| 没展示 generalization / OOD | 77% | −0.02 |
| efficiency / latency | 42% | +0.02 |

**两个可操作的结论：**

1. **novelty 是唯一真正压分的意见，是第二名的两倍。**
2. 被提得最多的两条（generalization、efficiency）几乎不影响评分 —— 是仪式性意见。

（绝对百分比是关键词匹配，有噪声；**按 Δ 的排序才是可信的部分**。）

### 针对 3D 背景的专项统计

把标题含 3D / spatial / depth / point cloud 的 16 篇单独拆出来：
**接收率 31%（其余 34%）**，**novelty 意见 24.2%（其余 18.7%）**。
更关键的是 reviewer 现在会主动检查 3D 分支是否承重：

> "the real-world results show the Point Cloud policies **significantly underperform the RGB-only
> and RGBD variants**" —— EmbodiedMAE，评分 4

> "the ablations show that on CALVIN the performance is **similar between RGB-only or RGB-D**
> versions of their model" —— FALCON

同时：**评分 top-20 里有 7 篇是 benchmark / eval / infra 类，而它们在全样本里只占 12%。**
拥挤的是 architecture，不拥挤的是 measurement —— 这个梯度是实测出来的，不是猜的。

---

## 02 · 理论：3D 输入为什么常常不 work

- **Modality Competition**（ICML 2022 Spotlight, [arXiv 2203.12221](https://arxiv.org/abs/2203.12221)）：
  *证明* late-fusion 网络在梯度下降下各 modality 会互相竞争，encoder 只学到其中一部分，
  「输掉的」modality 的特征永远不会被发现 —— 所以联合训练可以**劣于**最好的单模态。
- **Gradient-Blending**（CVPR 2020）：不同 modality 过拟合 / 泛化速率不同，共用一套优化策略必然次优。
- **Probing the 3D Awareness of Visual Foundation Models**（CVPR 2024,
  [arXiv 2404.08636](https://arxiv.org/pdf/2404.08636)）：DINOv2 的 feature 本身已编码深度与法向；
  但 CLIP 系（含 SigLIP 家族）不行 —— 所以「3D 冗余」这个解释**依赖于用哪个 encoder**。

### 修正后的结论（不要过度概括）

**3D 在「小规模、from-scratch、显式 point cloud、几何主导的任务」上确实赢**
（Point Cloud Matters, NeurIPS 2024 D&B, [2402.02500](https://arxiv.org/abs/2402.02500)；
DP3, [2403.03954](https://arxiv.org/abs/2403.03954)，10 条 demo、72 个任务、+24.2%）。

**它失败的是「作为旁路挂在 web-pretrained VLM 上、在平面 tabletop benchmark 上评测」这一种情形。**

而且 Point Cloud Matters 明确发现：**2D depth map 无论单用还是与 RGB 融合都受限，起作用的是显式
3D 表示** —— 这正好否定了「加一个 depth 通道」这类做法。

---

## 03 · 击杀清单：8 个方向，全部否决

每条都在写任何代码之前完成 prior-art 检查，平均耗时数分钟。
标注「**（我提的）**」的是我本人提出的方向。

### K1 · VLA 表征的多视角一致性 — 否决：前提不成立

设想模型接了多个相机但内部并未融成一致的 3D 场景，可以测出来。
**问题在于「大家都假设 X」是假的** —— 领域内早已公认多视角融合是未解问题，并有一批方法在修它。
测量一个大家已经承认的问题，没有「therefore」。

> MV-Actor (2606.10899) · ReMAP-DP (2603.14977) · OmniD (2508.11898) · RoboTransfer · VistaBot

### K2 · 几何从输入端搬到输出端 — 否决：撞车

设想 VLM 只出 3D 目标，几何模型负责把目标解码成动作，避开输入端的 modality competition。**被完全覆盖。**

> ★ **OASIS** ([2605.25829](https://arxiv.org/html/2605.25829v1), 2026-05) —— 用 SE(3) 末端轨迹预测
> 对齐观测与动作空间，已报告优于 VLA / WAM baseline
> · **BridgeVLA** ([2506.07961](https://arxiv.org/pdf/2506.07961)) · GeoPredict (2512.16811)

### K3 · 等变性作为几何先验挂在冻结 VLM 上 — 否决：撞车

设想几何不作为数据输入，而作为 action head 上的对称性约束 —— 参数量近零，不参与梯度竞争。
**一个月前已发表，连「不改动 pretrained VLM 权重」这句都一样。**

> ★ **EquiVLA** ([2606.19784](https://arxiv.org/pdf/2606.19784), 2026-06) · LangPose
> · 邻近：EquAct ([2505.21351](https://arxiv.org/abs/2505.21351))、Equivariant Diffusion Policy (CoRL 2024)

### K4 · GeoBalance：诊断 3D 分支被「饿死」并用梯度再平衡修复 — 否决：撞车（含我设想的改进点）

设想 3D-VLA 的负结果不是表征问题而是优化问题 —— 3D 分支在前几千步输掉梯度竞争，用 OGM-GE 类方法
可恢复；我还想加「按阶段而非全局的平衡」作为新意点。
**诊断、场景、修复、连那个「按阶段」的改进全部已有**，而且出自 OGM-GE 原作者组。
CoRE-VLA 甚至已顺手报出 depth ablation：`97.6% → 97.0%`。

> ★ **GAP** — *When would Vision-Proprioception Policies Fail in Robotic Manipulation?*
> (gewu-lab, https://gewu-lab.github.io/GAP/) —— Gradient Adjustment with **Phase-guidance**
> · **CoRE-VLA** ([2607.03693](https://arxiv.org/pdf/2607.03693))

### K5 · 把 SAM2 / SAM3 redirect 成 policy —（我提的）— 否决：撞车 + 机制不成立

设想既然 GAM 能把 DA3 变成动作模型，SAM2 能精准分割并把首帧目标跟到末帧，应该也能 redirect。
**双重否决：**

1. SAM2Act 已在 ICML 2025 做了，连 memory bank 那部分都做了（RLBench 18 任务 86.8%）。
2. 机制上也不成立 —— mask 不在 action 空间里；且 SAM2 的 tracking 是**看得见未来帧的传播（插值）**，
   不是预测。

> ★ **SAM2Act** ([2501.18564](https://arxiv.org/abs/2501.18564), ICML 2025)
> · 对照：GAM ([2606.17046](https://arxiv.org/html/2606.17046))

### K6 · 把 6D pose 基础模型 redirect 成 policy — 否决：半撞车 + 架构不适配

「pose 当中间表示喂给 policy」已被占满；「GAM 式 redirect」没人做，但**做不了** ——
FoundationPose 是 render-and-compare 的迭代精化架构，没有可切开的 encoder→decoder latent，
且依赖 CAD / reference view 先验。

> ★ **SPOT** ([2411.00965](https://arxiv.org/abs/2411.00965), ICRA 2025)
> · PRISM-DP (2504.20359) · OMNI-PoseX (2604.02759) · DexMan (2510.08475)

### K7 · 把 optical / scene flow 模型 redirect 成 policy — 否决：撞车（本月）

按判据筛，flow 是最像 DA3 的一格：一次前向、稠密输出、输出是位移场（天然就是 action 空间）、
不需先验。**但 redirect 那一半也刚被占，论文是这个月挂的。**

> ★ **FlowWAM** ([2607.13017](https://arxiv.org/html/2607.13017), 2026-07) ——
> "In policy mode, the model generates future flow that is decoded into executable actions"
> · **FOFPred** ([2601.10781](https://arxiv.org/abs/2601.10781)) · RoboFlow4D (2605.17522)
> · FlowVLA (2508.18269)

### K8 · AdaJEPA × LaWAM：latent WAM + test-time adaptation —（我提的）— 否决：撞车

设想 LaWAM 会预测 latent subgoal，执行后能观测到真实 latent —— 「预测 vs 实际」就是白送的自监督
信号，正好补上 AdaJEPA 在 VLA 上缺的那一环。
**推理链没问题，但同样的洞见已被写成论文，还多加了一个 action-variance gate 来过滤归因不清的步骤。**

> ★ **T³VF** ([2605.08215](https://arxiv.org/abs/2605.08215)) ——
> "the predicted future image and its subsequent observation form a **natural supervision pair**"
> · EWAM (2606.12690) · AdaWorldPolicy (2602.20057)

### 最该带走的一条

把这些杀手按时间排：

```
2026-01  FOFPred
2026-05  OASIS, RoboFlow4D
2026-06  GAM, EquiVLA
2026-07  FlowWAM, CoRE-VLA        ← 本月
```

**七个月之内，「把某个 vision foundation model redirect 成 policy」这张表被逐格填满，
最后一格是这个月填的。** 而一个 idea 从构思到成文需要 3–4 个月 ——
**填格速度快于研究周期**。所以「在这张表里找空格」是结构上赢不了的，必须换搜索方式。

---

## 04 · 幸存下来的方向

三个通过检查的方向，全部是 measurement / falsification 类 —— 与第 01 节实测的拥挤度梯度一致。

| ID | 方向 | 状态 |
|---|---|---|
| S1 | **任务的「3D 必要性」度量** —— 量化每个 manipulation task 究竟需要多少 3D，并**用它预测**各论文报出的 RGB vs RGB-D 差距。VLM 的 spatial reasoning benchmark 很多（SpatialBench、3DSRBench、Spatial457），但没人度量 *task* 的 3D 必要性 | 通过检查 |
| S2 | **传感器噪声的剂量–响应曲线** —— 3D policy 的增益都是在仿真的完美深度下报的；用真实立体相机误差模型扫描，找出增益归零的临界点 | 通过检查（待读 1 篇） |
| S3 | **Flow ≠ Affordance**（我提的） | **已选定 · 已开工** |

---

## 05 · 选定方向：Flow ≠ Affordance

### 论证

1. 一批工作（ATM、Track2Act、Im2Flow2Act、General Flow、FlowPolicy、PPI…）把 point flow 称作
   *unified / embodiment-agnostic action representation*。
2. **但物体的 point flow 对 grasp 的选择是 invariant 的** —— 把杯子从 A 移到 B，抓把手还是抓杯沿，
   物体点的 flow 完全一样。**一个对 Y 不变的量不可能决定 Y。**
3. 所以 flow 提供不了 affordance，抓取信息必然另有来源：heuristic grasper、单独的 keypose
   interface、或隐式来自 BC 数据。
4. **但没有人算过这笔账** —— 聚合成功率区分不了「flow 帮我抓对了」和「flow 帮我搬对了」。

### 决定性证据：PPI 自己表里一个未被解释的数字

| 配置 | 成功率 | 增量 |
|---|---|---|
| Continuous only | 47.6% | — |
| + Keypose | 53.6% | **+6.0** |
| + Pointflow | 74.3% | +26.7 |
| **Both (PPI)** | 80.8% | **+6.5** |

**Keypose 单独加 +6.0，加在 pointflow 之上 +6.5 —— 交互项约等于零。**
若 pointflow 已部分提供 grasp 信息，叠加应当衰减。
PPI 的自述（「两个互补的 interface 增强空间定位」）解释不了这个零；
**grasp-invariance 恰好预测它。**

### 定位：什么是已知的，什么是我们的

| 命题 | 状态 | 出处 |
|---|---|---|
| flow 修不好抓取位置 | 已知，但只是 limitation 里一句话，**无量化无实验** | Track2Act failure mode；Im2Flow2Act "may not direct the policy to grasp the cube precisely" |
| 把 keypose 与 pointflow 拆成两个 interface | 已知，但只是**架构选择，作者未解释原因** | PPI, RSS 2025 ([2504.17784](https://arxiv.org/abs/2504.17784)) |
| **按阶段做功劳归因** | 无人做过 | — |
| **零交互 = grasp-invariance 的必然结果** | 无人提出 | — |
| **跨方法定量律** | 无人提出 | — |

> **写作定位（防 obviousness，而非 novelty）：** Abstract 第一句写**「交互项为零」这个数字**，
> 不要写「我们发现 flow 不提供 affordance」—— 后者读者会说「本来就是」。
> 定量结构没人猜得到，定性结论人人都觉得显然。

### 实验设计

把 episode 按 gripper 开合切成 **approach** 与 **transport** 两段，分别对 interface 做
**shuffle ablation**（不用 zero —— zero 会改激活统计量，掉分可能是分布 shift 的假阳性）：

| 组 | 条件 | 预期 | 测的是什么 |
|---|---|---|---|
| A | 完整 | baseline | — |
| B | approach 掐 flow | 几乎不掉 | flow 对「抓哪儿」有没有贡献 |
| C | transport 掐 flow | 大跌 | flow 对「往哪走」有没有贡献 |
| D | approach 掐 keypose | 大跌 | keypose 是否承担 affordance |
| E | transport 掐 keypose | 几乎不掉 | 职责是否互斥 |
| F | **全程掐掉（sanity）** | 必须明显掉 | **先看这组**，否则说明 hook 挂错 |

**无论结果朝哪个方向都有结论：** B 不掉 + C 大跌 → 「flow as action representation」是 overclaim，
它其实是 transport representation；B 也掉 → flow 确实隐含本体信息，那就要回答「它是怎么隐含的」，
同样成篇。

### 三块内容

1. **归因** —— 在 PPI 与 ATM 上做分阶段 ablation，把 +6.0 / +26.7 拆开（**Figure 1**）
2. **机制** —— 跨方法定量律：*approach 阶段的贡献量 ∝ 追踪点集中落在机器人本体上的比例*（**Figure 2**）
3. **可预测性** —— 从任务的阶段构成预测加 contact interface 的收益，不跑实验就能预测（**Figure 3**）

---

## 06 · 今日实际进度

### G0.1 ✅ ATM 的采点方式（读代码确认，非推测）

- `preprocess_libero.py:127` — `sample_from_mask(np.ones((H,W,1))*255, 1000)`，mask 全 255，
  **整幅图无任何 robot masking**。
- `preprocess_libero.py:96-110` — variance filter **保留高方差（会动）的点**。LIBERO 里会动的只有
  机械臂（全程）与被操作物体（抓后），静态背景被丢掉 → **保留点系统性偏向机器人本体**。
- 两个视角之一 `robot0_eye_in_hand` 是腕部相机，画面很大一块恒定是夹爪。
- policy 端 `sample_tracks_nearest_to_grids(num_samples=32)` 仍是纯几何采样，无语义筛选。

> **意外收获：ATM 内部自带一组对照** —— `eye_in_hand`（几乎全是机器人）vs `agentview`
> （机器人只占一部分）。若定量律成立，approach 阶段掐掉前者应显著比掐后者更疼。
> **这是组内对照，比跨方法对比更干净、更便宜，而且不依赖 PPI。**

### G0.2 ✅ PPI 侧

- checkpoint、数据、**Table VII 的四个 ablation 脚本全部已开源**（`scripts/ablation/`，
  靠 `what_condition` / `predict_point_flow` 一行开关切换）→ +6.0 / +6.5 可复现。
- **keyframe 定义可直接当阶段边界**：`_keypoint_discovery_bimanual` 在 `gripper_open` 变化时触发，
  且左右手分别追踪 —— 双臂不同步的问题作者已替我们解决。
- PPI 的采点是 GroundingDINO → SAM → `sample_farthest_points` **在物体 mask 内部**
  = **纯物体、零机器人**，与 ATM 构成 x 轴两个极端，两端均由代码证实。
- ⚠️ **风险**：eval 栈重 —— CoppeliaSim 4.1 + PyRep + RLBench2 + pytorch3d + SAM/GDINO 权重，
  本机全无，估计 2–4 天。

### ATM baseline ✅ 端到端跑通

`success_env_avg = 0.62`（libero_spatial，10 tasks × 5 rollouts，单张 RTX 5090）。
踩到并修好六个坑，全部固化在 `run_eval_gpu0.sh`：

| # | 坑 | 处理 |
|---|---|---|
| 1 | `environment.yml` 钉 `python=3.8.18` + `torch==2.0.1`，早于 sm_120 | 基座换 py3.10 + torch 2.11+cu128 |
| 2 | eval 脚本硬编码 8 卡（`[0,1,2,3]`/`[4,5,6,7]`） | 新 launcher，全部 `[0]` |
| 3 | `~/.libero/config.yaml` 是**全局共享**的，指向另一个项目且缺键 | **未覆盖**，改用 `LIBERO_CONFIG_PATH` 建项目本地 config |
| 4 | torch ≥ 2.6 的 `weights_only=True` 打死 LIBERO init_states | 三处 `torch.load` 补 `weights_only=False` |
| 5 | lightning ≥ 2.3 要求登记非 forward 方法 | `model.mark_forward_method("act")` |
| 6 | eval 似乎需要 CoTracker 预处理 | **其实不需要** —— 只要 `env_meta.json`，写脚本直接生成，**省掉 1000 点 × 2 视角 × 40 task 的预处理** |

> **排期约束：** 10 tasks × 5 rollouts × horizon 600、单卡 ≈ **25–30 分钟**；
> 按论文默认 20 rollouts 外推 → **一个 suite 一个条件 ≈ 2 小时**；
> Phase 2 六组条件 × 3 seeds = 18 次 → **单 suite 约 36 GPU 小时**。
> 目前只有 GPU0 可用（GPU1 被占），主实验建议只做一个 suite。

---

## 07 · 下一步

1. **立刻**：在 `atm/policy/vilt.py` 上挂 hook，**先只跑 F 组**（全程 shuffle）作为 sanity check。
   半小时内就能知道 hook 是否生效。
2. 跑通后铺开 ATM 的组内对照（`eye_in_hand` vs `agentview` × approach/transport）→
   **Figure 1 的第一版，不依赖 PPI**。
3. 并行推进 PPI 的 CoppeliaSim 栈（2–4 天），用于跨方法定量律的另一端。
4. 写作前须做的两件事：精读 MiraBench (2605.29360) 与 Foresight (2606.23085)；
   扫一遍 ATM / Track2Act / PPI 的引用图确认无新撞车。

### 流程上的改变（这是今天最持久的产出）

1. **先查 prior art，再写代码** —— 今天 8 个方向的否决平均只花几分钟，
   而 GD-4D 那次同样的碰撞是在 23 个 commit 之后才发现的。
2. **最快的查重不是搜索引擎** —— 是读该方向最新一篇论文的 **baseline 表**（谁占了这格）和
   **limitation 节**（作者自己知道什么毛病）。两者合计 3 分钟，比几十次关键词搜索都准。
3. **idea 的形状要是「大家把 Y 归功于 X，但 X 对 Y 是不变的」**，而不是「A + B」。
   前者今天 2/2 存活，后者 0/8。

---

## 附：本地文件索引

| 文件 | 内容 |
|---|---|
| `/workspace/research/d4rt/PROGRESS_REPORT_2026-07-26.md` | 本文档 |
| `/workspace/research/d4rt/PROGRESS_REPORT_2026-07-26.html` | 同内容的 HTML，可双击打开 / 打印 PDF |
| `/workspace/research/d4rt/FLOW_AFFORDANCE_LOG.md` | 运行日志（每天的数字、踩坑记录） |
| `/workspace/code/ATM/run_eval_gpu0.sh` | GPU0-only 的 ATM eval launcher |
| `/workspace/code/ATM/setup_atm5090.sh` | 可复现的环境配方 |
| `/workspace/code/ATM/scripts/make_env_meta_only.py` | 跳过 CoTracker 只生成 `env_meta.json` |
| conda env `atm5090` | py3.10 + torch 2.11+cu128（sm_120 可用） |
