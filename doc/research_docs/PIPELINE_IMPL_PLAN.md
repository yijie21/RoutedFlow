# Phase-Gated Flow Routing · 分阶段实现计划（进度追踪文档）

> 创建：2026-07-29 · 本文档是实现阶段的**进度追踪主文档**——每完成一项就更新对应的状态框和「更新记录」表。
> 设计文档（理论/证据/进化史/pipeline 总图）：`PHASE_GATED_FLOW_ROUTING.md`
> 实验日志：`FLOW_AFFORDANCE_LOG.md` · 先前测量结果：`PROGRESS_REPORT_2026-07-28.md`

---

## 更新记录

| 日期 | 更新内容 |
|---|---|
| 2026-07-29 | 文档创建；四个阶段的计划 + 待决定事项表 + kill criteria 初版 |
| 2026-07-29 | 阶段〇重写为可执行级别：核实 ATM 代码（BC 不用 GT track，干预点在 `vilt.py:track_encode` 的 `reconstruct` 之后）；明确三变体 gating 定义、训练 config、四块待写代码、执行顺序 |
| 2026-07-29 | 阶段〇全部代码落地于 `/workspace/code/RoutedFlow`（单测 8/8、mask 交叉验证 9%/21%、OOM×2 修复定稿 batch 32×accum 4）；3×101 epochs 训练 + eval 接线 + n=400 eval 链全部执行 |
| 2026-07-30 | **阶段〇结果记录 + 暂缓判定**（用户指示：单模块不足以下全局结论）；见新增 §1.7 |
| 2026-07-30 | 阶段一设计定向：**两因子分解**（接触点 heatmap × 朝向/开度）写入 §2.2；坐标系约定写入 §2.1；文献 sweep 更新（AFUN / Affordance-R1 等）写入 §2.6；D1 选型从二选一改三选一（新增 AFUN 冻结前端案，等 spike 定稿） |
| 2026-07-30 | AFUN spike 执行完毕（空间指代 3/5，选对时 mask ≤9px）→ **D1 修订为 C′**（自训前端 + AFUN mask prior 通道）；详情 `RoutedFlow/experiments/stage1_afun_spike/` |
| 2026-07-30 | **训练策略修订（用户提出）**：staged 预训 → end-to-end 联合微调 + auxiliary losses（C head / flow 为辅助监督，action 为主监督），配套因果诊断（latent probe + 干预测试）；写入 §0.3，决策表新增 D7 |
| 2026-07-30 | 阶段三两处理论澄清（回应用户担忧）：§4.1.5 监督来源（物体 flow 无需人工标注——特权状态 / **GT action 经恒等式无损变换** / tracker 伪标签）；§4.3.2 刚体假设边界细化（晃动=闭环吸收+免费打滑检测；铰接物体=T(t) 取被抓链节即成立，flow 天然编码铰接约束；真边界=非抓握操作，v1 排除） |
| 2026-07-30 | **功能性链路 v0.1 审查**（用户提出链路，含 LLM decomposer 占位）写入 §2.7：4 修正（C 监督用 E(t_g) 状态非 action / C head 只补 (R,w) 平移由 heatmap 导出 / **flow 模块 text condition 移除**防因果旁路 / 初始帧→当前帧闭环）+ 2 补全（decomposer 每原子恰一个 t_g 的合同；查询点须覆盖 gripper 且铺开 + flow 带 3D）；§3.2 conditioning 表同步修订 |
| 2026-07-30 | 链路 v0.1 用户确认；**学习模块清单摘出**写入 §2.8（5 个 learned：L1 前端 / L2 C head / L3 robot-flow / L4 action expert / L5 object-flow，各配候选填充方法 + D7 curriculum 映射 + text 不对称设计要点）；决策表新增 D8（flow 3D 化）D9（L3/L5 共享 backbone） |
| 2026-07-30 | **latent condition 冷启动分析**（回应用户担忧）写入 §0.3：v1 课程表 by construction 消除；从零联训也可行（内部激活 + anchor loss 快收敛，PPI arXiv 2504.17784 实证：无 staging，接口 loss 权重 20×）；真风险 = latent 漂移 + 下游忽略，防护已在位；λ 启示进 D7 |
| 2026-07-30 | **监督信号总表**写入 §0.3（三套 loss 全免费：L_C 特权重放 / L_flow FK+物体位姿投影 / L_action demo 直用；一次重放全部标签）；**统一标签提取器落地并开跑**（`RoutedFlow` `run_stage1.py extract`，500 demos；smoke 验证：接触点正落碗沿、lift_gap ~3cm 系统偏差实测、obs 子帧时序差 9mm 与标签无关） |
| 2026-07-30 | **标签提取 500/500 完成 + QC 全过**（§2.5）：被抓物体运动判定 10/10 任务正确；lift 偏差实测（悬空抓取单像素 11.6cm → 窗口 robust 5.0cm，B2 证据）；多峰性直查 = 单峰（§2.3 #2 经验回答）。下一步 = L1 模型选型细化 + 开训 |
| 2026-07-30 | **⚠ mask 定向 bug**：标签可视化暴露 `get_camera_segmentation` 本身返回正立图，convert/eval 各多翻一次 → **阶段〇两视角路由标签全颠倒，§1.7 结果作废待重跑**（IoU 实测 0.004→修复后 0.971）；三处代码修正 + 650 个 h5 离线修复完毕；详见 §1.7 追加块 |
| 2026-07-30 | **L1–L5 复用度评估**写入 §2.8：真设计只有 L1；L2/L4 ≈0（L4=阶段〇 `BCViLTPolicyGated` 换 flow 源）；L3=ATM 小改造（D3 注入 + mask 查询）；L5 最接近现成（ATM 原生 text conditioning，GT 可从 c_labels 纯投影导出）；几何链纯实现+单测。唯一数据缺口：L3 的 link 位姿（倾向扩展提取器补存） |
| 2026-07-30 | **L1 设计选项全清单**写入 §2.2（5 案：a=C′ 具体化推荐 / b=VLM+LoRA 留 v2 / c=CLIPSeg 降 baseline / d=检测两段式排除（不可微断 latent）/ e=全共享 ATM backbone 排除）+ a 案 v1 规格 + fallback 阶梯预注册 |
| 2026-07-30 | **AFUN prior 角色澄清**写入 §2.2（回应用户）：叶子输入不断可微（梯度路径 = loss→L4→L3→z→L1 可训参数，完整）；上限不被卡（监督来自特权标签，AFUN 错时标签仍对，融合可学 override；最坏退化为噪声通道）；预注册保险 = channel dropout + spike 失败任务 A1 单独达标 |
| 2026-07-30 | **阶段一训练 pipeline 数据流定稿**写入 §2.2（a 案）：3 个离线一次性作业（标签✅ / AFUN 推理 2–3h / DINO 特征分钟级）→ 预计算特征训练（秒级 epoch，~20M 可训参数）→ L_C 三项 loss → split 协议（8 任务训 / 2 整任务 held-out）+ A1 id/ood + A2 probe + channel-dropout ablation |
| 2026-07-30 | **AFUN mask 错误处置**写入 §2.2：软特征非门控（无框死失效模式）/ 训练时错 mask=教 override 的解药 / **prior 正确率量化 QC（预注册，②作业后即测）**+ A1 按 prior 对错分层 / channel 可零成本摘除（dropout 保底 + ablation 决定去留） |
| 2026-07-30 | **阶段一全部代码落地**（grill 会话拍板：范围=L1+L2 全量+L3/L5 骨架、L3 GT=扩展提取器、D8=a 定稿、GPU=先 AFUN 后重训；新开 D10=L4 吃 L3 预测+加噪）。离线作业②③完成（**AFUN 500 张实测 3 分钟**；prior 正确率 26%/58%，C′ 坐实）；link 位姿 17×500 补存；L1+L2 7.5M 参数 + 10 单测 + smoke 全链路通；5-fold/prior 分层/z_random 对照进 A1/A2 协议；L3/L5 骨架（注入块 identity-at-init）。fold0 正式训练启动 |

---

## 0. 总览

### 0.1 Pipeline 一句话回顾

按接触时刻 $t_g$（夹爪闭合）把任务切成两半，**flow 的语义按阶段路由**：

```
接触前（approach）:  图片+指令 ──> C 的概率分布          [阶段一]
                    C-latent + 条件 ──> 机器人 flow ──> actions   [阶段二]
—— 夹爪闭合，C 冻结 ——
接触后（transport）: 物体 mask+指令 ──> 物体 flow ──> 解 IK ──> actions  [阶段三]
```

### 0.2 阶段划分与依赖关系

| 阶段 | 内容 | 依赖 | 状态 |
|---|---|---|---|
| **阶段〇** | 守门实验：BC-only 因果测试（路由前提是否成立） | 无（全部基建已就绪） | ✅ 已执行（2026-07-30）·**判定暂缓**，见 §1.7 |
| **阶段一** | C 分布预测头：图片+指令 → P(C) | 无（可与〇并行做数据部分） | 🔄 设计定向中（分解式设计 + 选型三选一，见 §2） |
| **阶段二** | approach：robot flow 预测 → action 生成 | 阶段一（要它的 latent） | ☐ 未开始 |
| **阶段三** | transport：object flow → SVD → IK | 无（可与一/二并行） | ☐ 未开始 |
| **阶段四** | 整合：phase switch 推理时接线 + 端到端评测 | 一+二+三 | ☐ 未开始 |

**并行策略（GPU0 单卡）**：阶段〇先跑（最便宜、能一票否决整个方向）；跑训练的空档做阶段一的数据提取和阶段三的纯几何模块（后者几乎不吃 GPU）。

### 0.3 训练策略（2026-07-30 修订：end-to-end 联合训练 + auxiliary supervision）

> 动机（用户 2026-07-30 提出）：end-to-end VLA 靠整体训练拿高 SR；分块冻结的 pipeline 会在接口处损失成功率，且「模块化」不该以 SR 为代价。

- **终态**：共享 backbone → C head（辅助监督）→ C-latent → flow predictor（辅助监督）→ action head（主监督），L = λ₁·L_C + λ₂·L_flow + λ₃·L_action，全程可微联合训练。PPI 同构（其 pointflow/keypose 也是辅助输出）。
- **课程表**：① 先训 C head（latent 语义初始化）→ ② 加 flow 分支 → ③ 全 unfreeze + action loss 联合微调。每步 checkpoint = 论文 ablation 行。原计划的「冻结 latent 分块训」是①②阶段的归因脚手架，不是产品形态。
- **必须配套的因果诊断**（教训来源：阶段〇三变体 BC loss 无差别 + 用户对伪泛化的担忧对我们自己同样成立——辅助 head 的 loss 低 ≠ latent 被下游使用）：A2 latent probe（必要条件）+ **eval 时干预测试**（换/坏 C-latent → SR 必须可预测地掉，充分条件）。这也是对黑盒 VLA 的差异化论点：同等 SR + 中间量的因果证据。
- **边界**：阶段三（SVD+IK 纯几何，无参数）不进联合训练；D6 的 phase exposure bias 在联合训练中原样存在，phase 加噪/scheduled sampling 是配套项。
- **数据侧推论**：robot flow 的 GT 无需 CoTracker——仿真里机器人 link 表面点经正运动学逐帧投影 = 零噪声真值 track（复用阶段〇重放机器）。

**Latent condition 的冷启动问题（2026-07-30，回应用户担忧「训练初期 C-latent 是混沌，怎么当 condition」）**：

- **v1 不存在此问题（by construction）**：课程表 ① 先把 L1+L2 预训到 A1/A2 验收，latent 定型后 ② 才挂 L3（v1 甚至冻结 L1，§3.4）。联合微调 ③ 从预训权重出发，latent 起点不是混沌，auxiliary L_C 锚住它不漂移。
- **即使从零联训也不致命**：latent 是网络内部激活而非外部固定输入——「上游特征在初期对下游是噪声」是任何深度网络层间边界的标准动力学；有监督的 anchor loss（L_C 是回归/分类，比 policy 任务好学得多）使 latent 远快于下游收敛，下游 attention 早期自动降权该通道、随后 co-adapt。
- **PPI（arXiv 2504.17784, RSS'25）实证同构可行**：单一 diffusion transformer，keypose/pointflow 接口 token 与 action token 同网联训，**无 staging、无 scheduled sampling**；GT 只进 loss 不进 condition（action token 经单向 attention 读接口 token 的 hidden state = 我们的 latent conditioning）；loss 权重 L = 0.05·L_c + 0.05·L_k + **1·L_F**——接口 loss 权重 20 倍，保证接口先收敛。RLBench2 +16.1%。**λ 启示（D7）：初期 λ_C, λ_flow ≫ λ_action**。
- **真正要防的不是冷启动**，是两个别的：(a) 联训后期 action 梯度把 latent 重塑漂离 C 语义 → auxiliary loss + ③ 阶段低 lr + A2 复测；(b) 下游学会永远忽略 latent → 已由 §2.7 修正 #3（L3 不吃 text，latent 是唯一任务通道）+ 干预测试防住。**PPI 自己有 (b) 的暴露面**（其 action token 同时 attend scene/language token，可绕过接口；只做了 ablation 没做因果诊断）——这是我们对它的差异化点之一。
- **反面教材**：teacher forcing（训练时把 GT C 值直接当 condition 喂下游）会引入 exposure bias（D6 同族病），不采用；PPI 也没采用。

**监督信号总表（2026-07-30，回应「监督不止 C 吧」）**——三套 loss，全部免费，全部出自同一批 raw demos 的一次重放：

| 模块 | Loss | 标签来源 | 激活（curriculum） |
|---|---|---|---|
| L1+L2 前端+C head | L_C = heatmap KL + bins CE + 开度 L1 | 特权重放 E(t_g)（提取管线见 §2.5 / `RoutedFlow` `extract_c_labels.py`） | ①②③ 全程 |
| L3 robot flow | L_flow^robot（track MSE） | FK/重放：link 表面点逐帧投影，零噪声 | ②③ |
| L5 object flow | L_flow^obj | 物体特权位姿逐帧投影（§4.1.5 来源①） | ②③ |
| L4 action expert | L_action（MSE v1） | demo GT actions 直接用 | ③ |

工程推论：**一次重放、全部标签**——统一提取器落盘 t=0 渲染 + 全帧位姿轨迹 + 相机参数，此后任何 flow 标签都是离线纯投影，不再进仿真。

### 0.4 环境与不变约束

- GPU：只有 GPU0（`CUDA_VISIBLE_DEVICES=0`），GPU1 被占。
- 代码基座：`/workspace/code/ATM`（track transformer + BC 头 + 完整 eval 机器，A_full=0.710 参照系）。
- 数据：LIBERO（robosuite 仿真）。仿真里深度图（`camera_depths=True`）和机器人/物体分割（`segmentation=True` 重放）都是**免费真值**——v1 全部用真值 mask，SAM/GroundingDINO 属于 scale-up 故事，不进 v1。
- `~/.libero/config.yaml` 是全局共享的，**不许改**，用 `LIBERO_CONFIG_PATH`。

---

## 1. 阶段〇 · 守门实验（先跑，别跳过）——详细版

**为什么放在实现计划里**：它就是整个 pipeline 的最小可行性证明——不训练任何新模块，只用现成的冻结 track transformer + 3 份重训的 BC policy，验证「按阶段路由 flow 语义」这件事本身有没有因果收益。如果它失败，阶段二/三的新模块全都不用写。

### 1.0 两个容易搞混的概念（先说清楚）

1. **不需要"准备"一个 point flow prediction transformer。** ATM 自带训好的 track transformer checkpoint（`results/track_transformer/libero_track_transformer_libero-spatial/model_best.ckpt`），阶段〇全程**冻结**它。它预测的是**任意查询点**的未来轨迹，所以机器人通道和物体通道用的是**同一个** transformer——区别只在"哪些点的预测被送进 BC policy"。这正是阶段〇便宜的原因：唯一要训的是 BC policy（~几百万参数的小模型），track transformer 零训练。
2. **gripper 闭合的 action 不是 C，也不用来"当成预测的 C"。** 阶段〇里**没有任何东西在预测 C**——C（`inv(T_obj) @ T_site` ∈ SE(3)+开度）根本不出现在这套实验里，预测 C 是阶段一的事。gripper 闭合信号在这里只有一个角色：**phase 切换的时刻标记 $t_g$**（也就是 C 被冻结的那一瞬间）。训练时从 demo 的 action 序列里读（`actions[:, -1] > 0` 首次为真后锁存），推理时从 policy 自己输出的 gripper 维度读（同样锁存）。

### 1.1 代码现状（已逐行核实，2026-07-29）

ATM 的 BC 数据流跟直觉不一样，干预点由此决定：

- **BC 训练不用数据集里的 GT track。** `atm/policy/vilt.py` 的 `forward()` docstring 明说：`track: not used for training, only preserved for unified interface`（L380）。dataloader 返回的 GT track 只喂给可视化。
- **真实数据流**：每个 forward（训练和推理相同），`track_encode()`（`vilt.py:240-274`）在当前帧上取一个**固定的 32 点 double grid**（`sample_double_grid(4)`，确定性的归一化坐标，L255），连同 BERT task embedding 一起喂给冻结的 track transformer（`self.track.reconstruct`，L262，`torch.no_grad`），得到每个视角、每个时间步、32 个点、未来 16 步的预测 track。
- **预测 track 被用了两次**：① 经 `track_proj_encoder` 变成 token 进 spatial transformer（L271）；② 原始坐标直接 concat 到 temporal feature 上进 policy head（L386-387）。masking 必须在 `reconstruct` 返回之后立刻做，两条使用路径就都覆盖了。
- **augmentation 不影响标注**：dataloader 里 `obs` 和 GT track 会被增广，但喂 track transformer 的 `track_transformer_obs` **不增广**（`bc_dataloader.py:26-28`）——所以在原始帧上算的 mask 标签始终有效。

### 1.2 三个 BC 变体到底是什么

三份 **BCViLTPolicy** 的完整重训。架构、超参、数据、seed 协议全部相同，唯一区别是插在 `track_encode()` 里的一个 **gating 操作**——决定 32 个预测 track 里哪些保留信息：

| 变体 | approach（闭合前） | transport（闭合后） | 对应假设 |
|---|---|---|---|
| (i) object-only | 只留物体/场景点，机器人点抹平 | 同左 | flow 的价值全在物体运动 |
| (ii) robot-only | 只留机器人点，物体点抹平 | 同左 | flow 的价值全在机器人运动 |
| (iii) phase-switched | 只留机器人点 | 只留物体点 | **路由假设（我们的设计）** |

- **点的标注**：每个时间步，把 32 个固定 grid 点的像素坐标查当前帧的机器人分割 mask——落在机器人上的是"机器人点"，其余是"物体/场景点"。标签逐帧变化（机器人在动），不是只标 t=0。预期比例校验：wrist 视角约 21% 机器人像素、agentview 约 9%（测量阶段的数字）。
- **"抹平"操作（v1 定义）**：被排除的点，预测 track 替换成"查询点原地重复 16 步"（零位移，静止点）。选它的理由：静止点在自然数据里本来就存在（背景），比 ATM 自己 ablation 用的 `use_zero_track`（坐标清零到角落）更接近分布内。
- **freeze 之死的教训已内建**：上次 freeze 干预失败是因为"训练时见真 track、评测时喂假 track"的 train/eval 不一致。这次 gating 在**训练和评测用同一套操作**，policy 学的就是这个分布，没有 OOD 问题。
- **已知泄漏（记为 limitation）**："哪些点被抹平"这个 pattern 本身以 32 点的粗粒度泄漏机器人轮廓位置（但不泄漏运动）。若 margin 可疑，robustness check：训练时把被排除点替换成 batch 内其它样本的 track（shuffle-train），连 pattern 也毁掉。

### 1.3 训练的输入 / 输出 / 配置（原样照抄 ATM，零改动）

**Config**：`conf/train_bc/libero_vilt.yaml`。要点：

| 项 | 值 |
|---|---|
| 优化 | AdamW，lr 5e-4，weight_decay 1e-4，CosineAnnealingLR，batch 128，**epochs 101**，clip_grad 100 |
| 输入 | 2 视角（agentview + eye_in_hand）RGB 128×128 × frame_stack 10 帧；BERT task embedding（768d）；extra_states = `joint_states` + `gripper_states`；track_obs（喂 track transformer 的帧） |
| flow | num_track_ids 32（固定 grid），num_track_ts 16（未来 16 步），冻结 track transformer 现场预测 |
| 网络 | PatchEncoder(8×8, embed 128) → spatial transformer(7 层 8 头) → temporal transformer(4 层 6 头) → DeterministicHead(MLP 1024×2) |
| 输出 | 7 维 action（6 pose + 1 gripper），**MSE loss**（`policy_head.py:41-43`） |
| 数据 | `data/atm_libero/libero_spatial/<task>_demo/bc_train_10`，10 任务 × 10 demos |

**启动命令**（照 `scripts/train_libero_policy_atm.py` 的模板，改单卡 + 加 gating 参数）：

```bash
python -m engine.train_bc --config-name=libero_vilt train_gpus="[0]" \
  experiment=stage0_variant_iii_phase_switched \
  train_dataset="[...bc_train_10 列表...]" val_dataset="[...val 列表...]" \
  model_cfg.track_cfg.track_fn=./results/track_transformer/libero_track_transformer_libero-spatial/ \
  model_cfg.track_cfg.use_zero_track=False \
  model_cfg.track_gate_cfg.mode=phase_switched \    # ← 新增的唯一配置项
  seed=0
```

Seed 协议：v1 每变体先跑 seed=0（共 3 个 101-epoch run）；若 iii 对 max(i,ii) 的 margin < 5 pts 再补 seed 1、2。

### 1.4 需要写的代码（就这四块）

1. **mask 重放脚本**（新，~1 天）：对每条 demo 用 robosuite `segmentation=True` 重放，渲染每帧、每视角的机器人分割；直接落盘两样东西：全分辨率 bool mask（备用）+ **32 个 grid 点的逐帧标签数组 `(T, V, 32)`**（训练用的就是它）。
2. **gating 模块**（新，~半天）：`track_gate_cfg = {mode: none|object_only|robot_only|phase_switched}`，插进 `track_encode()` 的 `reconstruct` 之后；输入 = 预测 track + 逐帧点标签 + phase 位。`mode=none` 时严格恒等（保证与 A_full 兼容）。
3. **dataloader 改动**（~半天）：`bc_dataloader.py` 的 `__getitem__` 额外返回点标签切片和 phase 位（phase 从 `demo["root"]["actions"][:, -1] > 0` 锁存预计算，每条 demo 一个一维数组）。
4. **eval 侧接线**（~1 天）：eval 环境开启在线分割渲染（robosuite `camera_segmentations`），逐步算 32 点标签；phase 位由 policy 自己输出的 gripper 维度锁存。误切换率（闭合但没抓到）单独计数上报。

### 1.5 执行顺序（sanity check 优先，吸取 freeze-lockup 教训）

1. ✅ 2026-07-29 BC 单 epoch 墙钟 = **~150s**（OOM 两次后定稿 batch 32 × grad_accum 4 = 等效 128，峰值 21.1G/31.4G；教训：ATM 的 batch 128 是 4×A100 每卡的量）
2. ✅ 2026-07-29 mask 重放 + 标签可视化（`sanity_overlays/` 人工过目通过；**比例交叉验证：agentview 0.088 / wrist 0.217 vs 测量阶段 9%/21%**；replay RGB diff 2.9/6.9 远小于错误翻转的 52/62）
3. ✅ 2026-07-29 gating 单测 8 项全过（`tests/test_stage0_units.py`，含 grid 与 ATM 逐元素相等、(i)/(ii) 互补、(iii) 闭合前后语义、静止替换）
4. ✅ 2026-07-29 3 × 101-epoch 训练完成（GPU0 串行 12.6h；三者 train loss 几乎相同 0.0034–0.0037，**iii 的 val loss 最低 0.0327**——BC loss 区分不出变体，判定全在 rollout）
5. ✅ 2026-07-30 n=400 eval × 3 完成（eval 接线含在线分割 `RobotGridLabelWrapper` + phase 自锁存；实弹 smoke 先行验证）
6. ☐ 重扫 arXiv 确认 phase-switch 接线仍无人占（上次确认 2026-07-28）——**移到论文写作前执行**

**代码状态（2026-07-29 完成）**：§1.4 四块全部完成并验证。实现在 `/workspace/code/RoutedFlow`（结构和用法见其 README；实验主页 `experiments/stage0_routing_causal_test/`，含状态板和逐日 changelog）。

### 1.6 预注册预测与 kill criterion（不变）

- **P1**: iii ≥ i；**P2**: iii ≥ ii；**P3**: iii ≈ A_full 0.710（路由不该丢掉整体性能）。
- **Kill**：iii − max(i, ii) < 3 pts → 路由无因果收益，整个 pipeline 方向重审。
- 顺带的免费读数：(i) vs (ii) 的差重现测量阶段的 Bw/Ba 结论（approach 价值住在机器人点里）——训练侧的独立复证。

### 1.7 执行结果（2026-07-30）与暂缓判定

**结果（n=400，10 任务，`runs/<mode>_seed0/eval_model_final_n40.json`）**：

| 变体 | SR | val loss（epoch 100） |
|---|---|---|
| (i) object_only | 0.630 | 0.0360 |
| (ii) robot_only | 0.6525 | 0.0383 |
| (iii) phase_switched | **0.1975** | **0.0327（最低）** |

配对 t（iii vs max(i,ii)，10 任务 df=9）：diff = −0.503，t = −9.57，**p = 5.1×10⁻⁶**。表面判定：P1/P2/P3 全部不成立，kill criterion 触发。

**⏸ 判定暂缓（2026-07-30 用户指示）**：单模块（BC-only 因果测试）不足以对整个 pipeline 下结论——阶段〇只训练了 flow→action 一环，其余环节（C 头、flow predictor 改造、几何解算）都还不存在。留待 pipeline 全貌后回看。

**已记录的机制线索（不作结论，只作追溯）**：
1. iii 的 latch 统计异常：首次闭合中位 **160 步**（demo 真值 45、(i)/(ii) 评测 63–68），**27% episode 从不闭合**，0% 过早闭合。失败模式 = 迟疑不闭合，不是抓错。
2. val loss 最低 + rollout 崩盘的组合 → 疑似 train/eval 失配而非能力缺失。候选机制：**phase 信号在部署时自指**——训练时 gate 由 demo 真值 phase 驱动（policy 可把「gate 切换」当阶段提示的捷径），评测时 gate 等 policy 闭合、policy 等 gate 切换，互相等待。freeze-lockup 教训的深化版：这次不是标注 bug，而是 BC 用 oracle phase 训练带来的 exposure bias。
3. 候选诊断（未跑）：iii 重评测但 phase 强制在第 44 步锁存（demo 均值，~1.2h）。SR 恢复 → 路由内容成立、只是切换信号来源问题（修复向：训练时 phase 加噪 scheduled-sampling / 评测用 gripper 状态或接触检测等物理信号）；SR 不恢复 → 负结果坐实。
4. 免费读数：(i) 0.630 vs (ii) 0.6525，逐任务差方差大、不显著——**训练侧砍通道后 policy 能从 RGB+proprio 补偿**（与评测侧 shuffle 的 Bw −0.095 显著不同：那测的是「已训好的 policy 依赖什么」，这测的是「没有该通道能不能学会」——两个问题，都是定理③冗余/泄漏故事的证据）。

**❌ 2026-07-30 追加：mask 定向 bug 发现，上表结果作废（须重跑）**。阶段一标签可视化时发现 robosuite 的 `get_camera_segmentation` **本身返回正立图**（内置 `[::-1]`，与 raw `sim.render` 相反）；convert 又按 rgb 的 flip 选择翻了 mask → **两视角的 robot_seg / grid_labels 全部上下颠倒**（实测 IoU(存盘, 正立真值)：agentview 0.004 / wrist 0.000；翻回后 0.963/0.902）。eval 侧 `RobotGridLabelWrapper` 同一假设 → 同样颠倒——train/eval 自洽但**路由语义全错**（"robot 点" 实为机器人的垂直镜像位置）。当时 overlay 目检没发现是因为 t49 机械臂在画面中央行（垂直翻转对中央行不变）。含义：
- 上表三变体的数字**不可解释为 object/robot 路由的因果对比**；(iii) 崩盘的 phase 自指假说（线索 2）不受此 bug 直接影响（phase 机制与 mask 无关），但需在正确标签下重验。
- **修复已完成（2026-07-30）**：三处代码修正（`convert_libero_raw.py` / `eval_env.py` 不再翻 seg；`extract_c_labels.py` 同）+ 存盘数据离线翻回（150 个阶段〇 h5 的 robot_seg+grid_labels 重算；500 个阶段一 seg0）+ IoU 复验 0.971/0.923。
- **待决定**：阶段〇三变体重训（3×101 epochs ≈ 12.6h + n=400 eval）与阶段一 L1 开训的 GPU0 排期。
- 教训入库：**方向约定必须实测锚定**（一个空间断言 = 一次实弹验证），翻转类 bug 用「行心/IoU 对真值」数值检验，目检小图不可靠。

---

## 2. 阶段一 · C 分布预测：图片 + 指令 → P(C)

### 2.1 目标与输入输出

- **输入**：初始帧 RGB + 语言指令。
- **输出**：C 的概率分布 P(C | I, L)，以及**输出分布之前的 latent feature**（阶段二的 condition，必须保留接口）。
- C 的定义：`C = inv(T_obj) @ T_site` ∈ SE(3) + 夹爪开度，在 $t_g$ 冻结。

**坐标系约定（2026-07-30 定稿）**——C 本体是物体系相对量（这是它场景不变性的来源，**不是**绝对 pose）；需要绝对坐标的是执行层量，统一用 **robot base 系**：

| 量 | 坐标系 | 锚定来源 |
|---|---|---|
| C | 物体系（相对量，故意的） | 只在造标签/分析时显式计算；执行时不出现 |
| 接触点 heatmap p | 图像像素系 | 内参+深度 lift → 相机系 3D |
| E(t_g)（approach 目标） | robot base 系 | 相机系经外参 T_base←cam 变换（仿真已知；真机手眼标定） |
| T_rel(t)（transport 刚体变换） | 相机系算出 → 共轭到 base 系 | flow SVD；T_rel_base = T_bc · T_rel_cam · T_bc⁻¹ |
| E(t) 执行目标 | robot base 系 | E(t) = T_rel(t) · E(t_g)，E(t_g) 来自 FK |

### 2.2 基本方法

**两因子分解（2026-07-30 定稿方向，设计核心）**：

```
P(C | I, L)  =  P(p | I, L)  ×  P(R, w | p, I)
                └ 接触点 heatmap        └ 朝向 R + 开度 w
                  语言决定「抓哪」          几何决定「怎么抓」
```

平移分量 = 接触点 p + 深度 lift，不单独预测。分解依据：语言只影响第一因子（抓杯柄还是杯沿是任务语义），给定接触点后第二因子基本是纯几何（与指令无关）→ 贵的 VLM 只花在需要语言的因子上。文献版图也按此分裂（affordance grounding 有语言没 SO(3)；grasp generation 有 SO(3) 没语言；胶水未被占，见 §2.6）。

**模型选型（待决定，2026-07-30 从二选一改为三选一，当前倾向 C，等 spike 定稿）**：

| 选项 | 方案 | 优点 | 缺点 |
|---|---|---|---|
| A | DINOv2/v3 + CLIP text，cross-attention 融合 + 可训头 | 轻、快、latent 完全可控，GPU0 友好 | 语言理解上限低，全部从头训 |
| B | 小 VLM + LoRA，AffordanceLLM 式 mask token（`<mask_token>` hidden state → decoder 出 heatmap；token embedding = C-latent） | latent 接口与 v2 同构，VLM 世界知识 | 单卡训练慢（AffordanceLLM 原文用 8×A100 全参微调） |
| **C** | **冻结 AFUN 当前端**：functional mask 当接触点分布、MetaQuery embedding 当 C-latent；只训朝向+开度头 + latent adapter | 白拿 foundation-scale 泛化；训练量比 B 小一个数量级；[代码+权重已公开](https://github.com/EricWang12/AFUN)（inference-only） | 依赖外部权重；LIBERO 仿真域 mask 质量未知（**spike 待验**） |

**定稿 gate**：半天 spike——AFUN 权重跑 LIBERO agentview 图+指令，看 functional mask 质量。过关选 C，不过关退 B。

**L1 细化设计选项（2026-07-30 全清单，评估轴 = latent 接口质量 / 空间指代 / 单卡可训 / D7 联训兼容 / 实现时长）**：

| # | 方案 | 优点 | 致命/主要短板 | 判定 |
|---|---|---|---|---|
| a | **C′ 具体化：冻结 DINOv3 patch 特征 + 文本（BERT task emb，与 ATM 同源）+ 2–4 层 cross-attn 融合 + 学习型 [C] token + 轻量上采样 decoder；AFUN mask 投影后并入输入通道** | 可训参数 ~15–25M；latent=[C] token 干净可 probe；联训 ③ 内存友好；实现 ~3–4 天；空间指代靠监督学（LIBERO 模板化语言 ~10 种关系，可学） | 语言上限低于 VLM（靠 fallback 阶梯兜底） | **推荐** |
| b | Qwen3-VL-8B + LoRA mask-token（AffordanceLLM/B 案；模型已在盘） | 语言/空间指代上限最高 | 8B 进 D7 联训循环单卡内存爆表；实现 1–2 周；对模板化语言杀鸡用牛刀 | v2 升级项 |
| c | CLIPSeg 微调（现成 text→mask decoder） | 最快立起（~1 天）；开箱权重 | decoder 分辨率粗（A1 10px 存疑）；latent 是 FiLM 向量非 token，接口弱；CLIP text 空间组合差 | 降级为 baseline 行 |
| d | 检测两段式（GroundingDINO/OWL-ViT 候选 + 语言排序 → 物体上出 heatmap） | 空间指代可显式做 | **离散选择不可微 → latent 接口/D7/A2 全断**；且正是 §2.6 已被占满的"VLM 过滤胶水"区 | 排除 |
| e | 与 L3/L5 全共享 ATM backbone + heatmap 头（D9 推到极致） | 参数最省、联训最自然 | 视觉先验弱（只见过 LIBERO）；v2 无 scale-up 路；held-out 语义泛化弱 | 排除（D9 保持只在 L3/L5 之间） |

**a 案 v1 规格**：DINOv3 ViT-B/14 冻结；文本 = BERT task emb（ATM 缓存复用）；融合 = text tokens + [C] token cross-attend patch features（2–4 层）；heatmap decoder = 轻量 FPN 上采样到 128²（A1 在 512 尺度双线性）；AFUN mask 1 通道 project 进 patch 特征；z = [C] token 输出（384–512 维）；L2 头吃 z + feat(p̂)。
**Fallback 阶梯（预注册）**：A1 在空间模板上不过 → 换/加强文本编码器（T5-small 微调，关系组合更强）→ 仍不过 → 升 b 案（LoRA 只训 L1 阶段、联训时冻结）。

**AFUN prior 通道的角色澄清（2026-07-30，回应「冻结 AFUN 是否卡死上限/断可微」）——两个担心都不成立**：
- **可微性**：AFUN 是**叶子输入**（和 RGB、depth 同级的数据预处理，离线跑一次），不在 z→loss 的计算路径中间。D7 要求的是 action loss → L4 → L3 → z → L1 可训参数这条梯度路径完整——它完整。不往 AFUN **里面**回传梯度和不往相机里回传梯度一样，不是问题（冻结 DINO 同理：冻结 ≠ 链路不可微，只是该权重不更新；③ 阶段想解冻 DINO 末几层是备用杠杆）。
- **上限**：卡死上限的是**选项 C**（AFUN mask 直接当输出）——已因此被否。a 案里 mask 是 advisory 通道：监督来自特权标签，**AFUN 错的样本标签照样是对的**，融合层会学到"该通道与其它证据冲突时压低其权重"。最坏情形 = AFUN 通道退化为噪声 → 模型学到零权重 → 上限回到 DINOv3+text 的监督学习上限，与 AFUN 无关。
- **两个预注册保险**：① 训练时 **channel dropout**（随机置零 AFUN 通道）——防依赖 + 免费拿到有/无 prior 的 ablation 行；② spike 两个失败任务（table center / next to cookie box）的 A1 单独报告，**必须 ≥ 全任务平均**——实证 override 能力，防"跟着 mask 走"的捷径。

**阶段一训练 pipeline 数据流（a 案定稿，2026-07-30）**：

*离线预处理（3 个一次性作业；①已完成）*：
| # | 作业 | 输入→输出 | 成本 |
|---|---|---|---|
| ① | `extract_c_labels`（✅） | raw LIBERO → `c_labels/<task>.h5`（rgb0/depth0/seg0/contact/ee_quat[t_g]/gripper_q/相机参数） | 已完成 |
| ② | AFUN 推理 | rgb0 + 指令 → `afun_prior/<task>.h5`（500 张 512² mask） | ~2–3h GPU，一次性 |
| ③ | DINOv3 特征 | rgb0 → `dino_feats/<task>.h5`（(37²,768) f16/张，共 ~1G） | 分钟级，一次性 |

预计算特征 ⇒ 训练循环只跑融合+decoder（秒级/epoch）；代价 = v1 放弃图像增广（DINO 在线算才可增广，留作杠杆）。文本 = ATM 的 BERT task emb 缓存，零成本。

*Dataloader 每样本*：`{dino (1369,768) f16, afun_patch (37,37), bert (768), heatmap_tgt (128²，σ=8px 高斯 @ contact/4，归一), yaw_bin/pitch_bin（ee_quat[t_g] 世界系→base 系→离散 36/12 bins，夹爪 ±π 对称折叠）, w（gripper_q[t_g]）, contact_xy}`。

*前向*：patch tokens = Linear([dino ∥ afun_patch]) → (1369,384)；+ text token + 学习型 [C] token → 3 层 transformer（全 self-attn，1371 tokens）→ ① img tokens 重排 37² → 上采样 conv → 128² heatmap logits；② z = [C] token (384)。L2 头：feat(p̂) = decoder 特征图在 **GT 接触点**处双线性 gather（训练 teacher-forced 坐标，推理用预测峰；已知轻微 exposure gap，记录不处理）→ MLP([z, feat]) → yaw/pitch logits + ŵ。

*Loss*：L_C = 1.0·KL(heatmap) + 0.5·(CE_yaw + CE_pitch) + 0.5·L1(w)。AdamW 只更新融合/decoder/[C]/heads（~20M）。

*Split 与验收协议*：train = 8 任务 × 45 demos；val_id = 同 8 任务 × 5 demos；**val_ood = 2 个整任务全 held-out**（选 on_the_stove + next_to_the_ramekin）。A1 分 id/ood 报告 + spike 失败任务单列；A2 = z 上线性/浅 MLP probe（回归 contact/yaw/pitch/w）vs 冻结 DINO pooled 特征基线。channel-dropout 训练 ⇒ eval 免费出有/无 AFUN prior 两行。

*算力*：训练 <1h/run（特征已缓存）⇒ 多 seed + fallback 阶梯迭代都便宜；唯一大头是 AFUN 推理的一次性 2–3h。

*AFUN mask 错误的处置（2026-07-30，回应「万一 mask 不对」）*——三条设计不变量 + 一个量化动作：
1. **软特征，非门控**：mask 从不裁剪/限制 heatmap 支撑集、不参与任何 argmax——错 mask 的最大伤害 = 一个噪声通道，不存在"错了就把答案框死在错物体上"的失效模式（硬 prior 设计才有，已排除）。
2. **训练时的错 mask 是解药不是毒药**：标签来自特权状态，mask 错、标签对的样本正是教会融合层「mask 与文本/视觉证据冲突时压权重」的监督信号。
3. **量化动作（离线作业 ② 的 QC 步骤，预注册）**：AFUN 推理完 500 张后立刻算 **prior 正确率**（GT 接触点是否落在 mask 内 / ≤20px），按任务出表；A1 按「prior 对 / prior 错」分层报告——「万一错」从担忧变成测量值。
4. **零成本退出**：channel dropout 训练保证无 prior 也能跑；若 ablation 显示 ood 上「无 prior ≥ 有 prior」，直接摘掉该通道（一个 flag），架构无任何变动。诚实预期：AFUN 错误可能与难样本相关（文本歧义处它也错）——那些样本回落到 DINO+text 上限，是「本来就难」，不是「被 prior 带偏」，分层 A1 能看到。

**朝向+开度头（第二因子，三案通用）**：v1 = 离散朝向 bins + 开度回归，condition on C-latent + 接触点局部 feature；v2 = GraspGen-X 式 SE(3) diffusion + discriminator。

**C 的表示形式（待决定，当前倾向 a）**：

| 选项 | 表示 | 分布形式 |
|---|---|---|
| **a** | 像素 heatmap（接触点 u,v）+ 离散化朝向 bins + 开度回归 | heatmap 天然是空间分布；朝向是 categorical |
| b | 直接在 SE(3) 上离散 bins | 分辨率与维数打架 |
| c | diffusion / GMM head 采样式输出 | 表达力最强，v2 升级项 |

选 a 的理由：多峰性（马克杯抓柄 vs 抓沿）直接体现在 heatmap 的多个峰上，可视化和 debug 都直观；从 heatmap 峰 + 深度图 lift 到 3D 得到完整 SE(3)。

**监督信号**：仿真特权信息直接算标签——在每条 demo 里用夹爪闭合信号定位 $t_g$，取 `C = inv(T_obj(t_g)) @ T_site(t_g)`，投影回初始帧像素得 heatmap 标签。**不需要人工标注。**

**Loss**：heatmap 用 KL/focal；朝向 bins 用 cross-entropy；开度用 L1。

### 2.3 需要考虑的地方

1. **latent 必须携带 C 的信息**（一票否决问题的定量化）——见 2.4 的 latent probe 验收。
2. **多峰监督**：一条 demo 只给一个 C 样本；同一任务多条 demo 可能抓不同位置。heatmap 标签按任务聚合还是按 demo 独立？（倾向：按 demo 独立、让模型自己学出多峰。）
3. **朝向的表示**：离散 bins 的粒度（yaw 优先？LIBERO 桌面任务大多 top-down，可能 yaw+俯仰两自由度就够）。
4. **相机视角**：用 agentview 还是双视角？（测量结论：approach 的价值住在 wrist 视角的机器人像素里，但阶段一预测的是场景级的 C，agentview 更合适。）
5. **指令里没有唯一解时**（"pick up the mug"没说抓哪）P(C) 应该是多峰的——评测指标要能容忍多峰（top-k 而不是 top-1）。

### 2.4 验收标准（预注册）

- **A1（预测质量）**：held-out 任务上，heatmap top-5 峰在 10px 半径内命中真值接触点的比例 ≥ 80%。
- **A2（latent probe，关键）**：从 latent 用线性/浅层 probe 回归 C，误差显著小于从冻结 DINO feature 直接 probe → 证明 C 的信息确实进入了 latent，阶段二的 conditioning 才有意义。**A2 不过 = 阶段二白搭，先修阶段一。**

### 2.5 状态

- ✅ 2026-07-30 **AFUN spike** 完成：空间指代 3/5（between/drawer/next-to-plate ✅，table-center/next-to-cookie-box ❌ 都选了显眼碗），选对时 mask 精度 ≤9px@512；motion 曲线方向合理。判定 → D1=C′（自训前端 + AFUN mask 当输入 prior；冻结单用上限 ~60% 不可接受；AFUN 训练代码未放出）
- ✅ 2026-07-30 **统一标签提取完成**（500/500 demos，`RoutedFlow/data/c_labels/`；一次重放同时落盘 flow 标签所需位姿轨迹）。QC 全过（montage 20/20 落碗沿）。三个副产物：① 被抓物体判定改用**运动判定**（nearest 启发式 1.6% 错）；② **lift 系统偏差实测** 中位 2.5cm（robust 化后），悬空抓取单像素 lift 达 11.6cm——B2 的现成证据 + lift 必须窗口 robust 化；③ 多峰性直查：libero_spatial 示教高度**单峰**（§2.3 #2 经验回答：按 demo 独立监督即可）。详情 `RoutedFlow/experiments/stage1_c_labels/README.md`
- ✅ 2026-07-30 模型选型定稿（a 案，§2.2 五案清单 + grill 会话确认）；**全部代码落地**：离线作业②③（AFUN prior 实测 3 分钟/500 张——原 2–3h 估计全是模型重复加载；DINO 特征 1G 缓存）、link 位姿补存（17×500，L3 GT 原料）、L1+L2（7.5M 参数）、A1/A2 评测（5-fold 协议 + prior 分层 + z_random 对照）、10 项单测（含轴约定数据锚定）。**prior 正确率实测：in-mask 26% / ≤20px 58%（n=500）**——C′ 判定坐实。smoke 全链路通。实验主页 `RoutedFlow/experiments/stage1_l1_training/`
- 🔄 fold0 正式训练 + A1/A2 验收；☐ 5-fold 轮换 + no-prior ablation
- ✅ L3/L5 骨架（`flow_models.py`：D3 两案注入块零初始化 identity、D8a 深度通道、text 不对称接口断言；ATM surgery 随 L3 训练 PR 落地）

### 2.6 文献基础与撞车状态（2026-07-30 sweep）

**接触点因子（heatmap 前端）候选，按推荐序**：

| 论文 | 时间 | 要点 | 对我们 |
|---|---|---|---|
| **AFUN** (arXiv 2606.02551) | 2026-06 | RGB-D+指令 → 任务条件 functional mask + 3D post-contact 运动曲线；VLM+SAM3 经 **MetaQuery** 连接；Bézier 运动表示；跨 8 测试集领先，接触点命中 +12.7~61.3% | **前端首选**（选项 C）；MetaQuery token = 现成 C-latent 接口；代码+权重公开（inference-only），https://github.com/EricWang12/AFUN |
| Affordance-R1 (arXiv 2508.06206) | 2025-08（rev 2026-05） | GRPO RL + CoT rewards 训 MLLM affordance；ReasonAff 数据集；代码已放 | 训练方法学升级项；明确批评 special-token 路线的 OOD 弱点（= 选项 B 的已知短板） |
| AffordanceLLM (arXiv 2401.06341, CVPR'24) | 2024-01 | LLaVA-7B + `<mask_token>`→decoder heatmap；OWL-ViT 768² 换掉 CLIP 224²（定位分辨率）；伪深度注入；AGD20K hard split KLD 1.661 vs LOCATE 1.829 | 选项 B 的模板；mask-token 接口思想已被 AFUN 的 MetaQuery 迭代 |
| VideoAfford (arXiv 2602.09638) | 2026-02 | 人类 HOI 视频 → 3D affordance via MLLM | 潜在标签/预训练来源，非前端 |

**朝向因子（grasp generation）**：GraspGen-X (arXiv 2606.00998)——SE(3) diffusion + discriminator、swept-volume 12 维 gripper embedding、2B grasps、zero-shot 新夹爪。**没语言不是缺陷**：我们的分解本来就不让朝向因子碰语言。三个用法：v2 朝向头模板 / 跨形态 embedding（Phase 2 故事）/ 真实数据 demo-free 标签 teacher（候选生成 + 我们的 heatmap 按任务过滤）。语言条件 grasp diffusion 子领域（ECCV'24 negative-prompt 6-DoF、LLGD 2407.17967、coarse-to-fine 2512.21065、OmniDexVLG 2512.03874）= 完整阶段一的**竞争对手**而非零件。

**撞车状态**：「image+language→任务相关 grasp」的**离散候选 + VLM 过滤胶水**版全被占（VLM-guided bimanual 2604.08726、VLAD-Grasp 2511.05791、Springer'25 3D affordance grounding→grasp、3DFlowAction 的 GPT-4o+AnyGrasp、MolmoMotion 外包给 MolmoBot、TaskGrasp/GraspGPT/FoundationGrasp 族）。**未被占**：可微分布输出 + 被下游 flow conditioning 消费的 latent 接口。阶段一单独不成文章，必须和阶段二/三绑定讲。
**⚠ AFUN 双面性**：它的 post-contact 运动曲线已覆盖我们阶段三的一部分（物体运动先验）。它没做的：SE(3) 朝向+开度、phase 切换、C 合成 E=T·C、可微 latent 进 policy。叙事必须从「预测 affordance」上移到「routing + 合成 + 可微集成」，否则被它压住。

### 2.7 功能性链路 v0.1 与 condition 审查（2026-07-30，用户提出，占位模块待选型）

**用户提出的链路（原始版）**：LLM 把指令分解为若干「approach+transport」原子语句 → 每个原子的 approach prompt + 图片 → affordance 分布模块 → 末层 latent → head 预测 C（用序列中第一个夹爪闭合的 action 监督）→ (latent, 初始图, 深度, SAM3 robot mask, approach prompt) → robot point-flow 模块 → (flow, robot state) → action expert → actions。

**审查结论：骨架成立（与 §2.2 两因子分解、§3.2 conditioning、§0.3 联合训练全部兼容），4 处修正 + 2 处补全**：

| # | 类型 | 修正内容 |
|---|---|---|
| 1 | ill-posed 监督 | C 的监督不能用 t_g 那一步的 **action**（OSC delta 命令，不含绝对 pose）——用 t_g 的**状态** E(t_g)（proprio/FK 特权状态），与 §2.2 既有约定一致 |
| 2 | 冗余（双头冲突） | C head 不预测完整 C：平移分量由 heatmap 峰 + 深度 lift **导出**（否则 heatmap 说 A 点、C head 说 B 点，两头冗余且可冲突）；C head 只补第二因子 (R, w)，condition on latent + p̂ 处局部 feature |
| 3 | 冗余（因果旁路，最重要） | **flow 模块的 text prompt condition v1 移除**：C-latent 是任务信息进入 flow 的唯一入口；并行喂 text 给 flow 制造 gradient shortcut——flow 可绕过 latent 自行 grounding，A2 probe / 干预测试（§0.3 因果诊断）全部失效。text 降级为 ablation 行。**本条修正 §3.2 原 conditioning 表** |
| 4 | 措辞→约定 | 「初始图片」→ **当前帧**：flow 每个 replanning horizon 从当前帧起算（闭环，与 §4.3.2 晃动吸收机制同一约定） |
| 5 | 补全（decomposer 合同） | LLM decomposer 的输出合同：**每个原子语句恰好包含一个闭合事件 t_g**（phase 语义绑定到原子粒度）；LIBERO v1 任务本身即原子 → decomposer = identity placeholder，选型推迟 |
| 6 | 补全（查询点几何） | robot mask 的真实角色是**查询点采样域**；查询点必须覆盖 gripper 且空间铺开——刚体 ≥3 非共线点的 flow 才能编码到达 C 的完整 6D（与 §4 transport 的 SVD conditioning 同一要求）；flow 需带 3D 信息（3D flow 或 2D track + 查询点深度），否则 action expert 无法闭环到米制空间 |

**修正后链路 v0.1**：

```
指令 ──LLM decomposer (v1=identity)──▶ {(approach_promptₖ, transport_promptₖ)}，每原子恰一个 t_g
RGB + approach_prompt ──▶ 前端(D1=C′) ──▶ heatmap P(p|I,L) ─┐
                                        └─▶ latent z          ├─▶ 平移 = lift(p̂, depth)；C head(z, feat(p̂)) → (R, w)
                                                              │        监督 = E(t_g) 特权状态（非 action）
(z, 当前帧 RGB, depth, robot mask→查询点) ──▶ robot-flow 模块 ──▶ flow（3D，覆盖 gripper，K 步）
(flow, proprio state [, image tokens: ATM 原样]) ──▶ action expert ──▶ actions
到达 C + 闭合 → phase 切换（D6 未解项照旧）→ transport 分支（§4）
```

action expert 保留 image tokens 是 ATM 原架构（B1/B2 对 SR 友好）；expert 看不到 text，无法自行 re-ground 指令，因果接口不受损。

### 2.8 学习模块清单与候选填充方法（2026-07-30，用户确认链路 v0.1 后摘出）

全链路 **5 个 learned 模块**（其余全部 frozen 或零参数几何，见下）。**命名（2026-07-31 用户定）：L1 对外称 C-VLM，L2 称 C head**（图面/论文用语；文档内 L1/L2 编号继续作内部索引）：

| # | 模块 | v1 填充 | 候选 / 升级项 | 决策状态 |
|---|---|---|---|---|
| L1 | 前端（RGB+prompt → heatmap + latent z） | **D1=C′**：DINOv2/v3 + CLIP text cross-attn + 可训 heatmap decoder，AFUN mask 当输入 prior 通道，仿真特权标签自训 | B 案小 VLM+LoRA mask-token（Qwen3-VL-8B 已在盘上）；Affordance-R1 的 GRPO 训练法（升级）；AFUN 冻结 = baseline 行 | 已定 C′（spike 判据），待用户最终确认 |
| L2 | C head（z + feat(p̂) → R, w） | 离散朝向 bins（yaw+pitch）+ 开度 L1 回归 | v2: GraspGen-X 式 SE(3) diffusion + discriminator | 表示已定（§2.2 选项 a），bins 粒度待定 |
| L3 | robot-flow 模块（approach） | **ATM track transformer 改造**：查询点限 robot mask + C-latent 注入 | 注入方式 cross-attn 加一路 vs AdaLN（待定）；3D 化两案见下「新开决策」 | v1 基座已定（§3.3），两个子决策open |
| L4 | action expert（approach） | **ATM ViLT BC 头**（flow-conditioned BC，现成） | v2: diffusion policy 头 / flow-matching 头（π0 式） | 已定 v1 |
| L5 | object-flow 模块（transport） | ATM track transformer 同款，查询点限 object mask，condition on **transport prompt** + object mask | AFUN post-contact Bézier 曲线当 frozen prior / 蒸馏 teacher；Im2Flow2Act、General Flow、EC-Flow（铰接）参考；3DFlowAction = 竞争对手 | 基座倾向与 L3 共享 backbone（D7 联合训练友好），未定稿 |

**text 的不对称（设计要点，勿混淆）**：L3 **不吃** text（§2.7 修正 #3，任务信息只走 C-latent）；L5 **必须吃** transport prompt——transport 没有 latent 接口，语言是它唯一的任务信息入口（"放到哪"）。两条分支的语言路径不同是 by design。

**Frozen / 零参数模块**（不训）：LLM decomposer（v1=identity，之后也用现成 LLM 冻结）；SAM3 / robosuite 分割（mask）；DINO feature；depth；平移导出 lift(p̂,depth)；transport 的 SVD+E=T·C+IK（零参数，§0.3 明确不进联合训练）；phase 切换（v1 用策略 gripper 输出锁存，D6 未解）。

**D7 curriculum 映射**：① 先训 L1+L2（C 监督预训）→ ② 加 L3+L5（flow 监督，L1 冻结或低 lr）→ ③ 加 L4 全解冻联合微调（action 主 loss + C/flow 辅助 loss，λ 待定 D7）。

**新开决策（进 §6 决策表）**：
- **D8 flow 表示 3D 化**：§2.7 补全 #6 要求 flow 带 3D。两案：(a) ATM 2D track + 查询点深度通道（改动最小，v1 倾向）；(b) 真 3D flow predictor（General Flow 式，改动大）。
- **D9 L3/L5 是否共享 backbone**：共享（参数省、D7 联合训练自然）vs 分开（归因干净）。

**复用度评估（2026-07-30，回应「哪些基本不用设计」）**——真正要设计的只有 L1：

| 模块 | 设计量 | 现成件 | 新写的部分 |
|---|---|---|---|
| L1 | **大（唯一的真设计）** | DINO/CLIP/AFUN 推理全现成 | 融合结构 + heatmap decoder + latent token + AFUN prior 接法 + 训练循环 |
| L2 | ≈0 | — | bins CE + 开度 L1 的小头，挂 L1 训练器，~1 天 |
| L3 | 小（改造） | **ATM track transformer + checkpoint + 阶段〇训练引擎/OOM 配置全复用** | conditioning 注入（D3 一个决定）+ 查询点限 robot mask（阶段〇 mask 机器复用）+ robot-flow GT 管线（见下） |
| L4 | ≈0 | **阶段〇 `BCViLTPolicyGated` 就是它** | 换 flow 来源（从冻结 track transformer 换成 L3 输出） |
| L5 | **最接近直接用现成** | ATM track transformer **本来就吃 text conditioning**（task emb） | 只需查询点限 object mask；GT 今天就能从 c_labels h5 纯投影导出（seg0+depth0+obj 位姿+相机全存了） |
| 几何链 | 0 设计、纯实现 | robosuite OSC 控制器 | Umeyama/SVD + E=T_rel·E(t_g) + 单测（用仿真 GT 验证到数值精度） |

**唯一的数据缺口**：L3 的 robot-flow GT 需要机器人 link 表面点的逐帧位姿——c_labels 只存了 EE site 轨迹。两案：(a) 扩展提取器补存各 link 位姿（半天，重放机器现成）；(b) 沿用 ATM 原生 CoTracker 伪标签管线（third_party 里有）。倾向 (a)（零噪声）。L5 的 GT 无此问题。

---

## 3. 阶段二 · approach：robot flow 预测 → action 生成

### 3.1 两条路线（都要实现）

**备选路线（baseline，必须实现）**：从 P(C) 取峰 + 深度 lift 到 3D → 直接解 IK / OSC 到达 C 位置。
它是对照组：主路线的存在理由是"避免碰撞、走出合理路径"，**没有 baseline 就没法证明这个理由成立**。预期它在无障碍任务上能用、在 clutter 任务上碰撞失败。

**主路线（routed flow）**：flow 预测网络 → robot flow → action generation module → actions。

### 3.2 主路线的 conditioning 设计

| Condition | 作用 | 来源 |
|---|---|---|
| C-latent（阶段一分布输出前的 latent） | 告诉 flow "要去哪、怎么抓"——C 的信息唯一入口 | 阶段一，冻结取出 |
| ~~text prompt~~（**2026-07-30 移除**，见 §2.7 修正 #3） | ~~任务语义~~ → 与 C-latent 冗余且造成因果旁路（flow 绕过 latent 自行 grounding，A2/干预诊断失效）；降级为 ablation 行 | — |
| 机器人 mask | 限定 flow 查询点的采样范围 = 只预测机器人身体的运动 | v1: robosuite 渲染分割真值；scale-up: SAM |
| DINO feature | 场景外观（障碍物在哪） | 冻结 DINOv2/v3 |
| 深度图 | 3D 几何（避障需要） | 仿真真值深度 |

### 3.3 基本方法

1. **Flow predictor v1 = ATM track transformer 改造**：ATM 本来就预测任意查询点的 track，把查询点采样限制在机器人 mask 内 + 注入 C-latent conditioning，就是 robot-flow predictor——不用从零写网络。改造点：conditioning 注入方式（cross-attention 加一路 / AdaLN，待定）。
2. **Action generation module v1 = ATM 的 BC 头**（flow-conditioned BC，现成）。v2 升级项：diffusion policy 头。
3. **Decode fork（已在设计文档定过）**：v1 只解 EE 轨迹（Cartesian，跨形态可迁移，Cloak 证据）；whole-arm 版本留给避障专门实验。

### 3.4 需要考虑的地方

1. **到达判定（phase 切换的前半）**：什么时候认为 approach 结束？v1 用策略自己输出的夹爪闭合动作触发（与阶段〇一致）；备选：EE-C 距离阈值。误触发的行为要在 eval 里单独统计。
2. **C-latent 冻结 or 联合微调**：v1 冻结（模块边界干净、归因清楚）；联合训练留 v2。
3. **flow → action 的时间对齐**：预测的 flow horizon（ATM 默认 16 步）和 action chunk 长度的匹配。
4. **碰撞怎么量化**：LIBERO 原生 SR 不区分"碰撞失败"和"没抓到"——eval 里要加 contact-force / 碰撞事件的计数，否则主路线 vs baseline 的对比讲不出故事。
5. **mask 时变性**：机器人在动，mask 每帧都变；查询点在 t=0 采样后 track 会离开 mask——这正是 flow 的语义（跟踪机器人身体），不是 bug，但要确认 dataloader 不会重采样。

### 3.5 验收标准（预注册）

- **B1**：主路线的抓取成功率（approach 段成功 = 夹爪闭合时物体在指间）≥ IK-baseline。
- **B2**：在有障碍/clutter 的任务子集上，主路线碰撞率显著低于 IK-baseline。**B2 是主路线的存在理由：不成立就退回 IK-baseline + 把创新叙事集中到阶段一/三。**

### 3.6 状态

- ☐ IK/OSC baseline 实现（robosuite OSC_POSE 控制器，估 2–3 天）
- ☐ ATM track transformer 的 conditioning 改造
- ☐ mask 内查询点采样（复用阶段〇的标注代码）
- ☐ 训练 + B1/B2 评测

---

## 4. 阶段三 · transport：object flow → 解 IK

### 4.1 关键化简：这一段根本不需要显式的 C

$E(t) = T(t) \cdot C$，而 $C = T(t_g)^{-1} E(t_g)$，代入得：

$$E(t) = \underbrace{T(t)\,T(t_g)^{-1}}_{T_{rel}(t)\text{：flow SVD 直接给出}} \cdot \underbrace{E(t_g)}_{\text{proprioception 免费读出}}$$

- $T_{rel}(t)$：物体从 $t_g$ 到 $t$ 的相对刚体变换——物体 flow 的 3D 点对应关系做 Umeyama/SVD 就是它，**不需要绝对物体位姿估计器**。
- $E(t_g)$：抓取瞬间的 EE pose，正运动学直接读。
- 所以 transport 段只依赖：物体 flow + 深度 + 抓取瞬间的 proprioception。这正是定理②"transport 方向 flow→action 平凡"的构造性实现。

### 4.1.5 监督信号来源（2026-07-30 补，回应「物体 flow 没 GT 要人工标注」的担忧——不需要）

| 来源 | 适用 | 说明 |
|---|---|---|
| ① 仿真特权状态 | v1 | mujoco state 里有 T(t)，物体表面采样点逐帧变换+投影 = 零噪声 GT track（复用阶段〇重放机器） |
| ② **GT action 的无损变换** | v1 + 真机 | E(t)=T(t)·C 反用：demo 的 EE 轨迹（proprio/FK）→ T_rel(t)=E(t)·E(t_g)⁻¹ → 作用于物体点 = flow 标签。**transport 的 action 监督没有被丢弃，只是换了坐标表达**（定理② 的反方向）。依赖刚体+抓稳（C 漂移实测 2-3mm） |
| ③ tracker 伪标签 | scale-up | CoTracker/SpatialTracker 跑真实视频（ATM/Im2Flow2Act/ChronoFlow/MolmoMotion 的惯例） |

边界（limitation）：②依赖刚体+抓稳（打滑/软物体失真）；release 后的物体运动三条都不覆盖。v2 选项：D7 联合训练下 Umeyama/SVD 与 E(t) 合成均可微，action-consistency loss 可反传给 flow predictor。

### 4.2 基本方法

1. **Object flow predictor**：condition = 物体 mask + text prompt。v1 = 同一个 ATM track transformer，查询点采样在物体 mask 内（和阶段二共享 backbone，只换 mask——工程上最省）；scale-up 叙事 = MolmoMotion 式的大模型 flow。
2. **2D flow → 3D**：深度图 + 相机内参 lift 查询点到 3D。
3. **刚体拟合**：帧间 3D 对应 → RANSAC + Umeyama → $T_{rel}(t)$。
4. **执行**：$E(t) = T_{rel}(t) E(t_g)$ → OSC_POSE / IK 逐步跟踪。
5. **Release**：预测 flow 收敛（位移 < 阈值持续 N 帧）或指令完成信号 → 开夹爪。

### 4.3 需要考虑的地方

1. **遮挡**：transport 时夹爪常挡住物体的一部分——物体 mask 内的点会有一批 track 失效。对策：查询点多采样 + RANSAC 剔除 + 只要 ≥3 个内点就能解刚体变换。
2. **刚体假设的边界（2026-07-30 细化，回应晃动/铰接担忧）**：
   - **晃动/打滑**：E(t)=T_rel·E(t_g) 逐 horizon 闭环执行（跨 horizon 在当前帧重新锚定），单步残差≈单步滑移量不累积；实测漂移 2-3mm 对粗放任务无感。**预测 flow vs 观测 flow 的差 = 免费的打滑检测信号**（可触发 regrasp；纯 action 方法无此可观测性，论文加分项）。整体掉落 = 所有方法共同失败，不算本理论边界。
   - **铰接物体（开门/抽屉）**：恒等式只要求被抓链节是刚体——T(t) 取**被抓链节（门板）的位姿**即成立；flow 天然编码铰接约束（门=弧线、抽屉=直线，FlowBot3D/EC-Flow 同理），是 flow 表示对整物体 6D pose 的优势项。工程要求：SVD 只用被抓链节的点（per-link geom mask，仿真免费）+ 点撒满链节面防病态。
   - **真边界 = 非抓握操作**（推合笔记本盖等）：无 gripper closure 事件（phase 语义失效）+ 接触点滚动/滑移（C 时变）。v1 排除，扩展方向（contact-event 检测 + 时变 C）记 future work。
   - 软物体同前：v1 排除，limitation。
3. **深度噪声**（真机才有，仿真真值无此问题）：v1 不管，scale-up 时用 NAF 类 feature upsampler / 多帧滤波。
4. **误差累积**：$T_{rel}$ 相对 $t_g$ 帧计算（不是相邻帧递推），天然不累积——实现时务必锚定 $t_g$ 帧，别写成帧间递推。
5. **flow 预测的是未来还是跟踪过去**：执行时需要的是**未来** flow（plan），ATM 本来就是预测未来 track 的，语义对得上；但要确认 horizon 内物体真的按预测走（闭环重规划频率，待定：每步重预测 vs 每 K 步）。

### 4.4 验收标准（预注册）

- **C1**：oracle 抓取（脚本抓好）+ 阶段三执行 transport 的 SR ≥ 对应任务 BC baseline 的 transport 段表现。
- **C2**：$T_{rel}$ 拟合误差（仿真真值可算）：平移 < 5mm、旋转 < 5° 的帧占比 ≥ 90%。

### 4.5 状态

- ☐ 纯几何模块（lift + RANSAC/Umeyama + $E(t)$ 合成）：不吃 GPU，可先写 + 用真值 flow 单测
- ☐ 真值 flow 喂给几何模块的 oracle 上限实验（C2）
- ☐ object-mask 查询的 flow 预测接入
- ☐ C1 评测

---

## 5. 阶段四 · 整合与端到端评测

### 5.1 推理时的 phase switch

- **切换信号**：夹爪闭合动作输出后锁存（与阶段〇、训练标注一致）。
- **误切换处理**：闭合但没抓到（指间无物体，可用夹爪开度残差检测）→ 回退到 approach 模式重试？v1 先只统计不处理，把误切换率作为单独指标报告。
- **状态机**：approach → (闭合锁存) → transport → (release) → done。就三个状态，别过度设计。

### 5.2 评测协议

- benchmark：LIBERO spatial（现成机器，n=400 协议，配对 t 检验 across 10 tasks）。
- 参照系：A_full = 0.710（ATM 原版）；阶段〇的变体 iii 结果。
- 核心对比：完整 pipeline vs ATM baseline vs 各阶段 ablation（IK-baseline 替换阶段二 / oracle-C 替换阶段一 / 真值 flow 替换预测 flow）——每个模块的贡献单独归因，呼应测量论文的方法论。

### 5.3 状态

- ☐ phase 状态机实现
- ☐ 端到端 eval + ablation 矩阵
- ☐ 结果并入论文叙事（诊断 §3 + 修复 §4）

---

## 6. 横切的待决定事项（决策表）

| # | 决定项 | 选项 | 当前倾向 | 必须定稿的时点 |
|---|---|---|---|---|
| D1 | 阶段一前端 | A. DINO+CLIP 从头 / B. 小 VLM+LoRA mask-token / C. 冻结 AFUN / **C′. 自训前端 + AFUN mask 输入通道** | **C′**（2026-07-30 spike 后修订：AFUN 空间指代 3/5，冻结单用上限 ~60% 不可接受；选对时 mask ≤9px 极准 → 降级为 prior 通道。spike 详情见 `RoutedFlow/experiments/stage1_afun_spike/README.md`） | ✅ 已定（待用户确认） |
| D2 | C 的分布表示 | heatmap+bins / SE(3) bins / diffusion | heatmap+bins（a）；由两因子分解承载（§2.2） | 阶段一开训前 |
| D3 | conditioning 注入方式 | cross-attention 加路 / AdaLN / concat | 未定 | 阶段二改造前 |
| D4 | action 头 | ATM BC 头 / diffusion policy | ATM BC 头（v1） | 阶段二开训前 |
| D5 | 闭环重规划频率 | 每步 / 每 K 步 | 未定 | 阶段三接入前 |
| D6 | 阶段〇 phase 自指问题的处置 | 强制锁存诊断 / 训练时 phase 加噪 / 物理信号切换 / 接受负结果 | 未定（判定暂缓中，线索记录在 §1.7）；**联合训练下依然存在，见 §0.3** | 阶段四整合前必须回来处理 |
| D7 | 训练策略 | 分块冻结 / **staged 预训 → end-to-end 联合微调 + auxiliary losses** | 后者（2026-07-30 定向，用户提出；配套因果诊断见 §0.3） | 阶段二开训前定 λ 权重 |
| D8 | flow 表示 3D 化（§2.7 补全 #6 引出） | a. ATM 2D track + 查询点深度通道 / b. 真 3D flow predictor（General Flow 式） | **✅ a 定稿（2026-07-30 grill 会话用户拍板）**；`QueryDepthEmbed` 已实现（零初始化） | 已定 |
| D10 | L4 的 flow 来源（②→③ 断层，grill 会话新开并当场定向） | a. GT flow teacher forcing（exposure bias）/ b. ③ 阶段 L4 吃 L3 预测（带梯度）+ flow 加噪增广 | **b**（用户默认同意 2026-07-30） | ③ 开训前复核 |
| D9 | L3/L5 flow 分支是否共享 backbone | 共享（省参数、D7 联合训练自然）/ 分开（归因干净） | 共享 | 阶段三开训前 |

## 7. 风险清单（top 5）

1. **phase 信号自指（阶段〇已实测暴露，见 §1.7）**：gate 依赖 policy 自身输出时出现互相等待的迟疑失败（iii 0.1975，27% 从不闭合）。整合段（§5.1）用同一机制——**这是当前最大的已知技术风险**，D6 必须在阶段四前解决。
2. **A2 latent probe 失败**（latent 不携带 C）→ 阶段二的 conditioning 是空转；修复方向：加 probe loss 显式约束 latent。
3. **B2 失败**（flow 路线避障不比 IK 强）→ 主路线卖点消失；退路：IK-baseline + 阶段一/三叙事。
4. **AFUN 挤压叙事**（2026-07-30 新增）：其 functional mask + post-contact 运动曲线覆盖我们阶段一前端 + 阶段三先验；对策 = 用它当零件（选项 C）+ 叙事上移到 routing/合成/可微集成。
5. **撞车**：phase-switch 接线的 no-precedent 声明停留在 2026-07-28——论文写作前重扫 arXiv（原「每阶段开工前」放宽，理由：核心格子的占据状态一个月内变化有限，而 sweep 成本高）。

## 8. 时间预算（GPU0 单卡，粗估）

| 阶段 | GPU 时间 | 人力时间 | 备注 |
|---|---|---|---|
| 〇 | ✅ 实测：训练 12.6h（150s/epoch ×3）+ eval ~12h | ✅ 实际 1 天（代码+数据+跑） | 已完成；预算参照系：n=400 eval ≈ 4h/条 |
| 一 | 1–2 天训练（选 C 案则更少：只训头） | AFUN spike 半天 + 数据提取 1 天 + 模型 2–3 天 | |
| 二 | 1–2 周 | 2–3 周 | 最重的阶段（改造+baseline+双评测） |
| 三 | <1 天（几何模块不吃卡） | 1 周 | 可最早并行启动 |
| 四 | eval 矩阵 ~4h×条件数 | 1 周 | |

## 9. 文件索引

| 文件 | 内容 |
|---|---|
| `PHASE_GATED_FLOW_ROUTING.md` | 设计文档（理论/证据/进化史/pipeline） |
| `PHASE_GATED_FLOW_ROUTING_part1.html` | 离线导出（理论+证据+进化史） |
| `FLOW_AFFORDANCE_LOG.md` | 实验运行日志 |
| `PROGRESS_REPORT_2026-07-28.md` | 测量阶段总结（面向初读者） |
| `C_HEAD_PRIOR_ART.md` / `FLOW_TO_ACTION_SURVEY.md` | 撞车检查记录 |
| `/workspace/code/RoutedFlow` | **实现代码库**（src/doc/third_party 结构，README 有强制 changelog，规则见其 README） |
| `/workspace/code/ATM` | 代码基座（track transformer / BC / eval / `flow_probe.py`），已 symlink 为 `RoutedFlow/third_party/ATM` |
