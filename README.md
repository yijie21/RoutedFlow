# RoutedFlow

Phase-Gated Flow Routing 的实现代码库。按接触时刻 $t_g$（夹爪闭合）把操作任务切成 approach / transport 两段，flow 的语义按阶段路由：接触前预测机器人 flow，接触后预测物体 flow。

- 设计文档：`/workspace/research/d4rt/PHASE_GATED_FLOW_ROUTING.md`
- 分阶段实现计划（进度追踪）：`/workspace/research/d4rt/PIPELINE_IMPL_PLAN.md`
- 实验日志：`/workspace/research/d4rt/FLOW_AFFORDANCE_LOG.md`

## 目录规则（约定，勿破坏）

| 位置 | 放什么 |
|---|---|
| `src/` | 所有有用的正式代码（模块、库、工具函数） |
| `doc/` | 需要保留的文档/笔记/结果文件 |
| `third_party/` | 被引用的外部代码——先放这里，再参考它们在 `src/` 下写自己的实现 |
| 根目录 | 总领式脚本（`run.py`、main 入口、批处理 shell 脚本） |

**README 维护规则：每次环境变更或代码变更，必须在下方 Changelog 加一条**，让下一个使用者知道改了什么。

## 环境

- Python 环境：conda env `atm5090`（沿用 ATM 的环境，见 `third_party/ATM/setup_atm5090.sh`）
- GPU：只用 GPU0（`CUDA_VISIBLE_DEVICES=0`），GPU1 被其它项目占用
- LIBERO 配置：`~/.libero/config.yaml` 是全局共享文件**不许改**，需要自定义时用 `LIBERO_CONFIG_PATH` 环境变量
- `third_party/ATM` 是指向 `/workspace/code/ATM` 的 symlink（不是拷贝——里面有大 checkpoint 和数据集，原地引用）

## Changelog

### 2026-07-29
- 初始化仓库结构：`src/` / `doc/` / `third_party/` + 本 README
- `third_party/ATM` → symlink 到 `/workspace/code/ATM`（阶段〇的代码基座：冻结 track transformer checkpoint、BC 训练/评测机器、LIBERO demo 数据）
- 尚无代码。下一步（阶段〇，见实现计划 §1.4）：mask 重放脚本、track gating 模块、`bc_dataloader` 补丁、eval 侧在线分割

### 2026-07-29（当天第二批）— 阶段〇代码全量落地
- 新增 `src/routedflow/`：`grid.py`（32 点固定 grid + 标签查询）、`phase.py`（gripper 锁存 phase）、
  `track_gate.py`（干预本体，4 种 mode）、`gated_policy.py`（`BCViLTPolicyGated`，只 override `track_encode`）、
  `gated_dataset.py`（`GatedBCDataset`，多返回 labels+phase）、`convert_libero_raw.py`（raw LIBERO → ATM 格式
  + robot_seg/grid_labels/phase，**无需 CoTracker**，状态重放零漂移）、`engine_train_stage0.py`（训练引擎，
  单卡 Fabric strategy=auto——**deepspeed 未安装**，不要照抄 ATM 的 strategy）
- 新增 `experiments/stage0_routing_causal_test/`（配置/README/状态板/sanity_overlays），`experiments/CURRENT` symlink 指向它
- 根目录 `run_stage0.py`：`prep` / `test` / `smoke` / `train --mode <gate>` / `status`
- `tests/test_stage0_units.py` 8 项全过（含 grid 与 ATM `sample_double_grid(4)` 逐元素相等）
- 数据落在 `data/atm_libero_gated/libero_spatial/`（10 任务 × 15 demos，每任务 bc_train_10 + val）
- 验证结果：replay RGB diff 2.9/6.9（对照错误翻转 52/62）；robot 像素占比 agentview 0.088 / wrist 0.217，
  与测量阶段的 9%/21% 交叉吻合
- 环境注意：跑任何东西用 `run_stage0.py`（它设好 `MUJOCO_GL=egl`、`LIBERO_CONFIG_PATH`、`PYTHONPATH=src:third_party/ATM`、atm5090 解释器）；退出时的 2 行 EGLError 是析构噪音，无害

### 2026-07-29（当天第三批）— OOM 修复 + 正式训练启动
- **OOM ×2 修复**：ATM 的 batch 128 是 4×A100 每卡的量（全局等效 512）；单张 32G 5090 上 128 直接 OOM，
  64 在 epoch 1（AdamW 状态物化后）距天花板 0.5G 撞线。定稿 `batch_size 32 + grad_accum 4`（等效 128），
  峰值 21.1G，~150s/epoch。engine 加了 epoch 边界 `torch.cuda.empty_cache()`；`run_stage0.py` ENV 加
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- engine `run_epoch` 支持 gradient accumulation（micro-batch 尾巴也会 step）
- smoke（mode=none, 2 epochs）通过：loss 0.130→0.060，val 0.071
- pre-flight：三个 gated mode 各过 1 个真实 batch 的 forward+backward，全过
- **正式训练链启动**（串行：object_only → robot_only → phase_switched，各 101 epochs，预计 ≈12.6h）；
  输出在 `experiments/stage0_routing_causal_test/runs/<mode>_seed0/`

### 2026-07-29（当天第四批）— eval 侧在线分割接线
- 新增 `src/routedflow/eval_env.py`：`RobotGridLabelWrapper`（子环境级，subprocess 内渲染分割 →
  obs["robot_grid_labels"] (v,32)，`merge_dict` 自动堆叠成 (b,v,32)；首次 reset 时自动检测朝向）+
  `make_libero_env_masked` / `build_env_masked`（上游函数的插桩拷贝，注意与 upstream 同步）
- 新增 `src/routedflow/engine_eval_stage0.py`：gated rollout（phase 由 policy 自身 gripper 输出锁存，
  每 episode 重置；已知 1 帧边界近似——闭合那一步 policy 仍看到 approach gating，训练标签记为 transport）；
  结果 json 含 per-episode successes（之后做配对统计用）+ latch 统计
- `run_stage0.py` 加 `eval --mode <m> [--nroll 40] [--vec 10]` 子命令
- object_only（变体 i）训练完成：101 epochs，final train loss 0.0036 / val 0.036；链条已进入 robot_only
- 修 bug：eval 里 `model_cfg` 不能 `OmegaConf.to_container` 转 plain dict（ATM `__init__` 属性访问
  `track_cfg.track_fn`）——保持 DictConfig 直接 `**cfg.model_cfg`
- eval 实弹 smoke 通过（object_only ckpt，1 任务 ×10 episodes：SR=0.400，latch 行为合理）；
  正式 n=400 eval 等训练链结束后再跑，避免与训练抢 GPU

### 2026-07-30（第二批）— Stage-1 统一标签提取
- 新增 `src/routedflow/extract_c_labels.py`（每 demo 一次零漂移重放：C 标签 + EE/物体逐帧位姿轨迹
  + t=0 三渲染 512²（RGB/深度/geom-id 分割）+ 相机参数——之后 flow 标签可纯投影导出，不再进仿真）、
  `src/routedflow/qc_c_labels.py`（统计 + overlay 抽查 + 单任务接触点散布图）
- 根目录新增 `run_stage1.py`：`extract` / `qc` / `status`；`experiments/CURRENT` → `stage1_c_labels`
- 数据落 `data/c_labels/libero_spatial/<task>.h5`；QC 指标解释见实验 README
  （lift_gap ~3cm = lift 方案系统偏差实测；replay_ee_maxdiff = obs 子帧时序差，非重放漂移）
- **全量 500/500 完成 + QC 全过**（montage 20/20 落碗沿）。QC 加了两个离线指标：
  `grasped_by_motion`（t_g 后位移判被抓物体，10/10 任务正确；nearest 启发式 1.6% 错，弃用）、
  `robust_lift_gap`（9×9 窗口最小深度；悬空抓取 11.6→5.0cm——单像素 lift 在深度不连续处失效的实测）
- 坑：`extract` 的 skip-existing 会跳过 smoke 半成品 h5——重跑前先删对应任务文件

### 2026-07-30（第三批）— ⚠ mask 定向 bug 修复（阶段〇结果作废）
- 标签可视化（新增 `src/routedflow/viz_c_labels.py`，`run_stage1.py viz`：四联板 RGB+轨迹+朝向轴 /
  深度 / 分割 / 开度-phase）暴露：robosuite `get_camera_segmentation` **内置 `[::-1]` 返回正立图**
  （与 raw `sim.render` 相反）——convert 按 rgb flip 又翻一次、eval wrapper 同一假设也翻一次
  → **阶段〇 robot_seg/grid_labels 两视角全颠倒**（IoU 对正立真值 0.004/0.000），train/eval 自洽但语义全错
- 修复：`convert_libero_raw.py` / `eval_env.py`（删 `_detect_flip`）/ `extract_c_labels.py` 不再翻 seg；
  存盘数据离线翻回（阶段〇 150 h5 重算 grid_labels + 阶段一 500 demo seg0）；IoU 复验 0.971/0.923
- **阶段〇三变体需重训重评**（计划 §1.7 追加块）；阶段一 C 标签不受影响（接触点/深度/RGB 与 seg 无关）
- 教训：方向约定必须数值实测（行心/IoU 对真值），128px 目检不可靠

### 2026-07-30 — AFUN spike（阶段一 D1 选型）
- `third_party/AFUN` clone + conda env `afun`（cu130 torch 2.10，作者适配 Blackwell）；afun.pt 1.3G +
  Qwen3-VL-8B 17G；SAM3 走 `/workspace/huggingface_cache` 的既有 token/缓存（symlink 进 `.hf_home`）
- `experiments/stage1_afun_spike/`：`make_inputs.py`（512² 渲染 + GT 投影，翻转约定实弹验证）→
  `run_spike.sh`（`--no-refine` 直喂真值深度）→ `score.py`
- 结果：空间指代 3/5，选对时 mask ≤9px；**D1 修订为 C′**（自训前端 + AFUN mask prior 通道），
  判定全文见 spike README

### 2026-07-30（第四批）— 训练 pipeline 论文级架构图 v0.1
- `doc/pipeline_fig_v01.svg`（矢量、自包含、嵌入真实标签小图）+ `doc/pipeline_fig_v01_preview.png`；
  artifact: https://claude.ai/code/artifact/12bf11ca-7294-40c6-acb1-69a18af26e5d
- 内容：双 lane（approach 绿 / transport 橙）、L1–L5 + 零参数几何链、三套监督虚线、
  phase 路由开关、D7 curriculum 缎带；小图素材由 `data/c_labels` 的 demo_0 生成

### 2026-07-30（第五批）— 图制作流程入库（可原地修改）
- 图脚本正式入库：`src/routedflow/fig_assets.py`（素材生成 → `doc/fig_assets/`）+
  `src/routedflow/fig_build.py`（SVG/HTML 排版 → `doc/`）+ 根目录 `run_fig.py assets|build|all`；
  复跑验证 SVG 字节一致（确定性）
- **改图指南：`doc/PIPELINE_FIG_HOWTO.md`**——三层结构（改哪层）、helper API、全元素坐标地图、
  空余空间、QA 回路（http.server + playwright 截图；7 条踩坑自查）、同 URL 发布方法、
  与计划文档的同步义务。之后优化 pipeline 环节改图，按它原地改

### 2026-07-30（第六批）— 阶段一（L1+L2）全量代码 + L3/L5 骨架
- grill 会话拍板：范围 L1+L2 全量 + L3/L5 骨架；L3 GT=提取器补 link 位姿；**D8=a 定稿**（2D track+查询点深度）；
  新开 **D10**（L4 在课程③吃 L3 预测+flow 加噪，不做 GT teacher forcing）
- 新增 `src/routedflow/stage1/`：`dino_feats.py`（作业③，500×1369×768 f16 缓存）、
  `afun_prior.py`（作业②，模型载一次批量推理——**500 张实测 3 分钟**，可断点续跑；prior QC：
  in-mask 26% / ≤20px 58%，与 spike 3/5 一致）、`augment_links.py`（17 links×500 demos，无渲染重放）、
  `dataset.py`（heatmap σ2@128 / yaw36+pitch12 ±π 折叠 / w；5-fold split）、`model.py`（L1FrontEnd 7.5M
  + CHead，channel dropout 0.3）、`engine_train.py`（2s/epoch）、`eval_a1a2.py`（A1 峰值 NMS + prior
  分层按名索引；A2 ridge probe ×3 特征组含 z_random 对照）
- 新增 `src/routedflow/flow_models.py`（L3/L5 骨架：CondCrossAttn/CondAdaLN 零初始化 identity、
  QueryDepthEmbed=D8a、role 断言强制 text 不对称；ATM surgery 随 L3 训练 PR 落地）
- `tests/test_stage1_units.py` 10 项全过（轴约定拿 50 条真实 demo 锚定：>80% 俯仰 <30°）
- `run_stage1.py` 新子命令：`dino-feats / afun-prior / augment-links / train-l1 / eval-l1 / test`
  （train/eval 直通引擎参数）；`experiments/CURRENT` → `stage1_l1_training`
- smoke：10 epochs val 4.32→2.23；eval 全链路 json 正常。fold0 正式训练（200 epochs）启动
- 坑：h5py 文件锁——AFUN 作业读 c_labels 时 augment 的 r+ 打不开（BlockingIOError），错峰执行

### 2026-07-30（第七批）— pipeline 图 v0.2（随阶段一实现更新）
- 5 处文案更新（L1 实现细节 / prior 58% / D8a ✔ / D10 / bins 规格）+ 图注同步；
  `doc/pipeline_fig_v01.svg` 内容更新（文件名沿用 v01，版本语义在页面标题）、
  `doc/pipeline_fig_v02_preview.png`；artifact 同 URL 重发

### 2026-07-30（第八批）— pipeline 图 v0.3（通俗化重写）
- 用户反馈图内黑话看不懂（特权重放/防旁路/半截话）→ 图面文字全部通俗化 + 内部代号清零 +
  图下新增「术语注解」5 条（注①–⑤ 挂接）；HOWTO 加新规矩：图面语言给外人、代号留文档
- `doc/pipeline_fig_v03_preview.png`；artifact 同 URL 重发

### 2026-07-31 — 仓库上 GitHub
- 远程：git@github.com:yijie21/RoutedFlow.git；`.gitignore` 排除 `data/`（重建：`run_stage0.py prep`、
  `run_stage1.py extract|dino-feats|afun-prior`）、`third_party/`、checkpoint/h5、`runs*/`
- 导出物入库：`doc/c_labels_viz.html`（标签可视化页源码）、`doc/research_docs/`（研究文档快照，
  canonical 在 `/workspace/research/d4rt/`，push 前手动刷新）

### 2026-07-31（第二批）— 阶段〇重训中止，阶段一 5-fold 队列启动
- 用户指示优先阶段一：夜链在变体 ii ep11 处 kill（变体 i 重训完成保留；iii/evals 未跑，重启说明
  见 stage0 实验 README）；夜链曾在变体 i ep9 被外部占用饿死 14.4h（单次，未复发）
- `stage1_chain.sh`：fold1–4 训练 + fold0 no-prior ablation + 6 个 run 的 A1/A2 评测（估 ~1.2h）

### 2026-07-31（第三批）— fold0 逐 episode 可视化评测
- 新增 `src/routedflow/stage1/eval_viz.py`（`run_stage1.py eval-viz --run <name>`）：val_id 40 集
  × 5 文件（AFUN mask / C 预测投影 / 冻结 ATM 基座 flow + GT 参照 / 相位边框示教视频 / info.json）
- 发现：w 推理误差 0.039 vs teacher-forced 0.003 → **feat-gather exposure gap 实测**；
  on_the_wooden_cabinet 5/5 近失（14–27px）
- 诚实标注：flow 图为无 conditioning 的冻结基座（L3 未训练）；视频为示教回放（无 policy）

### 2026-07-31（第四批）— approach 分支全链（阶段二）实现 + 联合训练启动
- grill 拍板：一步联合 warm start（L1=fold0 ckpt，L3=ATM track transformer，L4 从零）/
  双视角图像+仅 agentview flow（wrist flow 置零）/ **FK 链点方案（D11）**：32 个运动链插值点，
  身份跨帧固定，零噪声 GT + 查询深度免费 + 部署走 proprio+FK，robot mask 从 L3 输入退役 /
  rollout 成功=闭合后脚本提起 5cm 且夹持宽度合理
- 新增 `src/routedflow/stage2/`：`convert_light.py`（500 demos 无重放打包 → `data/atm_libero_light/`，
  分钟级）、`chain_points.py`（chain_uv/chain_z 入 c_labels + QA overlay，**t=0 100% in-mask**）、
  `flow_l3.py`（**D3 v1 = z 占 text 槽**；深度通道/QueryDepthEmbed 零初始化；ATM warm start）、
  `joint_model.py`（ApproachPolicy 外部 flow 源 + JointApproachModel 三 loss，λ=0.5/1.0/0.1，
  flow 加噪 0.01=D10）、`engine_train.py`、`eval_env2.py`（子环境 ChainStateWrapper：chain_uv/z/ee_z/
  rgb512@reset）、`eval_rollout.py`（512 首帧现场 DINO→z，prior 置零；闭合锁存→脚本提起判定；
  POLICY 水印视频）
- 根目录 `run_stage2.py`：convert / chain-prep / test / smoke / train / eval / status
- 单测 3/3；训练 smoke（三 loss 齐动）；rollout smoke 全通路 rc=0
- **联合训练运行中**：13669 approach 窗口，29.7M 可训参数，~240s/epoch × 60 ≈ 4h
- 坑 ×3：atm.model 先于 atm.policy.vilt import → 循环 import 坏缓存（flow_l3 顶部 order guard）；
  vilt 的 train() 会调 self.track.eval()（del 不得，用 nn.Identity 占位）；eval 侧 obs 键名是
  robot0_joint_pos/robot0_gripper_qpos（engine/utils.py 的 obs_key_mapping）

### 2026-07-31（第五批）— 首次端到端 rollout：approach 分支 WORK
- 三轮诊断链：① 包装器 512 渲染毁 128 视口（整集观测中心裁切）→ 环境原生 512；② gripper 通道
  0 附近抖动 + 首次>0 即锁存 → ~26 步半空过早闭合（**D6 切换信号脆弱性的执行层实证**）→
  连续 3 步 >0.5 鲁棒锁存；③ 训练在 ep24 被 RAM OOM 静默击杀（worker CoW 泄漏，已修
  persistent_workers）——用 ep~20 的 ckpt_best 出的数
- **结果（10 eps/任务，闭合后提起 5cm 判定）：train-8 = 0.2875**（next_to_plate/ramekin 0.5、
  stove 0.4、on_cookie_box 0.0），**ood-2 = 0.10**；rollout 时 AFUN prior 置零
- 已知增益杠杆（未做）：补完 60 epochs、λ_action 日程、latch 细化、ood 差距随 C-VLM held-out
  弱项复合（stage-1 A1 ood 0.33）
- 坑：后台命令 `; echo rc=$?` 会把任务退出码遮成 0——真实 rc 只在输出文件里

### 2026-08-04（第三批）— goal 混训判决 + 实验总结 artifact
- goalmix 25k 跑完（途中三次启动：①会话进程退出连坐后台任务——改 `setsid nohup` 脱钩；
  ②邻居实验 54G 挤爆 cgroup 被 OOM 杀——engine 加 `--stream-train` 流式模式（~25G→~8G，
  步速仅慢 35%）；③流式模式跑通全程）
- 行为层：探针曲线 0.05→**0.15**(5k-7.5k 峰，触及冠军同协议标定线)→过训衰减（第九批现象
  第五次重演）；ckpt@7500 全量 rollout **train-8 0.1375 / ood-2 0.15**（ood 首次方向性向好，
  train 8 任务 7 个非零；冠军 0.2875 仍未被超越）
- 机制层：**真实 z 分化测试 = 0.36%**（同场景 8 指令的真 z，flow 几乎不变）——数据歧义方向
  正确但 D3 v1 单 token 注入太弱，歧义窗口少数（手臂一动目标即泄露），模型宁吃均值亏不学 z；
  行为层收益归因于数据正则化。下一轮候选：方案 A（z 直连 L4 语言槽，复用 use_language_token 管线）
- 实验总结 artifact 发布（Motivation 三痛点三回答 + 注解架构图 + Contribution 四主张带证据
  状态 + 附录 A 三个关键测量的自包含定义）：claude.ai/code/artifact/751189da

### 2026-08-04（第二批）— goal 混训启动（数据层断旁路）
- convert_light 扩到 libero_goal（分钟级）；Stage2Dataset 加 `extra_suites`：额外 suite 的
  含抓取任务全量进 train 窗口（零闭合任务经 c_labels 空校验自动排除），fold/val/rollout
  协议保持 spatial 不变；s1 对齐经 Stage1Dataset extra_suites 同路径
- 机制：goal 十任务共享同一场景、仅指令不同——分布内视觉无法识别任务，梯度被迫走
  z→flow→action；成功判据 = 训练后 z 置换测试从 0.5-0.9% 显著上升 + goal 训练窗口的
  action loss 能降
- joint_fold0_seed0_goalmix 启动：32,339 窗口（spatial 13,669 + goal 18,670，goal 占 58%），
  expand L1 warm start（stage-1 开小灶时已见过 goal），探针照旧

### 2026-08-04 — task_emb 论断撤正 + 接口正式摘除
- 控制变量复测：task_emb 对策略输出影响 **精确 0.0**（`use_language_token: false`
  自阶段〇配置起恒关，文本从未进过策略）——第三批"task_emb 旁路"论断**撤正**；
  幸而用户坚持"直接去掉"触发上游代码核读，避免一场无效的 7h 断旁路重训
- **真旁路 = 场景布局**：spatial 各任务摆放不同，分布内视觉即可识别任务（flow 对动作
  影响 1.32 vs task_emb 0.0 vs z→flow 0.5-0.9%）；解释分布内有分/ood 0.1/z 装饰品全链
- ApproachPolicy 接口摘除 task_emb（forward_loss/act 新签名，内部零占位喂 upstream
  的 compute-and-discard 管道；third_party 不动）；joint_model/eval_rollout 调用点同步；
  单测 +接口断言与零文本 act 实跑 = 3/3
- 断旁路唯一可行路径 = **数据层视觉歧义**（libero_goal 同场景 10 目标混训，
  z 置换测试在 goal 任务上应翻转为毁灭性 = 现成因果诊断实验）；待拍板
- 文档同步撤正：PHASE_GATED_FLOW_ROUTING §4.3 / PIPELINE_IMPL_PLAN 更新记录 /
  walkthrough ⚠ 标注

### 2026-08-03（第三批）— 标定裁决 + z 旁路实锤 + walkthrough 文档
- 独占 GPU 标定：harness 无罪（冠军复现 0.2/0.4）；freezeL1 真零；探针数字全程可信
- **z 敏感度测量：所有模型（含冠军）z 对 flow 影响仅 0.5-0.9%**——L4 的 task_emb 输入
  是旁路，策略靠 task_emb+图像记任务，z 通道形同虚设；24h 的 L1 优化优化了不承重通道；
  低 SR 区间 5-20 集评测方差大，新旧 L1 run 的 SR 差异多在噪声内
- V/L/A 动静不匹配 story 写入 PHASE_GATED_FLOW_ROUTING.md §4.3（z 旁路为"必须结构强制"
  的实证；task_emb 开/关即现成 bypass 消融）；待用户拍板：断旁路重训
- 新增 `doc/TRAINING_PIPELINE_WALKTHROUGH.md`：全 pipeline 输入→输出代码导读
  （离线数据链字段表、单样本张量表、三段 forward、循环/探针/产物、部署对照、阅读顺序）

### 2026-08-03（第二批）— 毒窗口假设证伪 → 冻结 L1 方案
- expand2（干净窗口 + 扩数据 L1）探针照样贴地（2500-17500 步全 ≤0.1，train 任务归零）——
  6.8% 毒窗口是真实污染但**不是崩塌主因**；18k 步处用户拍板杀掉（省 1.5h）
- 嫌疑收敛（三 run 对照：旧 L1@8.5k 0.2875 / 旧 L1@7k 0.1625 / 新 L1×2 ≈0）：
  ① joint 阶段用 360 张 spatial 样本继续微调新 L1，把 A1 ood 0.70 的好 z 磨掉的速度快过
  策略学会消费它；② step 制 T_max=25k 的慢退火让 lr 高位停留更久（旧冠军是 60-epoch 快退火）
- engine 加 `--freeze-l1`（文献"两阶段冻结"姿势）：L1+CHead requires_grad=False + 移出
  optimizer + λ_C=0 + **恒 eval 模式**（训练时 z 无 dropout 噪声 == rollout 分布）；
  joint_fold0_seed0_freezeL1 启动（探针裁决：恢复≈0.29 ⇒ 嫌疑①坐实且方案即产品）

### 2026-08-03 — ⚠ t_g v2 毒化 stage-2 窗口：探针曲线全程贴地 → 解耦修复
- joint_fold0_seed0_expand 跑满 25k：探针 SR 全程 0-0.1（train 任务归零；唯 ood between
  升到 0.2-0.4，与更强 z 一致）——探针机制首战即抓到问题，val loss 又一次全程失明
- 根因（数据实证）：t_g v2（最后闭合）让 fumble demo 的 approach 窗口延伸过失败闭合段，
  **6.8% 窗口（1015/14875）含"闭合-失败-张开"序列** = 把 D6 过早闭合当行为教给策略，
  latch 对此零容忍（3 步闭合→错位提起→判负）
- 修复 = 解耦 t_g 两种用途：C 标签（stage-1）保持最后闭合（A1 ood 0.70 收益保留）；
  **stage-2 窗口边界回退首次闭合**（读 phase latch，argmax）→ 窗口数精确回到 13669 净基线
- 决定性 A/B 启动：joint_fold0_seed0_expand2 = 干净窗口 + 扩数据 L1（探针曲线直接对比
  expand 的贴地曲线与旧 0.2875）
- AFUN 补跑竞态事故：守候循环在两次训练的间隙触发，随即撞上 expand2 启动——900/900 demos
  全部 CUDA OOM，失败路径存**零 mask**（afun_prior.py:89）→ 两 suite 缓存整体无效
  （QC 0/500、0/400 露馅），已隔离至 stage1_cache/*_invalid_oom/（dataset 回退零 prior；
  fold0_seed0_expand 训练早于这些文件生成，不受影响）。教训：GPU 空档守候会与后续排队
  任务竞态，重资源补跑必须显式排在训练完成之后

### 2026-08-02（第十一批）— 🎯 扩数据打中靶心：A1 ood 0.33 → 0.70
- fold0_seed0_expand（1260 样本 + 增强 + t_g v2，4000 步，best val@~600-800）A1 对比旧基线：
  **val_ood 0.33→0.70**（table_center **0.00→0.82**，between 0.66→0.58）、val_id 0.775→0.75（噪声内）；
  prior 分层恢复正序（correct 0.76 > wrong 0.36，旧模型倒挂）——语义单点饥饿诊断证实
- stage-2 engine 装**闭环探针**（--probe-every 2500：子进程 rollout 前 4 任务（2 ood+2 train）
  ×5 eps，SR 进 wandb/jsonl + ckpt_step 快照留档；robomimic 式 rollout-based selection；
  探针失败只告警不伤训练）；smoke 含探针全链通过
- joint_fold0_seed0_expand 启动：新 L1 warm start + 探针 + ckpt_last，25k 步（这次会拿到
  完整的"闭环 SR vs 步数"曲线，直接检验第九批的 BC 崩塌现象）

### 2026-08-02（第十批）— C 头扩数据开工（grill：范围/t_g 规则/喂法/增强）
- 闭合周期实测（去抖前 raw 边缘）：spatial/object ~10% 补抓（fumble）、goal 100 条零闭合
  （push 类，自动跳过）、libero_10 71% 多原子——多子句拆分方案已档（周期切分 + 逐周期渲染 +
  grasped_body↔子句对齐），v1 先做 object+goal，libero_10 留 v2
- `phase.grasp_cycles(debounce=3)`：去抖 close/open 周期（D6 鲁棒锁存教训在标签侧复用）；
  **t_g 规则 v2 = 最后一次去抖闭合**（fumble demo 首闭是失败抓取）；attrs 加 n_cycles；
  旧 spatial 标签备份于 data/c_labels_v1_tgfirst/ 后全量重提
- 三 suite 提取链后台运行（extract→links→chain ×3 + 新 suite dino/afun 缓存）
- Stage1Dataset 加 raw/augment/extra_suites 模式（保留 cached 路径给 Stage2Dataset）：
  接触点保持的 random resized crop（无翻转/旋转——空间语言与 base 系朝向标签约束）+
  颜色抖动；heatmap/prior-grid 变换后重算；engine 在线 DINO（--cached-feats 走旧路）
- 单测 +1（grasp_cycles 去抖/补抓语义）= 11/11；坑：`cat >>` 会把测试追加到
  `if __name__` 块后——runner 收集不到，必须插在 guard 之前
- 提取链落地实况：三 suite 标签+links+chain 完成（char test 5/5——重放确定性使 spatial 快照
  逐位复现）；goal 2 个任务全零闭合（push_the_plate 等）→ chain_points/dino_feats 加空任务守卫；
  `turn_on_the_stove` 会闭爪拧旋钮所以有标签；**新 suite 的 AFUN prior 因邻居占 GPU 推迟**
  （缺失时 has_prior=0 + channel-dropout 兜底 = rollout 同款零 prior 路径；后台守候显存自动补）
- 扩数据 stage-1 正式训练启动：fold0_seed0_expand，4000 步，train ~1260 样本（360 spatial +
  object/goal 全量），在线 DINO + 增强

### 2026-08-02（第九批）— ⚠ 关键发现：BC 训练越久，闭环 SR 越差
- 三点对比（同 rollout 协议，10 eps/任务）：旧 ckpt@~8.5k 步 **0.2875** / 新 ckpt_best@7k
  **0.1625** / 新 ckpt_final@25k **0.025**（train-8；ood-2 三者均 0.10）
- 同 run 内 7k→25k 从 0.1625 崩到 0.025 是干净证据：BC 记忆越锐利，闭环 covariate shift
  下越脆；**val 一步 BC loss 全程平台 0.014-0.016，对此完全失明**——模型选择不能再靠 val loss
- 混杂因素（诚实记录）：新 run 在 4k 处无 moments 续跑（warm-restart 扰动），故旧@8.5k vs
  新@7k 不是纯步数对比；但趋势方向由同 run 内对比锚定
- 当前冠军仍是旧 run 的 ckpt（experiments/stage2_approach_joint/runs/joint_fold0_seed0/）；
  两次新 rollout 均已上报 wandb（rollout_joint_fold0_seed0_s25k / _final）
- 待决策（下轮）：训练中周期性闭环探针（2 任务×5 eps ≈ 10 min）做 rollout-based 选择；
  周期性快照（每 2500 步留档）替代只存 best/last

### 2026-08-02（第八批）— 25k 步重训完成（wandb 首个正式 step 制 run）
- joint_fold0_seed0_s25k 跑满 25000 步（中途 OOM 一次，resume 自 step 4000 续毕）；
  wandb: runs/t7l8gse3（0-6.3k，被杀段）+ runs/uhv4lo2u（4k-25k，续跑段）
- 曲线判读：train act 0.0018（持续降），**val l_action 自 ~1k 步起平台 0.014-0.016**
  （best 0.0137@7k），val 总 loss 一路涨（C-VLM 在 held-out demo 过拟合）——一步 BC loss
  是弱代理，结论交给闭环 rollout（ckpt_best@7k 已开测）
- 顺手：val 每次落 ckpt_last（本次运行未生效——进程先于改动加载；下次起 resume 不再回退到 best）

### 2026-08-02（第七批）— OOM 第二击 → resume 机制 + val 流式读
- 25k 步正式重训（joint_fold0_seed0_s25k）在 **step 6320** 再次被 cgroup OOM 击杀（rc=247）：
  同机他人实验涨到 ~41G（/venv/main 三进程），85 GiB 上限被挤爆，killer 挑 RSS 最大的我们
- 修复①：engine 加 `--resume <ckpt>`——snapshot 现在带 opt.state_dict + best，恢复模型/优化器/
  cosine 进度（旧格式 ckpt 兼容：无 opt 则 moments 重建）；每次 OOM 损失有界化
- 修复②：val 数据集 `cache_all=False` 流式读（BCDataset 索引表本就不依赖缓存）——train/val_id
  读同 8 个任务目录，双份全量缓存纯浪费 ~8G；val loader 恢复 workers=2（无缓存可 CoW）
- Stage2Dataset 的 cache_all assert 移除；重训从 step 6000 ckpt_best 续跑，--val-every 1000

### 2026-08-02（第六批）— 训练循环改 step 制（VLA 惯例）
- stage1/stage2 engine 弃 epoch：dataloader 无限 cycle，`--steps` 定训练长度（optimizer step），
  LR cosine T_max=steps；`--log-every`（默认 20）窗口均值上报 train metric，`--val-every`
  （stage2 500 / stage1 100）跑全量 val + 存 ckpt_best；wandb/jsonl 的 x 轴统一为 step
- 理由：数据集大小可大可小，epoch 无跨实验可比性；step 对齐计算量（OpenVLA/π0 同款做法）
- ckpt 元数据 `epoch` → `step`（eval_rollout 读取处向后兼容旧 ckpt）；默认量程等换算旧配置
  （stage2 25000 步 ≈ 旧 60 epochs；stage1 1200 步 ≈ 旧 200 epochs）；smoke = 8 步含 val
- 坑（环境）：本容器 cgroup 内存上限 **85 GiB**（`free` 显示的是宿主 125G，别信）；smoke 曾被
  OOM killer 击杀（rc=247 = SystemExit(-9)，SIGKILL 无 traceback）——同机他人实验占 ~17G 时
  加载期 cgroup 水位可顶到上限；重跑通过但峰值贴顶，开长训前先看 `/sys/fs/cgroup/memory.current`

### 2026-08-02（第五批）— wandb logging 接入（grill 决策：默认开）
- 新增 `src/routedflow/wandb_util.py`：fail-safe 包装（初始化/上报全 try-except + init_timeout 30s），
  wandb 不可用只打印警告降级，训练绝不因 log 挂掉；jsonl/summary.json 仍是本地真相源
- 接入三处：stage1/stage2 `engine_train.py`（逐 epoch 镜像 jsonl 全字段 + lr + sec，
  summary 记 best val）、`eval_rollout.py`（train8/ood2 + per-task SR 标量 + wandb.Table，不传视频）
- 组织：project=`routedflow`，run 名=本地 run 名，job_type=stage1/stage2/rollout；
  `--no-wandb` 关（smoke 命令自带）；历史 jsonl 不补录
- 三路径验证过：在线（server 端确认 run 落库）/ `--no-wandb`（零痕迹）/ 断网（警告+照跑）；
  `wandb/` 本地缓存入 .gitignore

### 2026-08-02（第四批）— flow 广播错位复核：**假阳性，撤销**
- 动手修复前先跑运行时探针：`act()` 内 track_obs 实测 (b,v,**t=1**,fs=10,c,h,w)——upstream ViLT
  把历史帧堆进 **fs 维**（queue cat 在 dim=2），t 恒为 1，时序语境走 latent_queue
- `track_encode` 按 t 维配 flow ⇒ rollout 的 `pred[:,None]` (t=1) 严丝合缝，无广播；
  每个历史 latent 在它那步 act 已配过自己的 flow，train/rollout 配对语义一致
- **结论：无 bug、无需修复、rollout 无需重跑**（同 ckpt 会复现同一组数）；spec §5-1 已改记假阳性
- 教训入档：只读训练侧代码推断 act 侧张量布局不可靠，此类 train/test 错位指控必须探针实证后立案

### 2026-08-02（第三批）— retrofit 阶段3：top-2 逆向 spec + characterization test
- 逆向 spec 落盘（只写实际行为，歧义单列）：`.scratch/retrofit-rollout-chain/spec.md`
  （eval_rollout+eval_env2）、`.scratch/retrofit-label-extraction/spec.md`（extract/augment/chain/convert_light）
- 一致性核查 5 项通过（text emb mean(0)、DINO 预处理逐行同、L3 单帧、chain_uv 约定、BCDataset forward 窗口）；
  **抓到 1 个真 train/test 错位**：训练 flow ctx (b,10) 每历史槽配自己帧的 flow，rollout `pred[:,None]`
  把当前 flow 广播进 10 槽——不崩、能出 0.29，但分布不一致（候选修复：rollout 维护 flow 队列；已有网，可安全改）
- characterization test 7/7 绿：`tests/test_char_labels.py`（500 demos 全量清点零出视野/零 NaN +
  demo_0 数值快照 + light 文件跨源 phase 相等 + in-mask ≥0.95 定向锚）、`tests/test_char_rollout.py`
  （env obs 契约 + upright 亮度锚 + env/离线两条投影路径 chain_uv 统计 0.006 内互证 + 纯函数快照）；
  `run_stage2.py char` / `char-env` 触发
- 事实修正：chain_uv **非** [0,1]——实测 [-0.41, 0.78]，t=0 约半数链点在画面外（train/rollout 一致，非 bug）
- eval_env2 docstring 假 `rgb512` 键删除（代码从未写入）；风险图第三名按方法论**不碰**

### 2026-08-02（第二批）— retrofit 阶段2：静态护栏开通
- 新增 `pyrightconfig.json`（basic 模式，venv=atm5090，extraPaths=src+third_party/ATM）、
  `ruff.toml`（E4/E7/E9/F/PLE；third_party/data/doc/experiments 显式排除=无人看管区，勿静默扩大）
- 工具：ruff 0.16.1（装进 atm5090）、pyright 1.1.411（npx，需 `PATH=$HOME/.nvm/.../bin`）
- 分诊结果：**运行期错误四类清零**。ruff 运行期类 0 条（123 条全风格）；pyright 183 errors
  基线全为 stub 噪声——h5py 联合类型 ~30、numpy overload ~138、PIL 常量 3（运行时存在）、
  torch.hub object 2、`draw_star(d, *ndarray)` 变长解包 2、复合布尔守卫 narrowing 2
- 复跑基线：`npx pyright` 应 ≈183 errors / `ruff check . --select E9,F63,F7,F82,F811,PLE` 应 0 条；
  显著偏离即新引入问题

### 2026-08-02 — agent skills 仓库配置（/setup-matt-pocock-skills）
- 新增根目录 `CLAUDE.md`（Agent skills 配置段）+ `docs/agents/`：`issue-tracker.md`
  （issue 用本地 markdown，`.scratch/<feature>/` 一目录一 feature，不用 GitHub Issues）、
  `triage-labels.md`（默认五标签）、`domain.md`（single-context：根 `CONTEXT.md` + `docs/adr/`，惰性创建）
- 注意：`docs/`（agent 约定文档）与既有 `doc/`（图/研究文档快照）并存，勿混用
