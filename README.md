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
