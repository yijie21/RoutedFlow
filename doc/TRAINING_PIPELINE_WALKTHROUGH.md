# 训练 pipeline 代码 walkthrough（从输入到输出）

> 2026-08-03 整理。对应当前代码状态（step 制训练 + 闭环探针 + freeze-l1 开关）。
> 建议对照源码阅读，每节都给了文件与函数锚点。已知结构问题用 ⚠ 标出。

---

## 0 · 一张图看全局

```
【离线，跑一次】
raw LIBERO hdf5 ──┬─→ convert_light.py ──→ data/atm_libero_light/   （视频+动作，策略吃）
                  └─→ extract_c_labels.py → data/c_labels/           （C 标签+几何）
                          └→ augment_links.py（补 link 位姿）
                                └→ chain_points.py（补 chain_uv/chain_z = L3 的查询点与 GT）
                       dino_feats.py / afun_prior.py → data/stage1_cache/（特征缓存）

【阶段一，可选 warm start】
stage1/engine_train.py：L1(C-VLM)+CHead 单独预训练 → runs/<name>/ckpt_best.pt

【阶段二，主训练】
run_stage2.py train → stage2/engine_train.py
  Stage2Dataset ─→ JointApproachModel.forward_loss ─→ L = 0.5·L_C + 1.0·L_flow + 0.1·L_action
                                                        ↑ 全程可微：L_action→L4→flow→L3→z→L1

【评测】
eval_rollout.py：真实环境闭环（训练中的探针也是它，子进程小规模跑）
```

---

## 1 · 离线数据链：磁盘上每个文件是怎么来的

### 1.1 `src/routedflow/stage2/convert_light.py` —— 策略的"食粮"

输入是 LIBERO 原始 demo（`/workspace/datasets/libero/hdf5/<suite>/<task>.hdf5`）。
对每条 demo 只做纯 IO（不开仿真器），写出
`data/atm_libero_light/<suite>/<task>/all/demo_X.hdf5`：

| 字段 | 形状 | 说明 |
|---|---|---|
| `root/actions` | (T, 7) | 7 维 = 3 平移 + 3 旋转 + 1 夹爪（>0 为闭合） |
| `root/<view>/video` | (1, T, 3, 128, 128) uint8 | 原始 obs 竖直翻转成 upright；view ∈ {agentview, eye_in_hand} |
| `root/extra_states/*` | (T, d) | joint_states、gripper_states 等本体状态 |
| `root/phase` | (T,) uint8 | 0=approach，1=transport（首次闭合后锁存） |
| `root/<view>/tracks, vis` | 零假体 | 只为兼容 ATM dataloader，训练不读 |

### 1.2 `src/routedflow/extract_c_labels.py` —— C 标签与场景几何

对每条 demo 做一次零漂移重放（逐帧 `set_state`，只在 t=0 渲染），写出
`data/c_labels/<suite>/<task>.h5`。关键字段：

| 字段 | 说明 |
|---|---|
| `rgb0 / depth0 / seg0` | t=0 的 512² 渲染（C-VLM 的输入图 / 深度 / 分割） |
| attrs `t_g` | 抓取时刻 = **最后一次去抖闭合**（`phase.grasp_cycles`，v2 规则） |
| `contact_rowcol` | E(t_g)（抓取时的手爪位置）投影到 t=0 相机的像素坐标 —— heatmap 的监督 |
| `ee_pos / ee_quat / gripper_q` | 重放得到的手爪轨迹与指宽（朝向与 w 的监督取 t_g 帧） |
| `phase` | 与 light 文件同源（首次闭合锁存）——注意与 t_g 的"最后闭合"是两个概念 |

### 1.3 `augment_links.py` → `chain_points.py` —— L3 的查询点与零噪声 GT

`augment_links` 重放补存每帧 17 个机器人 link 的世界系位姿；`chain_points` 把它们
变成 **32 个 FK 链点**（D11：沿运动学链按 t=0 弧长取固定凸组合权重，逐帧变换后投影），
追加进 c_labels：

- `chain_uv` (T, 32, 2)：归一化像素坐标 (x=col/512, y=row/512)。注意**不限于 [0,1]**
  （t=0 约半数链点在画面外，实测范围 [-0.41, 0.78]）；
- `chain_z` (T, 32)：相机系深度（米）。

这两个量给了 L3：任意帧的查询点 + 未来 16 帧的 flow GT（`chain_uv[t:t+16]`），
不需要每帧分割或深度模型；部署时同样的点由本体感知 + FK 算出（eval_env2 的 wrapper）。

### 1.4 特征缓存（stage-1 用）

- `dino_feats.h5`：每张 rgb0 过冻结 DINOv2 ViT-B/14（512→518）得 (1369, 768) patch 特征；
- `afun_prior.h5`：AFUN 对 rgb0+指令输出的 512² 亲和 mask（弱先验，训练带 0.3 dropout，
  rollout 时置零）。

---

## 2 · 训练入口

`python3 run_stage2.py train [...]` 只做两件事：设环境变量（GPU、EGL、PYTHONPATH），
然后把参数转给 `src/routedflow/stage2/engine_train.py` 的 `main()`。

`main()` 的顺序（对照源码读）：

1. 建数据集与 DataLoader（§3）；
2. 建 `JointApproachModel`（§4），可选 `--l1-ckpt` warm start、`--freeze-l1`；
3. AdamW 参数组：L3 lr=1e-4，L4 lr=3e-4（未冻结时 L1+CHead lr=1e-4）；
   cosine 退火，T_max = `--steps`（默认 25000 个 optimizer step）；
4. step 制主循环（§5）。

## 3 · Stage2Dataset：一个训练样本长什么样

`src/routedflow/stage2/dataset.py`。继承 ATM 的 `BCDataset`（负责视频窗口切片），
叠加三层自己的逻辑：

- **窗口过滤**：只保留 approach 窗口——窗口末端 ≤ 首次闭合 + 2 帧
  （⚠ 这里刻意用 `phase` 的首次闭合而不是 attrs `t_g`：t_g v2 曾让 6.8% 窗口
  含"闭合-失败-张开"序列，教坏策略，见文件内注释）；
- **fold 划分**：与 stage-1 完全同一套（fold0 = 8 任务训练，每任务字典序前 45 条 demo）；
- **链数据与 s1 样本**：按 (task, demo) 对齐附加。

`__getitem__` 返回 10 元组（bs=8 时 collate 后的形状）：

| 张量 | 形状 (batch 后) | 给谁 | 语义 |
|---|---|---|---|
| `obs` | (8, 2, 10, 3, 128, 128) | L4 | 双视角 10 帧图像窗口 |
| `track_obs` | (8, 2, 10, 1, 3, 128, 128) | L3/L4 | 每窗口步 1 帧（track_obs_fs=1） |
| `track` | 零假体 | — | 兼容占位 |
| `task_emb` | (8, 768) | 无人 | 已从 L4 接口摘除（2026-08-04）；dataset 仍返回但 `_task_emb` 弃用 |
| `actions` | (8, 10, 7) | L4 | BC 监督 |
| `extra_states` | dict, (8, 10, d) | L4 | joint/gripper 状态 |
| `chain_query` | (8, 10, 32, 2) | L3 | 每窗口步的链点查询（归一化 xy） |
| `chain_depth` | (8, 10, 32) | L3 | 查询点深度（D8a 通道） |
| `chain_gt` | (8, 10, 16, 32, 2) | L3 | 每步未来 16 帧的零噪声 flow GT |
| `s1` | dict | L1 | 该 demo 的 stage-1 样本（dino 1369×768、prior 37²、text 768、heatmap 128²、yaw/pitch bin、w、contact） |

## 4 · JointApproachModel.forward_loss：一个 batch 的完整前向

`src/routedflow/stage2/joint_model.py`。三段：

**第一段 · L1 + CHead（每窗口一次，不逐帧）。**
`hm_l, z, fm = l1(s1.dino, s1.prior, s1.text)`。L1 内部（`stage1/model.py`）：
1369 个 DINO patch 各拼上 1 维 prior 标量 → 线性投影成 token，再拼一个 text token
和一个可学习的 [C] token，过 3 层自注意力；**z = [C] token 的输出**（384 维），
patch 侧上采样成 128² heatmap logits。CHead 用 z + GT 接触点处的特征预测
yaw(36 bins)/pitch(12 bins)/指宽 w。`stage1_loss` = KL(heatmap) + CE(朝向) + L1(w)
→ **L_C**。（`--freeze-l1` 时这段只前向不训练，λ_C=0。）

**第二段 · L3 robot-flow（逐窗口步）。**
把 track_obs 的 agentview 帧摊平成 (8·10, 1, 3, 128, 128)，链点查询与深度同样摊平，
z 复制到每个窗口步。`flow_l3.py` 的 `L3RobotFlow` 是 ATM TrackTransformer 的手术版：
**z 经 cond_proj 占据原 text token 的位置**（D3：z 是 flow 唯一任务通道），查询点
附加零初始化的深度嵌入。输出 (8·10, 16, 32, 2) 的未来轨迹，对 chain_gt 做 MSE
→ **L_flow**。

**第三段 · L4 action expert（吃 L3 的预测，梯度不断）。**
预测 flow 重排回 (8, 10, 16, 32, 2)，训练时加 0.01 高斯噪声（D10：不用 GT teacher
forcing），`l4.set_flow(...)` 存进 `ApproachPolicy`。`ApproachPolicy` 是 ATM
`BCViLTPolicy` 的手术版：内部冻结 track transformer 换成 `nn.Identity`，
`track_encode` 改为读外部 flow（agentview 槽位放预测 flow，wrist 槽位置零）。
之后走 upstream 原味前向：图像 patch + flow token + extra states
→ 时空 transformer → 动作头，BC loss → **L_action**。task_emb 已从接口摘除
（2026-08-04；实测其影响本为 0.0——`use_language_token: false` 一直关闭）。

**方案 A（`--z-to-l4`，2026-08-04 落地，默认关）**：z (384) 顶替原 BERT emb 走这条
曾经闲置的语言槽——`language_encoder_spatial/temporal` 输入改 384、两个
`use_language_token` 打开。z 以 spatial 语言 token + temporal 语言 token 两个位置
进入 L4，L_action 由此获得**不经 L3 flow 的直达梯度**到 L1。raw text 仍然不进
L4（z 依旧是唯一任务通道）；rollout 时 eval_rollout 从 ckpt cfg 读到 flag，把
每 episode 算一次的 z 传给 `act(obs, extra_states, z=...)`。

⚠ **真正的旁路是场景布局**：spatial 各任务物体摆放不同，策略与 L3 只看图像就能
在分布内识别任务，实测 z 对 flow 影响仅 0.5-0.9%（z 未承重）。对策在数据层：
混入"同场景、多目标、仅语言可分"的任务（libero_goal）迫使梯度走 z。

**汇总**：`loss = 0.5·L_C + 1.0·L_flow + 0.1·L_action`，一次 backward。
梯度路径：L_action → L4 → 预测 flow → L3 → z → L1 融合层（全程可微，这是
"整链联合训练"的落点；但因旁路存在，z 支路的实际梯度贡献目前接近零）。

## 5 · 主循环、验证、探针、产物

`engine_train.py` 的 step 制循环（VLA 惯例，dataloader 无限 cycle）：

- 每个 optimizer step 累积 `--accum 4` 个 micro-batch（bs 8 → 有效 32）；
- 每 `--log-every 20` 步：窗口均值写 `metrics.jsonl` 一行 + wandb（step 轴）；
- 每 `--val-every` 步：全量 val（流式读盘，不占内存）→ `ckpt_last.pt` 无条件落盘，
  val L_action 创新低时另存 `ckpt_best.pt`（⚠ robomimic 告诫 + 我们三次实证：
  这个 val loss 与闭环成功率几乎无关，只作参考）；
- 每 `--probe-every 2500` 步：存 `ckpt_step<N>.pt`，**子进程**跑 `eval_rollout.py`
  （前 4 个任务 × 5 集），per-task SR 写回 wandb `probe/*` —— 这才是真正的模型选择依据；
- 结束存 `ckpt_final.pt`。

产物目录 `experiments/stage2_approach_joint/runs/<name>/`：
`metrics.jsonl`（本地真相源）、各 ckpt、wandb run（project=routedflow, job_type=stage2）。

## 6 · 部署侧对照（eval_rollout.py 怎么消费训练产物）

- **z 每集只算一次**：reset 帧（512²）现场过 DINOv2 → L1 → z，prior 置零，整集冻结
  ——这就是 story 里"L 静态摊销一次"的实现；
- 每步：环境 wrapper（`eval_env2.py`）用 FK 现算 32 链点 → L3 预测 flow →
  `set_flow` → L4 出动作（act 路径 t=1，历史语境由 ViLT 的 latent 队列承担）；
- 策略连续 3 步夹爪指令 >0.5 触发锁存 → 脚本提起 25 步 → 成功 = 升高 ≥3cm 且
  指间宽度落在 (0.004, 0.075)。

## 7 · 建议阅读顺序

1. `stage2/dataset.py`（一个样本是什么）→ 2. `stage2/joint_model.py`（一个 batch
怎么变成三个 loss）→ 3. `stage1/model.py`（z 从哪来）→ 4. `stage2/flow_l3.py`
（z 怎么占 text 槽）→ 5. `stage2/engine_train.py`（循环与探针）→
6. `stage2/eval_rollout.py` + `eval_env2.py`（闭环怎么判分）。

配套逆向 spec（更细的字段级契约与坑）：`.scratch/retrofit-rollout-chain/spec.md`、
`.scratch/retrofit-label-extraction/spec.md`。
