# 逆向 spec：rollout 评测链（实际行为，非应然）

Status: ready-for-human
对象：`src/routedflow/stage2/eval_rollout.py` + `eval_env2.py`（356 loc）
写法：只写代码**实际**在做什么。歧义/嫌疑单独列在 §5。
产出配套：`tests/test_char_rollout.py`（characterization，钉住下述契约）

## 1. 环境栈（eval_env2）

包装顺序（内→外）：`OffScreenRenderEnv(512²)` → `LiberoResetWrapper(init_states 切片)` →
`LiberoTaskEmbWrapper` → **`ChainStateWrapper`** → 向量化 → `LiberoImageUpsideDownWrapper` →
`LiberoSuccessWrapper` → `LiberoObservationWrapper(agentview + robot0_eye_in_hand)`。

`ChainStateWrapper` 每步在子进程内追加（不做任何额外渲染——视口坑教训）：

| 键 | 形状/类型 | 语义 |
|---|---|---|
| `chain_uv` | (32,2) f32 | FK 链点 `(x=col/S, y=row/S)`，S=构造时 img_hw=**512**；upright 行约定 |
| `chain_z` | (32,) f32 | 相机系深度 (m)，`inv(extrinsic) @ p_world` 取 z |
| `ee_z` | (1,) f32 | `gripper0_grip_site` 的世界系 z (m) |

- 链点权重：reset 后**首帧姿态**算 arc-length 权重（`polyline_weights`），整集固定——与离线
  `chain_points.py` 用 demo t=0 的做法同构。MAIN 9 body →24 点，双指各 4 点，共 32。
- 投影矩阵在 `_setup()` 用 `get_camera_transform_matrix(sim,"agentview",512,512)` 取一次，整集复用
  （相机静止成立）。

## 2. 每集流程（eval_rollout.main）

1. `env.reset()`；`l4.reset()` 清 ViLT 滚动缓冲。
2. **z 一次性计算**（`compute_z`）：reset 的 `agentview_image` (B,512,512,3) → torch bilinear→518 →
   ImageNet 归一化 → DINOv2 ViT-B/14 patch tokens → `l1(feats, prior=0, text)` 取 z。
   AFUN prior **恒置零**（训练时 channel-dropout 0.3 兜底）。text = BERT 缓存 emb，
   ndim>1 时 `mean(0)`。整集 z 冻结。
3. 每步：512 obs 双视角 PIL resize→128（默认 bicubic）→ `l3(frame[:,None], chain_uv, z, chain_z)`
   → `(b,16,32,2)` → `l4.set_flow(pred[:, None])` → `l4.act(obs_v, task_emb, {joint_states,
   gripper_states})`。act 后 `_flow_ctx=None`。
4. 状态机（每个子环境独立，mode 0 approach / 1 lifting / 2 done）：
   - **latch**：策略动作 gripper 通道 `act[i,-1] > 0.5` **连续 3 步** → mode 1，记 `ee_z_latch`。
     翻转当步起动作即被替换为 `LIFT_ACTION=[0,0,0.9,0,0,0,1]`（策略自己的闭合动作**不执行**，
     闭合由 lift 动作的 gripper=1 完成）。
   - timeout：mode 0 且 t ≥ 120 → mode 2 判负（同一步内 latch 判定先于 timeout）。
   - lift 25 步后判定：`rose = ee_z - ee_z_latch ≥ 0.03` **且** 指宽
     `gw = gripper_qpos[0] - gripper_qpos[1] ∈ (0.004, 0.075)` → 成功。空抓 gw≈0、全开 gw≈0.08
     均落界外。
   - mode 2 后动作置零（sim 继续跑直到全部 done）。
5. 集数 = vec(5) × rounds(2) = 10/任务；init_states 按子环境切片轮转，不重复。
6. 视频：每任务前 3 个子环境，256² + 每帧水印 "POLICY ROLLOUT - approach branch" + 相位文字。
7. `summary.json`：per-task SR + train8/ood2 聚合（split 按 `fold_split(tasks, ckpt里的fold)`）。

## 3. 与训练侧一致性（逐项核实过）

| 项 | 训练 | rollout | 判定 |
|---|---|---|---|
| text emb | `mean(0)` (stage1/dataset.py:107) | `mean(0)` | ✅ |
| DINO 预处理 | torch bilinear 512→518, /255, ImageNet | 逐行相同 | ✅ |
| L3 vid 帧数 | track_obs_fs=1（单帧） | `img_a[:,None]` 单帧 | ✅ |
| chain_uv 约定 | 离线 (col/512,row/512) upright | wrapper 同式同序 | ✅ |
| 窗口方向 | BCDataset forward `[off,off+t)` | 逐帧在线，无窗口错位 | ✅ |
| **L4 flow ctx 时间维** | (b,**10**,tl,n,2)：窗口各步配自己帧的 flow | `pred[:,None]`，act 的 t **恒为 1** | ✅ §5-1 复核 |

## 4. 已钉契约（characterization test 覆盖）

- obs 键集与形状/dtype/取值域（512 uint8 图、chain_uv∈[0,1]附近、chain_z>0、ee_z 合理域）；
- 链点几何：t=0 chain_uv 落在机器人像素上的比例（数值锚，防 mask/投影翻转回归）；
- `watermark`/`short`/`LIFT_ACTION` 纯函数快照。

## 5. 看不懂 / 歧义 / 嫌疑（bug 高发区）

1. **flow ctx 广播 —— 复核后撤销（假阳性，2026-08-02）**：最初判定 rollout 把当前 flow
   广播进 10 个历史槽。运行时探针（probe_act）实测 `act()` 内 track_obs =
   (b, v, **t=1**, fs=10, c, h, w)——upstream ViLT 的 act 把历史帧堆进 **fs 维**、t 恒为 1，
   逐步时序语境由 latent_queue 承担；而 `track_encode` 按 **t 维**配 flow ⇒ `pred[:,None]`
   (t=1) 与之严丝合缝，每个历史 latent 在它自己那步 act 时已配对过自己的 flow。
   训练（t=10 窗口逐步配对）与 rollout（逐步 act + latent 队列）配对语义一致。**无需修复。**
   教训：仅读 forward_loss 侧推断 act 侧张量布局不可靠——act 的队列 cat 在 dim=2（fs），
   与直觉的 t 维堆叠相反；此类判断必须探针实证后才能立案。
2. **文档-代码不一致**：`eval_env2` 模块 docstring 宣称 reset 时提供 `obs["rgb512"]`——代码从未写入
   （C-VLM 输入实际取自常规 obs 流）。误导后续维护者。
3. **512→128 用 PIL 默认 bicubic**：训练视频是原生 128（LIBERO 原始 obs 直存），rollout 是 512 降采样
   ——轻微分布差；且 PIL 默认滤波器随版本可变，未显式指定。
4. **成功判据依赖 `ee_z` 绝对上升**：桌面高度/任务几何变化时 3cm 阈值语义会漂；当前 10 任务同桌面，成立。
5. **auto-reset 边界未审**：sub-env 若因 LiberoSuccessWrapper done 触发向量环境自动 reset，mode 2 之后的
   残余帧（仅进视频）可能跳变。不影响 SR 判定（ok 在 lift 结束当步已锁定）。

## Comments
