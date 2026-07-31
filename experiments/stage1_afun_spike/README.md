# 实验：Stage-1 AFUN spike（D1 选型 gate）

> **当前实验**（`experiments/CURRENT` 指向这里）。计划出处：`PIPELINE_IMPL_PLAN.md` §2.2/§2.5。

## 目的

半天判断：**冻结 AFUN 当阶段一前端（D1 选项 C）是否可行**——AFUN（arXiv 2606.02551，VLM+SAM3+MetaQuery）
在 LIBERO 仿真域的 functional mask 质量够不够。过关 → D1=C；不过关 → 退 B（小 VLM+LoRA mask-token）。

## 方法

1. `make_inputs.py`：5 个 libero_spatial 任务，demo_0 的 t=0 状态重放，agentview **512²** 渲染
   RGB + 真值米制深度（转 mm）+ 内参；GT = demo 的 EE 位置在 t_g 投影回像素
   （投影约定已实弹验证：`gt_overlay_both.png`，蓝点正中目标碗——robosuite 的 transform 无需额外翻转）。
2. `run_spike.sh`：对每个输入跑 `demo.py --depth-dir ... --no-refine`（**必须 --no-refine**：
   仿真深度是干净真值，lingbot 的 RealSense 去噪反而有害）。
3. `score.py`：GT 点是否落在预测 mask 内 / 到 mask 最近距离 / mask 面积 / confidence。

## 判定标准（预注册，宽松——这是 spike 不是 eval）

- **过关**：≥ 3/5 任务 GT 点在 mask 内或距 mask ≤ 20px（512 分辨率），且 mask 大致在目标物体上（人工过目 `pred_seg.png`）。
- **不过关**：mask 系统性落在错误物体 / 空白 / 不响应语言中的空间限定词（"between the plate and the ramekin"）。
- 特别观察项：**空间指代**能力——LIBERO spatial 的任务全靠空间限定词区分同类物体，这是比 AGD20K 更难的语言要求。

## 状态

| 步骤 | 状态 |
|---|---|
| AFUN clone → `third_party/AFUN` + conda env `afun`（cu130 torch 2.10） | ✅ 2026-07-30 |
| afun.pt checkpoint（1.3G）+ backbone 权重 | ✅（SAM3 曾被 gated repo 挡住；token 和 SAM3 缓存都在 `/workspace/huggingface_cache`——已拷 token + symlink 缓存进 `/workspace/.hf_home`） |
| 5 任务输入渲染 + GT 投影验证 | ✅（投影翻转 bug 已修，`gt_overlay_both.png` 实弹验证） |
| 推理 ×5 + 评分 | ✅ 2026-07-30 |
| D1 判定写回 `PIPELINE_IMPL_PLAN.md` | ✅（判定见下） |

## 结果（2026-07-30，`results.json` + `outputs/*/pred_seg.png`）

| 任务（空间限定词） | 选对物体？ | GT→mask 距离(px@512) |
|---|---|---|
| between the plate and the ramekin | ✅ | 0.3（GT 在 mask 内） |
| in the top drawer of the wooden cabinet | ✅（柜顶干扰碗二选一选对） | 4.0 |
| next to the plate | ✅ | 8.8 |
| from table center | ❌ 选了右缘的碗 | 153.9 |
| next to the cookie box | ❌ 选了远处的碗 | 186.5 |

**汇总**：选对物体 3/5；选对时 mask 精度极高（≤9px）；错的两例都是**空间指代失败**（都选了显眼/孤立的碗而不是解析空间关系——典型 grounding 捷径）。运动曲线方向合理（先提起、朝目标拐）。

## 判定

预注册线（≥3/5 ≤20px）**恰好压线通过**，但两个失败恰恰全是 LIBERO spatial 的命门能力（同类物体空间消歧）。
**冻结 AFUN 单独当前端 = pipeline 上限被封在 ~60% 选对率**（参照系 A_full 0.710），不可接受。
AFUN 训练代码未放出（repo TODO 明写），fine-tune 它要自己写训练——成本回到 B 案量级。

**D1 推荐修订为 C′ 混合案**：自训前端（A 案 backbone + 仿真特权标签，LIBERO 的公式化空间指代靠训练解决）
+ **AFUN mask 作为辅助输入通道 / prior**（保留 foundation 先验，帮 held-out 泛化）；AFUN 同时充当论文里的
zero-shot baseline 行和 v2 scale-up 故事。latent 接口不变（learnable token，A2 probe 照测）。
