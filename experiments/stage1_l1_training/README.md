# 实验：Stage-1 L1+L2 训练（curriculum ①）

> **当前实验**（`experiments/CURRENT` 指向这里）。计划出处：`PIPELINE_IMPL_PLAN.md` §2.2（a 案数据流定稿）。

## 方案

a 案（C′ 具体化）：冻结 DINOv2 ViT-B/14 patch 特征 + BERT task emb + 3 层 self-attn 融合 +
学习型 [C] token + conv 上采样 decoder；AFUN mask 经 area-pool 成 37² 通道并入 patch 输入；
channel dropout 0.3。可训参数 **7.5M**。L_C = 1.0·KL(heatmap) + 0.5·CE(yaw36+pitch12) + 0.5·L1(w)。

- 入口：`run_stage1.py train-l1 [--fold K] [--no-prior] ...` / `eval-l1 --run <name>` / `test`
- 缓存：`data/stage1_cache/libero_spatial/{dino_feats.h5, afun_prior.h5, prior_qc.json}`
- 代码：`src/routedflow/stage1/{dataset,model,engine_train,eval_a1a2,dino_feats,afun_prior,augment_links}.py`
- 单测：`tests/test_stage1_units.py`（10 项，含轴约定拿全量数据锚定 + 骨架注入块 identity-at-init）

## 预注册协议（写代码前定好的，勿事后改）

- Split：fold k held-out 任务 [2k, 2k+1] 整任务；train=8 任务×45，val_id=8×5，val_ood=2×50；**5-fold 轮换**报 A1。
- A1：top-5 峰 NMS ≤10px@512；报 val_id / val_ood / 逐任务 / spike 失败任务单列（须 ≥ 平均）/ **prior 对错分层**。
- A2：train 上拟合 ridge probe，val_id 评测；三组特征 = z_trained / dino_pool / **z_random（同架构未训练，probe 容量对照）**。
- AFUN prior 正确率（作业②QC）：**in-mask 26%，≤20px 58%**（n=500）——与 spike 3/5 一致，C′ 判定坐实。

## 状态

| 步骤 | 状态 |
|---|---|
| 离线作业③ DINO 特征（500×1369×768 f16, 1G） | ✅ 2026-07-30 |
| 离线作业② AFUN prior（500 mask，实测 **3 分钟**——耗时估计 2–3h 全是模型加载的错觉）+ prior QC | ✅ 2026-07-30 |
| link 位姿补存（17 links × 500 demos，L3 GT 原料） | ✅ 2026-07-30 |
| L1+L2 代码 + 10 项单测 | ✅ 2026-07-30 |
| smoke（10 epochs + eval 全链路） | ✅（val 4.32→2.23；eval json 正常产出） |
| fold0 正式训练（200 epochs）+ A1/A2 | 🔄 |
| 5-fold 轮换 + no-prior ablation | ☐ |
| L3/L5 骨架（`src/routedflow/flow_models.py`：CondCrossAttn/CondAdaLN 零初始化 identity、QueryDepthEmbed=D8a、text 不对称接口断言） | ✅ 骨架（训练待 D3 定稿 + 阶段〇重训排期后） |
