# 实验：Stage-1 统一标签提取（C labels + 可导出 flow 标签的位姿轨迹）

> **当前实验**（`experiments/CURRENT` 指向这里）。计划出处：`PIPELINE_IMPL_PLAN.md` §2.5（C 标签提取）+ §0.3 监督信号总表。

## 目的

一次零漂移重放遍历，产出 L1/L2 训练所需的全部 C 标签，并把**之后 L3/L5 flow 标签所需的位姿轨迹一并落盘**（此后 flow 标签 = 纯投影计算，不再进仿真）。

## 方法（设计要点）

- 数据源：raw LIBERO hdf5（`/workspace/datasets/libero/hdf5/libero_spatial/`，10 任务 × 50 demos）。
- 每 demo：`t_g` = `actions[:,-1]>0` 首次锁存；重放各帧只记位姿（EE site + 所有 free-joint 物体 body），**只在 t=0 渲染** agentview 512² RGB / 米制深度 / geom-id 分割。
- C 标签：E(t_g) 位置经已验证投影约定（AFUN spike，`rgb[::-1]` upright、无额外翻转）打到 t=0 帧像素 = 接触点；`ee_quat[t_g]` = 朝向标签原料；`gripper_q[t_g]` = 开度。
- 被抓物体判定：t_g 时距 EE 最近的 free body（QC 记录距离）。
- 相机 K / world_to_pix / extrinsic、geom→body 映射、robot base 位姿都存在任务级 attrs——后续任意帧任意点可离线投影。

## 产出

- `data/c_labels/libero_spatial/<task>.h5`（格式见 `src/routedflow/extract_c_labels.py` docstring）
- `qc/`：`qc_stats.json`、`overlay_*.png`（抽查 2/任务）、`spread_*.png`（单任务 50 个接触点散布 = 多峰性直查，计划 §2.3 #2）、`qc_montage.png`

## QC 指标含义（smoke 实测，2026-07-30）

| 指标 | smoke 值 | 解释 |
|---|---|---|
| grasp_dist_m | ~6.0cm | EE site 到物体 body **原点**的距离（碗原点在底部中心，site 在指尖间）——量级正常即可 |
| lift_gap_m | ~3.0cm | **lift(p̂, depth₀) 重建 vs 真值 E(t_g) 的系统偏差实测**：深度图打在物体表面、site 在指间空隙。这是推理时平移=lift 方案的固有偏差下界，设计上由闭环 flow/action 吸收 |
| replay_ee_maxdiff_m | ~1.4cm max / 9mm median | obs 记录与 state 快照的**子帧时序差**（随运动速度放大，t=0 静止时→0），非重放漂移；我们的标签是同一 sim 内自洽闭环，此量不影响标签质量 |

## 状态

| 步骤 | 状态 |
|---|---|
| 提取器 + QC + `run_stage1.py` 落地 | ✅ 2026-07-30 |
| smoke（1 任务×2 demos）+ overlay 目检（接触点正落碗沿） | ✅ 2026-07-30 |
| 全量 500 demos 提取 | ✅ 2026-07-30（500/500，0 skip；任务 1 曾因 smoke 半成品被 skip-existing 跳过，已删除重提） |
| QC 全量（20 抽查 + 多峰散布 + 统计） | ✅ 2026-07-30（montage 20/20 落碗沿） |
| 结果写回计划文档 §2.5 | ✅ |

## 全量结果（2026-07-30，`qc/qc_stats.json`）

- **500/500 demos 提取成功**，0 个 never-latch；t_g 中位数 31–52（各任务）。
- **被抓物体（运动判定）**：10/10 任务全部 = `akita_black_bowl_1_main`，与任务语义一致。
  nearest-body 启发式错 8/500（1.6%，碗 body 原点在底部导致）——**之后 object-flow 标签一律用运动判定**
  （t_g 后位移 argmax，`qc_c_labels.grasped_by_motion`），轨迹已落盘、离线可算。
- **lift 系统偏差实测**（推理时平移 = lift(p̂, depth) 方案的误差下界）：
  单像素 lift 中位 2.7cm，但**悬空抓取**（bowl on ramekin）掉到 11.6cm——接触点投影落在轮廓边缘、
  射线穿到远处背景（深度不连续失效模式）；9×9 窗口最小深度 robust 化 → 5.0cm，全局中位 2.5cm。
  **设计含义**：naive IK-to-lift baseline 在高处/悬空抓取上会系统性失败——这正是 §3.5 B2
  （主路线存在理由）的现成证据来源；lift 实现必须用 mask/窗口 robust 化，不能单像素。
- **多峰性直查**（`spread_*.png`，§2.3 #2 的经验回答）：同任务 50 条 demo 的接触点紧密聚在
  同一碗沿位置——libero_spatial 示教高度**单峰**，v1 按 demo 独立监督即可，top-k 多峰仲裁不急。
