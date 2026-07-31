# C-head /「pose 对齐 + morphology adaptation」pipeline 的 prior-art 检查

**日期** 2026-07-28 · **触发**：用户提出「物体 flow ⇒ 6D pose 轨迹；flow→action = 物体 pose 与
机械臂 pose 对齐；适应部分 = 每个 gripper 学一个 C；少量 demo ⇒ few-shot 迁移」。
消歧后 = 上一轮的 C-head 方向（$C$ 作为唯一 embodiment 接口）。

**判定方法**：按格子查占位。每格给出占位者、占到什么程度、原句证据。

---

## 结论先行

**pipeline 作为独立 contribution：死。** 五个组成格子全部有人，其中组合本身
（object-only flow → $T$ → 用 $T$ 约束抓取 → $E=T\cdot C$ → 跨 embodiment 不重训）
已被 **3DFlowAction (2506.06199)** 于 2025 年 6 月整条搭出来。
ΔC-常数性假设不但有人量过，而且已经是**领域内的标准 baseline 技巧**（"commonly used"）。

**measurement 论文：无伤，反而加强。** 3DFlowAction 的 contact 同样是外部搬的
（GPT-4o 选部位 + AnyGrasp 出候选）——survey 的「没有一个方法从 flow 得到 contact」
多了一个 2025 年的新证据。另有一处**必须修正我们自己的记录**（见 §7）。

---

## 逐格占位表

### 格 1 · 「只预测物体 flow，保持 embodiment-agnostic，组合出 E=T·C」

**占位者：3DFlowAction（arXiv 2506.06199，2025-06）。占满。**

- flow 定义："predicts the future movement of the interacting objects in 3D space" ——
  纯物体，动机与我们一致："understanding how the objects should move in the 3D space is
  a critical clue for guiding actions. **This clue is embodiment-agnostic**"
- flow→action：对预测 keypoint 最小二乘 $\min\sum_i\|k^i_{init}-k^i_{pred}(t)\|^2_2$，
  SVD 出变换 $T$，"transform all candidate grasp poses using T"，IK 验证可达性
- 跨 embodiment 实验：**Franka 67.5% / Dobot XTrainer 70.0%，零 per-robot 训练，只换 IK**
- 局限（自报）：刚体假设；"non-rigid deformations…unable to output effective actions"

### 格 2 · 「contact/grasp 从哪来」（3DFlowAction 的答案）

**照旧是外部搬的 —— survey 新增一行：**

> "use **AnyGrasp** to generate a series of grasping poses around the object part"
> （part 由 GPT-4o 按任务指令选）

即 contact = VLM 语义 + 现成 grasp detector。flow 只做**可达性过滤**（用 $T$ 变换候选后查 IK）。
**没有从 flow 推出 contact。** → 归入 survey 表 2 的「✅ 明说」栏。

### 格 3 · 「gripper-conditioned 的 C 预测（跨 gripper 泛化）」

**占位者：UniGrasp → AdaGrasp → GraspGen-X。占满，且规模已到 2B。**

- **UniGrasp**（RA-L'20）：输入物体点云 + gripper 属性 → contact points；
  "transfer between a diverse set of N-fingered robotic hands"，2/3 指爪 >90% 成功
- **AdaGrasp**（ICRA'21）：gripper 与场景 shape encoding 做 cross-convolution，
  单一 policy 泛化到 novel grippers
- **GraspGen-X**（2606.00998，2026-06）："train our cross-embodiment model with
  **procedural grippers** and a large-scale dataset of **2 Billion grasps**"，
  "**zero-shot** generalization to real-world novel grippers"，fine-tune 是可选项
- 共同点：**全部 task-agnostic**（只出稳定抓取，不 condition 在任务/轨迹上）

### 格 4 · 「ΔC-常数性：固定 per-gripper 修正量做抓法迁移」

**不但被占，而且已经沦为 baseline。**

- **MultiGripperGrasp**（2403.09841）：11 个 gripper × 345 物体 × 30.4M 抓取，
  迁移机制就是 palm 对齐 —— "The translation component **aligns the palm center** of each
  gripper, and the orientation aligns the canonical pose of the gripper palm"
- 另一个惯用法（fingertip-offset）被描述为 "**commonly used** for adapting a 6-DoF grasp
  pose to a new gripper"
- 而且领域已经在超越它："it is more promising to learn an end-to-end cross-embodiment
  model **rather than applying a simple pose correction technique**"

→ **「量 ΔC 跨物体是否恒定」的 E0 pilot 不用做了：答案是「大体恒定，恒定到成了标准技巧，
且端到端模型已在超越它」。** few-shot-via-ΔC 不是可发表的 claim。

### 格 5 · 「task/轨迹-conditioned 的 C 选择」

**部分占位，且我们自己的 pilot 证明了这格的剩余价值很低：**

- 语言条件版：TaskGrasp / GraspGPT / FoundationGrasp / DexTOG / GCNGrasp-VP —— 成熟子领域
- placement 条件版：**Pick2Place**（2304.04100）"task-aware grasp estimation…encodes the
  relationship between placement scene geometry and the object"
- 轨迹条件版（可达性形式）：**3DFlowAction 已做**（用 $T$ 变换候选查 IK）
- **我们 2026-07-27 的 grasp-feasibility pilot 恰好量的就是这一步的边际价值**：
  自由空间 headroom ≈ 0–3 点；紧公差下 feasibility 根本不是 $(C,T)$ 的函数。
  （沙盒不同：LIBERO replay vs 真机，逻辑可迁移、数字不可直接引用。）

### 格 6 · 「imitation pipeline 里的 per-embodiment 模块」

- **KITE**（2606.22113）：sparse contact → latent intent → per-embodiment decoder，
  代价 URDF + 1M 合成对 + 800 epochs，自称 "non-trivial pre-deployment cost"（survey 已录）
- **ChronoFlow-Policy**（2606.31493）：gripper canonical keypoints（TCP 刚性挂接）+
  物体 FPS keypoints 两组显式分开；**单机器人**（Flexiv Rizon + Robotiq），
  无任何 cross-embodiment 实验；contact 隐式（端到端 BC）

---

## 我们还剩什么（诚实版）

1. **measurement 论文本体** —— 无人做过按阶段/按相机的泄漏量化。3DFlowAction 属于
   「干净的 object-only 端」，正好是我们光谱上缺的一个新点（它自己没量任何东西）
2. **对 3DFlowAction 的可检验批评**（可作为论文一节）：它的「用 T 过滤抓取」步骤，
   按我们 pilot 的逻辑在自由空间 headroom≈0、紧公差下不可计算 ——
   即**该系统里 flow 对 grasp 选择的实际贡献可能≈0，其跨 embodiment 能力全部来自 AnyGrasp**。
   这是一个别人 pipeline 上的 flow≠affordance 实例，主张形状与主线完全一致
3. **ChronoFlow 的 ablation 是我们最近的邻居，必须引**（见 §7 的修正）——
   但它是单任务/可变形物体/无阶段切分/无统计，我们的测量在所有这四个维度上更强

## 死掉的（明确放弃）

- ✝ pipeline 作为独立 contribution（= 3DFlowAction）
- ✝ ΔC-常数性 few-shot claim（= MultiGripperGrasp 的 palm 对齐，"commonly used"）
- ✝ E0 pilot（问题已被文献回答）
- ✝ 学一个 task-agnostic 的 gripper-conditioned C-head（= GraspGen-X，2B 抓取）

---

## §7 · 必须修正我们自己的记录

**FLOW_TO_ACTION_SURVEY.md 和 PROGRESS_REPORT 里写的
「ChronoFlow-Policy…没有测两者各自的贡献」是错的。** 本次读到 HTML 全文：

> "Removing object flows (87%→47%) causes a larger Stage II drop than removing gripper
> flows (87%→80%), indicating that object flows are the primary signal for
> deformable-object tracking while **gripper flows provide complementary cues**."

即他们在 Fold Towel（真机、可变形）上做了一次分组 ablation：去掉 gripper flow 掉 7 点。
**但**：单任务、无 approach/transport 切分、无统计检验、无 cross-embodiment 讨论，
且把 gripper flow 的贡献框架为「complementary cues」而非 embodiment 泄漏/可迁移性代价。
我们的空位从「没人测过」收窄为「只有一次单任务粗测，无人按阶段拆、无人给统计、
无人指出这就是 embodiment 泄漏」。**引用它，然后在四个维度上超过它。**

---

## 本次检索的来源

- 3DFlowAction: https://arxiv.org/abs/2506.06199
- ChronoFlow-Policy: https://arxiv.org/html/2606.31493v1
- GraspGen-X: https://arxiv.org/abs/2606.00998
- MultiGripperGrasp: https://arxiv.org/abs/2403.09841
- AdaGrasp: https://arxiv.org/abs/2011.14206
- UniGrasp: RA-L 2020（经 ResearchGate 摘要页核对）
- Pick2Place: https://arxiv.org/pdf/2304.04100
- 另见（未深读，同格佐证）：RobotFingerPrint 2409.14519、D(R,O) Grasp、
  EfficientGrasp 2206.15159、Rigid-to-Soft Grasp Matching 2602.17110

---

## 附录（2026-07-28 深夜）：VLA 输入域的补充检查 —— ACE-Ego-0 与 Cloak 精读

**触发**：用户正确指出主表中所有形态条件化占位者都依赖特权几何输入（URDF/mesh/点云），
RGB-only 的 VLA 输入域是另一根轴。精读两个最近邻，钉 delta 边界。

### ACE-Ego-0（2606.17200）

- morphology token = **GNN over URDF graph**：$H^{(\ell+1)}=H^{(\ell)}+\phi_\ell([H^{(\ell)};\bar A_r H^{(\ell)}])$，
  kinematic tree 上做 message passing，body-level + chain-level 两级 embedding
- 只在 action decoding 时注入（"keeping our VLM backbone embodiment-agnostic"）
- 人类数据：**per-dataset 的 learned surrogate embedding**（反向传播学出，不从像素提取）
- **关键数字：去掉 morphology token 只掉 1.9 点**（RoboCasa 72.8%→70.9%）——
  URDF token 在分布内的实测价值很小
- 无 zero-shot 到未注册 URDF 的机器人

### Cloak（2606.22836）

- mask 管线：$q$ --FK--> 各 link 位姿 + **URDF link meshes** $V_\ell$ + 腕相机外参 $T_{ec}$ + 内参 $K$
  --投影光栅化--> mask：`rasterize(⋃_ℓ K T_wc(q)^−1 T_wℓ(q) V_ℓ)`；填充用「同图滚动拷贝」inpaint
- action 迁移 = **tip-pose retargeting**：`FK(q_src)` 出指尖位姿 → 目标机器人 IK。
  **即以指尖位姿（≈ E）为通用接口，C 原样带过去** —— 在双指类内 ΔC≈恒等的又一实证
  （与 MultiGripperGrasp palm-alignment 相互印证）
- zero-shot：Franka+Robotiq 源 88.0% → UMI 85.1% / Sharpa 五指手 81.8% / YAM 86.3%
  （π0.5-droid 在 Sharpa 上只有 54.4%）
- 自报限制："covers manipulation expressible via **two fingertips** and not skills that demand
  richer contact or in-hand reorientation"；"Transfer is not lossless"

### delta 边界结论

「从像素提 morphology token、无 URDF」这格**精读后仍无人占**，但边界收窄成：

1. **价值天花板被 ACE-Ego-0 压低**：URDF 版 token 实测只值 +1.9 点（分布内）
2. **需求面被 Cloak 挤掉一大块**：双指域内「擦除 + tip-space action」zero-shot 已达 81–86%，
   **根本不需要 policy 理解形态** —— 编码派只在 tip 抽象失效的域（灵巧手/富接触）有存在理由
3. 剩余格子 = 「像素出形态 × 富接触/灵巧 × 无标定数据」—— 存在但窄，且视觉难度
   （公制尺度、接触时遮挡）与数据多样性墙叠加

**对 measurement 论文的红利**：两篇都是可引用的测量 ——
ACE-Ego-0 的 +1.9 是 morphology-token 效应的公开数字；
Cloak 的 85% zero-shot 说明「删掉机器人视觉 + E-空间动作」近乎无损 ——
**机器人像素通道携带的任务信息基本可被 state 替代，直接支持我们的 leak-removal 修法框架**
（该修法需引 Cloak；我们的差异：flow/track 级 mask + 阶段分辨的测量根据）。
