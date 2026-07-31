# Flow → Action：现有方法逐篇核实表

**日期：** 2026-07-27
**用途：** Flow ≠ Affordance 一文的 related work 底稿 + Figure 1 的数据来源
**核实标准：** 每篇按三列填 —— ① flow 怎么用（追什么点、怎么选、2D/3D）② flow→action 这一步怎么做 ③ contact 从哪来

---

## 阅读状态

**20 篇全部逐字读过原文正文**，下表所有引文可直接引用。

**2026-07-28 新增第 21 篇：3DFlowAction（2506.06199，HTML 全文）** —— A 组新条目，见表 1；
它同时是 C-head pipeline 撞车检查的主角，详见 `C_HEAD_PRIOR_ART.md`。

**2026-07-28 深夜新增第 22 篇：MolmoMotion（2606.18558，HTML 全文，AI2）** ——
语言条件的 3D 点轨迹预测（Molmo 2 backbone，公制世界系）。**自报只做 post-grasp**
（"focus on the **post-grasp stage** where the policy must lift, transport and place"）；
grasp 来源："the original **MolmoBot policy performs grasping**, after which control is
handed to our trained policy" —— **$C$ 外部来源的第 22 个实例**。
flow→action 是**学出来的** flow-matching action head（不是解析执行）。

其中 **ATM 与 PPI 另有源码级证据**（本机跑过、读过预处理代码）；
**ToolFlowNet 与 AVDC** 因发表年份早于 arXiv HTML 渲染，读的是 PMLR / arXiv PDF 原文。

**已排除 1 篇：`FlowPolicy` (AAAI'25, 2412.04987)** —— 这里的 "flow" 是
**flow matching（生成模型）**，定义为 "an ODE whose solution transmits the noise
$x_0\sim p_0$ to the data $x_1\sim p_1$"，与「点的运动」无关。**同名不同义，不收。**
同理需警惕：`PointFlowMatch`、`CoLA-Flow`、`Trajectory-Consistent Flow Matching`。

**总纲引用：** `From Video to Control: A Survey of Learning Manipulation Interfaces from
Temporal Visual Data` (arXiv 2604.04974)，称这一族为 **"2D pixel-trajectory interfaces"**：

> "use predicted motion paths as interpretable targets that downstream controllers explicitly track or follow"

---

## 表 1 · 主表（按「追什么」从纯物体到纯机器人排序）

### A 组 · 只追物体

| 方法 | ① 追什么 / 怎么选 / 维度 | ② flow → action | ③ contact 从哪来 |
|---|---|---|---|
| **Im2Flow2Act**<br>CoRL'24, 2407.15208 | "object flow—the **exclusive motion of the manipulated object, excluding any background or embodiment movement**"。GroundingDINO bbox 内均匀采样。**2D** | **学出来的** flow-conditioned diffusion policy：$p(a_t\mid\mathcal{F}_{0:T},s_t,\rho_t)$ | **隐式**，藏在仿真 play data 里："The robot **first picks up the object** and starts executing 6DoF random trajectories" |
| **Track2Act**<br>ECCV'24, 2405.01527 | 物体点（筛掉不动的），位置随机化。**2D → 3D**（首帧深度） | $\bar a_t = T_t e_1$，$T_t=\arg\min\sum_i(\|x^i_t-u^i_t\|+\|y^i_t-v^i_t\|)$ | **启发式 + 残差 policy**："the end-effector moves to the **center of the 3D points**…with same orientation as $e_0$"，然后 "we execute a grasp" |
| **PPI**<br>RSS'25, 2504.17784 | 物体点，GroundingDINO → SAM mask → 随机采 $N_q$ 个。**3D** 世界系 | 双 interface 经 attention："the noised keyframe action token attends to the pointflow token…The final continuous action token attends to all the previous tokens" | **另一条 interface（keypose）**："use a **heuristic algorithm** to identify the keyframe timesteps…its corresponding action label can be directly retrieved" |
| **SPOT**<br>2411.00965 | 物体 SE(3) 位姿轨迹（非点集）。**3D** | $A_{t:t+h}=T^{obj}_{EE}\cdot T_{t:t+h}$，一次矩阵乘法 | **demo**："relying on **pre-generated grasp poses (derived from the demonstration)**" |
| **Object-Part Scene Flow**<br>2409.10032 | 物体部件，LISA VLM 定位，prompt 为 "Where should I grasp if <<<task>>>?"。**3D** | $\min_{T_n\in SE(3)}\|T_np'_{n-1}-(p'_{n-1}+s'_{n-1})\|^2_2$，前提 "the target object part is **rigidly attached to the end effector**" | **现成 grasp detector**："using **generalized grasp pose estimation methods**" |
| **A₀**<br>2504.12636 | contact point + post-contact 轨迹，$x_t=(u,v)\in[0,1]^2$。**2D**（后反投影） | 解析映射：反投影 → **查 GraspNet** → 选最近候选 → VLM 定高度 → SE(3) | **100 万条人标 contact point** + **GraspNet** 出 6-DoF |
| **F2F-AP**<br>2604.02408 | 物体关键点，分割 mask 上 K-means (K=4)。**2D** | flow **画到图像上**再喂 policy："Paint(·) represents the process of **rendering the flow vectors onto the current observation**" | **无模块**，diffusion policy 端到端学 |
| **Dex4D**<br>2602.15828 | "object-centric point tracks"，**不含手**。2D 跟踪后升维。**3D** | Paired Point Encoding：current 3D 点 + target 3D 点拼成 6 维 → policy 出 22-DoF | **仿真 RL（带特权信息）**：teacher policy 用 PPO "with **privileged states and fully observed object geometry**" |
| **General Flow**<br>CVPR'24, 2401.11439 | **夹爪周围 10cm 内的场景点**：`Query Points Q ← Radius(P_scene, g, 10cm)`。**3D** | `SE(3) Transformation T ← SVD-Alignment(F)` + 阻抗控制 | **人手摆**："we **manually position the robotic arm** for task initiation." 明确只做 "post-grasp motion" |
| **3DFlowAction**<br>2506.06199 | **纯物体** 3D flow："This clue is **embodiment-agnostic**"。**3D** | 最小二乘 $\min\sum_i\|k^i_{init}-k^i_{pred}(t)\|^2$ → SVD 出 $T$ → "transform all candidate grasp poses using $T$" → IK 验可达。跨 embodiment：Franka 67.5% / Dobot 70.0%，零 per-robot 训练 | **GPT-4o 选部位 + AnyGrasp 出候选**："use **AnyGrasp** to generate a series of grasping poses around the object part"。flow 只做可达性过滤 |

### B 组 · 物体 + 机器人都追（或不分）

| 方法 | ① 追什么 / 怎么选 / 维度 | ② flow → action | ③ contact 从哪来 |
|---|---|---|---|
| **ATM**<br>RSS'24, 2401.00025 | **全图**。policy 端 "a fixed set of **32 points on a grid**"；预处理源码 `sample_from_mask(np.ones((H,W,1))*255, num_samples=1000)`。**2D** | token **early + late fusion**："the predicted tracks are fed into the policy **both before and after** the transformer" | **无 grasp 模块。** 端到端 BC ← **我们量的就是这一格** |
| **Tra-MoE**<br>2411.14519 | **建立在 ATM 之上**，同样 grid 采样，32 点。**2D** | 轨迹画进一个 mask modality，"**concatenate with the image observations on the channel dimension**" | **无 grasp 模块**，端到端 BC |
| **ChronoFlow-Policy**<br>2606.31493 | **物体 + 夹爪显式分开两组**。夹爪："predefine $N_m$ **canonical keypoints on the gripper geometry**…rigidly attached to the TCP"；物体：FPS + TAPIP3D。**3D** | 两段分解：$\pi(a\mid o,P)=\pi_\phi(a\mid P_{t:t+h})\,\pi_\theta(P_{t:t+h}\mid o,P)$ | **无显式 contact 建模**，靠 co-motion 隐式表达 |
| **AVDC**<br>ICLR'24, 2310.08576 | **整帧 dense optical flow**（现成 GMFlow）。物体 mask "extracted by external segmentation methods…**or simply specified by the human**"。2D flow + 首帧深度 → 3D | 刚体拟合：$\mathcal{L}_{\text{Trans}}=\sum_i\|u^i_t-\frac{(KT_tx_i)_1}{(KT_tx_i)_3}\|^2+\|v^i_t-\frac{(KT_tx_i)_2}{(KT_tx_i)_3}\|^2$ | **随机采一个抓取**（见下方引文） |
| **DAWN / Pixel Motion Diffusion**<br>2509.22652 | **整幅图的 dense pixel motion**，每个像素的 $[u,v]$。**2D** | Motion Director 生成一张 pixel motion 图 → **Action Expert** 条件其上出动作 | **无来源**，BC 端到端 |
| **FOFPred**<br>2601.10781 | **整幅图的 dense optical flow**，**故意包含机械臂**（见下方引文）。**2D** | 预测的未来光流灌进 diffusion policy network | **未处理** |

### C 组 · 只追机器人 / 机器人控制的刚体

| 方法 | ① 追什么 / 怎么选 / 维度 | ② flow → action | ③ contact 从哪来 |
|---|---|---|---|
| **ToolFlowNet**<br>CoRL'22, 2211.09006 | **机器人手里那个工具**上的 dense per-point flow，从已分割点云中 "extract…the subset of $N'\le N$ points…corresponding to all points on the **tool**, while ignoring points belonging to other object classes"。**3D** | **可微 SVD**：$\hat a=(\hat R,\hat t)=\pi_\theta(o)=\text{SVD}(F_{\text{tool}})$ | **工具默认已在手里**（scooping / pouring 任务） |
| **EC-Flow**<br>2507.06224 | **只有机器人**："pixel-wise flow…for randomly sampled points of the **embodiment**"，$N_p=400$ 全在机器人 mask 内。**2D** | 学出来的 flow → action 模块 | **mask 质心 + 人手摆初始位姿** |
| **Flow as Flow**<br>2606.23090 | **机器人末端速度场**，图上均匀初始化 10×10 点。**2D** | Flow Generation → Action Generation，后者 "predict the end-effector pose $a_t$ conditioned on…the generated flow" | **无专门机制** |

### 相邻方法族（非 flow，但同一个结构性问题）

| 方法 | 中间表示 | contact 从哪来 |
|---|---|---|
| **KITE**<br>2606.22113 | 稀疏 contact set → latent intent → per-embodiment decoder | **URDF + 100 万合成 intent-configuration 对 + 800 epochs + 重训 shared latent space** |
| **V-JEPA 2-AC**<br>2506.09985 | latent world model + MPC | **人给的两张 sub-goal 图** |

---

## 表 2 · contact 从哪来 —— 这就是 Figure 1

按「有多明确」从最直白排到最隐蔽：

| 方法 | contact 来源 | 论文承认了吗 |
|---|---|---|
| **AVDC** | **随机采一个抓取** | ✅ 白纸黑字 |
| **General Flow** | **人手摆机械臂** | ✅ 明说，且明确只做 post-grasp |
| **EC-Flow** | 分割 mask 质心 + 人手摆初始位姿 | ✅ 明说 |
| **V-JEPA 2-AC** | 人给的两张 sub-goal 图 | ✅ 明说 |
| **SPOT** | demonstration 里的 grasp pose | ✅ 明说 |
| **Track2Act** | 启发式（点云中心 + 保持初始朝向）+ 残差 policy | ✅ 明说 |
| **Object-Part Scene Flow** | 现成 grasp detector | ✅ 明说 |
| **A₀** | 100 万条人标 contact point + GraspNet | ✅ 明说 |
| **KITE** | URDF + 100 万合成对 + 800 epochs + 重训 latent | ✅ 明说，自称 "non-trivial pre-deployment cost" |
| **Dex4D** | 仿真 RL，teacher 用**特权状态 + 完整物体几何** | ✅ 明说 |
| **ToolFlowNet** | 工具默认已在手里 | ✅ 前提，不是输出 |
| **PPI** | 另一条 interface（keypose），监督来自 demo 启发式关键帧 | ✅ 明说 |
| **Im2Flow2Act** | 仿真 play data 里「机器人先把物体拿起来」 | ⚠️ 写了，未当作 contact 来源讨论 |
| **F2F-AP** | 无模块，diffusion policy 端到端 | ❌ 未讨论 |
| **Flow as Flow** | 无专门机制 | ❌ 未讨论 |
| **ChronoFlow-Policy** | 无显式 contact 建模（但对 gripper-flow 分量做过一次单任务 ablation，见「我们的空位」§3 修正） | ❌ 未讨论 |
| **3DFlowAction** | GPT-4o 选部位 + AnyGrasp 出候选，flow 只做可达性过滤 | ✅ 明说 |
| **MolmoMotion** | 另一个 policy（MolmoBot）执行抓取后再接手；自限 post-grasp | ✅ 明说 |
| **DAWN** | 无来源 | ❌ 未讨论 |
| **FOFPred** | 未处理 | ❌ 未讨论 |
| **Tra-MoE** | 无 grasp 模块（继承 ATM） | ❌ 未讨论 |
| **ATM** | **无。全图 flow 里混着机器人像素** | ❌ **无人注意到** ← 我们量的就是这一格 |

**二十个方法，没有一个从 flow 里得到 contact。**
前十二个明码标价地从外部搬进来 —— 随机采、人手摆、demo、grasp detector、
人工标注、sub-goal 图、URDF、特权 RL。
后八个没讨论。**只有 ATM / Tra-MoE 这一支是隐性的**：机器人像素混在被追踪的点里，
收益被记在 flow 名下。

---

## 可直接进论文的原句

### 1. AVDC —— 最直白的一句

> "given inferred object transformations, we can use existing off-the-shelf robotics primitives to generalizably infer actions in the environment. In particular, if the object is graspable, **we randomly sample a grasp on the object** and then compute the target robot end-effector pose based on the target object pose and the grasping pose."
>
> "**We treat the grasp/contact point as the first subgoal.**"

**object flow 推不出抓哪儿，所以随机采一个。** 没有比这更直接的承认了。
注意第二句同时给出了 $E = T\cdot C$ 的又一个实例。

同页还有一句，把 grasping 明确列为「flow 之外必须另行获得的知识」：

> "directly learning explicitly regress actions using a learned inverse dynamics requires a substantial number of action labels so that a neural network can learn existing knowledge such as inverse dynamics, **grasping** and motion-planning."

### 2. General Flow —— 自我矛盾最锋利的一处

标题是 `General Flow as Foundation **Affordance**`。它这样论证：

> "Rooted in Gibson's theory, affordance concentrates on the potential actions associated with an object, **remaining neutral to specific manipulators**."

实验里：

> "we **manually position the robotic arm** for task initiation."

**用来论证「这是 affordance」的那条性质（manipulator-neutral），恰恰是让它无法成为
可执行 affordance 的那条性质** —— 与 manipulator 无关，就说不出 manipulator 该放哪儿。
而他们用「人手摆」证明了这一点。

（Gibson 的 affordance 是「物体对**某个特定 agent** 提供什么动作可能」，agent 是定义的一部分。
抽掉 manipulator 之后剩下的量，恰好把 affordance 之为 affordance 的东西抽掉了。）

### 3. Track2Act —— 作者自报的头号失败模式

> "The main failure modes we observe are **inability to grasp the object at the right location**, and inability to recover from intermediate failures."

### 4. F2F-AP —— 同样的失败模式，另一篇

> "**Object Slippage**…**Suboptimal grasping poses often result in insecure contact**, causing the object to be ejected or slip out"

### 5. FOFPred —— 一个独立作者组做出了同样的诊断，然后**故意**把机器人放进来

> "methods relying on localized trajectories…often limit their focus to specific object movements, **overlooking crucial global information, such as the overall movement of a manipulator**."
>
> "our FOFPred predicts spatially dense future optical flow" 以捕捉 "global information, such as the **overall movement of a manipulator**"

**他们和我们的诊断完全一致：只追物体会漏掉机械臂的运动。**
区别在于他们把这当成一个应该修的缺陷（改用 dense 全图光流），
而**从未量过这一改动带来了多少收益** —— 那正是我们在量的东西。

### 6. ToolFlowNet —— 光谱端点，flow 就是 action 本身

它对「机器人控制的刚体」预测 flow，然后 SVD 出变换。作者自己点破：

> "this method to detect flow means that it **reflects the 'intended' action from the robot**, which may differ from the true positions of the tool points in 3D space after the robot executes the action; for example, when a collision happens with a wall, the tool points might not move, even though the robot intended for them to move."

**追机器人刚体控制的东西，flow 就是 action 的投影（甚至是「意图」而非实际运动）。
追物体，flow 就不是。** 这是这根轴两端最干净的对照。

### 7. EC-Flow —— 光谱另一端的反证

走到极端只追机器人（400 点全在机器人 mask 内）：

> "the grasp point is the **centroid of the segmented mask**"
> "starting from a **manually set initial pose**"

**如果机器人 flow 能定出抓取，EC-Flow 就不需要这两样。**
另注：仿真只用 Sawyer、真机只用 Franka Research 3，**从未做过跨 embodiment 实验，
且无任何一条 limitation 提到 embodiment specificity**。

### 8. V-JEPA 2-AC —— 换个方法家族，同一个泄漏

> "For the *pick-and-place* tasks we present **two sub-goal images** to the model in addition to the final goal."

limitation：

> the model "must therefore implicitly infer the action coordinate axis from the monocular RGB camera input," which proves problematic **without visible robot base references**.

**看不见机器人就不好使。** 而且他们**从未讨论过「不同 action 产生相同 latent」的简并性**。

### 9. PPI —— 作者自己指出的缺口

> "**cross-embodiment evaluation on different robotic platforms is essential** to assess PPI's generalization across hardware."

本次核实到的 Table VII：

```
Continuous only          47.6
Continuous on Pointflow  74.3
Ours (Best Ckpt)         82.6
```

⚠️ **与之前记录的 keypose 边际贡献（+6.0 / +6.5）对不上，
写进论文前必须重新核对完整的 Table VII。**

### 10. ATM —— 源码级证据

`scripts/preprocess_libero.py:127`：

```python
points = sample_from_mask(np.ones((H, W, 1)) * 255, num_samples=num_points)
```

**全 1 的 mask —— 全图均匀采样，完全不区分物体与机器人。**
论文侧对应 "a fixed set of **32 points on a grid** for the policy"。
不是推断，是他们的代码。

---

## 我们的空位

把两张表合起来，空着的格子是：

1. **没有人量过**「flow 的收益里，有多少来自被一起追踪的机器人像素」。
   FOFPred 甚至**明确诊断出了这一点并据此改了设计**，但没有测量。
2. **没有人拆过** approach 阶段 vs transport 阶段，尽管 SPOT / General Flow / ToolFlowNet
   都把自己明确限定在 post-grasp（也就是只做 transport 那一半）。
3. **ChronoFlow-Policy 已把物体点和夹爪点显式分成两组，并在 Fold Towel（真机）上
   做过一次分组 ablation** —— ⚠️ 修正（2026-07-28，读 HTML 全文后）：此前记录的
   「没有测两者各自的贡献」**是错的**。原句：
   > "Removing object flows (87%→47%) causes a larger Stage II drop than removing gripper
   > flows (87%→80%), indicating that object flows are the primary signal for
   > deformable-object tracking while **gripper flows provide complementary cues**."
   但它是**单任务、可变形物体、无 approach/transport 切分、无统计检验、单机器人**
   （Flexiv Rizon + Robotiq，无任何 cross-embodiment 讨论），且把 gripper flow 框架为
   「complementary cues」而非 embodiment 泄漏。**这是离我们最近的一次测量，必须引用，
   然后在阶段切分 / 统计 / 视角对照 / 泄漏定性四个维度上超过它。**

我们当前的 ATM 结果正好填第 1、2 格（配对 t 检验，按 10 个 task 配对，df=9；
2026-07-28 终版，全部 n=400）：

```
A_full              0.710   (两 seed 均为 0.710)
Bw_approach_wrist   0.615   Δ=-0.095   t=-3.58   p=0.0060
Ba_approach_agent   0.712   Δ=+0.003   t= 0.12   p=0.9043   95%CI=[-0.043,+0.048]
Bw vs Ba            Δ=-0.098  t=-4.17  p=0.0024
B_approach          0.637   Δ=-0.072   t=-1.95   p=0.0829   （跨零，见主日志）
C_transport         0.420   Δ=-0.290   t=-4.64   p=0.0012
F_all               0.337   Δ=-0.372   t=-9.69   p<0.0001
```
