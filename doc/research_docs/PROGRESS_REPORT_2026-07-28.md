# 进展汇报 2026-07-28：主结论已定，pipeline 的核心 module 被判死

**覆盖区间** 2026-07-26 16:14（上一份进展汇报）→ 2026-07-28 02:30
**作者** yd3007@nyu.edu · **上一份** `PROGRESS_REPORT_2026-07-26.md`
**日常日志** `FLOW_AFFORDANCE_LOG.md` · **文献表** `FLOW_TO_ACTION_SURVEY.md`

---

## 怎么读这份文档

**第一次看** → 按顺序读 §0（项目在做什么）→ §1（术语和符号表）→ §2（主结果）。
这三节自足，看完就能读懂后面所有的表。

**只有 5 分钟** → 读 §0.1 和 §2.4「三句核心结论」。

**已经熟悉这个项目** → 跳到 §3.2（freeze 干预失效）和 §7（当前状态），这两节是这轮新增的。

---

## 00 · 这个项目在做什么

### 0.1 一句话

**机器人学界近两年流行用「flow」（物体上的点未来往哪走）来指导机器人操作，
并且常常把它当成 affordance（该抓哪儿）来宣传。我们要证明的是：flow 里根本不含
「该抓哪儿」这个信息，那些方法的性能提升另有来源 —— 它们偷偷把机器人自己的像素
也一起追踪了。**

### 0.2 什么是 flow

给一段机器人操作的视频，在第一帧上撒一批点（几十到上千个），然后**预测这些点在
接下来若干帧里分别移动到哪**。每个点的这条位置序列就叫它的 **track**，
所有点的 track 合起来叫 **flow**（也叫 point flow / point track / 轨迹）。

一个「flow-based policy」的流程通常是：

```
当前画面 + 任务描述  →  [预测 flow]  →  flow + 画面  →  [policy]  →  机器人动作
```

代表工作：ATM (RSS'24)、Track2Act (ECCV'24)、Im2Flow2Act (CoRL'24)、PPI (RSS'25) 等，
详见 `FLOW_TO_ACTION_SURVEY.md` 里读过的 20 篇。

### 0.3 什么是 affordance，为什么说 flow 不是它

**affordance** 在这里指的是「物体上哪里可以被怎样操作」——
具体到机械臂就是**该抓哪个位置、以什么姿态抓**。

**核心论证（不变性论证）：**

> 把一个杯子从 A 点搬到 B 点。**不管你抓的是杯把还是杯壁，杯子上那些被追踪的点，
> 运动轨迹一模一样。**

所以：**物体的 flow 对「抓哪儿」是不变的（invariant）**，
一个对某个量不变的信号，**不可能携带关于那个量的信息**。

用符号写清楚（后面会反复用到）：

| 符号 | 含义 |
|---|---|
| $T(t)$ | **物体**在时刻 $t$ 的位姿（SE(3) 刚体变换）—— 这就是 flow 描述的东西 |
| $C$ | **contact / grasp offset** —— 夹爪相对物体的固定偏移，「抓哪儿」就是它 |
| $E(t)$ | **末端执行器**（夹爪）在时刻 $t$ 的位姿 —— 这才是机器人真正要执行的 |

抓住之后三者的关系是：

$$E(t) = T(t)\cdot C$$

**给定 $T$（flow 给的）还差一个 $C$ 才能算出 $E$（动作）。而 $C$ 恰恰是 flow 里没有的。**

### 0.4 那些方法的性能是从哪来的？

这是我们要回答的问题。假设：**从被一起追踪的机器人像素里来的。**

ATM 的预处理代码是这样采样追踪点的（`scripts/preprocess_libero.py:127`）：

```python
points = sample_from_mask(np.ones((H, W, 1)) * 255, num_samples=num_points)
```

`np.ones(...)` 是一张**全 1 的 mask** —— 意思是**整幅图均匀采样**，完全不区分
「这个点在物体上」还是「这个点在机械臂上」。所以它追踪的点里**混着大量机械臂本体的点**。

而**机械臂本体的 flow 就是动作本身**（夹爪未来的轨迹 = 未来要执行的动作序列）。
所以「预测 flow 帮助了 policy」这件事，可能只是「policy 从另一条路看到了答案」。

### 0.5 怎么验证 —— 实验的基本思路

拿一个公开的、用 flow 的方法（**ATM**），**在评测时动态破坏它的 flow 输入**，
看成功率掉多少。分阶段、分相机分别破坏，看效应落在哪一格。

关键的一刀是**按阶段切开**：

| 阶段 | 定义 | 物体在动吗 | 机械臂在动吗 |
|---|---|---|---|
| **approach**（接近） | 从 episode 开始 → 夹爪闭合（抓住东西）那一刻 | ❌ 静止 | ✅ 在动 |
| **transport**（搬运） | 夹爪闭合之后 | ✅ 跟着手走 | ✅ 在动 |

**逻辑：**

- **transport 段 flow 有用是平凡的** —— 物体在动，flow 就是它的运动，这是废话
- **approach 段物体压根没动**，所以「物体的 flow」在这一段应该毫无信息
- **⇒ 如果 approach 段破坏 flow 还是掉分，那掉的分只能来自画面里另一个在动的东西：机械臂本身**

这就是整个实验的支点。

---

## 01 · 术语与符号表（读表之前必看）

### 1.1 实验设置词

| 词 | 含义 |
|---|---|
| **task** | 一个具体任务，比如「把黑碗从桌子中央拿起放到盘子上」。我们用 LIBERO 的 `libero_spatial` 套件，共 **10 个 task** |
| **episode / rollout** | 跑一遍任务，从初始状态到成功或超时。成功记 1，失败记 0 |
| **成功率** | 一批 episode 里成功的比例。**本文所有 0.710 之类的数字都是成功率**，1.0 = 全成 |
| **seed** | 随机种子。同一个条件用两个不同 seed 各跑一遍，用来看结果稳不稳 |
| **条件 / condition** | 一种破坏方式（破坏哪个阶段、哪个相机、用什么手法）。每个条件跑 10 tasks × 20 rollouts × 2 seeds = **400 episodes** |
| **ablation** | 消融实验 —— 把系统的某个部件拿掉/破坏，看性能掉多少，以此判断这个部件贡献了多少 |
| **baseline / 基线** | 不做任何破坏的原始性能。本文是 `A_full` = 0.710 |
| **probe / 探针** | 我们写的那段代码，在评测时挂进 ATM 内部拦截并篡改 flow。文件：`atm/utils/flow_probe.py` |

### 1.2 两个相机（这个区分是全文的关键）

LIBERO 给 policy 两路图像，探针可以分别破坏其中一路：

| 代号 | 相机 | 画面里是什么 | 被采样的点里机器人占比 |
|---|---|---|---|
| `agentview`（视角 0） | 架在场景外的**第三人称**相机 | 桌面、物体、机械臂的一部分 | **9%** |
| `robot0_eye_in_hand`（视角 1） | 装在**手腕**上的相机 | 一大块恒定是夹爪本身 | **21%** |

**腕部相机的画面里机器人占比是第三人称的 2 倍多** —— 所以如果「approach 段的效应来自
机器人像素」这个假设成立，**掐腕部应该比掐第三人称掉得多**。这正是 `Bw` vs `Ba` 这组对照。

### 1.3 三种破坏手法

破坏 flow 有很多种做法，选哪种直接决定结论能不能成立：

| 手法 | 怎么做 | 破坏了什么 | 为什么这样设计 |
|---|---|---|---|
| **shuffle**（主力） | 把这个 env 的预测 track 换成**同一时刻、同一 task、另一个并行 env 的真实 track** | 「哪些点」「它们在哪」「它们怎么动」全破坏 | **不用置零**：置零会让输入变成 policy 从没见过的取值，掉分可能只是因为「输入变奇怪了」。换成另一个 env 的**真实** track，输入的边缘分布一模一样，只是**和当前画面对不上**。这样掉的分只能归因于「配对关系被破坏」 |
| **freeze** | 把整条轨迹压成 t=0 那一帧的位置重复 | 只想破坏「怎么动」，保留「哪些点」「在哪」 | **已实测失效，见 §3.2** |
| **anchored**（新，替代 freeze） | 取另一个 env 的**位移场**，从**本 env 的 t=0 位置**发出 | 只破坏「怎么动」 | freeze 的正确版本，见 §3.2 |

> **derangement（错位排列）**：shuffle 用的置换必须**没有不动点** ——
> 即不能有任何一个 env 拿到自己的 track，否则那个 env 等于没被破坏。

### 1.4 条件命名

| 条件 | 破坏哪个阶段 | 破坏哪个相机 | 用什么手法 |
|---|---|---|---|
| `A_full` | **不破坏**（基线） | — | — |
| `B_approach` | approach | 两个都破坏 | shuffle |
| `Bw_approach_wrist` | approach | **只破坏腕部** | shuffle |
| `Ba_approach_agent` | approach | **只破坏第三人称** | shuffle |
| `C_transport` | transport | 两个都破坏 | shuffle |
| `F_all` | **全程** | 两个都破坏 | shuffle（破坏上限的参照） |
| `G_freeze_all` | 全程 | 两个都破坏 | freeze |
| `Gb_freeze_approach` | approach | 两个都破坏 | freeze |
| `Gc_freeze_transport` | transport | 两个都破坏 | freeze |

后缀 `_seed0` / `_seed1` 是两次独立重复。

### 1.5 统计符号（第一个表格里的每一列）

以主结果表的这一行为例：

```
Bw_approach_wrist        400   0.615   -0.095   -3.58   0.0060   [-0.155,-0.035]
                          n     mean      Δ      配对t     p          95%CI
```

| 符号 | 全称 | 含义 | 这一行怎么读 |
|---|---|---|---|
| **n** | sample size | 参与统计的 episode 总数 | 400 个 episode |
| **mean** | — | 这个条件的**成功率** | 61.5% 成功 |
| **Δ**（delta） | difference | **这个条件的成功率 减去 基线的成功率**。负数 = 掉分 | 0.615 − 0.710 = **−0.095**，即比基线**掉了 9.5 个百分点** |
| **配对 t** | paired t-statistic | 效应相对于波动有多大。绝对值越大越明显。见下方详解 | −3.58，方向是掉分 |
| **p** | p-value | **假设这个条件其实完全没有效应，纯靠运气能看到这么大（或更大）差值的概率**。习惯上 **p < 0.05 算显著** | 0.006 = 只有 **0.6%** 的概率是运气，所以**这个效应是真的** |
| **95% CI** | 95% confidence interval，置信区间 | 真实效应值有 95% 把握落在这个范围里。**区间不跨零 ⇒ 显著** | [−0.155, −0.035] 全是负数，**不跨零** ⇒ 确实掉分，掉 3.5 到 15.5 个点之间 |

#### 「配对 t 检验」是什么，为什么必须用它

**朴素的做法（错的）**：把 400 个 episode 当成 400 次独立抛硬币，
直接比两个条件的成功比例（这叫 naive z-test）。

**为什么错**：同一个 task 里的 20 个 episode **高度相关** ——
同样的场景布局、同样的物体、同样的难度。把它们当独立样本会**高估有效样本量、低估方差**。

**配对 t 检验（对的）**：

1. 对**每一个 task**，算出它在基线下的成功率、和在该条件下的成功率
2. 相减，得到 **10 个差值**（10 个 task 各一个）
3. 对这 10 个差值做单样本 t 检验：「它们的均值显著不等于 0 吗？」

这样每个 task 自己和自己比（**配对**），场景难度的差异被抵消掉了。

> **df = 9** 是自由度（degrees of freedom）= 10 个 task − 1。t 分布的形状由它决定。

**两种口径给的答案确实不一样，而且方向不一致：**

| 条件 | 朴素 z | 配对 t | 差异方向 |
|---|---|---|---|
| `C_transport` | −6.34 | **−4.52** | 配对更**保守** |
| `Bw_approach_wrist` | −2.86 | **−3.58** | 配对更**激进** |

一个变松、一个变紧 —— **不能事后统一打个折了事，也不能事后选一个好看的**。
**论文必须报配对检验**，而且这个选择是在看结果之前定的。

#### 「等价检验」是什么

普通的显著性检验回答「有没有效应」。当 p 很大（比如 0.90）时，
**「没测出效应」和「确实没有效应」是两回事** —— 前者可能只是样本太少。

**等价检验（equivalence test）** 换个问法：**「效应最大能有多大？」**
答案就是**置信区间的边界**。

例：`Ba_approach_agent` 的 95% CI = **[−0.043, +0.048]**。

这不是「我们没测出来」，而是：**n=400 的样本量足以排除任何大于 4.8 个百分点的效应。**
第三人称相机的 flow 对 approach 阶段的贡献，**上界就是 4.8 点，点估计是 +0.3 点。**

这是一个**有信息的零**，不是无知的零。**审稿人会认这个，不会认「没测出来」。**

---

## 02 · 主结果：ATM 按阶段 ablation

### 2.1 实验怎么做的

**探针挂在 `model.track.reconstruct`。** 为什么是这里：ATM 的 track 有**两条**路进 action ——

1. `track_encode()` → track token → transformer
2. `act()` 里 `rec_tracks` **被直接 concat 进 policy head 的输入**：
   `feat = torch.cat([x[:, -1], rearrange(rec_tracks[...])], dim=-1)`

两条都源自 `track.reconstruct`，**所以那是唯一能一次切断两条的单点**。

- **阶段判据**：gripper 一旦闭合就永久进入 transport（`action[:, -1] > 0`）。
  与 PPI 的 keyframe 定义一致，不是我们自创的
- **规模**：每个条件 10 tasks × 20 rollouts × 2 seeds = **400 episodes**

### 2.2 结果表

**pass2 于 2026-07-28 03:30 收工，六个条件全部达到 n=400（终版）：**

```
条件                       n    mean       Δ     配对t        p          95%CI
A_full (基线)             400   0.710       —       —        —
Bw_approach_wrist        400   0.615   -0.095   -3.58   0.0060   [-0.155,-0.035]  ✅
Ba_approach_agent        400   0.712   +0.003    0.12   0.9043   [-0.043,+0.048]  等价检验
B_approach               400   0.637   -0.072   -1.95   0.0829   [-0.157,+0.012]  ✗ 跨零
C_transport              400   0.420   -0.290   -4.64   0.0012   [-0.431,-0.149]  ✅
F_all                    400   0.337   -0.372   -9.69   0.0000   [-0.459,-0.286]  ✅
────────────────────────────────────────────────────────────────────────────
Bw vs Ba (同阶段直接对比)  400           -0.098   -4.17   0.0024   [-0.150,-0.045]  ✅ 头条
```

（列的含义见 §1.5；条件的含义见 §1.4）

**逐行白话翻译：**

| 行 | 白话 |
|---|---|
| `A_full` | 什么都不破坏，成功率 **71%** |
| `Bw_approach_wrist` | 只在 approach 段破坏**腕部**相机的 flow → 掉 **9.5 点**，**显著**（p=0.006） |
| `Ba_approach_agent` | 只在 approach 段破坏**第三人称**相机的 flow → **几乎没变**（+0.3 点），且能确定效应不超过 4.8 点 |
| `B_approach` | approach 段两个相机都破坏 → 掉 7.2 点，**CI 仍跨零，不显著**。原因见 §2.3.1 —— **这与主结论一致** |
| `C_transport` | transport 段破坏 → 掉 **29 点**，显著。（符合预期，transport 段 flow 有用是平凡的） |
| `F_all` | 全程破坏 → 掉 **37.2 点**。这是破坏的**上限参照** |
| `Bw vs Ba` | 把上面两条**直接对比**：差 9.8 点，**显著**（p=0.0024）。**这一行是全文最关键的一个数** |

> **加倍样本没有动摇任何一个显著结论。** `C_transport` 从 −0.300 微调到 −0.290（p 0.0014→0.0012）。

### 2.2.1 为什么 `B_approach` 不显著 —— 这与主结论一致，不是矛盾

补到 n=400 后 p 从 0.273 → 0.083，CI 收到 [−0.157, +0.012]，**仍然跨零**。

原因：`B_approach` 是**两个相机都掐**，而这两路一实一虚：

| 条件 | Δ | 每-task 标准误 |
|---|---|---|
| `Bw`（只掐腕部） | −0.095 | 0.027 |
| `Ba`（只掐第三人称） | +0.003 | — |
| `B`（两个都掐） | −0.072 | **0.037**（比 `Bw` 高 40%） |

**掐第三人称不带来效应，只带来方差。** 效应量没增加而 task 间标准误高了 40%，
所以显著性反而更差。

**这正是「approach 段的效应全在机械臂像素上」的另一个侧面证据：
往里加一个纯噪声通道，只稀释信噪比。**

→ **论文的头条数必须是 `Bw` vs `Ba`（p=0.0024），不是 `B_approach`。**
`B` 是个更粗的条件，把一个真效应和一个零混在一起。

### 2.3 关键对照：`Bw` vs `Ba`

这两个条件**唯一的差别是掐哪个相机**：

- 同一个阶段（approach）
- 同一种手法（shuffle）
- 同样的物体、同样静止

**结果从 0 变成 −9.5 点。**

对应的是：腕部画面里夹爪占采样点的 **21%**，第三人称里机器人只占 **9%**。

### 2.4 三句核心结论

1. **approach 阶段掐掉腕部视角的 flow：−0.095，p=0.006**（显著掉分）
2. **同一阶段掐掉第三人称视角的 flow：+0.003，p=0.90，95% CI = [−0.043, +0.048]**
   —— 等价检验，**效应上界 4.8 点**
3. **两者直接对比：−0.098，p=0.0024**（显著）

### 2.5 为什么这直接回掉了最强的那个质疑

审稿人最可能说的是：

> 「approach 阶段物体根本没动，flow 当然没用，你们测了个废话。」

**这个质疑在物体通道上完全成立** —— `Ba_approach_agent`（第三人称，物体主导）
效应恰好是零，**它就是这个质疑的确认**。

**但这个质疑预测的是「两个视角都是零」** —— 而腕部不是零，是 −9.5 点。

approach 阶段唯一还在起作用的 flow 是**机械臂自身的 flow**，因为**夹爪的未来轨迹
就是 action 本身**。质疑推不出「恰好是零，而剩下那点全落在机械臂像素上」。

**这一步是 `Bw` vs `Ba` 这组对照挣来的，不是靠论证。**

### 2.6 seed 稳定性，以及一个附带发现

同一条件跑两个不同随机种子，看结果稳不稳：

```
条件                  seed0 / seed1    |差|
A_full              0.710 / 0.710   0.000   ← 完全一致，仪器本身稳定
Ba_approach_agent   0.720 / 0.705   0.015
C_transport         0.430 / 0.410   0.020
B_approach          0.625 / 0.650   0.025
Bw_approach_wrist   0.645 / 0.585   0.060
F_all               0.275 / 0.400   0.125   ← 仍是次大值的 2 倍，异常
```

`A_full` 两个 seed **完全一致** —— 说明整套评测流程本身没有随机噪声问题。

**`F_all` 的 seed 方差异常在补满 n=400 后依然存在。**
假设（未验证）：它是唯一全程双视角都被破坏的条件，
policy 完全失去 flow、行为近乎崩溃，**崩溃状态对初始条件更敏感**。

> **原本登记的检验作废了**：本来打算用 `G_freeze_all`（保留 grounding 和定位、只抹运动）
> 的 seed 方差来验这条 —— 但 freeze 已被判死（§3.2）。
> **改由 anchored 组来验**：若上述解释成立，anchored 的 seed 方差应显著小于 0.125。

不给 `F_all` 加第三个 seed —— 它的 CI 已经够窄且远离零，只是个上界参照。

---

## 03 · 论点的两次修正

### 3.1 修正一：pointflow 不只是 flow

原来的论证跳得太快：

> 「object flow 对 grasp 不变」→「所以 flow 的收益必然来自 transport」

中间漏了一步 —— **所谓 pointflow 这个输入，至少捆绑了三个不同的信号：**

| # | 信号 | 从哪来 | 受不变性论证约束吗 |
|---|---|---|---|
| 1 | **哪个物体**（语言 grounding） | GroundingDINO 用任务文本当 prompt | ❌ 不受 |
| 2 | **它现在在哪**（定位） | SAM 出 mask | ❌ 不受 |
| 3 | **它接下来往哪走**（运动） | 预测的位移 | ✅ **只有这一个才是不变性论证所指的 flow** |

> **grounding** 在这里指「把任务描述里的名词对应到画面里的具体物体」。
> 七个物体的场景里，光是知道「该看哪一个」本身就解决了任务的一大块，
> **而这跟 grasp 姿态无关。**

**这会 confound（混淆）跨方法定量律**：ATM 和 PPI 都可能在 approach 掉分，但机制不同 ——
ATM 是因为追了机械臂本体的点，PPI 是因为 pointflow 里自带 grounding。
**同一个观测量，两种机制。**

**当时的方案：加一组只破坏「运动」的干预（freeze）**

| 干预 | 破坏了什么 | 保留了什么 |
|---|---|---|
| shuffle（B/C/F 组） | grounding + 定位 + 运动 | — |
| **freeze（G 组）** | **只破坏运动**（轨迹压成 t=0 静止点重复） | grounding + 定位 |

**两者之差 = 纯运动分量的贡献。**

### 3.2 ⚠️ 2026-07-28 更新：freeze 这个干预失效了

pass2 一开跑就把 G 组自己判死了：

```
条件                        每个 task 的成功率                                 avg
A_full   (基线)            0.45 1.0  0.9  0.8  0.8  1.0  0.65 0.65 0.55 0.3   0.710
F_all    (shuffle 全程)    0.20 0.35 0.35 0.30 0.15 0.45 0.15 0.35 0.25 0.20  0.275
C_transport (shuffle)      0.0  0.8  0.75 0.15 0.45 1.0  0.25 0.3  0.15 0.25  0.410
────────────────────────────────────────────────────────────────────────────────
G_freeze_all               0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.000
Gb_freeze_approach         0.05 0.0  0.0  0.0  0.0  0.15 0.0  0.0  0.0  0.0   0.020
Gc_freeze_transport        0.15 0.0  0.45 0.0  0.0  0.2  0.4  0.15 0.0  0.35  0.170
```

#### 判死它的是一条单调性自检

freeze 破坏的东西是 shuffle 的**真子集**：

| 干预 | 哪些点 | 它们在哪 | 它们怎么动 |
|---|---|---|---|
| shuffle | ❌ 破坏 | ❌ 破坏 | ❌ 破坏 |
| freeze | ✅ 保留 | ✅ 保留 | ❌ 破坏 |

**破坏得更少 ⇒ 成功率必须 ≥ shuffle。**
实测 **0.000 < 0.275**，**严格更差** —— 这在逻辑上不可能，
**除非 freeze 引入了计划外的第三条破坏通道。** 找到了两条：

**① 分布外取值（out-of-distribution）。**
恒定不动的 track 在训练数据里**从不出现**（相机相对运动至少让它动一点点），
而 ATM 把 `rec_tracks` **直接 concat 进 policy head**，
恒定值是 policy 从没见过的取值。10 个 task **精确为 0**（不是「低」）—— 手臂基本没动。

**② 阶段状态机被锁死。**
阶段判据是「gripper 是否闭合」。freeze 让手臂不动 → **夹爪永远不闭合** →
**永远停在 approach** ⇒ **`freeze_approach` 退化成了 `freeze_all`**。
0.020 ≈ 0.000 不是巧合，**是同一个条件**。

> shuffle 没有陷阱 ②：它给的是另一个 env 的**真实** track，手臂照动。
> `B_approach = 0.650` 说明 episode 正常越过了 approach 阶段。

#### `Gc` 把两条病因分开了，并量出 OOD 惩罚 = 24 点

`Gc_freeze_transport = 0.170` —— **不是零，而且 task 间散布正常（0.0–0.45）。
这是一个活着的 policy。**

原因：Gc 的 **approach 段完全不受干预**，手臂正常伸过去、正常闭合夹爪，
**状态机正常翻转** ⇒ **陷阱 ② 在 Gc 上不触发**。于是它只剩陷阱 ①。

**同为 transport 段、同为双视角，只换手法：**

| 手法 | 条件 | 成功率 |
|---|---|---|
| shuffle | `C_transport` | 0.410 |
| freeze | `Gc_freeze_transport` | 0.170 |
| | **差 = 纯 OOD 惩罚** | **−0.240** |

**单调性第三次被违反，幅度 24 点。这是判死 freeze 的定量依据：
OOD 通道本身值 24 点，而我们要测的 approach 效应只有 9.5 点 —— 噪声是信号的 2.5 倍。**
freeze 在任何阶段都不可能量出想量的东西。

**G 组三个数现在有了完整的机制解释**，不是「policy 崩了」这种含糊说法：

| 条件 | 陷阱 ① OOD | 陷阱 ② 阶段锁死 | 结果 |
|---|---|---|---|
| `G_freeze_all` | ✅ 全程 | （本来就全程，无所谓） | 0.000 |
| `Gb_freeze_approach` | ✅ | ✅ **退化成 all** | 0.020 ≈ G |
| `Gc_freeze_transport` | ✅ | ❌ 不触发 | **0.170** |

**seed1 的三个 freeze 条件应跳过** —— 机制已经完全清楚，再跑一个 seed 买不到新信息。

#### 修正方案：anchored shuffle

取**另一个 env 的位移场**，从**本 env 的 t=0 位置**发出：

```python
donor = x[perm] - x[perm][:, :, :1] + x[:, :, :1]
#       ↑ 另一 env  ↑ 减掉它的起点    ↑ 换成本 env 的起点
```

| | 哪些点 | 它们在哪 | 它们怎么动 | 输入在训练分布内吗 |
|---|---|---|---|---|
| freeze | ✅ 保留 | ✅ 保留 | ❌ 破坏 | **❌ 恒定值，没见过** |
| **anchored** | ✅ 本 env | ✅ 本 env 的 t=0 | ❌ 另一 env 的位移 | **✅ 真实位移幅度** |

这才是原本想要的「只剥掉运动分量」，且不制造 OOD 输入、不锁死状态机。

**已实现**：`flow_probe.py` 新增 `anchored` / `anchored_approach` / `anchored_transport`
三个 mode，现有 shuffle / freeze 行为逐元素不变（已回归验证）。
单元测试 `tests/test_flow_probe_anchored.py` **六条全过**：

```
PASS  test_anchored_preserves_t0_exactly        t=0 位置逐元素不变
PASS  test_anchored_motion_comes_from_another_env
PASS  test_anchored_is_not_constant             与 freeze 的关键区别，位移幅度未塌
PASS  test_phase_gating_is_complementary        approach/transport 门控严格互补
PASS  test_view_selectivity                     只掐指定相机
PASS  test_off_is_identity
```

### 3.3 修正二：flow→action 这一步，对 transport 是平凡的，对 approach 是未定义的

**原来打算主张的**：「flow→action 是个黑箱、是个 open problem」。

**这条站不住** —— SPOT (arXiv 2411.00965) 已经把它写成**一次矩阵乘法**：

$$A_{t:t+h} = T^{obj}_{EE}\cdot T_{t:t+h}$$

也就是 §0.3 那个 $E(t) = T(t)\cdot C$：**末端位姿 = 物体轨迹 ∘ 常量 contact offset。**

推论：**给定 $T$ 和 $C$，机器人的 flow 就被完全决定了** ——
它作为一个**预测目标**是冗余的，**任何它买到的性能都是 $C$ 在泄漏。**

**修正后的主张：**

> flow→action **对 transport 段是平凡的**（一次矩阵乘法），
> **对 approach 段是未定义的**（$C$ 从哪来？）。
> **现有方法通过泄漏机器人像素，把 approach 段的问题藏了起来。**

比原版更保守（承认了平凡的那一半），也更锋利（把问题精确定位到 $C$ 上）。
**§5 的文献表就是这条主张的证据库。**

### 3.4 一个更强的 claim：affordance 含量与可迁移性直接冲突

追踪机械臂本体的点**确实**能让 flow 决定 action —— 但那等于放弃了 flow 最初的卖点：

| 追踪什么 | 能决定 grasp？ | embodiment-agnostic（换个机器人还能用）？ |
|---|---|---|
| **只追物体点** | ❌ | ✅ 所以能从**人类视频**学 —— 这是这类方法的核心卖点 |
| **加上机械臂点** | ✅ | ❌ Franka 手臂的 flow 迁不到别的本体 |

**两者不可兼得，而没有一篇论文把这条 trade-off 写出来。**

这比原来的「flow 不提供 affordance」更强 ——
**flow 的 affordance 含量与它的可迁移性直接冲突。**

→ 定量律的 x 轴（被追踪点里机器人本体的占比）因此**不是干扰变量，
它就是这条 trade-off 曲线本身。**

---

## 04 · pipeline：提了三个方案，死了两个

出发点是那个正确的问题：**验证完 flow ≠ affordance 之后，pipeline 级别的创新是什么？**

初始构想 **AffordanceFlow**：只预测物体 flow（保住 cross-embodiment），
用 affordance-VLM 给出粗略可抓区域，再用「物体轨迹 $T(t)$」去给候选抓法**打分/筛选**。

### 4.1 方案 A：transport-conditioned grasp scoring → **死**

#### 要测什么

**知道物体接下来的轨迹 $T(t)$，能从「语义上都合法」的候选抓法里排除掉多少个？**

比方说要把碗放进一个窄抽屉，从上方抓可能会撞到抽屉顶 ——
如果 $T(t)$ 能提前排除掉这类抓法，这个打分步骤就有价值；排除不掉，就没价值。

#### 三个指标（先定义清楚）

| 指标 | 含义 |
|---|---|
| **oracle** | 知道完整轨迹，只挑可行的抓法 → 成功率上限 |
| **blind** | 完全不看轨迹，随便挑一个语义合法的抓法 → 下限 |
| **blind+reach** | 不看轨迹，但至少保证「够得着」（IK 有解） → 更公平的下限 |
| **headroom** | **oracle − blind+reach** —— **「知道轨迹」到底额外买到了多少** |

**先定死判据（在看数据之前）：headroom < 5 点 → 这个 module 死。**

#### 做了什么

1. **`tools/grasp_pilot_probe.py`** —— 只读探针，摸清 LIBERO 的 body/site/geom 命名、
   抓取发生的时刻、以及 $C_0$（demo 里真实用的那个 contact offset）
2. **`tools/grasp_pilot.py`** —— 候选抓法生成 + **plotly 3D 可视化**
   （物体网格 + 夹爪几何，都画在物体坐标系里，用来肉眼确认候选姿态合不合理）
   - 关键细节：必须**绕质心轴**旋转，不能绕 body 原点
     —— LIBERO 的 body 原点未必在几何中心，碗差了 9°
   - 网格直接从编译后的 MuJoCo model 里取（`mesh_vertadr/vertnum/faceadr/facenum`）
   - **肉眼确认通过**：夹爪到碗 0.9–3.6 mm，无穿模
   - 在线版：https://claude.ai/code/artifact/f62e5b2e-0562-4d59-957b-cf6ee41f7dc7
3. **`tools/grasp_feasibility.py`** —— K×J 可行性矩阵（K 个候选抓法 × J 条物体轨迹）
   - 候选族：8 方位角 × 3 倾角 × 3 高度 × 3 半径 = **216 个**，
     再按「贴着物体表面」过滤（穿透 ≤ 3 mm 且 间隙 ≤ 8 mm）

#### 判死它的是自检，不是主结果

加了一条自检：**「demo 自己真实执行过的那个抓法，必须被判为可行」**。
这条不需要任何领域知识就能写，而它当场暴露了问题：

```
task                                          自检   C漂移mean   C漂移max
put_the_bowl_on_the_plate                     1.00      3.1mm      9.5mm
open_the_top_drawer_and_put_the_bowl_inside   0.40     22.4mm     92.4mm   (两段式 task)
KITCHEN_SCENE4 碗放进底层抽屉                  0.30      2.3mm      6.1mm
KITCHEN_SCENE6 杯子放进微波炉                  0.40      5.2mm     17.4mm
```

> **「C 漂移」** = 抓住之后 $C$ 到底有多恒定。$E(t)=T(t)C$ 这个式子成立的前提是
> $C$ 是常量；漂移大说明抓取不是刚性的，公式本身不适用。

三个原因，**前两个是 bug（已修），第三个不是**：

1. ~~**transport 段跑过了松手点**~~ —— `..._and_close_it` 的 task 在放下之后还要回去关抽屉，
   强行令 $E(t)=T(t)C$ 会把夹爪拽回抽屉里 → 全 0。
   **已修**：`t_end` = 夹爪首次张开的那一刻
2. ~~**geom 级接触白名单被 box 分解击穿**~~ —— LIBERO 的 collision mesh 是几十个 box 拼的
   （碗 41 个、柜子 40+ 个），IK 解差几毫米就从 `white_cabinet_1_g35` 换成 `g37`，
   白名单当场失效。**改成 body 对 + 穿透深度判定，没救回来**
3. **紧公差场景下，可行性由毫米级间隙 + 7-DoF 冗余解算决定。**
   SCENE4 的 $C$ 漂移**只有 2.3 mm**（刚性抓取假设完全成立）而自检仍然只有 0.30。
   加了零空间偏置 DLS IK 也没改善（0.20→0.20，0.83→0.825）

> **7-DoF 冗余是什么**：Franka 有 7 个关节，但末端位姿只有 6 个自由度 ——
> 多出来的那 1 个意味着**同一个末端位姿对应无穷多组关节角**。到底用哪一组由 IK 求解器决定，
> 而不同的关节角会不会撞到柜子是完全不同的。
> 我们试的是**零空间偏置**：$dq = J^{+}e + (I - J^{+}J)\,k_{ns}(q_{ref}-q)$，
> 前一项解末端位姿，后一项在**不影响末端位姿的零空间里**把关节角拉向 demo 的真实姿态。没用。

#### 第三条同时判死了 pipeline 本身

> 在唯一值得打分的 regime（紧公差放置）里，**feasibility 不是 $(C, T)$ 的函数** ——
> 它还取决于毫米级间隙和 planner 的冗余解算。
> **我算不出来，一个学出来的打分器同样算不出来，因为输入里根本就没有这个信息。**
> 这不是实现问题，**是这个 module 的输入输出定义本身不成立。**

**可信的那部分数字说「没效应」**：自检 = 1.00 的桌面自由放置 task，
每条轨迹 25–39 / 197 个候选都可行，**headroom ≈ 0–3 点**，远低于 5 点的判据。

**所有 headroom 数字（+8.2 / +4.5 / +22 / +18 / +12）全部作废。**

**若要救**：只能把「几何可行性」换成「从执行结果学出来的 critic」——
不算 feasibility，直接学「这个抓法配这条轨迹最后成没成」。
**是个大得多的项目，就凭现在的证据不建议开。**

### 4.2 方案 B：latent object-centric JEPA + MPC → **死（论证层面）**

**构想**：既然 flow 只是给人看的，把它压进一个 latent（隐空间表示）；
用 world model 预测「某个 action 会导致 latent 怎么变」，
与「物体 latent 应该到的位置」求偏差，反解出 action；或者直接上 MPC。

> **JEPA** = Joint-Embedding Predictive Architecture，在 latent 空间里做预测的 world model。
> **MPC** = Model Predictive Control，用模型向前推演若干步、挑代价最小的动作序列。

**问题：简并性（degeneracy）不会因为换坐标系而消失。**

达到同一个物体 latent 变化的 action 集合 = $\partial f/\partial a$ 的**零空间**。
「抓杯把」和「抓杯壁」把杯子搬到同一个地方 ——
它们在 pixel flow 里不可区分，**在任何 latent 里同样不可区分，因为 latent 是 flow 的函数**。

> **一次坐标变换不能把一个非单射的映射变成单射的。**

MPC 同理：cost 定义在物体 latent 上，那么零空间里**所有 action 代价相同**，
MPC 会在里面任选一个 —— **这正是「随机采一个抓取」的连续版本。**

（**V-JEPA 2-AC 是活体证据**：它需要**人给两张 sub-goal 图**才能做 pick-and-place，
而且论文从未讨论过这个简并性。见 `FLOW_TO_ACTION_SURVEY.md` §8。）

### 4.3 剩下什么可以保留

1. **「只预测物体 flow」这个约束是对的**，而且现在有实测支撑：
   `Ba_approach_agent` 的等价检验说明第三人称（物体主导）的 flow 对 approach
   贡献**上界 4.8 点**。放弃机器人 flow 的代价是**可测量、可上界的**，不是空谈
2. **$E(t) = T(t)\cdot C$ 这个分解本身是对的**，只是不能拿它去做 scoring。
   它可以当**论文的分析框架** —— 把「flow 买到了什么」精确拆成 $T$ 的部分和 $C$ 的部分
3. **「flow 增强 affordance」而不是「affordance + flow」这个定位仍然成立** ——
   affordance 是静态的、对任务无感知；flow 是动态的、对 fine-grained 抓取无感知。
   **但具体怎么增强，现在没有可执行的方案**
4. **诚实的现状**：measurement 这一侧（论文主体）已经足够独立成文。
   pipeline 那一侧还没有形状，**不要为了凑一个 method 章节而硬造**

---

## 05 · 文献工作：`FLOW_TO_ACTION_SURVEY.md`

**20 篇 flow→action 方法逐字读过正文**（不是只读摘要）。
ATM / PPI 另有源码级证据；ToolFlowNet (2211.09006) 和 AVDC (2310.08576)
早于 arXiv 的 HTML 渲染，读的是 PMLR / arXiv 的 PDF 原文。

按三个标准整理成三张表：**① 追什么 ② flow 怎么变成 action ③ contact 从哪来**。

### 5.1 核心发现：二十个方法，没有一个从 flow 里得到 contact

> 回忆 §0.3：contact / $C$ 就是「该抓哪儿」。这一节在数：**大家都是从哪儿弄到 $C$ 的。**

| contact 来源 | 方法 |
|---|---|
| **明码标价从外部搬进来（12 个）** | 随机采一个（AVDC）、人手摆机械臂（General Flow、EC-Flow）、从 demo 里取（SPOT）、现成 grasp detector（Object-Part Scene Flow）、100 万条人工标注 + GraspNet（A₀）、另开一条 interface（PPI 的 keypose）、URDF + 100 万合成对（KITE）、特权信息 RL（Dex4D）、工具默认已在手里（ToolFlowNet）、人给 sub-goal 图（V-JEPA 2-AC）、仿真 play data（Im2Flow2Act） |
| **完全没讨论（8 个）** | F2F-AP、Flow as Flow、ChronoFlow-Policy、DAWN、FOFPred、Tra-MoE… |
| **隐性泄漏，无人注意到** | **ATM / Tra-MoE** —— 机器人像素混在被追踪的点里，**收益被记在 flow 名下** ← **我们量的就是这一格** |

### 5.2 四条最有用的原句

- **AVDC**："if the object is graspable, **we randomly sample a grasp on the object**"
  → object flow 推不出该抓哪儿，所以随机采一个。**最直白的承认。**
- **General Flow**：标题叫 "Foundation **Affordance**"，论证靠 "remaining neutral to specific
  manipulators"（对具体机械臂保持中立），实验里却写 "we **manually position the robotic arm**"。
  → **用来论证「它是 affordance」的那条性质，恰恰是让它不可能是 affordance 的那条性质。**
  （Gibson 的 affordance 是「物体对**某个特定 agent** 提供什么动作可能」，agent 是定义的一部分。
  抽掉 manipulator 之后剩下的量，恰好把 affordance 之为 affordance 的东西抽掉了。）
- **FOFPred**：一个独立作者组做出了**完全相同的诊断** —— 只追物体会 "overlooking crucial global
  information, such as the overall movement of a **manipulator**" —— 于是改用整幅图 dense 光流
  **故意**把机器人放进来，**但从未量过这个改动买到了多少**。
  → **诊断被独立验证了，测量的空位还留着。**
- **ToolFlowNet**：对机器人控制的刚体预测 flow，作者自己点破它
  "reflects the **'intended' action from the robot**"。
  → **追机器人刚体，flow 就是 action 的投影；追物体，就不是。** 这是这根轴两端最干净的对照。

### 5.3 我们的空位（三格）

1. **没有人量过**「flow 的收益里有多少来自被一起追踪的机器人像素」——
   FOFPred 甚至明确诊断出来并据此改了设计，**但没测量**
2. **没有人拆过 approach vs transport**，尽管 SPOT / General Flow / ToolFlowNet
   都把自己明确限定在 post-grasp（= 只做 transport 那一半）
3. **ChronoFlow-Policy 已把物体点和夹爪点显式分成两组**。⚠️ 修正（07-28 读 HTML 全文）：
   它在 Fold Towel（真机）上**做过一次分组 ablation**（去掉 gripper flow：87%→80%），
   此前记录的「没测两者各自贡献」是错的。但它是单任务/可变形/无阶段切分/无统计/单机器人，
   且把 gripper flow 框架为 "complementary cues" 而非 embodiment 泄漏 ——
   **是离我们最近的一次测量，引用它，然后在四个维度上超过它**

**当前 ATM 的结果正好填第 1、2 格。**

**已排除**：`FlowPolicy` (AAAI'25) 的 "flow" 是 flow matching（一种生成模型），同名不同义。
同类需警惕 `PointFlowMatch` / `CoLA-Flow` / `Trajectory-Consistent Flow Matching`。

---

## 06 · 工程与事故

### 6.1 PPI：栈跑通了，但主动挂起

```
Evaluating bimanual_push_box | Episode 0 | Score: 100.0
Evaluating bimanual_push_box | Episode 1 | Score: 100.0
```

CoppeliaSim + PyRep + RLBench2 + GroundingDINO + SAM + DINOv2 整条栈**搭对了**。
（2 个 episode 不足以复现论文成功率，只证明 pipeline 没搭错。）

**挂起的两个理由：**

1. **太慢**：episode 0 = 37 min，episode 1 = 52 min，约 **45 min/episode**。
   1 条件 × 20 episodes = 15 小时；6 条件 × 2 seed = **7.5 天**。而且它跟 ATM 抢 GPU0
2. **上游复现性存疑**：GitHub issue #7 有人用官方 checkpoint + 官方 pipeline 得到
   `bimanual_lift_ball` ~40%、`bimanual_handover_item_easy` ~0%，**作者未回复**

**重启前的检查单（顺序不能乱）：**

1. `ppi_agent.py:481,489` 已有 `ptc_time` / `dino_time` 变量但**从不打印**。
   先把三段耗时打出来：点云预处理 / DINOv2+GDINO+SAM / diffusion 采样。
   45 min ÷ 25 次 policy call ≈ **108 s/次**，先搞清这 108 秒的构成
2. **别凭猜降 `num_inference_steps`。** 瓶颈若是 Xvfb 软件渲染，砍 diffusion 是白砍
3. **若确认是 diffusion**：切 `DDIMScheduler` + `num_inference_steps: 20` ——
   这是**作者自己真机 config（`ppi/config/ppi_real.yaml:45`）的设置**，不是我们瞎改。
   改完先重跑 baseline 确认成功率没掉，然后所有条件统一

**什么时候需要它**：做「跨方法定量律」时（需要第二个方法点，确认
「approach 阶段的贡献 ∝ 被追踪点里机器人本体的占比」不是 ATM 独有的）。
在此之前不在关键路径上。

**复跑命令**：`cd /workspace/code/PPI && bash run_eval_box.sh <EPISODES> <SUFFIX>`

### 6.2 修补记录（PPI 两个真 bug）

- `RuntimeError: Can't find the demos for bimanual_push_box` → RLBench 期望的目录结构是
  `{dataset_root}/{task_name}/all_variations/episodes`，
  建了 `ppi_test/bimanual_push_box -> push_box_test` 符号链接
- `AttributeError: 'Tensor' object has no attribute '__array_interface__'` →
  仿真侧的 `semantic_feature_extractor.py` 是**训练期的 numpy 版本**，
  而 `real_world_deployment` 的 `_bf16` 变体已经改成收 tensor，仿真侧没同步。
  已在 `update()` 里加了带 range 检测的 tensor→numpy 转换（[-1,1] / [0,1] / [0,255] 三种）

### 6.3 事故：pass2 静默死亡，空转 14.5 小时

`chain_pass2.sh` 在 pass1 结束时**正确触发**，`G_freeze_all_seed0` 10:38:51 开跑，
**10:56 进程静默消失**，日志停在一个 0% 的 tqdm 进度条，**无 traceback、无 OOM 记录**。
结果目录存在但没有 summary CSV。
**8 个 pass2 条件一个都没完成，07-27 10:56 → 07-28 01:34 全空转。**

**注意：那个 batch 当时已经用 `setsid` 脱离了会话，照样死。死因未定位。**

**修法 —— 不赌「这次能活」，加看门狗** `/workspace/code/ATM/watchdog_pass2.sh`：

```bash
EXP=/workspace/code/ATM/results/policy/atm-policy_libero-spatial_demo10
TARGET=18          # 9 个条件 × 2 seed
for i in $(seq 1 40); do
  n=$(for d in $EXP/eval_results_*_seed*/; do [ -f "$d/summary_libero_spatial.csv" ] && echo x; done | wc -l)
  echo "[watchdog] 第 $i 轮启动  已完成 $n/$TARGET  $(date '+%F %T')"
  [ "$n" -ge "$TARGET" ] && { echo "[watchdog] 全部完成 $(date '+%F %T')"; break; }
  bash /workspace/code/ATM/run_formal_batch2.sh
  sleep 30
done
```

用**结果文件计数**判定完成，**不用 `pgrep`** —— `pgrep -f` 会匹配到调用它的 shell 自己，
本 session 已经因此吃过两次 `exit 144`。batch 自身有 skip 逻辑，重跑是幂等的。

### 6.4 踩坑记录

1. **写测量脚本必须先写自检。** 两次全 0 翻车（夹爪-物体接触被当成碰撞；transport 段
   跑过松手点）都能被「demo 自己的抓法必须判可行」这一条当场抓住。**第一版就该有**
2. **LIBERO 的 collision mesh 是 box 分解的**，任何 geom 级的白名单/黑名单都不可靠，
   必须在 body 级做
3. **长跑任务只 `setsid` 不够**，必须配一个基于产物计数的重启循环
4. **raw `MjData` 用 `xpos` / `xmat`**，不是 `body_xpos` / `body_xmat`（后者是 robosuite 的封装）

---

## 07 · 当前状态

| | |
|---|---|
| **pass1 + pass2** | ✅ **全部完成（03:30 收工）**，6 个条件全部 n=400，主结论显著（§2） |
| **G 组（freeze）** | ❌ **干预失效**（§3.2），三个数只作记录；**OOD 惩罚已量出 = 24 点** |
| **替代品 anchored** | ✅ 已实现 + 单元测试全过，**尚未开跑**（下一轮，约 2.5 小时） |
| **队列** | ⏹ 已停（看门狗 6726 → batch 6733 → python 127171 按 PID 依次 kill），未完成的 `eval_results_G_freeze_all_seed1/` 已删 |
| **GPU** | **GPU0 空闲**（2 MiB）。GPU1 被别的任务占着，硬约束，不动 |

**pass2 完整产出清单：**

| 条件 | seed0 | seed1 | n | 用途 |
|---|---|---|---|---|
| `A_full` | 0.710 | 0.710 | 400 | 基线 |
| `Bw_approach_wrist` | 0.645 | 0.585 | 400 | **头条对照的一半** |
| `Ba_approach_agent` | 0.720 | 0.705 | 400 | **头条对照的另一半 + 等价检验** |
| `B_approach` | 0.625 | 0.650 | 400 | 粗条件，不显著（§2.2.1） |
| `C_transport` | 0.430 | 0.410 | 400 | transport 段（平凡效应的确认） |
| `F_all` | 0.275 | 0.400 | 400 | 破坏上限参照 |
| `G_freeze_all` | 0.000 | — | 200 | **干预失效，仅作记录** |
| `Gb_freeze_approach` | 0.020 | — | 200 | **干预失效，仅作记录** |
| `Gc_freeze_transport` | 0.170 | — | 200 | **干预失效；但量出 OOD 惩罚 = 24 点** |

---

## 08 · 下一步（按优先级）

1. **跑 anchored 组** —— `anchored` / `anchored_approach` / `anchored_transport` × 2 seed，
   约 2.5 小时，GPU0 现在空闲。买到两样东西：
   **① 与 shuffle 之差 = 纯运动分量的贡献**（§3.1，这是原本 freeze 要干的活）；
   **② 验 §2.6 的 `F_all` seed 方差假设**（原登记检验随 freeze 判死而作废）
   代码和单元测试已就绪，需要新写一个 batch 脚本
3. **重核 PPI 完整 Table VII** —— 本次读到的 47.6 / 74.3 / 82.6 与之前记录的 keypose
   边际贡献（+6.0 / +6.5）**对不上**，而这是 PPT 幻灯片 P2 的依据。**写进论文前必须查清**
4. **（可选）** 量 ATM 单 epoch 的墙钟时间，估算「训练期去泄漏」实验的成本
   （track transformer 1001 epochs，BC 101 epochs）
5. **（可选）** V-JEPA 2-AC 的 Jacobian 零空间实验 —— 不用训练，公开 checkpoint 就够

---

## 09 · 方法论上的收获（这部分最持久）

1. **先定判据，再看数据。** grasp-feasibility pilot 的「headroom < 5 点 → 死」是
   **在看数据之前**定死的，所以 headroom ≈ 0–3 点出来时**没有任何回旋余地**。
   这是唯一能防住自己的机制
2. **自检是测量脚本的第一行代码，不是最后一行。** 两次全 0 翻车、以及最终判死这个 module，
   全部来自同一条自检
3. **判死一个 module 是产出，不是损失。** 死因（紧公差下 feasibility 不是 $(C,T)$ 的函数）
   **同时判死了「学一个打分器」这个 fallback** —— 因为输入里就没有那个信息。
   这个结论可以直接写进论文的 limitation / future work，**比一个跑不通的 module 值钱**
4. **一个 claim 被 prior art 削弱时，正确反应是精确化而不是放弃。**
   SPOT 把 flow→action 写成一次矩阵乘法 → 承认 transport 段平凡，
   把主张收缩到「approach 段未定义 + 现有方法靠泄漏机器人像素藏起来」——
   **收缩后的版本更难反驳**
5. **统计口径要在看结果之前定。** 配对 t 与朴素 z 在两个条件上**方向相反**，
   **事后选口径 = 事后选结论**
6. **干预之间的单调性是一条免费的自检。**
   `freeze ⊂ shuffle ⇒ SR(freeze) ≥ SR(shuffle)` 不需要任何领域知识就能事先写下来，
   而它**一条抓住了两个独立的 bug**（§3.2）。
   这是第 2 条教训在今天的**第二个实例**（第一个是 grasp-feasibility）
7. **当干预会改变阶段判据所依赖的信号时，「只在某阶段干预」是不成立的。**
   gripper 状态既是阶段判据、又受干预影响 → 反馈回路 →
   `freeze_approach` 退化成 `freeze_all`。**设计任何分阶段 ablation 都必须先检查这条**

---

## 附：文件索引

| 文件 | 内容 |
|---|---|
| `/workspace/research/d4rt/PROGRESS_REPORT_2026-07-28.md` | 本文档 |
| `/workspace/research/d4rt/PROGRESS_REPORT_2026-07-26.md` | 上一份（GD-4D → Flow≠Affordance 的转向） |
| `/workspace/research/d4rt/FLOW_AFFORDANCE_LOG.md` | 运行日志（每天的数字、踩坑、事故复盘） |
| `/workspace/research/d4rt/FLOW_TO_ACTION_SURVEY.md` | 20 篇 flow→action 论文的三张表 + 可直接引用的原句 |
| `/workspace/research/d4rt/PPT_GUIDE.md` | mentor-facing 幻灯片提纲 |
| `/workspace/research/d4rt/tools/grasp_pilot_probe.py` | LIBERO 只读探针（body/site/geom、抓取时刻、$C_0$） |
| `/workspace/research/d4rt/tools/grasp_pilot.py` | 候选抓法生成 + plotly 3D 可视化 |
| `/workspace/research/d4rt/tools/grasp_feasibility.py` | K×J 可行性矩阵（**已判死，保留作记录**） |
| `/workspace/code/ATM/atm/utils/flow_probe.py` | flow 探针（shuffle / ~~freeze~~ / **anchored** / 分阶段 / 分相机） |
| `/workspace/code/ATM/tests/test_flow_probe_anchored.py` | anchored 模式的 6 条不变量测试（全过） |
| `/workspace/code/ATM/run_formal_batch2.sh` | pass2 批处理（幂等，有 skip 逻辑） |
| `/workspace/code/ATM/watchdog_pass2.sh` | 看门狗重启循环（按产物计数） |
| `/workspace/code/PPI/run_eval_box.sh` | PPI eval（**挂起中**，重启前先看 §6.1 检查单） |
| conda env `atm5090` | py3.10 + torch 2.11+cu128（sm_120 可用） |
