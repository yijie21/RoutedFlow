# Pipeline 架构图：制作过程记录 & 原地修改指南

> 对象：`doc/pipeline_fig_v01.svg`（论文图 1 草稿）。
> 已发布 artifact：https://claude.ai/code/artifact/12bf11ca-7294-40c6-acb1-69a18af26e5d
> 首次制作 2026-07-30；设计内容对应 `PIPELINE_IMPL_PLAN.md` §2.7（链路 v0.1）/ §2.8（L1–L5）/ §0.3（D7 课程）。

## 0. 一句话流程

```
python3 run_fig.py assets   # ① 从 data/c_labels 生成小图素材 → doc/fig_assets/
python3 run_fig.py build    # ② 排版 SVG + HTML → doc/pipeline_fig_v01.svg / doc/pipeline_fig.html
# ③ QA：本地 http 服务 + playwright 截图（见 §4）
# ④ 发布：用 Artifact 工具发 doc/pipeline_fig.html（保 URL 方法见 §5）
```

全流程确定性：同样的标签数据 + 同样的代码 → 字节一致的 SVG（已验证）。

## 1. 三层结构，改哪层？

| 想改什么 | 改哪 | 然后 |
|---|---|---|
| 模块框文案、箭头、颜色、布局、loss 徽标 | `src/routedflow/fig_build.py` | `run_fig.py build` |
| 示意小图（换 demo/任务、换 flow 画法、加新模态图） | `src/routedflow/fig_assets.py` | `run_fig.py all` |
| 标签数据本身变了（重新提取过） | 不改代码 | `run_fig.py all` |

素材层和排版层**故意解耦**：排版只认 `doc/fig_assets/` 里的文件名。

## 2. `fig_assets.py` 要点

- 数据源：`data/c_labels/libero_spatial/<TASK>.h5` 的 `demo_0`（TASK 常量在文件顶部，换任务/换 demo 改这里）。
- 产出（512² 底 + `_crop` 裁剪版）：`rgb` / `heatmap`（接触点高斯 σ=18px）/ `depth`（turbo，clip 0.5–1.6m）/
  `mask_robot` `mask_object`（灰底着色）/ `flow_robot`（EE 路径 0→t_g，绿三流线 + 接触圈）/
  `flow_object`（被抓物体路径 t_g→末帧，橙）/ `c_label`（接触点黄盘 + R(t_g) 三轴）。
- 流线画法：`smooth()` 滑动平均 + `arrow_path()` 手绘三角箭头；偏移流线用 `(offset, trim)` 对控制，
  trim 是为了偏移线别越过目标物（v1 QA 教训）。
- 裁剪框在文件末尾的 crops 元组里——换任务后**必须重调**（目标物位置不同）。

## 3. `fig_build.py` 排版系统

**画布**：viewBox `1450 × 940`，白底（论文图故意单主题；页面外壳才做深浅色）。

**Helper API**（全部往全局列表 `E` 里 append SVG 元素，出现顺序 = 叠放顺序）：

```python
T(x, y, text, size, fill, weight, anchor)   # 文本，y 是基线
box(x, y, w, h, stroke, fill, dash, rx, sw) # 圆角框
thumb(x, y, w, h, "asset.png", caption)     # 素材图（JPEG data-URI 内嵌）+ 可选下方 caption
arrow([(x1,y1),(x2,y2),...], stroke, sw, dash, marker)  # 折线箭头；marker: arr(黑)/arrR(红)
```

**颜色语义**（改配色只动顶部常量）：`GREEN/GREEND`=approach、`ORANGE/ORANGED`=transport、
`GRAY`=零参数几何、`RED`=监督信号（一律虚线 `dash="5 4"` + `marker="arrR"`）、lane 底色 `LANE_G/LANE_O`。

**坐标地图**（y 向下；元素在代码里按此顺序出现，注释即锚点）：

| 区域 | x 范围 | y 范围 |
|---|---|---|
| 图例 | 700–1416 | 40–60 |
| 输入列（RGB→指令→LLM 分解器→两子句 chip） | 30–202 | 90–502 |
| approach lane 背景 | 228–1426 | 78–480 |
| L1 前端 | 252–464 | 190–320 |
| heatmap 缩略图 + 两行 caption | 506–638 | 110–252 |
| latent z pill | 506–606 | 292–330 |
| C 头辅助虚线容器（lift 芯片 + C head + 标签图 + L_C） | 676–1012 | 92–270 |
| 深度/robot-mask 小图（L3 输入） | 506–632 | 352–430 |
| L3 robot-flow | 700–934 | 310–430 |
| robot flow 缩略图（caption 在上方！） | 972–1122 | 300–424 |
| L4 action expert + aₜ 柱状 glyph | 1156–1420 | 318–422 |
| transport lane 背景 | 228–1426 | 500–822 |
| object mask 缩略图 | 252–370 | 552–668 |
| L5 object-flow | 430–654 | 560–688 |
| object flow 缩略图（caption 在上方） | 692–842 | 542–670 |
| 零参数几何链 | 902–1202 | 552–688 |
| phase 路由开关（跨两 lane） | 1258–1426 | 456–518 |
| 课程缎带 ①②③ | 30–1426 | 848–910 |

**空余空间**（加新元素优先往这放）：approach lane 内 x 240–500 / y 340–470；transport lane 内 x 240–420 / y 700–810；两 lane 之间 y 482–498 只有 16px，别放东西。

## 4. QA 回路（必做——首版就是这么抓出 7 处重叠的）

```
cd doc && python3 -m http.server 8763 --bind 127.0.0.1 &     # file:// 被 playwright 挡，必须走 http
# playwright: browser_navigate http://127.0.0.1:8763/pipeline_fig.html
# playwright: browser_take_screenshot fullPage → 人眼过一遍
# 完事记下 PID kill 掉（不要 pkill -f）
```

已踩过的坑（改版时逐条自查）：
1. lane 标题 (y≈102) 和 lane 内顶部缩略图打架——缩略图 y ≥ 110。
2. z pill 右侧别放东西——z→L3 主动脉和 z→C head 的箭头都从那出发。
3. 缩略图 caption 默认在图**下方**；若下方还要放 loss 徽标+红箭头，把 caption 挪到图上方（robot/object flow 两图就是这么处理的）。
4. 长距离箭头（如 transport 子句→L5）走 lane 底部空带，别横穿缩略图。
5. 箭头起点必须在源元素**边缘外**，别从图/框内部出发。
6. HTML 必须带 `<meta charset="utf-8">`——裸 http.server 不发 charset，中文会 mojibake。
7. SVG 里 emoji（🔥❄⚙）浏览器渲染没问题，但若将来要转 PDF/PNG 用无头浏览器截图，别用系统缺字体的环境。

## 5. 发布 & 存档

- **同 URL 更新**：本会话内重发 `doc/pipeline_fig.html` 即可；**换了会话**必须给 Artifact 工具传
  `url=https://claude.ai/code/artifact/12bf11ca-7294-40c6-acb1-69a18af26e5d`，否则会铸新 URL。
- 每次定稿后：SVG 已在 `doc/`（版本号进文件名，大改开 `_v02`），预览 PNG 用 QA 截图存
  `doc/pipeline_fig_v01_preview.png`，README changelog 加一条。
- 论文用法：SVG 直接进 Inkscape/Illustrator 微调导出 PDF；图注文案在 `fig_build.py` 底部的
  caption 段落里，与图同步维护。

## 6. 内容与计划文档的同步义务

图上固化了这些设计决定——**计划文档改了，图必须跟着改**：
L3 不吃 text（§2.7 修正 #3）；transport 无 action loss（§0.3 边界）；C 头辅助监督 + 平移 lift 导出（§2.2/§2.7）；
D6 未解标注（§1.7/§6）；D7 课程与 λ（§0.3）；D8/D9 以括号标注在模块框里。
反向也成立：改图前先想清楚是不是设计真的变了——图是设计的镜子，不是画板。

## 变更记录

- 2026-07-30 v0.1 首版（本文档 + `fig_assets.py` + `fig_build.py` + `run_fig.py` 入库；QA 两轮修 7 处重叠）。
- 2026-07-30 v0.2：随阶段一实现更新 5 处——L1 框（DINOv2+BERT+[C] token，7.5M）、prior 通道标注实测 58% 正确率、
  robot flow 标注 D8a 定稿（2D track+查询深度）、L4 加 D10 行（③ 吃 L3 预测+加噪）、C head 标 bins 规格；
  图注段落同步。同 URL 重发（用 `url=` 参数，因发布路径从 scratchpad 换到了 doc/）。
- 2026-07-30 v0.3（用户反馈：图内文字看不懂/半截话）：**图面全面通俗化**——内部代号（D3/D8a/D10/①②③徽标/
  「特权重放」「防旁路」「FK 投影 GT」等）全部清出图面；模块框只留 self-explained 短句；新增图下
  「术语注解」块（注①仿真真值监督 / 注②单一语言通道 / 注③AFUN 提示 mask / 注④几何解算 / 注⑤训练三步），
  图内用「注N」标记挂接。**新规矩：图面语言 = 给外人看的通俗话；内部决策代号只活在计划文档里。**
- 2026-07-31 命名更新（用户指定）：「L1·前端」→「C-VLM」、「C 头」→「C head」、「3D 平移」→「3D translation」
  （图面/缎带/图注/注⑤全同步）；lane 标题相应缩短避免贴框。
