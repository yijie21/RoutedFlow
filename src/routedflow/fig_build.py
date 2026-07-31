"""Build the two-branch training pipeline figure: standalone SVG + artifact HTML.

Reads doc/fig_assets/*.png, writes doc/pipeline_fig_v01.svg (self-contained
vector source for the paper) and doc/pipeline_fig.html (artifact page).
Layout is fixed-coordinate; see doc/PIPELINE_FIG_HOWTO.md for the coordinate
map and how to edit safely. Run via `run_fig.py build`.
"""
import base64
import io
import os

from PIL import Image

REPO = __file__.rsplit("/src/", 1)[0]
A = REPO + "/doc/fig_assets"
OUT_DIR = REPO + "/doc"

GREEN, GREEND = "#2e8b57", "#1e5c3a"
ORANGE, ORANGED = "#d97c2b", "#a35a1c"
GRAY, INK, SUB = "#8a8f94", "#22272b", "#5b6167"
RED = "#c0392b"
LANE_G, LANE_O = "#f0f7f2", "#fbf3ea"


def uri(name, max_w=300):
    img = Image.open(os.path.join(A, name)).convert("RGB")
    if img.width > max_w:
        img = img.resize((max_w, int(img.height * max_w / img.width)), Image.LANCZOS)
    b = io.BytesIO()
    img.save(b, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


E = []  # svg elements


def T(x, y, s, size=13, fill=INK, w="normal", anchor="start", style=""):
    E.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{w}" '
             f'text-anchor="{anchor}" {style}>{s}</text>')


def box(x, y, w, h, stroke, fill="#ffffff", dash="", rx=10, sw=2):
    E.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
             f'stroke="{stroke}" stroke-width="{sw}" {f"stroke-dasharray={dash!r}" if dash else ""}/>')


def thumb(x, y, w, h, name, caption=None, cap_fill=SUB):
    E.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="{uri(name)}" '
             f'preserveAspectRatio="xMidYMid slice" clip-path="inset(0 round 8)"/>')
    E.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="none" stroke="#c9cdd1" stroke-width="1.5"/>')
    if caption:
        T(x + w / 2, y + h + 16, caption, 11.5, cap_fill, anchor="middle")


def arrow(pts, stroke=INK, sw=2, dash="", marker="arr"):
    d = "M " + " L ".join(f"{px},{py}" for px, py in pts)
    E.append(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}" '
             f'{f"stroke-dasharray={dash!r}" if dash else ""} marker-end="url(#{marker})"/>')


W_, H_ = 1450, 940

# ---------- lanes ----------
box(228, 78, 1198, 402, "#cfe0d5", LANE_G, rx=14, sw=1.5)
T(246, 102, "APPROACH 分支（接近并抓取）— 学习：C head ＋ flow ＋ 动作", 13, GREEND, "700")
box(228, 500, 1198, 322, "#e8d5bd", LANE_O, rx=14, sw=1.5)
T(246, 524, "TRANSPORT 分支（抓稳后搬运）— 只学物体 flow；动作由几何解算，不学", 13, ORANGED, "700")

# ---------- legend ----------
T(700, 52, "🔥 需要训练", 12.5); T(796, 52, "❄ 冻结（不训练）", 12.5); T(920, 52, "⚙ 纯几何计算", 12.5)
E.append(f'<line x1="1040" y1="47" x2="1076" y2="47" stroke="{INK}" stroke-width="2" marker-end="url(#arr)"/>')
T(1082, 52, "数据流", 12.5)
E.append(f'<line x1="1150" y1="47" x2="1186" y2="47" stroke="{RED}" stroke-width="2" stroke-dasharray="5 4" marker-end="url(#arrR)"/>')
T(1192, 52, "监督信号", 12.5)

# ---------- input column ----------
thumb(30, 90, 150, 150, "rgb.png", "RGB  I₀（agentview）")
box(30, 272, 172, 66, "#b9bec3", "#fafafa", rx=8)
T(40, 290, "指令 L：“pick up the black bowl", 10.5, SUB); T(40, 304, "between the plate and the ramekin", 10.5, SUB)
T(40, 318, "and place it on the plate”", 10.5, SUB)
box(30, 356, 172, 46, "#9aa0a6", "#f1f2f3", rx=8)
T(116, 375, "LLM 指令分解 ❄", 13, INK, "600", anchor="middle"); T(116, 392, "拆成「抓取 ＋ 搬运」子句", 10, SUB, anchor="middle")
arrow([(116, 338), (116, 356)])
box(30, 424, 172, 34, GREEN, "#ffffff", rx=17)
T(116, 446, "抓取子句（approach）", 12, GREEND, "600", anchor="middle")
box(30, 468, 172, 34, ORANGE, "#ffffff", rx=17)
T(116, 490, "搬运子句（transport）", 12, ORANGED, "600", anchor="middle")
arrow([(90, 402), (90, 424)], GREEN); arrow([(146, 402), (146, 468)], ORANGE)

# ---------- L1 ----------
box(252, 190, 212, 130, GREEN, sw=2.5)
T(358, 216, "C-VLM 🔥（7.5M）", 14.5, INK, "700", anchor="middle")
T(358, 240, "冻结 DINOv2 视觉 ＋ BERT 文本", 11.5, SUB, anchor="middle")
T(358, 258, "3 层融合 ＋ 学习型 [C] token", 11.5, SUB, anchor="middle")
T(358, 276, "AFUN 提示 mask ❄（见注③）", 11.5, SUB, anchor="middle")
arrow([(182, 165), (216, 165), (216, 225), (252, 225)])
arrow([(202, 441), (226, 441), (226, 290), (252, 290)], GREEN)

# L1 outputs
thumb(506, 110, 132, 110, "heatmap_crop.png")
T(572, 236, "接触点 heatmap", 11.5, SUB, anchor="middle")
T(572, 252, "P(p|I,L)", 11, SUB, anchor="middle")
arrow([(464, 220), (484, 220), (484, 165), (506, 165)])
box(506, 292, 100, 38, GREEND, "#e7f2ec", rx=19, sw=2)
T(556, 316, "latent  z", 14, GREEND, "700", anchor="middle")
arrow([(464, 285), (485, 285), (485, 311), (506, 311)])

# ---------- C auxiliary block ----------
box(676, 92, 336, 178, GREEND, "#ffffff", dash="7 5", rx=12, sw=1.8)
T(692, 114, "C head（辅助监督）", 13, GREEND, "700")
box(692, 128, 158, 40, GRAY, "#f4f5f6", rx=8)
T(771, 145, "接触点＋深度 ⚙", 12, INK, "600", anchor="middle"); T(771, 161, "→ 3D translation", 11, SUB, anchor="middle")
box(692, 180, 158, 44, GREEN, rx=8, sw=2)
T(771, 198, "C head 🔥", 12.5, INK, "700", anchor="middle"); T(771, 215, "→ 抓取朝向 ＋ 开度", 11, SUB, anchor="middle")
thumb(866, 122, 128, 106, "c_label_crop.png")
T(930, 246, "监督：夹爪闭合时刻的", 11, RED, anchor="middle")
T(930, 260, "真实位姿（仿真真值，注①）", 11, RED, anchor="middle")
arrow([(866, 202), (850, 202)], RED, dash="5 4", marker="arrR")
arrow([(638, 163), (676, 152)])                              # heatmap -> lift
arrow([(606, 302), (660, 302), (660, 210), (676, 210)], GREEND)  # z -> C head

# ---------- L3 ----------
box(700, 310, 234, 120, GREEN, sw=2.5)
T(817, 336, "L3 · 机器人 flow 模块 🔥", 14.5, INK, "700", anchor="middle")
T(817, 358, "ATM track transformer 改造", 11.5, SUB, anchor="middle")
T(817, 374, "查询点取自 robot mask（附深度）", 11, SUB, anchor="middle")
T(817, 396, "任务信息只从 z 进入，不接语言（注②）", 11.5, GREEND, "600", anchor="middle")
arrow([(606, 311), (652, 311), (652, 345), (700, 345)], GREEND, sw=3)   # z artery
thumb(506, 352, 60, 52, "depth.png"); thumb(572, 352, 60, 52, "mask_robot.png")
T(569, 430, "深度 · robot mask · 当前帧", 10, SUB, anchor="middle")
arrow([(632, 382), (666, 382), (666, 405), (700, 405)], "#7a8087", 1.6)

# L3 output
T(1047, 300, "机器人 flow（未来轨迹）", 11.5, SUB, anchor="middle")
thumb(972, 306, 150, 118, "flow_robot_crop.png")
arrow([(934, 366), (972, 366)])
T(1047, 452, "监督：仿真真值轨迹（注①）", 11.5, RED, anchor="middle")
arrow([(1047, 440), (1047, 426)], RED, dash="5 4", marker="arrR")

# ---------- L4 ----------
box(1156, 318, 178, 104, GREEN, sw=2.5)
T(1245, 344, "L4 · action expert 🔥", 14, INK, "700", anchor="middle")
T(1245, 366, "输入：预测的 flow ＋ 机器人状态", 11, SUB, anchor="middle")
T(1245, 382, "输出：机械臂动作", 11, SUB, anchor="middle")
T(1245, 402, "不接收语言（注②）", 11, GREEND, "600", anchor="middle")
arrow([(1122, 365), (1156, 365)])
T(1245, 452, "监督：示教动作（主 loss）", 11.5, RED, anchor="middle")
arrow([(1245, 440), (1245, 426)], RED, dash="5 4", marker="arrR")
# action glyph
for i, h in enumerate((26, 40, 32, 46, 36)):
    E.append(f'<rect x="{1360 + i * 11}" y="{392 - h}" width="7" height="{h}" rx="2" fill="{GREEND}"/>')
T(1387, 412, "aₜ approach", 11, GREEND, "600", anchor="middle")
arrow([(1334, 360), (1352, 360)])

# ---------- transport lane ----------
thumb(252, 552, 118, 100, "mask_object.png", "object mask（SAM3 ❄ / 真值）")
arrow([(202, 485), (218, 485), (218, 736), (542, 736), (542, 688)], ORANGE)  # transport prompt
box(430, 560, 224, 128, ORANGE, sw=2.5)
T(542, 588, "L5 · 物体 flow 模块 🔥", 14.5, INK, "700", anchor="middle")
T(542, 610, "与机器人 flow 模块共享主干", 11.5, SUB, anchor="middle")
T(542, 626, "查询点取自 object mask", 11.5, SUB, anchor="middle")
T(542, 648, "条件：搬运子句 ＋ RGB", 11.5, ORANGED, "600", anchor="middle")
arrow([(370, 600), (430, 600)], "#7a8087", 1.6)
# L5 output
T(767, 542, "物体 flow（未来轨迹）", 11.5, SUB, anchor="middle")
thumb(692, 548, 150, 122, "flow_object_crop.png")
arrow([(654, 620), (692, 620)])
T(770, 700, "监督：仿真真值物体轨迹（注①）", 11.5, RED, anchor="middle")
arrow([(770, 688), (770, 672)], RED, dash="5 4", marker="arrR")

# geometry chain
box(902, 552, 300, 136, GRAY, "#f4f5f6", rx=12, sw=2)
T(1052, 580, "几何解算 ⚙（无学习参数，注④）", 13, INK, "700", anchor="middle")
T(1052, 608, "物体 flow → 刚体变换（SVD 拟合）", 12, SUB, anchor="middle")
T(1052, 630, "→ 末端目标位姿", 12, SUB, anchor="middle")
T(1052, 652, "→ 解 IK 得到动作", 12, SUB, anchor="middle")
arrow([(842, 608), (902, 608)])
for i, h in enumerate((30, 42, 34, 44, 30)):
    E.append(f'<rect x="{1240 + i * 11}" y="{628 - h}" width="7" height="{h}" rx="2" fill="{ORANGED}"/>')
T(1267, 648, "aₜ transport", 11, ORANGED, "600", anchor="middle")
arrow([(1202, 608), (1232, 608)])

# ---------- phase switch ----------
box(1258, 456, 168, 62, INK, "#ffffff", rx=10, sw=2)
T(1342, 478, "执行时相位切换", 13, INK, "700", anchor="middle")
T(1342, 496, "夹爪闭合 → 抓取切搬运", 11, SUB, anchor="middle")
arrow([(1387, 420), (1387, 438), (1370, 438), (1370, 456)], GREEND)
arrow([(1267, 560), (1267, 540), (1310, 540), (1310, 518)], ORANGED)

# ---------- curriculum ribbon ----------
box(30, 848, 1396, 62, "#d4d8db", "#fafbfb", rx=12, sw=1.5)
T(48, 872, "训练分三步（注⑤）", 12.5, INK, "700")
box(210, 860, 340, 38, GREEND, "#eef6f1", rx=19)
T(380, 884, "第一步：只训 C-VLM ＋ C head（接触监督）", 12, INK, anchor="middle")
arrow([(550, 879), (588, 879)])
box(588, 860, 340, 38, "#7d8a5e", "#f2f5ec", rx=19)
T(758, 884, "第二步：加入两个 flow 模块", 12, INK, anchor="middle")
arrow([(928, 879), (966, 879)])
box(966, 860, 436, 38, "#8a6d3b", "#f7f2e8", rx=19)
T(1184, 884, "第三步：全网端到端微调（动作 loss 为主）", 12, INK, anchor="middle")

svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W_} {H_}" font-family="ui-sans-serif,-apple-system,'Segoe UI','PingFang SC','Noto Sans CJK SC',sans-serif">
<defs>
<marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L8,4.5 L0,9 z" fill="{INK}"/></marker>
<marker id="arrR" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L8,4.5 L0,9 z" fill="{RED}"/></marker>
</defs>
<rect width="{W_}" height="{H_}" fill="#ffffff"/>
{''.join(E)}
</svg>"""

open(os.path.join(OUT_DIR, "pipeline_fig_v01.svg"), "w").write(svg)

html = f"""<meta charset="utf-8">
<title>训练 pipeline 图 · Phase-Gated Flow Routing</title>
<style>
:root {{ --bg:#f7f6f3; --ink:#1c2320; --sub:#5c645f; --card:#fff; --line:#ddd9d0; --acc:#2e7d4f; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#14181a; --ink:#e6e4de; --sub:#9aa39d; --card:#1d2326; --line:#2e3538; --acc:#4caf7d; }} }}
:root[data-theme="dark"] {{ --bg:#14181a; --ink:#e6e4de; --sub:#9aa39d; --card:#1d2326; --line:#2e3538; --acc:#4caf7d; }}
:root[data-theme="light"] {{ --bg:#f7f6f3; --ink:#1c2320; --sub:#5c645f; --card:#fff; --line:#ddd9d0; --acc:#2e7d4f; }}
body {{ background:var(--bg); color:var(--ink); margin:0; padding:2rem 1rem 4rem;
  font:15px/1.7 ui-sans-serif,-apple-system,"Segoe UI","PingFang SC","Noto Sans CJK SC",sans-serif; }}
main {{ max-width:1280px; margin:0 auto; }}
.eyebrow {{ text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; color:var(--acc); font-weight:700; }}
h1 {{ font-size:1.5rem; margin:.2rem 0 .4rem; }}
.figwrap {{ background:#fff; border:1px solid var(--line); border-radius:12px; overflow-x:auto; margin-top:1rem; }}
.figwrap svg {{ display:block; min-width:1100px; }}
.caption {{ color:var(--sub); font-size:.88rem; margin-top:.9rem; max-width:72rem; }}
.caption b {{ color:var(--ink); }}
.notes {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:.4rem 1.4rem .9rem; margin-top:1.1rem; max-width:72rem; }}
.notes h2 {{ font-size:1rem; margin:.8rem 0 .2rem; }}
.notes ol {{ margin:.4rem 0 0; padding-left:1.3rem; }}
.notes li {{ font-size:.88rem; color:var(--sub); margin:.45rem 0; }}
.notes li b {{ color:var(--ink); }}
</style>
<main>
<div class="eyebrow">RoutedFlow · 图 1 草稿 · 2026-07-30</div>
<h1>Phase-Gated Flow Routing — 双分支训练 pipeline（v0.3）</h1>
<div class="figwrap">{svg}</div>
<p class="caption"><b>图 1（v0.3）：</b>指令被拆成「抓取＋搬运」两个子句，任务也在夹爪闭合的瞬间切成两段。
<b>抓取段</b>（绿）学习三样东西：在哪抓（C head，辅助监督）、机器人怎么移过去（机器人 flow）、以及最终动作；
<b>搬运段</b>（橙）只学一样：物体应该怎么动（物体 flow）——因为抓稳之后物体跟着手走，动作可以用纯几何从物体 flow 反推出来，不需要学。
两段在执行时由夹爪闭合信号切换。图中所有输入/输出小图均为本项目实际提取的数据（libero_spatial，demo_0）。</p>

<div class="notes">
<h2>术语注解</h2>
<ol>
<li><b>仿真真值监督（图中「注①」）</b>：所有训练标签都来自仿真器内部可直接读取的真实状态——把示教在仿真里逐帧重放，
读出夹爪/物体的真实位姿即可得到标签。全程无人工标注。例如「抓取位姿」标签 = 重放到夹爪闭合那一帧、读出的夹爪真实位姿。</li>
<li><b>为什么 flow 模块和 action expert 都不接收语言（注②）</b>：任务语义被刻意约束成只能经 latent z 这一条通道进入下游。
如果下游也能直接看语言，它就可以绕开 z 自行理解任务——那样 z 是否真的承载了任务信息将无法验证。
只留一条通道，之后的探针实验（从 z 读出抓取信息）和干预实验（换/坏 z 看成功率掉不掉）才能证明 z 的因果作用。</li>
<li><b>AFUN 提示 mask（注③）</b>：一个外部基础模型给出的「大概在这里」候选区域，仅作输入提示，实测命中率 58%。
训练标签始终是仿真真值，所以提示错误的样本反而教会模型「提示与图像/语言矛盾时忽略提示」；
训练时还随机把该通道置零，保证没有提示也能工作。它只托底、不封顶。</li>
<li><b>几何解算为什么不用学（注④）</b>：抓稳后物体与夹爪相对位置不变。给定预测的物体 flow，
用 SVD 拟合出物体的刚体运动，夹爪跟着做同样的运动即可——目标位姿由此直接算出，再解 IK 得到关节动作。这一段没有任何可学参数。</li>
<li><b>训练三步（注⑤）</b>：第一步只训 C-VLM 和 C head（让 z 先学会承载抓取信息）；第二步加入两个 flow 模块；
第三步全网端到端微调，动作 loss 为主、其余降为辅助。第三步里 action expert 用的是 flow 模块的<b>预测</b>（外加噪声增强）
而不是真值 flow——避免训练时见到的都是完美 flow、部署时却面对有噪声的预测。</li>
</ol>
</div>
</main>
"""
open(os.path.join(OUT_DIR, "pipeline_fig.html"), "w").write(html)
print("svg+html written,", f"{os.path.getsize(os.path.join(OUT_DIR,'pipeline_fig.html'))/1e6:.2f}MB")
