# 逆向 spec：标签提取链（实际行为，非应然）

Status: ready-for-human
对象：`extract_c_labels.py` + `augment_links.py` + `chain_points.py` + `convert_light.py`
（+ 共享的 `phase.py`；stage-0 的 `convert_libero_raw.py` 仅作 build_env/命名工具被复用）
写法：只写代码**实际**在做什么。歧义/嫌疑单独列在 §5。
产出配套：`tests/test_char_labels.py`（characterization，钉住现有数据文件）

## 1. 数据流总图

```
raw LIBERO hdf5 (/workspace/datasets/libero/hdf5/<suite>/<task>_demo.hdf5)
  ├─ extract_c_labels  →  data/c_labels/<suite>/<task>.h5     （零漂移重放 + t=0 渲染）
  │    └─ augment_links  → 同文件追加 link_pos/link_quat/link_names（重放，无渲染）
  │         └─ chain_points → 同文件追加 chain_uv/chain_z（纯投影，无 sim）
  └─ convert_light     →  data/atm_libero_light/<task>/all/demo_X.hdf5（纯 IO，无重放）
```

## 2. c_labels 每字段契约（demo 组）

| 字段 | 形状/类型 | 单位/坐标系 | 来源 |
|---|---|---|---|
| attrs `t_g` | int | 帧号 | `argmax(latched_phase(actions))` = **首个 gripper>0 指令帧**（指令时刻，非物理接触时刻） |
| attrs `T` | int | | actions 长度 |
| `rgb0/depth0/seg0` | (512,512,{3/-/-}) u8/f32/i32 | upright | t=0 重放渲染；rgb/depth 手动 `[::-1]`，seg 由 `get_camera_segmentation` **内部已翻**（勿再翻——7-30 mask 事故根因） |
| `contact_rowcol` | (2,) f64 | (row,col) 浮点像素，upright 512 | `world_to_pix @ ee_pos[t_g]`，E(t_g) 投进 **t=0 静止相机** |
| `contact_depth0` | () f64 | m | `depth0[round(row),round(col)]`，即接触像素处 **t=0 场景表面深度**（非 EE 深度——lift-by-depth 的设计输入）；出视野时 **NaN** |
| `ee_pos/ee_quat` | (T,3)/(T,4) f32 | 世界系 / xyzw | 重放 `gripper0_grip_site` |
| `obj_pos/obj_quat` | (T,n,3)/(T,n,4) f32 | 世界系 / xyzw（wxyz 经 `[[1,2,3,0]]` 转换） | 自由关节 body |
| `gripper_q` | (T,2) f32 | 关节位置 | 原始 obs 直拷 |
| `phase` | (T,) u8 | 0/1 | 锁存：首闭合后恒 1（释放不回 0） |
| `link_pos/link_quat` | (T,L,3)/(T,L,4) f32 | 世界系 / xyzw | augment_links；L = 有 geom 的 robot0*/gripper0*/mount0* body |
| `chain_uv` | (T,32,2) f32 | (x=col/512, y=row/512) | 权重取 **lp[0]** arc-length，逐帧凸组合再投影 |
| `chain_z` | (T,32) f32 | 相机系 z (m) | `inv(extrinsic)` |

任务级：attrs `task_language/camera/res/robot_base_*/skipped_demos`；数据集
`K/world_to_pix/extrinsic/geom_id_to_body/robot_geom_ids/obj_names/link_names`。
QC attrs：`grasp_dist_m`（t_g 时最近自由体距离，兼定 `grasped_body`）、`lift_gap_m`
（pixel+depth0 反投 vs 真值 E(t_g) 的相机系距离）、`replay_ee_maxdiff_m`（重放保真度）。

## 3. convert_light 契约（ATM 格式）

`root/`：`actions` (T,7)、`task_emb_bert`、`phase` (T,) u8、`extra_states/{gripper_states,
joint_states, ee_ori, ee_pos, ee_states}`；每视角 `video` (1,T,3,128,128) u8
（原始 obs **竖直翻转** `[:, ::-1]` 后 CHW）、`tracks` 零假体 (1,T,32,2)、`vis` 全 1。
phase 与 c_labels 的 phase 同源同式（同一 `latched_phase(actions)`）→ 可作跨文件一致性锚。

## 4. 关键约定（跨模块必须同时成立）

1. **upright 行约定链**：`world_to_pix` 产出的 row 天然 upright（7-30 实证）＝ rgb0/depth0/seg0
   的存储方向 ＝ convert_light 翻转后的 video 方向 ＝ rollout wrapper 方向。任何一环再翻一次即回归
   7-30 事故。数值锚：chain_uv[0] 落在 robot mask 内比例（QA 全任务 ~100%）。
2. **四元数全库 xyzw**（mujoco wxyz 处处经 `[[1,2,3,0]]`）。
3. **demo 排序两套并存**：extract/augment/convert 用 `natsorted`（demo_0..demo_49 数值序）；
   fold 划分（stage1/stage2 dataset）用 **`sorted` 字典序** → fold0 val_id = demo_5..demo_9
   （非 demo_45..49）。两处 dataset 已互相对齐，但排序语义不同是长期地雷。
4. 跳过语义：整任务文件存在即跳（extract 无 demo 级续跑）；augment/chain 按 demo 键续跑；
   `phase.any()==False` 的 demo 整条剔除并记 `skipped_demos`。

## 5. 看不懂 / 歧义 / 嫌疑（bug 高发区）

1. **出视野无下游守卫**：`in_view=False` 时 `contact_rowcol` 仍按原值写入（可能越界），
   `contact_depth0=NaN`；Stage1Dataset 生成 heatmap 时未见越界/NaN 检查。当前数据是否存在
   此类 demo 由 char test 全量清点（若 0 条则现状安全，但代码路径裸奔）。
2. **`grasped_body` = t_g 时刻最近自由体**：两物体贴近时可能标错；当前仅作 provenance 字段，
   无训练消费——但若未来拿它选 object-flow 目标（L5），此启发式会成为静默错误源。
3. **`t_g` 是指令帧不是接触帧**：E(t_g) 的手爪还未闭合、位姿略早于真实抓取姿态；C 标签
   （contact_rowcol/ee_quat[t_g]/gripper_q[t_g]）全部继承这个"提前量"。系统性、全库一致，
   模型可学，但与"接触时刻"的字面语义有偏差——文档层面从未明说。
4. **chain_gt 越过 t_g**（Stage2Dataset）：obs 窗口约束 `end ≤ t_g+2`，但 flow GT 窗口
   `uv[off+k : off+k+16]` 最远伸到 **t_g+17**——"approach-only" 的 flow 监督实际含抓取后
   最多 ~15 帧 transport 运动。设计上说得通（L3 预告闭合后去向），但与命名不符，未见文档。
5. **`natsorted(os.listdir)` 无 `.hdf5` 过滤**（convert_libero_raw main / convert_light 有过滤，
   extract 有过滤；stage0 converter `task_files = natsorted(os.listdir(suite_dir))` 无过滤）——
   suite 目录混入杂文件即崩。stage-0 已冻结，仅记录。
6. **`extract_c_labels` 的 `keys` 作用域**：最终 print 用 `len(keys)`，`keys` 在 with 块内定义，
   demos_limit 与 skip-existing 组合时统计口径含糊（纯日志问题）。

## Comments
