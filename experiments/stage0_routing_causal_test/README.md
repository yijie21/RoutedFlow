# 实验：Stage-0 routing causal test（守门实验）

> **这是当前正在进行的实验**（`experiments/CURRENT` 指向这里）。
> 计划出处：`/workspace/research/d4rt/PIPELINE_IMPL_PLAN.md` §1。

## 一句话

track transformer 冻结，重训 3 份 BC policy，唯一区别是哪些预测 track 携带信息
（object-only / robot-only / phase-switched），验证「按接触阶段路由 flow 语义」有没有因果收益。

## 预注册预测与 kill criterion

- P1: `phase_switched` ≥ `object_only`；P2: `phase_switched` ≥ `robot_only`；P3: `phase_switched` ≈ A_full 0.710
- **Kill**: `phase_switched` − max(object_only, robot_only) < 3 pts → 路由无因果收益，pipeline 方向重审
- 免费读数：`robot_only` vs `object_only` 之差 = 测量阶段 Bw/Ba 结论的训练侧复证

## 怎么跑（都从 repo 根目录）

```bash
python3 run_stage0.py test          # 单测（CPU，秒级）
python3 run_stage0.py prep          # 数据转换 + mask 渲染（一次性，~2-5h）
#   先 python3 run_stage0.py prep --tasks-limit 1 试跑一个任务，看 sanity_overlays/
python3 run_stage0.py smoke         # mode=none 跑 2 epochs：测单 epoch 墙钟 + 全链路检查
python3 run_stage0.py train --mode object_only     # 变体 (i)，101 epochs
python3 run_stage0.py train --mode robot_only      # 变体 (ii)
python3 run_stage0.py train --mode phase_switched  # 变体 (iii)
python3 run_stage0.py status        # 查看当前进度
```

## 状态板（每步完成后更新）

| 步骤 | 状态 | 结果/备注 |
|---|---|---|
| 代码 + 单测 | ✅ 2026-07-29 | 模块在 `src/routedflow/`，单测 `tests/test_stage0_units.py` |
| prep（数据转换+mask） | ✅ 2026-07-29 | 10 任务×15 demos，2.4G；overlay 过目通过；robot 占比 0.088/0.217 vs 测量阶段 9%/21% 交叉吻合；replay diff ≈3/6 |
| smoke（epoch 墙钟） | ✅ 2026-07-29 | **121.8 s/epoch** → 101 epochs ≈ 3.4 h/变体，3 变体串行 ≈ 10.3 h。OOM 教训：batch 64 + grad_accum 2（=等效 128） |
| pre-flight（3 mode 各过 1 真实 batch） | ✅ 2026-07-29 | object_only 0.2599 / robot_only 0.2470 / phase_switched 0.2498，forward+backward 无误 |
| 训练 (i) object_only | ✅ 2026-07-29 | 101 epochs，train 0.0036 / val 0.0360 |
| 训练 (ii) robot_only | ✅ 2026-07-29 | 101 epochs，train 0.0037 / val 0.0383 |
| 训练 (iii) phase_switched | ✅ 2026-07-29 | 101 epochs，train 0.0034 / **val 0.0327（三者最低）** |
| 正式 eval 链（n=400 × 3） | ✅ 2026-07-30 | object_only 0.630 / robot_only 0.6525 / phase_switched 0.1975 |
| 判定 P1/P2/P3 | ⏸ **暂缓判定**（2026-07-30 用户指示） | 单模块（BC-only 因果测试）不足以对整个 pipeline 下结论；phase_switched 的 latch 统计（中位 160 vs demo 45，27% 从不闭合，val loss 却最低）指向 eval 时 phase 信号自指的 train/eval 失配，候选诊断=强制第 44 步锁存重评。留待 pipeline 全貌后再回看 |
| eval 接线 + 实弹 smoke | ✅ 2026-07-29 | object_only ckpt 任务1 SR=0.400（n=10）；latch_rate 1.0，成功 episode latch 于 44-63 步、失败的 100-433 步 |
| n=400 eval × 3 | ☐ | 等训练链结束后跑：`run_stage0.py eval --mode <m> --nroll 40`（别与训练并发抢 GPU） |
| 判定 P1/P2/P3 | ☐ | |

## 目录

- `configs/stage0_vilt.yaml` — 训练配置（照抄 ATM `libero_vilt.yaml`，加 `track_gate_cfg`）
- `runs/<experiment>_seed<seed>/` — 训练输出（ckpt、metrics.jsonl、config.yaml）
- `sanity_overlays/` — prep 生成的 grid-点标签可视化（必须人工过目）
- 可复用代码在 `src/routedflow/`：`track_gate.py`（干预本体）、`gated_policy.py`、`gated_dataset.py`、`convert_libero_raw.py`、`engine_train_stage0.py`

## 已知未完成 / 注意事项

1. **eval 侧在线分割接线还没写**（policy 已留好 `act_gated()` 入口）——训练跑起来之后写，不阻塞训练。
2. mask 抹平的 pattern 以 32 点粒度泄漏机器人轮廓位置（不泄漏运动）——记为 limitation；margin 可疑时用 shuffle-train robustness check。
3. `mode=none` 与 A_full 的对照：本实验的 `none` 用的是我们转换的数据（无 CoTracker GT track，但训练本来就不用它），理论上应复现 A_full 0.710±noise——smoke 之后值得跑一个完整 `none` 作为自家 baseline（决定：先跑 3 变体，`none` 101 epochs 视 GPU 预算插队）。
4. phase 是锁存的：release 之后仍算 transport（LIBERO demo 在 release 后很快结束，尾巴很短）。

## ❌ 2026-07-30 追加：mask 定向 bug —— 本实验结果作废，须重跑

阶段一标签可视化暴露：`get_camera_segmentation` 本身返回正立图，convert 与 eval 各多翻一次
→ 两视角 robot_seg/grid_labels 全颠倒（IoU 对正立真值 0.004/0.000，翻回后 0.963/0.902）。
train/eval 自洽但路由语义全错——上方结果表不可解释为 object/robot 路由对比。
代码已修（`convert_libero_raw.py`、`eval_env.py`）、数据已离线修复（150 h5 翻回 + grid_labels 重算，
IoU 复验 0.971/0.923）。重训排期待定（3×101 epochs ≈ 12.6h + n=400 eval）。
详见 `PIPELINE_IMPL_PLAN.md` §1.7 追加块。

**2026-07-31 重训链中止（用户指示：优先阶段一训练）**：变体 i 重训完成（101/101，ckpt 保留）；
变体 ii 在 ep11 中止作废；变体 iii 与三个 eval 未跑。重启方式：`./night_chain_stage0_retrain.sh`
（会跳过已完成的变体 i？——不会，engine 无 resume，重启前把 runs/robot_only_seed0 删掉即可，
变体 i 的 runs/object_only_seed0 保留则重复训练，需手动把它从脚本循环里去掉）。
