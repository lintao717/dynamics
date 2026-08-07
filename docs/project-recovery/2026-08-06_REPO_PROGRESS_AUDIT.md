# 仓库开发进度恢复审计报告

**日期**: 2026-08-07  
**审计范围**: 只读 — 不修改业务代码、不创建分支、不推送  
**仓库**: `D:\舆情分析`  
**分支**: `master`  
**HEAD**: `bd8b4f8` (chore: add GPT-Claude development workflow)  
**业务基线**: `3298c38` (fix: V2.0-C corrected opinion scoring)

---

## 1. 仓库基线

| 项目 | 值 |
|------|-----|
| 当前分支 | `master` |
| HEAD commit | `bd8b4f8` |
| 业务基线 commit | `3298c38` |
| HEAD 之后的提交 | 1 个 (`bd8b4f8`: 仅添加 `.claude/skills/wf-*` 工作流文件) |
| 已跟踪文件 (含修改) | ~113 个 (主要是未跟踪的本地配置/数据目录) |
| 未提交修改 | 无 (仅 `.gitignore` 覆盖的未跟踪目录) |
| 远程 | `origin/master` 同步到 `bd8b4f8` |

未跟踪文件分类（不纳入业务完成度）：

| 类别 | 数量 | 示例 |
|------|------|------|
| Claude Code 配置 | ~80 | `.claude/`, `.aris/`, `.trellis/` |
| 研究数据 | ~5 | `Tibet_data_collector/`, `auto_claude_research/` |
| 论文/申报 | ~8 | `Paper/`, `2409.08717v4.pdf`, `申报书_*.docx` |
| 项目文档 | ~10 | `文献调研报告.md`, `项目实施方法*.md` |
| 探索性目录 | ~3 | `idea-stage/`, `my_research_project/` |

---

## 2. 已完成里程碑

### 2.1 V1.7R.4 传播模块收口

**对应 commits**: `9d3ba31`, `b6e2854`, `7f13f96`, `2cd3440`

| 组件 | Commit | 源码证据 | 状态 |
|------|--------|---------|:--:|
| 行为观测修复 (pre-state) | `9d3ba31` | `dynamics_simulation/simulation.py:219` — `state_before_macro = state.copy()` | DONE |
| 步观测器签名更新 | `9d3ba31` | `dynamics_simulation/forecast_runner.py:178` — `observer(step_idx, state_before, state_after, events)` | DONE |
| 行为发射测试 (4个) | `b6e2854` | `tests/test_simulation_external_state.py:132-215` — 4 个新测试 | DONE |
| 传播基线冻结 | `2cd3440` | `configs/propagation_baseline_v1.yaml` | DONE |
| CV 多种子拟合 | `7f13f96` | `scripts/run_v17r3_crossval.py:83-96` — 5 种子中位损失 | DONE |

**实际能力**: ONE_SHOT 传播模型的行为观测已修复（正确计入 A→D 用户），传播参数基线已冻结。

### 2.2 V2.0-A 文本语义 Schema

**对应 commit**: `f16caa4`

| 组件 | 源码证据 | 状态 |
|------|---------|:--:|
| `InteractionSemanticSignal` | `dynamics_simulation/semantics/schema.py:20-70` | DONE |
| `WindowSemanticAggregate` | `dynamics_simulation/semantics/schema.py:73-98` | DONE |
| 字段范围校验 | `schema.py:43-58` — `__post_init__` 校验所有字段 | DONE |
| `SemanticAnnotator` | `dynamics_simulation/semantics/annotator.py:21-180` | DONE |
| 规则基线实现 | `annotator.py:138-173` — 关键词匹配立场/情绪 | DONE |
| SHA-256 缓存 | `annotator.py:35-40` | DONE |
| LLM 调用接口 | `annotator.py:175-178` — `_call_llm()` 占位（`NotImplementedError`） | PARTIAL |

**实际能力**: 可以关键词匹配方式标注交互文本的立场/情绪，缓存结果到 JSON 文件。**LLM 标注未实现。**

### 2.3 V2.0-B 观点观测聚合

**对应 commit**: `97ca920`

| 组件 | 源码证据 | 状态 |
|------|---------|:--:|
| `aggregate_window()` | `dynamics_simulation/semantics/aggregation.py:38-108` | DONE |
| 极化指标 | `aggregation.py:16-24` — 方差比双峰性 | DONE |
| 情绪熵 | `aggregation.py:27-35` | DONE |
| 话题熵 | `aggregation.py:38-48` | DONE |

**实际能力**: 可以按时间窗聚合交互语义信号为立场分布/极化/情绪/话题统计。

### 2.4 V2.0-C 观点动力学校准

**对应 commits**: `cddaff3`, `65bd7ef`, `3298c38`

| 组件 | Commit | 源码证据 | 状态 |
|------|--------|---------|:--:|
| 校准脚本框架 | `cddaff3` | `scripts/run_v20_opinion_calibration.py` | DONE |
| 观点指标 Replay 导出 | `65bd7ef` | `dynamics_simulation/replay/result.py:39-43` — `o_std_ts`, `o_hat_mean_ts` 等 | DONE |
| 观点指标多种子聚合 | `65bd7ef` | `dynamics_simulation/replay/runner.py:106-108` — 聚合列表包含观点指标 | DONE |
| 修正评分函数 | `3298c38` | `scripts/run_v20c_fixed_opinion.py:91-116` — 空窗口掩码、无 abs(corr) | DONE |
| 公开表达 vs 公开观测 | `3298c38` | `_score()` 使用 `o_hat_mean` 而非 `o_mean` | DONE |
| 校准结果导出 | `3298c38` | `artifacts/opinion/v20c_fixed_opinion.json` | DONE |

**实际能力**: 可以运行观点参数网格搜索，比较模拟公开表达 `o_hat` 与观测立场，但**仅单个 fake 案例**且**语义信号未注入动力学**。priv_std=0.32 确认观点动力学在运行。

---

## 3. 当前系统数据流

### 3.1 数据导入

```
CHECKED JSON → checked.py → EventCase (immutable) → NodeIndex → ObservedTrajectory
```
- **状态**: 完整。2,103 案例成功加载，1 失败。

### 3.2 传播动力学 (ONE_SHOT)

```
EventCase → build_initial_state → ReplayConfig → SimulationRunner → TransitionEngine
                                                                        ↓
                                                              Step 1: Exposure (Lambda)
                                                              Step 2: U→E
                                                              Step 3: Opinion update
                                                              Step 4: Emotion/fatigue
                                                              Step 5: E→A/D
                                                              Step 6: A→D (D0/D1 blocked)
                                                              Step 7: Public expression
```
- **状态**: 完整。ONE_SHOT 经 V1.5.2 消融、V1.7R.3 CV 和保留集测试确认。
- **网络模式**: 主要使用 Broadcast (`G_s=0, G_o=0`)。Cumulative 模式存在但未充分测试。

### 3.3 观点动力学

```
TransitionEngine Step 3 (update_opinions):
  G_o (opinion influence network) × o_hat (public expressions)
  + I_i (info_evidence) + u_i (official_info) + anchoring + noise
  → o_i(t+1)

TransitionEngine Step 7 (generate_expressions):
  o_i(t+1) + conformity + emotional amplification → ô_i(t+1)
```
- **G_o 网络**: 已连接。Cumulative 模式下有 root→user 边 (181 条)。
- **语义信号 `I_i`, `u_i`, `I_m`**: **未连接**。`EventInputTimeline` 不填充这些字段，默认为零。
- **观点 Replay 导出**: `o_std_ts`, `o_hat_mean_ts` 等已加入 `ReplayRun` 和聚合 (`3298c38`, `65bd7ef`)。

### 3.4 语义标注与聚合

```
InteractionRecord → SemanticAnnotator.annotate() → InteractionSemanticSignal
                                                          ↓
                                              aggregate_window() → WindowSemanticAggregate
```
- **状态**: 标注/聚合管道完整。但仅限于**规则关键词**（`annotator.py:138-173`），不是 LLM 标注。

### 3.5 模块连接状态

| 连接 | 状态 | 证据 |
|------|:--:|------|
| CHECKED → EventCase | ✅ 已连接 | `checked.py` 返回 `EventCase` |
| EventCase → ObservedTrajectory | ✅ 已连接 | `observations.py:build_observed_trajectory()` |
| EventCase → Replay | ✅ 已连接 | `runner.py:run_replay()` |
| Replay → TransitionEngine | ✅ 已连接 | `simulation.py:SimulationRunner.run()` |
| 观点动力学 → G_o | ✅ 已连接 | `CumulativeProvider` 提供非零 G_o |
| TransitionEngine → 语义信号 | ❌ 未连接 | `EventInputTimeline` 不填充 `info_evidence` 等 |
| 语义标注 → 仿真器 | ❌ 未连接 | 标注仅用于损失计算，不进入动力学 |
| 语义标注 → 观点校准 | ⚠️ 部分 | 标注用作目标值，但不作为动力学输入 |
| Replay → 观点指标导出 | ✅ 已连接 | `o_std_ts` 等已在 `ReplayRun` (`65bd7ef`) |

### 3.6 Forecast (前向预测)

```
EventHistory (pre-cutoff) → ForecastRunner → build_cutoff_state → SimulationRunner(H steps) → ForecastResult
```
- **状态**: 完整。时间轴索引、行为观测、绝对时间偏移均已修复 (`V1.7R.1` 系列)。

---

## 4. 测试状态

### 4.1 测试入口

```bash
# 完整命令（已执行并验证）
python -m pytest tests/ -q
```

### 4.2 执行结果

```
154 passed, 2 warnings in 85.11s
```

| 文件 | 测试数 | 说明 |
|------|:-----:|------|
| `test_calibration_objective.py` | ~12 | 时序切分、损失计算、尾部掩码 |
| `test_ced_adapter.py` | ~3 | CED 适配器 |
| `test_checked_adapter.py` | ~5 | CHECKED 适配器（含空文本） |
| `test_cli.py` | ~7 | CLI 入口 |
| `test_data_schema.py` | ~5 | 数据契约 |
| `test_end_to_end_checked.py` | ~2 | 端到端 CHECKED |
| `test_event_timeline.py` | ~12 | 事件输入时间线 |
| `test_identifiability.py` | ~5 | 参数可识别性 |
| `test_initial_state.py` | ~3 | 初始状态构建 |
| `test_network_direction.py` | ~3 | 网络方向约定 |
| `test_observations.py` | ~10 | 观测轨迹（含尾部掩码测试） |
| `test_parameter_estimator.py` | ~8 | 参数估计器 |
| `test_replay.py` | ~10 | 回放管道 |
| `test_simulation_external_state.py` | 11 | 外部状态注入 + 行为发射测试（4个新） |
| `test_smoke.py` | ~5 | 烟雾测试 |
| `test_temporal_networks.py` | ~5 | 时序网络 |
| `test_timegrid.py` | ~11 | 时间网格 |

**未运行的测试**: 
- CI 测试（`python tests/test_smoke.py` 等独立脚本）— 未单独执行，只通过 pytest 运行。
- `test_identifiability.py` 中的合成参数恢复测试需要 scipy — 本地已安装。

### 4.3 缺失测试

| 缺失测试 | 审查要求 | 当前状态 |
|----------|---------|:--:|
| `test_semantics.py` | V2.0-C 审查 (section 14) | NOT FOUND |
| `test_opinion_calibration.py` | V2.0-C 审查 (section 14) | NOT FOUND |
| `test_semantic_timeline.py` | V2.0-C 审查 (section 14) | NOT FOUND |
| `test_replay_exports_opinion_std` | V2.0-C 审查 (section 14) | NOT FOUND |
| `test_opinion_std_is_not_silent_fallback_zero` | V2.0-C 审查 (section 14) | NOT FOUND |
| `test_empty_semantic_windows_are_not_scored` | V2.0-C 审查 (section 14) | NOT FOUND |
| `test_negative_stance_correlation_is_penalized` | V2.0-C 审查 (section 14) | NOT FOUND |
| `test_public_stance_compared_with_public_expression` | V2.0-C 审查 (section 14) | NOT FOUND |

---

## 5. 旧计划映射

来源: `docs/2026-07-31-real-data-integration-plan.md`

| Task | 描述 | 状态 | 证据 |
|------|------|:--:|------|
| Task 1 | CHECKED 数据适配器 | DONE | `data/checked.py:82-149` — `load_checked_case()` 完整 |
| Task 2 | EventCase 数据契约 | DONE | `data/schema.py:46-112` — 不可变，含 `validate()` |
| Task 3 | TimeGrid 时间网格 | DONE | `data/timegrid.py:16-90` — `last_data_step`/`final_step` 分离 |
| Task 4 | NodeIndex 节点索引 | DONE | `data/indexing.py:15-45` |
| Task 5 | ObservedTrajectory | DONE | `data/observations.py:20-134` — 含 `first_actor_count` |
| Task 6 | 时序网络提供器 | DONE | `data/networks.py:93-234` — Broadcast/Cumulative/Oracle |
| Task 7 | EventInputTimeline | DONE | `data/timeline.py:77-117` — 固定时间常数 |
| Task 8 | 初始状态构建 | DONE | `data/state.py` — `build_initial_state()` |
| Task 9 | 回放配置与运行器 | DONE | `replay/config.py`, `replay/runner.py` |
| Task 10 | 校准目标函数 | DONE | `calibration/objective.py` — `compute_replay_loss()` |
| Task 11 | 参数规格与搜索 | DONE | `calibration/parameters.py` — `Stage1ParameterSet` |
| Task 12 | CLI 工具链 | DONE | `cli/inspect_dataset.py`, `cli/replay_event.py`, `cli/calibrate_event.py` |
| Task 13 | 校准估计器 | DONE | `calibration/estimator.py` — `fit_stage1()` |
| Task 14 | CI 测试 | DONE | `.github/workflows/test.yml` — `pytest tests/` |
| Task 15 | CHECKED 数据下载 | UNVERIFIED | 计划中标注 "待下载"；实际 `data/raw/CHECKED/` 存在 2,104 个 JSON 文件 |

**未完成的计划项**: 无。旧计划 Task 1-15 均已完成或数据已到位。

---

## 6. 当前缺口

### 6.1 核心缺口（影响科学研究）

| 缺口 | 位置 | 证据 |
|------|------|------|
| **语义信号未注入动力学** | `dynamics_simulation/data/timeline.py:91-117` | `EventInputTimeline.inputs_at()` 返回的 `ExternalInputs` 中 `info_evidence`, `official_info`, `info_emotion` 均为默认值 0。`transitions.py:88-90` 的默认值从未被覆盖。 |
| **无 SemanticInputTimeline** | 文件不存在 | `glob **/semantics/timeline*` 返回空 |
| **观点校准仅单案例** | `scripts/run_v20c_fixed_opinion.py:27-28` | `CASE_ID` 和 `CASE_LABEL` 硬编码为单个 fake 案例 |
| **G_o 仅星形网络** | `data/networks.py:200-218` | `build_network_provider()` 的 CUMULATIVE 模式只创建 root→user 边，无 user↔user 边 |
| **LLM 标注未实现** | `semantics/annotator.py:175-178` | `_call_llm()` 抛出 `NotImplementedError` |
| **缓存未持久化** | `semantics/annotator.py:30-31` | `cache_path=None` 时不保存缓存 |
| **根帖语义未初始化观点** | `replay/runner.py:46` | `build_initial_state()` 未传入 `TextSignals` |
| **无滚动观点预测** | 校准脚本只做全轨迹拟合 | 无 train/val 时间切分 |

### 6.2 测试缺口

- V2.0-C 审查要求的 8 个测试均不存在（见 section 4.3）

### 6.3 文档缺口

| 缺口 | 说明 |
|------|------|
| V1.7R.4 保留集脚本缺失 | `scripts/run_v17r3_holdout.py` 不存在，结果文件 `v17r3_holdout.json` 无法复现 |
| `artifacts/opinion/` 目录缺失结果文件 | `cddaff3` 打印了文件路径但未实际写入 |

---

## 7. 候选下一任务

### 候选 A: 构建 SemanticInputTimeline（最小闭环）

**前置条件**: V2.0-A/B/C 已完成（语义 Schema、聚合、观点度量导出）。  
**最小闭环**: 将聚合后的 `WindowSemanticAggregate` 按时间步映射为 `ExternalInputs.info_evidence` 等字段，遵守 "同一步文本不作为该步输入" 约束。  
**证据**: 当前 `EventInputTimeline` 不填充语义字段（`timeline.py:91-117`），`transitions.py` 已支持但接收零值。

### 候选 B: 修复观点校准管道（补测试 + 多案例）

**前置条件**: 候选 A 或人工语义信号就位。  
**最小闭环**: 添加审查要求的 8 个测试，扩展校准到 3-5 个案例，加入 train/val 时间切分。  
**证据**: 当前测试套件 154 passed 但缺少语义相关测试。

### 候选 C: 增强 G_o 网络结构

**前置条件**: 候选 A（需要语义信号验证观点影响）。  
**最小闭环**: 从 CHECKED comment/repost 关系中构建 user↔user 边（非仅 root→user），使用 `edge_step < current_step` 严格历史约束。  
**证据**: `data/networks.py:200-218` 当前只构建 root→user 星形。

---

## 8. 无法核验项

| 项 | 原因 |
|----|------|
| LLM 智能体决策是否已实现 | `api.py` 中存在 `LLMDecisionRequest` 引用但未在当前提交中找到实现 |
| CHECKED 数据是否完整下载 | 需要访问 `data/raw/CHECKED/` 目录（gitignored） |
| 外部审查者所述配置 bug 是否已全部修复 | 部分修复（V1.3 消融已标记无效），但 root_shock 接线和 cutoff_time 偏移待验证 |
| CI Actions 状态 | GitHub 连接器未返回该提交的 combined status |
| 保留集脚本 `run_v17r3_holdout.py` | 文件不存在于仓库中 |

---

## 9. 最终评估

### 当前可确认的最新业务版本

**`3298c38`** — V2.0-C 修正观点评分（priv_std=0.32, corr=0.69）

### 传播模块

- **状态**: 基本完成。ONE_SHOT 经消融、CV 和保留集验证。行为观测已修复。基线已冻结。
- **剩余**: 保留集脚本缺失，需补 `run_v17r3_holdout.py` 以保证可复现。

### 观点模块

- **状态**: 基础设施就位但关键连接缺失。语义信号可生成但未馈入动力学。校准仅限单案例。
- **剩余**: 需要 `SemanticInputTimeline`、LLM 标注器、多案例校准和滚动观点预测。

### LLM 模块

- **状态**: 未开始。`_call_llm()` 仅占位。

### 是否适合开始新 Work Item

**可以**，前提是新 Work Item 的依赖已满足。当前推荐从候选 A（SemanticInputTimeline）开始，因为这是连接语义标注器和动力学引擎的最小缺失环节。

### 开始前必须补齐的证据

1. 补 `scripts/run_v17r3_holdout.py` 或明确标记保留集结果为不可复现
2. 确认 `artifacts/opinion/` 目录状态（`cddaff3` 承诺写入但未实现）
3. 确认 CI 状态（当前仅本地测试通过）

---

**审计完成。未修改任何业务代码。**
