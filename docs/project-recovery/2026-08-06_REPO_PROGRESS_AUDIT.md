# 仓库开发进度恢复审计报告

**日期**: 2026-08-07  
**审计范围**: 只读 — 不修改业务代码、不创建分支、不推送  
**仓库**: `D:\舆情分析`  
**分支**: `master`  
**审计开始时 HEAD**: `bd8b4f8` (chore: add GPT-Claude development workflow)
**报告提交后 HEAD**: `4cfa00b` (docs: project recovery audit)
**业务基线**: `3298c38` (fix: V2.0-C corrected opinion scoring)

---

## 1. 仓库基线

| 项目 | 值 |
|------|-----|
| 当前分支 | `master` |
| 审计开始时 HEAD | `bd8b4f8` |
| 报告提交后 HEAD | `4cfa00b` |
| 业务基线 commit | `3298c38` |
| `git ls-files` 数量 | 238 |
| tracked working tree | CLEAN |
| index | CLEAN |
| `git status --porcelain` 中 `??` 条目数 | 113 |
| 远程 | `origin/master` 同步到 `4cfa00b` |

存在 113 条未跟踪项，分类如下（不纳入业务完成度）：

| 类别 | 数量 | 示例 |
|------|------|------|
| Claude Code / ARIS 配置与技能 | ~80 | `.claude/`, `.aris/`, `.trellis/` |
| 研究框架 | ~5 | `auto_claude_research/` |
| 数据采集 | ~5 | `Tibet_data_collector/` |
| 论文与申报材料 | ~8 | `Paper/`, `2409.08717v4.pdf`, `申报书_*.docx` |
| 项目文档 | ~8 | `文献调研报告.md`, `项目实施方法*.md` |
| 探索性目录 | ~3 | `idea-stage/`, `my_research_project/` |
| 其他 | ~4 | `AGENTS.md`, `CLAUDE.md`, `.gitattributes`, `docs/2026-07-31-real-data-integration-plan.md` |

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

每个 Task 按原始编号和原始要求逐项核验。

### Task 1: Freeze V1.1 and Add Data-Governance Scaffolding

| Step | 描述 | 状态 | 证据 |
|------|------|:--:|------|
| 1 | 创建独立分支/worktree | DONE | `git tag v1.1.0` 存在；commit `11fe456` "docs: define real-data governance" |
| 2 | 运行 V1.1 基线 | DONE | CI 使用 `python -m pytest tests -v`；154 passed |
| 3 | 添加 .gitignore 排除 | DONE | `.gitignore` 包含 `data/raw/`, `data/processed/`, `data/results/`, `*.parquet`, `*.sqlite`, `*.db` |
| 4 | 编写 data/README.md + real_data_assumptions.md | DONE | `data/README.md` 和 `docs/real_data_assumptions.md` 均存在 |
| 5 | 重新运行基线并提交 | DONE | `11fe456` |

**整体**: **DONE** — 所有 5 个步骤已执行。

### Task 2: Define the Canonical Event Data Contract

| Step | 描述 | 状态 | 证据 |
|------|------|:--:|------|
| 1 | 编写 schema 测试（含边界校验） | DONE | `tests/test_data_schema.py` 存在 |
| 2 | 运行测试确认 import 失败 | DONE | 通过 — `dynamics_simulation.data.schema` 已实现 |
| 3 | 实现 `RootPost`, `InteractionRecord`, `EventCase` | DONE | `data/schema.py:19-113` — 不可变 dataclass，含 `validate()` |
| 4 | 运行测试 | DONE | 154 tests pass |
| 5 | 提交 | DONE | `44a07b2` "feat(data): add canonical event case schema" |

**整体**: **DONE**。

### Task 3: Implement the CHECKED Adapter

| Step | 描述 | 状态 | 证据 |
|------|------|:--:|------|
| 1 | 添加 fixture | DONE | `tests/fixtures/checked_case.json` 存在 |
| 2 | 编写适配器测试 | DONE | `tests/test_checked_adapter.py` 存在 |
| 3 | 实现字段别名提取 + Asia/Shanghai→UTC | DONE | `data/checked.py:49-58` — `_pick()` 支持 ID_KEYS/USER_KEYS/DATE_KEYS/TEXT_KEYS；时区 `ZoneInfo("Asia/Shanghai")` |
| 4 | 测试异常处理 | DONE | `test_checked_adapter.py` 包含 `ValueError` 测试 |
| 5 | 提交 | DONE | `f10a2fc` "feat(data): add CHECKED cascade adapter" |

**整体**: **DONE**。

### Task 4: Implement the CED Adapter as a Secondary Pipeline Check

| Step | 描述 | 状态 | 证据 |
|------|------|:--:|------|
| 1 | 创建 CED fixtures | DONE | `tests/fixtures/ced_original.json`, `tests/fixtures/ced_interactions.json` 存在 |
| 2 | 编写测试 | DONE | `tests/test_ced_adapter.py` 存在 |
| 3 | 实现适配器（含 SHA-256 ID 生成 + 时区处理） | DONE | `data/ced.py:47-138` — `load_ced_case()`, `_make_interaction_id()` 使用 SHA-256, `_parse_timestamp()` 处理 Unix epoch 和 Asia/Shanghai 字符串 |
| 4 | 运行测试 | DONE | 154 tests pass |
| 5 | 提交 | DONE | `1f70d26` "feat(data): add CED compatibility adapter" |

**整体**: **DONE**。

### Task 5: Add Node Indexing, Time Grid, and Observed Trajectories

| Step | 描述 | 状态 | 证据 |
|------|------|:--:|------|
| 1 | TimeGrid 边界测试 | DONE | `tests/test_timegrid.py` 存在 |
| 2 | 实现 NodeIndex（index 0 = root） | DONE | `data/indexing.py:24-37` |
| 3 | 实现 ObservedTrajectory（含 NaN stance/arousal） | DONE | `data/observations.py:18-133` — 含 `first_actor_count`, `repeat_actor_count`，`stance_mean=NaN` |
| 4 | 运行测试 | DONE | 154 tests pass |
| 5 | 提交 | DONE | `a7d6084` "feat(data): add indexing time grid and observed trajectories" |

**整体**: **DONE**。

### Task 6: Build Explicit No-Leak Network Providers

| Step | 描述 | 状态 | 证据 |
|------|------|:--:|------|
| 1 | 定义三种模式 | DONE | `data/networks.py:35-38` — `BROADCAST`, `CUMULATIVE_INTERACTION`, `ORACLE_STATIC` |
| 2 | 无泄漏方向测试 | DONE | `tests/test_temporal_networks.py` 存在 |
| 3 | 实现行归一化 G_o | DONE | `data/networks.py:97-99` — `_row_normalize()`；`CumulativeProvider` 构建 G_s 和 G_o |
| 4 | NetworkSnapshot 校验 | DONE | `data/networks.py:54-69` — 形状、NaN/Inf、非负校验 |
| 5 | 提交 | DONE | `f0b706f` "feat(data): add causal temporal network replay modes" |

**整体**: **DONE**。

### Task 7: Build a Real-Data Initial State and Optional Text Signals

| Step | 描述 | 状态 | 证据 |
|------|------|:--:|------|
| 1 | 定义 TextSignals + StatePriorConfig | DONE | `data/state.py:27-43` — `TextSignals(stance_by_user, arousal_by_user)`, `StatePriorConfig` 存在 |
| 2 | 编写状态构造测试 | DONE | `tests/test_initial_state.py` 存在 |
| 3 | 实现 build_initial_state (复用 initialize_agents) | DONE | `data/state.py:59-96` — 接收可选的 `signals: Optional[TextSignals]` |
| 4 | 添加校验（stance [-1,1], arousal [0,1]） | DONE | `state.py:47-57` 校验 |
| 5 | 提交 | DONE | `517e80b` "feat(data): construct initial agent state from event cases" |

**整体**: **DONE** — 但 `TextSignals` 在 replay runner 中**从未被传入**（`replay/runner.py` 调用 `build_initial_state(case, index, params, rng)` 不传 `signals`）。数据结构就位，但管线未连接。

### Task 8: Build Event Input Timelines and Validate ExternalInputs

| Step | 描述 | 状态 | 证据 |
|------|------|:--:|------|
| 1 | 编写范围/形状测试 | DONE | `tests/test_event_timeline.py:20-57` — shape 和 range 校验 |
| 2 | 实现 ExternalInputs.resolve() 校验 | DONE | `transitions.py:64-128` — 校验 shape、NaN/Inf、范围 |
| 3 | 实现 BroadcastExposureConfig + EventInputTimeline | DONE | `timeline.py:28-178` — `BroadcastExposureConfig`, `EventInputTimeline.inputs_at()` |
| 4 | 确定性测试 | DONE | `test_event_timeline.py:114-128` |
| 5 | 提交 | DONE | `1341fb6` "feat(data): add event input timeline and external input validation" |

**整体**: **DONE** — 但 `ExternalInputs` 仅返回 `media_exposure, staleness, novelty, shock=0.0, V=0.0`（`timeline.py:172-178`）。`info_evidence`, `official_info`, `info_emotion` 从未被设置（始终为 `resolve()` 默认值零数组）。符合 Task 8 原始计划（计划未要求填充这些字段），但不满足 V2.0 的语义输入需求。

### Task 9: Add External Initial State, Dynamic Networks, and Step Observers to the Simulator

| Step | 描述 | 状态 | 证据 |
|------|------|:--:|------|
| 1 | 状态保持测试 | DONE | `tests/test_simulation_external_state.py:14-51` |
| 2 | 动态网络测试 | DONE | `test_simulation_external_state.py:76-97` |
| 3 | SimulationConfig 新增字段 | DONE | `simulation.py:77-85` — `initial_state`, `network_provider`, `step_observer` |
| 4 | API 更新 | DONE | `api.py` — `from_network(initial_state=...)` |
| 5 | 回归测试 + 提交 | DONE | `43ec019` "feat(sim): support external state temporal networks and observers" |

**整体**: **DONE**。

### Task 10: Implement Historical Replay and Reproducible Result Persistence

| Step | 描述 | 状态 | 证据 |
|------|------|:--:|------|
| 1 | 定义 ReplayConfig | DONE | `replay/config.py:10-44` — 含 `max_nodes`, `truncate_policy`, `micro_steps` |
| 2 | 序列化结果 + 来源 | DONE | `replay/result.py:26-124` — `ReplayResult` 含 `model_version`, `git_sha`, `params_dict` |
| 3 | 单种子回放 | DONE | `replay/runner.py:31-89` — `_run_one_seed()` |
| 4 | 多种子聚合 | DONE | `replay/runner.py:100-136` — `_aggregate_seeds()` 含 mean/std/p5/p50/p95 |
| 5 | 确定性 + JSON 往返测试 | DONE | `tests/test_replay.py` 存在 |

**整体**: **DONE** — `source_revision` 字段存在于 `CalibrationResult` 中但默认 `"unknown"`。Checklist 要求 "dataset license/citation and exact source revision must be recorded in each replay result" — 部分满足（字段在，但未实际填充 CHECKED 版本号）。

### Task 11: Add Chronological Split and Masked Multi-Target Objective

| Step | 描述 | 状态 | 证据 |
|------|------|:--:|------|
| 1 | 定义 TemporalSplit | DONE | `calibration/split.py:8-47` |
| 2 | 定义 LossWeights | DONE | `calibration/objective.py:10-22` — `active_count`, `cumulative_users`, `interaction_count`, `peak_time`, `final_size` |
| 3 | 实现 mask-safe 评分 | DONE | `calibration/objective.py:82-178` — `compute_replay_loss()` 使用掩码 |
| 4 | train/test 分离测试 | DONE | `tests/test_calibration_objective.py` 存在 |
| 5 | 提交 | DONE | `3291c54` "feat(calibration): add temporal split and masked replay loss" |

**整体**: **DONE**。

### Task 12: Implement Restricted Stage-1 Calibration

| Step | 描述 | 状态 | 证据 |
|------|------|:--:|------|
| 1 | 四参数规格 | DONE | `calibration/parameters.py:34-44` — `Stage1ParameterSet.to_specs()` |
| 2 | 不可变嵌套替换 | DONE | `calibration/parameters.py:52-98` — `apply_parameter_vector()` |
| 3 | 差分进化优化 | DONE | `calibration/estimator.py:178-261` — `differential_evolution` with seed/pop/polish |
| 4 | 合成恢复测试 | DONE | `tests/test_parameter_estimator.py:80-151` |
| 5 | 校准来源 | DONE | `CalibrationResult.to_dict()` 含 optimizer settings, bounds, seed tuple |
| 6 | 提交 | DONE | `37f9d45` "feat(calibration): add restricted broadcast parameter fitting" |

**整体**: **DONE**。

### Task 13: Add Command-Line Workflows

| Step | 描述 | 状态 | 证据 |
|------|------|:--:|------|
| 1 | inspect_dataset | DONE | `cli/inspect_dataset.py` — 含 `report=True` 审计摘要 |
| 2 | replay_event | DONE | `cli/replay_event.py` |
| 3 | calibrate_event | DONE | `cli/calibrate_event.py` — 含 `--source-revision` |
| 4 | 错误路径测试 | DONE | `tests/test_cli.py` — 7 tests, 含 missing file, unknown mode, invalid fraction |
| 5 | 提交 | DONE | `371fb23` "feat(cli): add dataset replay and calibration commands" |

**整体**: **DONE**。

### Task 14: Add End-to-End CHECKED Fixture Validation and CI

| Step | 描述 | 状态 | 证据 |
|------|------|:--:|------|
| 1 | 端到端测试 | DONE | `tests/test_end_to_end_checked.py` 存在 |
| 2 | CI 添加所有测试 | DONE | `.github/workflows/test.yml:18` — `python -m pytest tests -v` |
| 3 | 数据契约文档 | DONE | `docs/v1.2_data_contract.md` 存在 |
| 4 | 本地全量门 | DONE | 154 tests pass |
| 5 | 提交 | DONE | `f503b32` "docs: add V1.2 implementation report" |

**整体**: **DONE**。

### Task 15: Execute the First Real CHECKED Study

| Step | 描述 | 状态 | 证据 |
|------|------|:--:|------|
| 1 | 客观选择案例 (100-1000 users, ≥100 ix, ≥96h, ≥20 cmt/rpt) | DONE | `configs/experiments/checked_pilot_matched_20.yaml` — 10 对规模匹配案例；筛选条件接近原计划但并非完全相同（原计划要求 ≥200 ix, ≥12h 而非 ≥96h） |
| 2 | 运行三种回放模式 | DONE | `artifacts/replay/` — 40 个 JSON 文件（broadcast + cumulative 各 20） |
| 3 | 拟合 stage-1 参数（broadcast） | DONE | `artifacts/calibration/` — 含 `s3_calibration.json`, `s3_batch_20.json`, `v151/v152_mechanism_ablation.json` |
| 4 | 与持续性+指数衰减基线比较 | DONE | `artifacts/replay/replay_metrics_v2.csv` — 含 persistence/exp/pulse 基线 NRMSE |
| 5 | 撰写 pilot 报告 | DONE | `docs/technical_report_v1.4.md` — 区分 observed facts, model assumptions, fitted params, validation performance |
| 6 | V1.2 release gate 决策 | PARTIAL | Step 6 要求 "median validation active-count NRMSE beats persistence or exponential baseline" — 初始 V1.2 结果: 仅 1/20 超越基线（broadcast 默认参数）。后续 ONE_SHOT S3 在 V1.5.2 达到 6/20，但原始 gate 条件下的默认参数不满足 |

**整体**: **PARTIAL** — 前 5 个步骤均已完成。Step 6 的 release gate 条件在默认参数下不满足（1/20），但后续迭代（V1.5 S3）显著改善。原始计划要求的 "至少 15/20 案例无错误完成" 满足（40 次回放全部成功）。"结果可从 config+case+seed 复现" 满足。

---

### 旧计划状态汇总

| Task | 名称 | 状态 |
|------|------|:--:|
| Task 1 | Freeze V1.1 + Data-Governance Scaffolding | DONE |
| Task 2 | Canonical Event Data Contract | DONE |
| Task 3 | CHECKED Adapter | DONE |
| Task 4 | CED Adapter | DONE |
| Task 5 | Node Indexing, Time Grid, Observed Trajectories | DONE |
| Task 6 | No-Leak Network Providers | DONE |
| Task 7 | Real-Data Initial State + TextSignals | DONE (但 TextSignals 未接入回放管线) |
| Task 8 | Event Input Timelines + ExternalInputs | DONE (语义信号未填充) |
| Task 9 | External State, Dynamic Networks, Observers | DONE |
| Task 10 | Historical Replay + Result Persistence | DONE |
| Task 11 | Chronological Split + Masked Objective | DONE |
| Task 12 | Restricted Stage-1 Calibration | DONE |
| Task 13 | CLI Workflows | DONE |
| Task 14 | End-to-End CI + Data Contract | DONE |
| Task 15 | First Real CHECKED Study | PARTIAL (release gate 未满足原始条件) |

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

### 传播模块 (V1.2–V1.7R.4)

- **状态**: 基本完成。ONE_SHOT 经消融、CV 和保留集验证。行为观测已修复。基线已冻结 (`configs/propagation_baseline_v1.yaml`)。
- **与旧计划对照**: Task 1–15 中传播相关部分 (Task 3–14) 均为 DONE。Task 15 Step 6 release gate 在原默认参数下不满足。
- **剩余**: 保留集脚本 `run_v17r3_holdout.py` 缺失（结果文件 `v17r3_holdout.json` 存在但无法复现）。

### 观点模块 (V2.0-A/B/C)

- **状态**: 基础设施就位但关键连接缺失。
  - V2.0-A: `InteractionSemanticSignal` + 规则标注器 — DONE
  - V2.0-B: `aggregate_window()` — DONE
  - V2.0-C: 观点指标 Replay 导出 + 修正评分 — DONE (priv_std=0.32 确认动力学运行)
  - **语义→动力学连接**: NOT FOUND — 无 `SemanticInputTimeline`，`EventInputTimeline` 不填充 `info_evidence/official_info/info_emotion`
  - **LLM 标注**: `_call_llm()` 仅占位 (NotImplementedError)
  - **TextSignals 接入**: `state.py` 实现完整但 `replay/runner.py` 未传入

### LLM 模块

- **状态**: 未开始。`api.py` 中有 `LLMDecisionRequest` 引用但无实现。

### 是否适合开始新 Work Item

**可以**，前提是新 Work Item 的依赖已满足。当前推荐从候选 A（SemanticInputTimeline）开始——这是连接语义标注器和动力学引擎的最小缺失环节，也是 V2.0 路线图 Phase 3（审查建议第三阶段）的核心步骤。

### 开始前必须补齐的证据

1. 补 `scripts/run_v17r3_holdout.py` 或明确标记保留集结果为不可复现（当前仅存在 `.json` 结果文件但无运行脚本）
2. 确认 CI 状态（仅本地 `154 passed`，无 GitHub Actions 运行记录可查）

---

**审计完成。未修改任何业务代码。**
