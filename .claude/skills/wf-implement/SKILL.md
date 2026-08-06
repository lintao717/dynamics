---
name: wf-implement
description: Implement one frozen workflow specification with minimal changes and evidence-backed tests. User-invoked only.
disable-model-invocation: true
argument-hint: <work-item-id>
---

对 Work Item `$0` 执行实现。

## 前置检查

1. 运行：`python .ai-workflow/workflow.py check --id "$0" --gate implement-start`；
2. 读取：
   - `.ai-workflow/CLAUDE_WORKFLOW_RULES.md`；
   - `00_TASK_BRIEF.md`；
   - `01_REPO_FACTS.md`；
   - `02_IMPLEMENTATION_SPEC.md`；
   - `status.json`。
3. 确认规范状态为 FROZEN；若仍是模板或 DRAFT，停止。
4. 确认 `02_IMPLEMENTATION_SPEC.md` 已删除 `WORKFLOW_PLACEHOLDER` 且状态明确为 FROZEN；
5. 运行：`python .ai-workflow/workflow.py mark --id "$0" --state SPEC_FROZEN`；
6. 检查当前分支。若处于受保护分支，先报告风险，不进行大范围修改。

## 执行规则

- 先填写 `03_IMPLEMENTATION_MAP.md`，每个 AC 映射到具体文件和测试；
- 每次修改必须对应一个 AC；
- 只实现 In Scope；
- 不处理 Backlog、P2/P3 和 Out of Scope；
- 不做无关重构、格式化、依赖升级或目录迁移；
- 优先最小兼容修改；
- 先建立或更新失败测试，再实施修复；
- 运行相关测试后再运行规范要求的完整测试；
- 不自动 commit、push、merge；
- 规范与仓库事实冲突时，停止并在 `04_IMPLEMENTATION_REPORT.md` 标记 BLOCKED，不自行改变需求。

## 输出

完成 `04_IMPLEMENTATION_REPORT.md`，必须包含：

- COMPLETE / PARTIAL / BLOCKED；
- AC → 代码 → 测试证据；
- 修改文件；
- 每条测试命令、退出码和结果摘要；
- 规范偏差和未验证事项；
- `git diff --stat`；
- `git status --short`。

完成后：

1. 运行：`python .ai-workflow/workflow.py check --id "$0" --gate implement-complete`；
2. 若 COMPLETE，运行：`python .ai-workflow/workflow.py mark --id "$0" --state IMPLEMENTED`；
3. 若 BLOCKED，运行：`python .ai-workflow/workflow.py mark --id "$0" --state BLOCKED`；
4. 最终汇报状态、测试结果和下一步命令 `/wf-package-review $0`。
