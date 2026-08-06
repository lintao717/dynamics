---
name: wf-fix
description: Fix only the closed list of P0/P1 blockers from the first acceptance review. User-invoked only.
disable-model-invocation: true
argument-hint: <work-item-id>
---

对 Work Item `$0` 执行定向修复。

## 前置检查

1. 运行：`python .ai-workflow/workflow.py check --id "$0" --gate fix`；
2. 读取 `05_REVIEW_REPORT.md`、`02_IMPLEMENTATION_SPEC.md`、`04_IMPLEMENTATION_REPORT.md`；
3. 从审核报告中提取唯一允许处理的 P0/P1 blocker 列表；
4. 确认结论为 `REQUEST CHANGES`，然后运行：`python .ai-workflow/workflow.py mark --id "$0" --state CHANGES_REQUESTED`。

## 硬约束

- 只修复原 blocker；
- 不处理 P2、P3、Next Milestone 或新的架构建议；
- 不重新实现已通过 AC；
- 不进行无关重构；
- 每个修改必须对应一个 blocker ID；
- 若审核中的 blocker 没有 AC/不变量依据，先在报告中指出，不擅自扩大范围；
- 修复后运行 blocker 对应测试和完整回归测试；
- 不自动 commit、push、merge。

## 输出

填写 `06_FIX_REPORT.md`：

- blocker → 代码 → 测试映射；
- 每个 blocker 的 CLOSED / OPEN；
- 回归测试命令、退出码和结果；
- 明确未处理的 P2/P3；
- Git diff 和 status。

完成后：

1. 运行：`python .ai-workflow/workflow.py mark --id "$0" --state FIXED`；
2. 运行：`python .ai-workflow/workflow.py export-recheck --id "$0"`；
3. 最终输出 `RECHECK_INPUT.md` 和 `FULL_DIFF.patch` 路径，并提示在 ChatGPT 项目中输入：`复验 $0`。
