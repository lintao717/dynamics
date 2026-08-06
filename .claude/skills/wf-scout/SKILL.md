---
name: wf-scout
description: Read-only repository reconnaissance for a workflow work item. Use only when the user explicitly starts the scout phase.
disable-model-invocation: true
argument-hint: <work-item-id>
---

对 Work Item `$0` 执行仓库侦察。

## 前置检查

1. 运行：
   `python .ai-workflow/workflow.py check --id "$0" --gate scout-start`
2. 读取：
   - `docs/work-items/$0/00_TASK_BRIEF.md`；
   - `.ai-workflow/CLAUDE_WORKFLOW_RULES.md`；
   - 根目录 `CLAUDE.md`；
   - 与任务相关的源码、测试、配置、文档和 Git 历史。

## 硬约束

- 不修改产品源码、测试、配置和业务文档；
- 只允许写入 `docs/work-items/$0/01_REPO_FACTS.md` 和 `status.json`；
- 不设计最终解决方案；
- 将“事实”“推断”“建议”“不确定项”严格分开；
- 不声称未实际运行的测试已通过；
- 使用文件路径、符号名、Git 命令和测试输出作为证据。

## 必须完成

1. Git 分支、HEAD、工作区状态和相关历史；
2. 仓库结构与相关模块；
3. 当前真实实现状态；
4. 调用链、数据流、接口和配置；
5. 当前测试命令及可安全执行的基线结果；
6. 可能修改和原则上不应修改的文件；
7. 约束、风险、冲突和未知信息；
8. 将结果完整写入 `01_REPO_FACTS.md`。

完成后：

1. 运行：`python .ai-workflow/workflow.py check --id "$0" --gate scout-complete`；
2. 运行：`python .ai-workflow/workflow.py mark --id "$0" --state SCOUTED`；
3. 运行：`python .ai-workflow/workflow.py export-plan --id "$0"`；
4. 最终只报告生成的 `PLAN_INPUT.md` 路径和仍未核验的事项。
