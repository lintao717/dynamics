---
name: wf-close
description: Close an accepted workflow work item and preserve backlog items. User-invoked only.
disable-model-invocation: true
argument-hint: <work-item-id>
---

关闭 Work Item `$0`。

1. 读取 `07_FINAL_ACCEPTANCE.md`；若不存在，但 `05_REVIEW_REPORT.md` 已是 `ACCEPT` 或 `ACCEPT WITH BACKLOG`，可以使用该报告作为验收依据；
2. 结论必须是 `ACCEPT` 或 `ACCEPT WITH BACKLOG`；否则停止；
3. 将审核报告中的 P2/P3 和 Next Milestone 合并到 `BACKLOG.md`，不得修改产品代码；
4. 运行：`python .ai-workflow/workflow.py mark --id "$0" --state ACCEPTED`；
5. 运行：`python .ai-workflow/workflow.py mark --id "$0" --state CLOSED`；
6. 输出最终任务状态、未提交修改、建议的人工 Git 操作；
7. 不自动 commit、push、merge、删除分支。
