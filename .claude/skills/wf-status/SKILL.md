---
name: wf-status
description: Show the current workflow state and missing artifacts for a work item. User-invoked only.
disable-model-invocation: true
argument-hint: <work-item-id>
---

运行：`python .ai-workflow/workflow.py status --id "$0"`。

根据输出说明：

- 当前状态；
- 已完成文件；
- 缺失文件；
- 唯一推荐的下一步命令。

不修改任何文件。
