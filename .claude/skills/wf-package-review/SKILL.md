---
name: wf-package-review
description: Package implementation evidence and Git diffs for a bounded ChatGPT acceptance review. User-invoked only.
disable-model-invocation: true
argument-hint: <work-item-id>
---

为 Work Item `$0` 打包验收材料。

1. 运行：`python .ai-workflow/workflow.py check --id "$0" --gate review`；
2. 不修改产品代码；
3. 核对 `04_IMPLEMENTATION_REPORT.md` 中测试证据与当前工作区一致；
4. 运行：`python .ai-workflow/workflow.py export-review --id "$0"`；
5. 运行：`python .ai-workflow/workflow.py mark --id "$0" --state UNDER_REVIEW`；
6. 最终只输出 `REVIEW_INPUT.md` 和 `FULL_DIFF.patch` 的路径，并提示在 ChatGPT 项目中输入：`验收 $0`。
