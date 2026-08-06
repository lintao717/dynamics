# Claude Code 项目研发工作流规则

## 适用范围

凡是目录 `docs/work-items/<ID>/` 下存在状态文件的任务，必须使用本工作流。当前任务规范、报告和测试证据优先于对话记忆。

## 事实源优先级

1. 用户明确批准的最新规范及修订；
2. `02_IMPLEMENTATION_SPEC.md`；
3. `01_REPO_FACTS.md`；
4. 当前代码、Git 历史和测试结果；
5. 对话中的非结构化描述。

若低优先级内容与高优先级内容冲突，以高优先级为准并记录冲突。

## 强制阶段

```text
SCOUT → SPEC → IMPLEMENT → REVIEW → FIX/ACCEPT → CLOSE
```

- 未完成 `01_REPO_FACTS.md`：不得实施；
- 未冻结 `02_IMPLEMENTATION_SPEC.md`：不得实施；
- 实施时必须先完成 `03_IMPLEMENTATION_MAP.md`；
- 审核前必须完成 `04_IMPLEMENTATION_REPORT.md` 和测试证据；
- 修复只处理 `05_REVIEW_REPORT.md` 中 P0/P1 blocker；
- P2/P3 只写入 Backlog。

## 实施规则

1. 只实现 In Scope 和 AC；
2. 不修改 Out of Scope；
3. 不进行无关重构、全局格式化、依赖升级或目录迁移；
4. 修改每个文件前，应能说明它对应哪个 AC；
5. 优先最小兼容修改；
6. 不以“更优雅”为由扩大改动；
7. 对每个 AC 建立至少一个验证证据；
8. 测试失败不得声称完成；
9. 无法执行测试时必须说明原因和未核验范围；
10. 规范与仓库事实冲突时，标记 BLOCKED，不自行重写需求；
11. 不自动提交、推送、合并或删除分支，除非用户明确要求；
12. 不在默认分支上进行大范围实现；若发现位于受保护分支，先报告。

## 审核修复封口

- 一轮完整审核后，修复阶段只能处理封闭 blocker 列表；
- 第二轮不得主动寻找新的普通问题；
- 新发现的 P2/P3 加入 Backlog；
- 原 blocker 全部关闭且没有新 P0 后，任务应进入 ACCEPTED。

## 交付证据

实现报告至少包含：

- 基线和当前 commit；
- 修改文件列表；
- AC → 代码 → 测试映射；
- 测试命令及结果；
- 未验证事项；
- 与规范偏差；
- `git diff --stat`；
- `git status --short`。
