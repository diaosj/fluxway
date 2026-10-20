---
name: "CI Failure / 流水线异常报告"
about: Report a failure in GitHub Actions CI, packaging tests, or Nightly builds
title: "[CI]: "
labels: ["ci", "bug"]
assignees: []
---

### Summary / 异常概述
<!-- Clear and concise description of the CI failure / 简要描述哪次 CI 或哪条流水线任务失败 -->

### Workflow Information / 流水线信息
- Workflow Name: [e.g. `CI`, `Nightly wheel`, `Release`]
- Run URL / Link: [e.g. https://github.com/fluxway-project/fluxway/actions/runs/...]
- Commit SHA / Tag: [e.g. `main` branch at `abc1234` or `v0.1.0`]
- Failed Job & Matrix: [e.g. `Lint and format`, `CPU tests (Python 3.13)`, `Build and install distributions`]

### Failure Logs / 失败日志
<!-- Paste the failed step's console output or traceback / 粘贴失败步骤的关键报错日志 -->
```text
<paste failed action logs here>
```

### Analysis or Root Cause / 初步分析（可选）
<!-- Any clues on dependency changes, environment issues, or recent commits / 是否为依赖版本变动（如 Ray/JAX 新版）或特定提交引起 -->

### Proposed Fix / 修复建议（如有）
<!-- Suggested fix, pin, or PR link / 建议的修复方式或关联 PR -->
