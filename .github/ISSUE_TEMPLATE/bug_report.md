---
name: "Bug report (General) / 缺陷报告"
about: Report a software defect, unexpected crash, or regression in Fluxway
title: "[Bug]: "
labels: ["bug"]
assignees: []
---

<!-- Please search existing issues before opening a new one. / 提交前请先在已有 Issue 中搜索是否已存在相同问题 -->

### Summary / 问题简述
<!-- A clear, concise description of what the bug is. / 用简明扼要的一两句话描述遇到了什么问题 -->

### Environment Info / 环境信息
<!-- Run `python -V` and check package versions. / 请提供系统与关键依赖版本 -->
- OS: [e.g. Ubuntu 22.04 / Linux x86_64]
- Python Version: [e.g. 3.12.3]
- Fluxway Version: [e.g. 0.1.0 or commit SHA]
- Key Backends (if used):
  - Ray Version:
  - JAX / CUDA Version:
  - PyTorch Version:

### Reproduction Steps / 复现步骤
<!-- Minimal, self-contained Python code or command to reproduce the issue. / 请提供尽量最小、自包含的复现代码或命令 -->
```python
import fluxway

# Minimal reproducible example
```

### Expected Behavior / 预期行为
<!-- What should have happened? / 按照设计预期应该出现什么结果？ -->

### Actual Behavior & Stack Trace / 实际表现与报错调用栈
<!-- Paste the full error traceback or unexpected logs here. / 请在此处完整粘贴终端报错或 Python Traceback -->
```text
<paste full traceback here>
```

### Additional Context / 补充背景
<!-- Any other relevant context, such as cluster topology, hardware, or logs. / 任何其他相关信息（如单机/多机、GPU 规格、拓扑等） -->
