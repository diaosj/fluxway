# Fluxway 文档

本文档目录包含 Fluxway 的公开稳定协议、接口规范与架构说明。英文目录见 [README.md](README.md)。

## 文档入口

| 主题 | 中文文档 | English |
| --- | --- | --- |
| 总体架构 | [ARCHITECTURE_zh.md](ARCHITECTURE_zh.md) | [ARCHITECTURE.md](ARCHITECTURE.md) |
| 执行生命周期 | [execution_lifecycle_zh.md](execution_lifecycle_zh.md) | [execution_lifecycle.md](execution_lifecycle.md) |
| 检查点与恢复 | [checkpoint_zh.md](checkpoint_zh.md) | [checkpoint.md](checkpoint.md) |
| 来源协调 | [source_coordination_zh.md](source_coordination_zh.md) | [source_coordination.md](source_coordination.md) |
| 来源任务协议 | [source_spmd_formal_protocol_zh.md](source_spmd_formal_protocol_zh.md) | [source_spmd_formal_protocol.md](source_spmd_formal_protocol.md) |
| Reader 接口 | [base_reader_zh.md](base_reader_zh.md) | [base_reader.md](base_reader.md) |
| 传输计划 | [transfer_planning_zh.md](transfer_planning_zh.md) | [transfer_planning.md](transfer_planning.md) |
| 数据面运行时 | [dataplane_runtime_zh.md](dataplane_runtime_zh.md) | [dataplane_runtime.md](dataplane_runtime.md) |
| 连贯性协议 | [coherence_protocol_zh.md](coherence_protocol_zh.md) | [coherence_protocol.md](coherence_protocol.md) |
| 放置配方 | [placement_recipe_zh.md](placement_recipe_zh.md) | [placement_recipe.md](placement_recipe.md) |
| 放置数学 | [placement_mathematics_zh.md](placement_mathematics_zh.md) | [placement_mathematics.md](placement_mathematics.md) |
| 工作进程算子 | [worker_process_operators_zh.md](worker_process_operators_zh.md) | [worker_process_operators.md](worker_process_operators.md) |
| 图可视化 | [graph_visualization_zh.md](graph_visualization_zh.md) | [graph_visualization.md](graph_visualization.md) |
| 用户 Profiler | [user_profiler_zh.md](user_profiler_zh.md) | [user_profiler.md](user_profiler.md) |

## 发布边界

公开文档描述稳定的所有权、状态机、数据格式、接口和失败语义。具体业务配置、远程数据目录、凭据和场景验收证据由下游应用仓库维护。本文档只定义通用 Fluxway 契约。
