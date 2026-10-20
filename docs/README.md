# Fluxway Documentation

中文目录见 [README_zh.md](README_zh.md)。This directory contains only stable public protocol, API, architecture, and usage documentation.

| Topic | English | 中文 |
| --- | --- | --- |
| Architecture and lifecycle | [ARCHITECTURE.md](ARCHITECTURE.md) | [ARCHITECTURE_zh.md](ARCHITECTURE_zh.md) |
| Execution lifecycle | [execution_lifecycle.md](execution_lifecycle.md) | [execution_lifecycle_zh.md](execution_lifecycle_zh.md) |
| Source coordination | [source_coordination.md](source_coordination.md) | [source_coordination_zh.md](source_coordination_zh.md) |
| Reader API | [base_reader.md](base_reader.md) | [base_reader_zh.md](base_reader_zh.md) |
| Checkpoint and recovery | [checkpoint.md](checkpoint.md) | [checkpoint_zh.md](checkpoint_zh.md) |
| Source work protocol | [source_spmd_formal_protocol.md](source_spmd_formal_protocol.md) | [source_spmd_formal_protocol_zh.md](source_spmd_formal_protocol_zh.md) |
| Transfer planning | [transfer_planning.md](transfer_planning.md) | [transfer_planning_zh.md](transfer_planning_zh.md) |
| Data-plane runtime | [dataplane_runtime.md](dataplane_runtime.md) | [dataplane_runtime_zh.md](dataplane_runtime_zh.md) |
| Coherence protocol | [coherence_protocol.md](coherence_protocol.md) | [coherence_protocol_zh.md](coherence_protocol_zh.md) |
| Placement | [placement_recipe.md](placement_recipe.md) | [placement_recipe_zh.md](placement_recipe_zh.md) |
| Operators and tooling | [worker_process_operators.md](worker_process_operators.md) | [worker_process_operators_zh.md](worker_process_operators_zh.md) |
| Placement mathematics | [placement_mathematics.md](placement_mathematics.md) | [placement_mathematics_zh.md](placement_mathematics_zh.md) |
| Graph visualization | [graph_visualization.md](graph_visualization.md) | [graph_visualization_zh.md](graph_visualization_zh.md) |
| User profiler | [user_profiler.md](user_profiler.md) | [user_profiler_zh.md](user_profiler_zh.md) |

Other documents cover profiling, visualization, and examples. Source identity, checkpoint cuts, and ownership definitions belong to the documents listed above; module documents link to them instead of duplicating those definitions. English and Chinese companions are updated together.

Downstream applications own scenario configuration, remote catalog operations, credentials, and acceptance evidence. This repository documents the generic Fluxway contract.
