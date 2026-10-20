# fluxway

[中文文档](README_zh.md)

Fluxway breaks through the global synchronization barriers and rigid cluster topologies of traditional SPMD by adopting **asynchronous state-machine MPMD**. It not only decouples **CPU memory I/O and GPU/TPU compute**, but also decouples **heterogeneous GPUs across different generations and memory tiers**. Through **elastic M:N topology matching** and **lock-free, zero-bubble overlap**, Fluxway unlocks peak efficiency across heterogeneous clusters.

### Key Workloads & Scenarios

* **Gigantic Recommendation Systems**: TB/PB-scale sparse embeddings with CPU-side deduplication & multi-hop transfer fully overlapped with GPU dense computation.
* **Large-scale Conditional Memory & Engram Architectures**: Disaggregate static memory lookups ($\mathcal{O}(1)$ N-gram hash tables & external knowledge bases) from dynamic neural reasoning. Massive parameters reside in host/tiered storage with deterministic prefetching completely hidden behind GPU Transformer layers.
* **Decoupled MoE (Mixture of Experts)**: Physically disaggregate Attention backbones and Expert pools across heterogeneous GPUs with zero-bubble dynamic token routing.
* **Disaggregated Prefill & Decode (PD Separation)**: Elastic asymmetric scaling between compute-bound Prefill nodes and memory-bandwidth-bound Decode workers with zero-copy KV cache exchange.
* **Multimodal World Models & End-to-End Networks (World Models & Omni)**: Asymmetric pipelining of spatiotemporal physics simulators/encoders and autoregressive LLM backbones across diverse accelerators without pipeline stalls.
* **Asynchronous RL & Alignment (RLHF / Self-Play)**: Non-blocking concurrency between high-throughput Rollout generation workers and Learner training clusters.

### Why Fluxway over Google Pathways?

Unlike monolithic distributed graph runtimes like [Google Pathways](https://arxiv.org/abs/2203.12533) that enforce rigid static graph abstractions and opaque compilation pipelines, Fluxway offers:
* **Flexible Native Python Component Orchestration**: Freely compose pipelines using standard Python functions, classes, and decorators. Operators seamlessly integrate with JAX, PyTorch, DLPack, and custom C++/CUDA kernels without proprietary walled-garden constraints.
* **Eager Execution Mode**: Full support for intuitive eager workflows and interactive debugging. Developers can prototype, inspect intermediate tensors, and evolve models dynamically without suffering through tedious, black-box full-graph recompilations.

## Documentation

- [Contributing](CONTRIBUTING.md) ([中文](CONTRIBUTING_zh.md)): local checks and CI coverage.
- [Releasing](docs/releasing.md) ([中文](docs/releasing_zh.md)): version tags, wheel builds, and PyPI publishing.

### Core Architecture & Execution
- [Architecture](docs/ARCHITECTURE.md) ([中文](docs/ARCHITECTURE_zh.md)): control-plane and runtime ownership.
- [Execution lifecycle and usage](docs/execution_lifecycle.md) ([中文](docs/execution_lifecycle_zh.md)): start, wait, cancel, checkpoint continuation, and readiness/activation contracts.
- [Checkpoints](docs/checkpoint.md) ([中文](docs/checkpoint_zh.md)): durable progress cut, save/restore, and retention lifecycle.
- [Data-plane runtime](docs/dataplane_runtime.md) ([中文](docs/dataplane_runtime_zh.md)): asynchronous transport, tensor ABI, DLPack boundaries, and lock-free delivery.
- [Coherence protocol](docs/coherence_protocol.md) ([中文](docs/coherence_protocol_zh.md)): endpoint-local MESI directory tree, access intents, and formal invariants.

### Data Ingestion & Transfer
- [BaseReader and Source Ingestion API](docs/base_reader.md) ([中文](docs/base_reader_zh.md)): reader coordinate binding, row-sparse tensors, and custom adapters.
- [Source coordination](docs/source_coordination.md) ([中文](docs/source_coordination_zh.md)): canonical work ownership and supported ingestion boundaries.
- [Source SPMD formal protocol](docs/source_spmd_formal_protocol.md) ([中文](docs/source_spmd_formal_protocol_zh.md)): algebraic planning, leases, and checkpoint fencing.
- [Compiled transfer plans](docs/transfer_planning.md) ([中文](docs/transfer_planning_zh.md)): shard intersections, execution groups, and whole-unit routing.

### Placement & Operators
- [Placement Recipe](docs/placement_recipe.md) ([中文](docs/placement_recipe_zh.md)): declare per-SPMD resources and configure affinity, objectives, runtime limits, and diagnostics.
- [Placement Mathematics](docs/placement_mathematics.md) ([中文](docs/placement_mathematics_zh.md)): formal semantics and solver boundaries for Recipe authors.
- [Worker-process operators](docs/worker_process_operators.md) ([中文](docs/worker_process_operators_zh.md)): `@fluxway.requires_worker_process`, process-affine runtimes, and JAX initialization.

### Observability & Tooling
- [User profiler](docs/user_profiler.md) ([中文](docs/user_profiler_zh.md)): custom scopes, instant markers, and Perfetto trace correlation.
- [Computation Graph Visualization](docs/graph_visualization.md) ([中文](docs/graph_visualization_zh.md)): Google Model Explorer export, dependency metadata, potential-overlap layout, remote viewing, trace correlation, and bundle semantics.

## Computation Graph Visualization

```bash
pip install "fluxway[visualization]"
```

```python
import fluxway

bundle = fluxway.visualize_graph(graph, "/tmp/training-graph")
fluxway.view_graph(bundle)
```

See [Computation Graph Visualization](docs/graph_visualization.md) for Model
Explorer usage, dependency metadata, potential-overlap layout,
remote viewing, trace correlation, and bundle semantics.
