# fluxway

Fluxway is a Flexible Large-scale Unified eXchange designed to maximize Workload Accelerated Yield across heterogeneous compute clusters. By replacing SPMD with Asynchronous State Machine MPMD, Fluxway dismantles the barriers between hardware constraints and model scale. Easy to train and infer gigantic Sparse Embeddings, Experts, Omni Models.

## Documentation

### Core Architecture & Execution
- [Architecture](docs/ARCHITECTURE.md) ([中文](docs/ARCHITECTURE_zh.md)): control-plane and runtime ownership.
- [Execution lifecycle and usage](docs/execution_lifecycle.md) ([中文](docs/execution_lifecycle_zh.md)): start, wait, cancel, checkpoint continuation, and readiness/activation contracts.
- [Checkpoints](docs/checkpoint.md) ([中文](docs/checkpoint_zh.md)): durable progress cut, save/restore, and retention lifecycle.
- [Data-plane runtime](docs/dataplane_runtime.md) ([中文](docs/dataplane_runtime_zh.md)): asynchronous transport, tensor ABI, DLPack boundaries, and lock-free delivery.
- [Coherence protocol](docs/coherence_protocol.md): endpoint-local MESI directory tree, access intents, and formal invariants.

### Data Ingestion & Transfer
- [BaseReader and Source Ingestion API](docs/base_reader.md) ([中文](docs/base_reader_zh.md)): reader coordinate binding, row-sparse tensors, and custom adapters.
- [Source coordination](docs/source_coordination.md) ([中文](docs/source_coordination_zh.md)): canonical work ownership and supported ingestion boundaries.
- [Source SPMD formal protocol](docs/source_spmd_formal_protocol.md) ([中文](docs/source_spmd_formal_protocol_zh.md)): algebraic planning, leases, and checkpoint fencing.
- [Compiled transfer plans](docs/transfer_planning.md) ([中文](docs/transfer_planning_zh.md)): shard intersections, execution groups, and whole-unit routing.

### Placement & Operators
- [Placement Recipe](docs/placement_recipe.md) ([中文](docs/placement_recipe_zh.md)): declare per-SPMD resources and configure affinity, objectives, runtime limits, and diagnostics.
- [Placement Mathematics](docs/placement_mathematics.md) ([中文](docs/placement_mathematics_zh.md)): formal semantics and solver boundaries for Recipe authors.
- [Worker-process operators](docs/worker_process_operators.md): `@fluxway.requires_worker_process`, process-affine runtimes, and JAX initialization.

### Observability & Tooling
- [User profiler](docs/user_profiler.md): custom scopes, instant markers, and Perfetto trace correlation.
- [Computation Graph Visualization](docs/graph_visualization.md): Google Model Explorer export, dependency metadata, potential-overlap layout, remote viewing, trace correlation, and bundle semantics.

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
