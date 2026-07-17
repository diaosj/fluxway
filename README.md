# fluxway
Fluxway is a Flexible Large-scale Unified eXchange designed to maximize Workload Accelerated Yield across heterogeneous compute clusters.    By replacing SPMD with Asynchronous State Machine MPMD, Fluxway dismantles the barriers between hardware constraints and model scale.   Easy to train and infer gigantic Sparse Embeddings, Experts, Omni Models.

## Documentation

- [Placement Recipe](docs/placement_recipe.md): declare per-SPMD resources and
  configure affinity, objectives, runtime limits, and diagnostics.
- [Placement Mathematics](docs/placement_mathematics.md): formal semantics and
  solver boundaries for Recipe authors.
- [Architecture](docs/ARCHITECTURE.md): control-plane and runtime ownership.

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
Explorer usage, dependency and transfer colors, potential-overlap layout,
remote viewing, trace correlation, and bundle semantics.
