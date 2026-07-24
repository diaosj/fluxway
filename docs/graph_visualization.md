# Computation Graph Visualization

Fluxway projects a compiled computation graph into Google Model Explorer. It
does not generate an SVG, DOT graph, or a separate visualization service.

## Installation

```bash
pip install "fluxway[visualization]"
```

The visualization extra installs `ai-edge-model-explorer`.

## Export And View

```python
from fluxway import view_graph, visualize_graph

bundle = visualize_graph(graph, "./output/training-graph")
view_graph(bundle)
```

`visualize_graph` reads two existing Python objects:

- `Graph` supplies the human-readable SPMD description.
- `CompiledGraph` supplies frozen SPMD declarations, compiled operators,
  dependencies, dependency codes, ports, TUUIDs, callable identities, and
  `AsyncPolicy`.

The function writes exactly one file:

```text
./output/training-graph/model_explorer.json
```

This JSON file is a Model Explorer export artifact. Fluxway does not use it to
rebuild operators, dependencies, callables, or worker state. Ray continues to
transport the native Python compiled graph and its user callables.

`view_graph` starts the Model Explorer server and blocks until it is stopped.
Export and viewing are separate, so a headless training process never opens a
browser or starts an HTTP server implicitly.

An existing output directory raises `FileExistsError`. Passing `overwrite=True`
requires the existing directory to contain a valid Fluxway Model Explorer
projection, then replaces the directory atomically. If the directory swap
fails, the previous projection is restored and the exception is propagated.

## Two Views

The exported Model Explorer collection contains two views of the same compiled
operators and dependencies.

| View | Question it answers |
|---|---|
| `Dependencies` | Which operator must precede which other operator, and what data moves between them? |
| `Potential overlap` | Which operators may become ready together, and which causal relations constrain that overlap? |

There are not two graph definitions. Both views are projections of the same
`CompiledGraph` and compiled `AsyncPolicy` schedule. The views deliberately use
different edge representations because they answer different questions.

SPMDs are Model Explorer namespaces and compiled operators are op nodes.
Selecting an operator exposes its `graph_id`, SPMD identity, function identity,
input/output signature IDs, mesh, resource range, dependency rank, and schedule
policy.

Searchable correlation fields include:

- `graph_id`
- `graph_generation`
- `spmd_id`
- `op_id`
- `function_id`
- `input_signature_id`
- `output_signature_id`
- `dependency_code`

Model Explorer may remove a namespace containing only one operator from the
overview. Fluxway therefore also places SPMD metadata on every operator. Enable
Model Explorer's `Keep layers with a single op node child` setting when each
SPMD must remain visible as a layer.

## Dependency And Transfer

A dependency means that a target operator cannot begin the corresponding step
until the source operator has satisfied that dependency. A transfer means that
a tensor also moves along the dependency.

```text
dependency: source operator -> target operator
transfers:  zero or more source port -> target port -> TUUID mappings
```

Zero transfers is valid. It is an ordering-only dependency and is implemented
by the transport compiler as a header-only frame rather than a tensor payload.
The visualization creates input/output metadata for that control relation so it
is visible in the same Model Explorer graph as tensor transfers.

The `Dependencies` view contains exactly one native edge per compiled
dependency. When one dependency carries multiple tensors, the source and
target metadata contain a structured `transfers` list with every transfer code,
logical TUUID, source port, and target port. It has no overlays.

The `Potential overlap` view contains one thin, unlabeled native layout edge
for each same-step causal operator pair. These edges use Model Explorer's
default ports and have no tensor metadata. The view aggregates compiled
dependencies and phase schedule relations by ordered operator pair. Tensor and
ordering dependencies do not create colored overlays because their same-step
causality is already represented by the native layout edge and their complete
metadata belongs to the `Dependencies` view. Only nonzero-step phase relations
create a teal overlay. They cannot become native edges because they must not
change the same-step layout rank.

## Visual Language

Operator box colors describe structural roles in the compiled dependency graph.

| Box | Meaning |
|---|---|
| Green | Source: no incoming dependency |
| Blue | Transform: both incoming and outgoing dependencies |
| Amber | Sink: no outgoing dependency |
| Purple | Member of a dependency cycle |
| Gray | Isolated operator |

The `Dependencies` view has no overlays. Its native Model Explorer edges are
the complete dependency projection and carry all tensor port metadata.

In `Potential overlap`, faint native edges are the complete same-step causal
layout. The only colored overlay is a nonzero-step phase constraint that cannot
participate in that layout.

| Color | Meaning |
|---|---|
| Teal `#007B83` | Nonzero-step phase dependency from `AsyncPolicy` |

Model Explorer's edge schema supports color, width, and labels, but has no
dashed or dotted stroke field. Fluxway does not emit unsupported JSON fields
that the viewer would silently ignore.

## Async Scheduling Facts

`AsyncPolicy` is the only source of asynchronous execution facts. Every
operator node exposes the policy values compiled for that operator, including:

- `async_mode`, `async_depth`, `async_commit`, and `async_max_staleness`
- `state_access` and `state_resource`
- `schedule_kind`, `schedule_base_priority`, and `schedule_phase`
- `schedule_phase_order` and `schedule_phase_dependencies`
- `schedule_lane`, `schedule_lane_order`, and `schedule_lane_priorities`
- `schedule_phase_priorities`

A phase-to-lane mapping is represented by each phase operator's
`schedule_phase` and `schedule_lane`. The ordered phase and lane collections are
also shown, so the Model Explorer details pane can reconstruct the full policy
without a second set of constants in the visualization module.

`compile_phase_schedule` is the shared schedule-semantics module. The worker
uses its precomputed gates on the execution hot path, while visualization uses
the same compiled relations. Phase ordering is therefore not reimplemented in
the visualization adapter.

These fields describe scheduling rules, not measured execution. Perfetto traces
remain the authority for actual queueing, transport, and device timing.

## Potential Overlap

Fluxway computes overlap groups without recursively traversing Python objects:

1. Combine compiled dependencies with same-step phase relations.
2. Collapse every same-step causal cycle into one component.
3. Compute the longest dependency rank on the resulting acyclic graph.
4. Put operators with the same rank into the same outer namespace.
5. Put schedule lanes inside each rank, following `lane_order` where present.
6. Keep each SPMD as the innermost namespace.

Every same-step causal relation between different components increases the
target rank. Cross-step phase relations are visible as teal overlays but do not
increase same-step rank. Operators in one rank are shown as a group rather than
being connected pairwise; pairwise overlap lines would grow quadratically.
Members of one group may become ready concurrently, but the view does not claim
that they overlap at runtime. Resource contention, conditions, different step
numbers, and backpressure can still serialize them.

## Remote Use

Bind the viewer to loopback and use SSH forwarding:

```python
view_graph(
    "./output/training-graph",
    host="127.0.0.1",
    port=8080,
    open_browser=False,
)
```

```bash
ssh -L 8080:127.0.0.1:8080 user@training-host
```

The exported file can also be copied to another machine and opened directly:

```bash
scp user@training-host:./output/training-graph/model_explorer.json \
  "$HOME/Downloads/training-graph.json"
model-explorer "$HOME/Downloads/training-graph.json"
```

To compare the graph with a trace, copy the `.pftrace` file and load it with
Perfetto:

```bash
scp user@training-host:./output/recommendation-traces/run.pftrace \
  "$HOME/Downloads/run.pftrace"
```

Use `graph_id`, `spmd_id`, `op_id`, and `function_id` to correlate Model Explorer
nodes with trace events.

## Scale Boundary

The iterative component and dependency-rank algorithms avoid Python recursion
and support graphs with more than 10,000 operators during export. Browser layout
is a separate Model Explorer cost: very dense graphs can take substantially
longer to open even when Fluxway exports them quickly.
