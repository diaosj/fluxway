# Fluxway Architecture

This document is the English companion to [ARCHITECTURE_zh.md](ARCHITECTURE_zh.md).
Both files describe the same current implementation and protocol boundaries.

## Runtime ownership

The compiled graph is the single source of truth for SPMD declarations,
operator identity, dependencies, tensor transfers, placement inventory, and
worker routes. Workers receive immutable graph projections and do not rebuild
sharding or tensor identity metadata.

The Controller owns graph lifecycle, placement, drivers, heartbeats, drain,
checkpoint orchestration, and failure propagation. Drivers supervise workers
and forward control commands. Workers own local execution, admission, pending
step lifecycle, transport delivery, and local plan execution.

## Coherence and data plane

The endpoint-local MESI directory is the sole coherence state owner. A
`CoherenceInstance(subject, epoch)` is the identity; `CoherenceDirectory.admit`
is the synchronous admission seam, while `CoherenceSequencer` orders per-subject
intents. Directory operations are event-loop local and do not own worker tasks
or transport queues.

`TransportTensor` is the canonical tensor ABI. Tensor slicing, frame assembly,
batching, routing, and lineage belong to the transport/data-plane layer rather
than the worker scheduler. Exact-step inputs are required; another step cannot
be substituted as a fallback.

## Drain and restore

One `DrainBoundary` reducer owns observed, admitted, drained, and committed
cuts. Restore projects only the durable `committed_cut` through
`DrainBoundary.from_committed_cut`; historical transient phase and ceiling
fields are not restored as restrictions for the new execution cycle.

Source termination, periodic drain, checkpoint publication, late completion,
and restore all use the same boundary reducer. Source discovery and local
prefetch are runtime state and are rebuilt after restore.

## Placement and execution

Placement declarations come from compiled SPMD declarations and
`ResourceProfile`. `PlacementRecipe` expresses constraints and lexicographic
objectives without duplicating resource ownership. The worker admission owner
contains gating and pending-state promotion; `SourcePipelineMaintenance`
contains coalesced source refill. The execution lane only consumes owned queue
items, evaluates admission, dispatches plans, and completes `task_done()`.

See [coherence_protocol.md](coherence_protocol.md),
[placement_recipe.md](placement_recipe.md),
[placement_mathematics.md](placement_mathematics.md), and the detailed Chinese
companion for the complete section-by-section specification.
