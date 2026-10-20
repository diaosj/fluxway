# Source coordination

## Ownership

`SourceScanBinding` is the session owner for topology, tail policy, scan adapter, and the
`initial_cursor_factory` resolver. The `IngestionSourceAdapter` only exposes
`scan_next(cursor, budget)` and pure `advance_cursor(cursor, records)`; it does not resolve the
session's initial position. The binding resolver may return `None` while an empty continuous
source has no scan view, allowing the controller to retry the same binding later. Once a cursor
is resolved, the planner owns that resolver for the session and never asks the adapter or a fresh
latest snapshot for an initial position. Scans return an immutable
`SourceScanResult` with ordered units and an `OpaqueSourceCursor`. The adapter owns
source identity, revision selection, locators and cursor encoding. Fluxway does
not interpret Iceberg snapshots or require producer-specific metadata.
Producers and consumers operate independently. Fluxway does not require producer notifications,
shared checkpoints, logs, or progress watermarks. Source progress is advanced by explicit scan
and acknowledgement events owned by the controller; the source-progress supervisor does not
poll on a timer.

Range reading and source scanning are separate capabilities. Reading specified rows
from a format does not establish bounded scanning or resumable cursor support; the
planning component requires the scan contract. Each adapter must implement and validate
that contract independently; the generic interface does not imply support for every
storage format.

`unit_cursors` gives the source-owned origin of every ordered unit. The adapter's
pure `advance_cursor(cursor, records)` advances within a unit without I/O; the
result's `cursor_at` selects the origin for a consumed prefix. These origins remain
in the control plane and are not worker layout fields.

`LayoutUnitDescriptor.slice` is the sole unit-slicing operation used by compilation,
window planning and the in-memory adapter. It preserves physical coordinates and
opaque metadata. `SourceScanStatus.from_scan` derives status from record count,
visible-range exhaustion and the finite-source contract; adapters supply facts
instead of implementing parallel status decisions.

`IngestionLayoutPlanningComponent` owns scan-result assembly, pending tails, compilation
and window publication. The compiler partitions logical steps before assigning
physical reader quotas. Readers consume only assigned immutable intervals; they
never replace a captured revision with the latest source state.

`LogicalWorkLedger` owns leases, acknowledgements and logical progress. A reader
acknowledgement records production, not downstream consumption or durable model
state. Global checkpoint publication is the only durable commit boundary.

The source ledger does not derive a common downstream-consumed prefix from the
training graph and does not require a linear path after the source. Its candidate cut
advances from contiguous source-work acknowledgements, and source acquisition is
bounded by the source's own in-flight window. Fan-out, joins, and conditional edges
remain owned by dependency flow control and whole-graph drain; they do not redefine
source identity or manufacture a source bookmark.

The existing credit topology covers remote dependencies but cannot replace source
leases and candidate-cut tracking. Arrival-triggered local completion counts also
cannot be written directly into a durable source position. Complex graphs still
require whole-graph drain and a common checkpoint to prove recovery; the generic scan
interface is not a recovery guarantee for arbitrary graphs.

## Budgets and windows

`SourceControlConfig` rejects unknown fields and provides:

| Field | Responsibility |
| --- | --- |
| `planning_window_steps` | Target complete logical steps per window |
| `max_in_flight_steps` | Acquisition credit relative to the source candidate cut |
| `max_planning_records` | Record budget |
| `max_planning_units` | Unit budget |
| `max_planning_descriptor_bytes` | Total encoded worker descriptor budget |

These quantities are independent. Source adapters own native metadata read limits;
Fluxway limits scan rows and units and measures actual compiled descriptor bytes. A full logical step must fit. Pending
units retain source order across scan results and revisions; nonterminal tails are not
padded or dropped. Finite termination applies the declared tail policy once.

Acquisition is bounded by the active window end and
`candidate_cut + max_in_flight_steps`. A worker that reaches the end of a published window
waits on the next contiguous activation; it does not poll, retry, or fail with `KeyError`.
Planning is triggered by explicit progress events and keeps a bounded lead derived from the
existing planning and in-flight limits. Reaching a source window boundary never forces a model
checkpoint. `READY`, `PROGRESS`, `IDLE` and `TERMINAL` describe source scan results, not
checkpoint states. An empty scan result is not end-of-source.

## Publication and lifecycle

Controller source-window progress is supervised by one `SourceProgressSupervisor` generation.
The supervisor owns event wakeup, planning execution, failure observation and stop/join. Controller
state transitions do not own a second polling task or timer, and checkpoint resume does not replay
source startup. Window planning runs outside the Controller FSM lock; only a bounded wakeup and
typed failure event cross the Controller boundary. A published activation wakes every worker
admission waiter for its newly available steps.

Capture and blocking metadata work use the existing adapter I/O lane. Compilation
uses the existing process executor. Accepted operations remain owned by the
component when a caller cancels its wait; close rejects new work, joins accepted
operations and releases executors. Failures remain observable.

Publication validates bounds and interval coverage before installing the window,
cursor, pending tail and ledger epoch. Activation delivers compiled worker layout
descriptors without creating a second planning identity. Physical binding and
placement are transient and do not change logical batch contents.

A window is a prefetch boundary, never a discard boundary. Units retained by the planner but not
yet consumed remain in its canonical pending state and are carried into the next contiguous
activation with the same source work identity. A terminal activation publishes the end position;
requests beyond that position return typed `SOURCE_EXHAUSTED`, which drives normal whole-graph
drain and checkpoint cut handling.

The planning API does not by itself prove an application has a complete event-driven
refill and shutdown loop. In particular, bounded scanning of results containing no
new records must be validated with the source adapter and controller lifecycle.

## Checkpoint and recovery

Use the existing Controller, Driver and Worker component registries. The planner
owns cut-aligned source state; source-control owns logical progress; model and
optimizer state are saved by Worker components. Driver components save only their
owned state. The current ingestion activation component has an empty checkpoint
image and rebuilds physical delivery after restore.

Recovery saves a source-owned consumption bookmark and replans and prefetches with
current window limits and reader topology. Logical batch
cardinality and ordered records must be preserved. Pure prefetch buffers, pending
tails and old plan digests are not durable source state; stateful operators still
save state that determines their future output. Physical actors, leases and iterators
are recreated. Required source history must remain available; missing history fails.

The planner saves the consumption bookmark and source constraints aligned with the
coordinated checkpoint; the source ledger saves logical work progress. Recovery replans
windows from the committed bookmark and rebuilds leases and assignments for the active
reader topology. Windows, tails, leases and prefetch caches are runtime state rather
than durable source checkpoint state. Model, optimizer and other state that affects
future results must participate in the same checkpoint.

Scan observations are reclaimed at durable cuts. Per-scan and per-window limits do
not bound the accumulated observations between checkpoints; long-running retention
and process memory still require validation.

See [formal work protocol](source_spmd_formal_protocol.md),
[reader API](base_reader.md), and [checkpoint lifecycle](checkpoint.md).
Scenario-specific integration configuration belongs to the downstream application repository;
this document defines only the generic Fluxway contract.
