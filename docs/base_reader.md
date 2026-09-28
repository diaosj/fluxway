# BaseReader and Source Ingestion API

`BaseReader` converts source records into coordinate-aware `TransportTensor`
values. A controlled source reads the exact intervals in `self.ctx.source_work_items`.
The planner owns immutable revisions and canonical batches; `LogicalWorkLedger`
owns leases and progress. Dispatcher only selects active workers and routes data.

Before a controlled source call, OperatorManager validates a nonempty tuple of
`SourceWorkItem` values for the current execution step. Items must agree on source,
schema and global batch size; duplicate assignment intervals are rejected.
Dispatcher does not derive source row offsets from physical or group-local ranks.

## Read Lifecycle

1. Controller publishes a canonical plan and Driver delivers transient worker layouts.
2. Worker takes the current step's leases from its grant cache, fetching a bounded
   batch of active steps from Controller when needed.
3. Worker injects those `SourceWorkItem` values into the active `OperatorContext`.
4. The custom reader reads their pinned revision, locator and logical interval.
5. Tensor helpers attach the work items' global row coordinates. Worker submits
   source ACKs through its bounded asynchronous queue.

ACK confirms source production. It is not a model-consumption or durable checkpoint
commit. External side effects need their own transaction or idempotency contract.

## Tensor Helpers

| API | Contract |
| --- | --- |
| `create_tensor(data, dtype=None, uniform_dense_partition=False)` | Wrap dense or nested data with validated source row coordinates. |
| `create_row_sparse_tensor(rows, *, width, dtype=None)` | Build row-partitioned sparse values with canonical global shape. |
| `self.ctx.source_work_items` | Exact acquired intervals for this logical step; never cache across invocations. |
| `self.rank`, `self.world_size` | Physical placement coordinates; do not derive ownership from them. |

The outer row count must equal the total acquired cardinality, and all items must
belong to the active step and the same logical batch. Dense uniform and row-sparse
helpers use the assignment's `global_batch_size` for global rows. Origin indices
concatenate each item's `global_start` and cardinality, including disjoint intervals.
Missing context or conflicting metadata raises an exception.

For an explicitly coordinated source without work items, the caller must supply
`global_source_offset`; BaseReader does not infer it from physical rank. This mode
alone does not establish checkpoint replay or topology-independent ownership.

## Custom Sources

Implement `IngestionSourceAdapter.capture_view`, `encode_cursor` and `decode_cursor`
for immutable metadata discovery. Source-specific metadata and binary cursors use
the shared MessagePack codec. `IngestionReadSession.read` can provide synchronous
interval chunks to `read_ingestion_layout`; `IngestionLayoutSession` offloads
iteration to a thread and validates consumed cardinality.

For a finite controlled-source example, see
[RecommendationReader](../fluxway/example/recommendation/reader.py) and its
[source descriptor](../fluxway/example/recommendation/source_adapter.py).
Its source descriptor supplies `build_reader_topology`, `build_ingestion_plan`
and the existing Driver activation component. Restoring a controlled source must
reconstruct canonical plans before restoring the logical cut; a physical reader
cursor must not become a second ownership authority.

## Streaming support

The planner component accepts empty nonterminal observations without creating an
epoch, carries incomplete batches across revisions, and checkpoints pending tails
and cursors. The runtime inherently supports strongly-consistent reading of pinned revisions even across Reader quota scaling events.

Controller startup activates one layout; automatic adapter discovery/refill is not
part of the startup API. Worker accepts contiguous immutable windows and owns their
installation identity; BaseReader supplies the installation hook. Implementing
the adapter alone does not enable automatic
unbounded streaming through a training graph. See
[source coordination](source_coordination.md) for the supported ingestion scope.
