# Source SPMD Formal Protocol

## Canonical Ownership

`LayoutUnitDescriptor` fixes revision, physical locator, cardinality metric,
interval and source-owned metadata. The compiler partitions the ordered stream
into logical steps before assigning physical worker quotas.

Interval coordinates are stable within their physical identity. Splitting a unit
changes its interval without rewriting the opaque source metadata; a source reader
resolves the assigned interval relative to the original encoded range. A new scan
or window must not renumber an existing physical interval. Compilation and window
planning both use `LayoutUnitDescriptor.slice`; slice composition preserves final
interval coordinates and identity independently of the slicing path.

`SourceWorkAssignment` describes one descriptor interval within a logical step.
`SourceWorkKey(epoch, step, assignment_id)` is its canonical identity. The
MessagePack digest is computed once by the assignment model. Rebinding the same
plan onto another worker count preserves assignments and their keys.

Logical batch cardinality must remain fixed across this rebinding. Changing the
number of physical workers does not authorize changing the data in a logical
step. The planner restore rejects a changed logical batch cardinality.

## Physical Leases

`CompiledSegment` binds part of an assignment to a worker and includes `step` and
`assignment_id`. `WorkerLayoutDescriptor` encodes these segments using the current
canonical MessagePack field set. No additional assignment binding protocol is used.

One assignment can span multiple workers. The ledger validates that published
segments cover each assignment exactly once, without gaps or overlaps. Batch
acquisition must match these published segments. Acquiring an already leased
segment fails; acquisition never silently steals ownership.

`SourceWorkItem` contains the assignment and the exact leased interval.
`SourceWorkLease` adds an opaque runtime `lease_id`.
`SourceWorkAcknowledgement(key, lease_id)` acknowledges only that lease's interval.
The parent descriptor is never acknowledged on behalf of other workers.

ACK batches are validated before mutation. Identical duplicates are idempotent while
the lease is retained; commit retires completed lease/ACK records. Retired leases
are stale and rejected.
Unknown or stale leases and conflicting assignment keys fail. Restore creates new
lease IDs, preventing delayed ACKs from the previous attempt from confirming replay.

## Frontier and Checkpoint

Each step has a remaining segment count. ACK updates touch the acknowledged
segments, without scanning the entire plan. `candidate_cut` advances only over
consecutive fully acknowledged steps. ACK records production, not model consumption.

`committed_cut` advances through the coordinated checkpoint transaction. Source,
model and optimizer must agree on that boundary. The existing checkpoint decision
and participant commit protocol remains the authority; source ACK is not a second
durable commit point.

The source recovery contract is a consumption bookmark and logical progress
at the shared cut. Replanning may change window boundaries and assignment keys while
preserving the ordered records in every logical batch. Leases, ACK caches, prefetched
layouts and tails are recreated; stateful operators retain required logical state.
Neither source component image carries a numeric protocol version. Source identity and
cursor compatibility are defined by [source coordination](source_coordination.md).

`max_in_flight_steps` bounds acquisition relative to the source candidate cut,
also capped by the active window end. Exhaustion waits asynchronously; a source
window does not force a model checkpoint. See [source coordination](source_coordination.md)
for configuration and checkpoint semantics.

## Execution Boundaries

Worker acquisition batches active steps up to operator pipeline depth. Controller
may return a nonempty complete-step prefix within its credit window. Worker checks
epoch, unique lease IDs, exact interval coverage and prefix completeness before
publishing a reply into its transient cache. Concurrent consumers share each
chunk's fetch task; cancelled callers do not abandon supervision. ACKs use a bounded
worker-local queue with batch submission, failure propagation and checkpoint drain.
Cache misses retain acquisition RPC latency; batching does not prove the absence
of a Controller throughput bottleneck.

Compiled execution layout defines physical membership and logical group ranks.
Dispatcher and send-plan lowering read that same immutable binding. Neither owns
source revision discovery, lease state or checkpoint cuts. Local zero-copy depends on
placement and transport conditions, independently of checkpoint correctness.

Automatic retry of arbitrary user side effects is not guaranteed safe. Recovery
must use the coordinated checkpoint; external effects require a user transaction
or an idempotency scheme that accounts for changing physical interval boundaries.

## Conformance requirements

A valid implementation preserves logical batch cardinality and exact interval coverage
when the physical topology changes. The planner stores the source consumption bookmark
at the coordinated checkpoint cut. Restore rebuilds physical assignments and leases
from that bookmark; the reader does not own an independent durable cursor or restore
an old window.

Source integrations must preserve explicit scan/refill, history retention and replay semantics
under the global model checkpoint. Source-window progress is event-driven; an adapter may use its
own storage protocol internally, but the Fluxway supervisor does not add timer polling. External
side effects require their own transaction
or an idempotency contract that remains valid when physical interval boundaries change.

See [source coordination](source_coordination.md) and [checkpoint](checkpoint.md).
