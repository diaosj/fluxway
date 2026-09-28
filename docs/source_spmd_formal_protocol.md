# Source SPMD Formal Protocol

## Canonical Ownership

`LayoutUnitDescriptor` fixes revision, physical locator, cardinality metric,
interval and source-owned metadata. The compiler partitions the ordered stream
into logical steps before assigning physical worker quotas.

`SourceWorkAssignment` describes one descriptor interval within a logical step.
`SourceWorkKey(epoch, step, assignment_id)` is its canonical identity. The
MessagePack digest is computed once by the assignment model. Rebinding the same
plan onto another worker count preserves assignments and their keys.

Logical batch cardinality must remain fixed across this rebinding. Changing the
number of physical workers does not authorize changing the data in a logical
step. The planner restore rejects a changed logical batch cardinality.

## Physical Leases

`CompiledSegment` binds part of an assignment to a worker and includes `step` and
`assignment_id`. `WorkerLayoutDescriptor` version 1 encodes these segments using
canonical MessagePack. No additional assignment binding protocol is used.

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

Logical ledger checkpoint version 3 contains the committed cut and immutable epoch
digests. Leases, ACK caches and physical owners are rebuilt. Planner checkpoint
version 2 stores canonical requests, fixed logical batch cardinality, pending tail
units and cursor state, excluding physical topology.

`max_in_flight_steps` bounds acquisition relative to the committed cut. Exhaustion
waits asynchronously; cancellation does not retain a ledger lock. Checkpoint
cadence must allow that window to advance, or acquisition will remain backpressured.

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

## Protocol Consistency Guarantees

The protocol provides the following strict consistency and elasticity guarantees:
- Handles split assignments, out-of-order ACKs, batch atomicity, isolation of stale ACKs, safe cancellation of waits, and revision conflicts.
- **Topology Elasticity**: After replacing the topology and restoring the ledger, the system guarantees seamless continued reading from the precise committed cut (e.g., safely scaling from three Readers down to two).
- **Stable Provenance**: Workers read along Canonical Assignment intervals. Any rebinding operations strictly preserve the original logical batch cardinality and data provenance.

**Architectural Flow**:
The Driver activation checkpoint state is empty. Upon restoration, the Controller always first rebuilds the canonical plan using the current reader topology before restoring the logical committed cut.

Controller startup publishes one layout. Worker source scopes accept contiguous
immutable windows after Reader installation. Automatic unbounded adapter discovery
and replenishment are outside the startup API. External storage integrations,
training convergence and zero-copy performance need workload-specific validation.

See [source coordination](source_coordination.md) and [checkpoint](checkpoint.md).
