# Data-plane runtime

This English companion mirrors [dataplane_runtime_zh.md](dataplane_runtime_zh.md)
and documents the current transport implementation.

`TransportTensor` owns payload, layout, routing indices, origin lineage,
receipts, access intent, and runtime validity. Metadata lengths must match the
routing entity count; partial batches must derive their shape and origin
descriptor from the actual sample count.

The data plane assembles transport frames, groups deliveries, applies bounded
backpressure, and performs asynchronous delivery. It does not decide worker
step admission, checkpoint durable position, or source ownership. Local-only
edges may pass references directly while preserving the same tensor ABI.

`CompiledSendPlan` uses compiled transfer relations. Whole-unit transfers select
one complete producer unit for each target; fragment transfers assemble source and
target shard intersections. Direct whole-microbatch delivery does not require
packing gather or multi-shard value concatenation. Execution groups map physical
sender ranks to group-local logical coordinates while frames retain physical
identity. See [transfer planning](transfer_planning.md) for declarations and semantics.

For controlled sources, Worker indexes compiled segments by step and acquires
complete step prefixes in batched Controller RPCs. `SourceWorkItem` coordinates determine
Reader provenance; dispatcher rank is not an ownership cursor. ACKs use the
bounded `SourceAcknowledgementRuntime` queue and flush during checkpoint drain.
This removes synchronous per-ACK waits below queue capacity, but acquisition RPC
latency remains on a grant cache miss. Worker source scopes accept contiguous
immutable windows after installation. Automatic adapter discovery/refill is outside
the startup API; see [source coordination](source_coordination.md).

Grant batch size depends on the source credit window as well as pipeline depth.
If a Reader executes once every `L` logical steps and credit covers `C` consecutive
logical steps, that interval contains at most `ceil(C / L)` of the Reader's active
steps. A grant is also limited by pipeline depth and the remaining planned steps.
Increasing physical Reader count with fixed logical parallelism and credit can
therefore reduce batching efficiency. Partial-prefix replies leave the ungranted
suffix available for a later request. Size credit together with checkpoint
intervals and the execution cycle; a configured depth is not a guaranteed batch size.

Frame admission is fenced after a protocol failure. Delivery completion and
coherence retirement are explicit asynchronous lifecycle operations; errors
propagate through the existing failure channel instead of being retried by a
fallback path.

One `CoherenceBatchRetireRuntime.retire_publication()` call owns the complete
target fanout produced for one local step. It derives the union of the batches'
`CoherenceInstance` values, registers each distinct `(instance, child-slot)`
directory receipt once, admits the publication once, then starts all target
enqueues concurrently. A target-local `TransportFrameBatch` is an efficient
carrier and is not an independent coherence ordering unit.

`_ChildDeliveryReceipts` is the publication's sole receipt owner and performs
the reverse-order terminal transition exactly once. Delegation failure, one
target enqueue failure, cancellation of unfinished siblings, and successful
receiver acceptance all use that owner. The original failure propagates; Fluxway
does not retry a target, wait for a timeout, or treat a sibling RPC result as
success.

Remote `receive_transport_frames()` completion is converted into a
`ReceiverAcceptanceReceipt(publication_id, instance, child_slot, target_rank,
batch_seq)` only after the receiver admitted the frame and deposited it into its
mailbox. The receipt is validated against the immutable `PublicationBatch`; an
RPC return value cannot complete a child by itself. The receiver then owns its
materialized carrier. `PayloadLifetime` owns producer frame references and calls
`PayloadRecycler.release()` exactly once after every receipt and local lease has
finished. `AllocationRegistry` only tracks identity and exact handoff, so a
watermark cannot keep a buffer alive or release it prematurely.
Same-worker aliases keep their `ConsumerLease` until the consumer releases it.
This ownership-transfer path avoids a universal ring allocator and its copies
for variable-size sparse batches. It does not prove that Ray transport is
zero-copy.

### Worker integration

The Worker supplies one publication callback for a whole data-plane frame plan:

```python
batch_submit = partial(
    coherence_batch_retire_runtime.retire_publication,
    compiled_send_plan,
)
```

`BatchingDataPlaneExecutor` calls that callback once with the complete immutable
tuple of target batches and an enqueue callable. Callers must not invoke
`CoherenceSequencer.commit_payload()` per target batch. Local-only consumers use
the existing lease path and are not reimplemented as a second publication API.

See [coherence_protocol.md](coherence_protocol.md) and
[ARCHITECTURE.md](ARCHITECTURE.md) for ownership boundaries.

## Delivery Identity and Numerical Reproducibility

Routing a canonical sample to the correct consumer does not select the version
of mutable operator state that its computation reads. In the recommendation
example, sparse lookup and update use `AsyncPolicy.bounded_stale` with
`max_staleness=7`. The Worker bounds execution lag, while lookup may run before
or after an earlier update. The data plane transports the resulting values under
the same routing contract in either case.

A restored checkpoint and an identical committed sample sequence do not imply
an identical numerical trajectory after execution resumes, including at unchanged
worker counts. Validate saved/restored state, sample provenance and numerical
behavior separately. Deterministic model replay requires an explicit computation
ordering or state-version contract; whole-unit delivery and checkpoint restoration
alone do not provide it.

The example's GPU sparse-gradient scatter also does not promise bitwise replay
for duplicate indices. Equal input arrays may yield different floating-point
accumulation results before the output reaches the data plane. Transport identity
checks should compare a produced value with its received value, separately from
comparisons between independently computed outputs.

## JAX and CuPy Export Boundary

A shared GPU pointer does not establish producer completion. Recommendation Dense
projects each JAX output to its sole addressable local shard and adopts it through
`HybridBackend.asarray` with an explicit device and `copy=False`. Parameter
gradients must be fully replicated before flattening. The backend uses DLPack to
transfer lifetime and negotiate the consumer stream; unsupported devices, missing
CUDA support and zero-copy requests across device domains raise exceptions.

Frame staging adopts values on the configured backend device with
`copy=False`. Cross-device adoption fails; it never silently copies through CPU.
Ragged length metadata may require separate buffers even when values are shared.

Physical locality and buffer aliasing are different properties. Same-process
aliasing, same-node Ray object-store delivery and cross-node transport must be
measured separately. No plan-level proof establishes cross-process zero-copy or
Controller throughput. Receiver producer caches are indexed by execution-cycle slot,
so recurring groups do not create a cache entry for every absolute step.

Reader/Input node affinity does not select a shared-memory Ray transport.
`RayActorDataPlane` submits CPU payloads to Ray, which owns their inter-process
delivery and receive allocation. A read-only NumPy payload can reside in private
process memory. Whole-unit Input assembly can alias that received payload without
establishing that Reader-to-Input delivery itself was copy-free.

## Local commit and delivery lock boundary

Delivery observation is outside the operator commit resource lock: `SEND` registers and starts delivery, `AWAIT_SEND` observes completion without the commit resource, and `BARRIER/COMMIT` acquires the ordered commit lock only to publish the contiguous watermark. Drain, checkpoint, export, and stop still use the delivery fence to await all registered deliveries.

```mermaid
flowchart LR
  S[SEND: register token] --> W[AWAIT_SEND: wait, lock free]
  W --> C[COMMIT: ordered publication lock]
  C --> M[contiguous committed watermark]
```
