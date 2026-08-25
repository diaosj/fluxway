# Source SPMD formal protocol

This English companion mirrors
[source_spmd_formal_protocol_zh.md](source_spmd_formal_protocol_zh.md).
It states the source protocol implemented by Fluxway.

## Durable identity and cuts

The durable identity is `(stream_id, position)`. Each stream has a durable cut,
published end, and optional sealed end. Cuts are half-open counts. A source
adapter owns discovery and metadata materialization; Fluxway owns checkpoint
coordination, drain, admission, and failure propagation.

## Exactly-once and elasticity

`receive(stream, position)` is an idempotent shared read. Virtual SPMD mapping
is derived from the current topology and interleave unit; owner rank is not
checkpoint state. Therefore changing worker count changes physical partitioning
without changing logical records or replay identity.

## Checkpoint and restore

Only the canonical committed cut enters the checkpoint envelope. Execution
cursor, mailbox contents, ready queues, physical owners, and local prefetch are
transient. Restore rebuilds an open execution boundary from the durable cut and
recreates source discovery, barriers, and peer credit for the new topology.

The protocol requires explicit terminal observation and does not infer source
completion from polling, empty batches, or Python `None` crossing the adapter
boundary.

See [source_coordination.md](source_coordination.md) and the Chinese companion
for the full algebra, state transitions, safety, liveness, and counterexamples.
