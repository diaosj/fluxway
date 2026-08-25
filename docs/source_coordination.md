# Source coordination

This English companion mirrors [source_coordination_zh.md](source_coordination_zh.md)
and describes source ownership in the current runtime.

The source driver owns discovery and source lifecycle. Workers execute already
owned logical steps. Source refill is a coalesced maintenance wakeup that keeps
the configured logical pipeline depth; it does not read checkpoints, advance a
durable cut, or decide completion.

The source admission owner reads the complete pending lifecycle, including
blocked, future, async-lag, phase-order, mailbox-ready, running, finished,
pipeline-ready, and queue membership states. Candidate selection, claim, and
publication occur in one event-loop turn. Queue membership transports owned
work; it is not an ownership registry.

Virtual interleave identity is `(stream_id, logical_position)`. Physical rank,
worker ID, and current world size are derived execution placement and are never
durable exactly-once identity.

See [source_spmd_formal_protocol.md](source_spmd_formal_protocol.md) for the
formal source protocol.
