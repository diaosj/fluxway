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

Frame admission is fenced after a protocol failure. Delivery completion and
coherence retirement are explicit asynchronous lifecycle operations; errors
propagate through the existing failure channel instead of being retried by a
fallback path.

See [coherence_protocol.md](coherence_protocol.md) and
[ARCHITECTURE.md](ARCHITECTURE.md) for ownership boundaries.
