# Worker-Process Operators

Fluxway normally keeps user code away from the `SPMDWorker` actor event loop.
The scheduler, mailbox receiver, prepare runtime, and delivery runtime must keep
making progress while user operators run.

Some classes still must execute inside the SPMD worker process. Use
`@fluxway.requires_worker_process` only when the class initializes or owns
process-local runtime state.

```python
import jax
import fluxway


@fluxway.requires_worker_process
class Trainer:
    def __init__(self):
        jax.distributed.initialize(
            coordinator_address=self.rank0_address,
            num_processes=self.world_size,
            process_id=self.rank,
        )
```

Common reasons:

- `jax.distributed.initialize`
- process-local GPU or accelerator runtime state
- libraries that must be initialized in the same process as the SPMD worker
- native handles that cannot move to a separate Ray execution actor

Do not add it to ordinary stateful operators. `@requires_worker_process` keeps
state in the worker process, so the class must be designed for that lifecycle.

## Runtime Semantics

`@requires_worker_process` does not mean synchronous inline execution on the
Ray actor event loop. Fluxway submits process-affine `CALL` nodes to
`WorkerProcessCallRuntime`:

- the actor event loop submits work and awaits a future
- synchronous user code runs in the same process on a worker-local thread
- if the callable returns an awaitable, the runtime completes it inside the
  worker thread
- `contextvars` such as rank/world/step are propagated
- exceptions propagate back to the LocalStepPlan and poison the worker normally

This preserves event-loop progress for:

- `receive_transport_frames`
- mailbox deposit and ready callbacks
- `LocalInputPrepareRuntime`
- `OutboundDeliveryRuntime`
- scheduler wakeups and drain fences

Pure Python CPU-heavy user code still competes for the CPython GIL. The runtime
is an event-loop isolation mechanism, not a promise of CPU parallel speedup.
JAX, CuPy, native kernels, Ray RPC waits, and many I/O paths can still overlap
because they release the GIL or are backend driven.

## State And Resource Rules

`AsyncPolicy` controls concurrency:

- `StatePolicy.exclusive()` serializes access through LocalStepPlan resource ids.
- `StatePolicy.thread_safe()` removes the class-instance lock and requires the
  underlying state/store to be safe for concurrent calls.
- `local_pipeline_depth` comes from `AsyncPolicy.depth`.

If a process-affine class has mutable Python state and is not explicitly
thread-safe, use the default `AsyncPolicy.sync()` or `StatePolicy.exclusive()`.

## Failure Mode

Fluxway fails fast when a class calls `jax.distributed.initialize` but is not
decorated:

```text
Fluxway Worker Process Error: ... calls jax.distributed.initialize but is not marked with @fluxway.requires_worker_process.
```

The internal protocol field `__fluxway_process_affine__` is intentionally hidden
behind the decorator. User code should use `@fluxway.requires_worker_process`.
