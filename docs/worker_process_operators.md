# Worker-process operators

Most Fluxway stateful operators do not need special execution settings. Fluxway
runs them in an isolated Ray execution actor so user code is separated from the
SPMD worker process.

Some operators must run inside the SPMD worker process itself. The common case is
JAX distributed setup:

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

Use `@fluxway.requires_worker_process` when the operator initializes or owns
process-local runtime state, for example:

- `jax.distributed.initialize`
- process-local GPU or accelerator runtime state
- libraries that must be initialized in the same process that executes the SPMD
  worker

Do not add it to ordinary stateful operators. Keeping ordinary operators in an
isolated execution actor protects the worker process from blocking calls, global
state changes, and library initialization side effects.

Fluxway fails fast for the most common beginner mistake. If a class calls
`jax.distributed.initialize` but is not decorated, worker startup raises:

```text
Fluxway Worker Process Error: ... calls jax.distributed.initialize but is not marked with @fluxway.requires_worker_process.
```

The old internal protocol field `__fluxway_process_affine__` is intentionally
hidden behind the decorator. User code should use the decorator instead of
writing internal Fluxway attributes.
