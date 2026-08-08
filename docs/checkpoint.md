# Checkpoints

A checkpoint is the complete, restorable state of one Fluxway execution at one
progress cut. Its public identity is intentionally small:

```text
checkpoint_id -> component_id -> checkpoint_object_id
```

`checkpoint_id` identifies the whole published checkpoint. `component_id`
identifies the subsystem that owns state, such as `dense`, `reader`, or
`embedding`. A component that writes more than one independently restorable
object gives each one a `checkpoint_object_id`. A single-object component uses
`"state"` by default.

Fluxway owns the checkpoint transaction, progress cut, topology, checkpoint
references, and cleanup decisions. A component owns serialization, physical
storage, restore mapping, and physical cleanup. This keeps Fluxway independent
of filesystems, object stores, Lance, catalogs, and data formats.

## Saving and restoring

Use `SaveCheckpointRequest.for_step(step)` for normal synchronous training.
It creates a unique save attempt, a step cut, and the step reference. Add names
when a checkpoint should also be selected as `best_model`, `final`, or another
application name.

```python
manifest = await checkpointer.save(
    SaveCheckpointRequest.for_step(1200, names=("best_model",))
)

await checkpointer.restore_step(1200)
await checkpointer.restore_named("best_model")
await checkpointer.restore_latest()
```

A step reference is a convenient lookup, not a storage-path convention. The
filesystem components happen to create a readable `steps/<step>/...` layout;
other components may store the same checkpoint in a catalog or object store.
Restore always resolves the published manifest, so it never needs a physical
path from application code.

For a non-step execution, construct `SaveCheckpointRequest` with a
`CheckpointCut` containing stream cuts and a named reference. The cut is the
only progress fact in the manifest.

## Component authoring

The extension point is `CheckpointComponent`. Implement the lifecycle once;
Fluxway supplies all transaction data through sessions rather than growing every
method signature.

```python
class MyComponent(CheckpointComponent):
    async def open(self, runtime): ...
    async def start_save(self, session): ...
    async def save(self, session) -> WorkerSaveResult: ...
    async def finish_save(self, session): ...
    async def commit_save(self, session) -> ComponentCheckpoint: ...
    async def rollback_save(self, session): ...
    async def on_checkpoint_available(self, session): ...
    async def restore(self, session): ...
    async def close(self): ...
```

Register a component with `register_checkpoint_component(component)`. For a
component with finite keep or external holds, pass its
`CheckpointObjectLifecycle` and `CheckpointObjectCleaner` to that same helper.
All stable authoring imports live in `fluxway.checkpoint`.

`SaveSession` is passed to worker-side save methods. It contains the component
instance, `checkpoint_id`, save attempt, commit permit, progress cut, SPMD,
rank, source topology, and component identity. `CommitSaveSession` adds every
worker's `WorkerSaveResult`. `RestoreSession` contains the selected checkpoint,
the saved topology, current topology, and the `ComponentCheckpoint` to restore.

Use session helpers instead of building manifest records by hand:

```python
async def save(self, session):
    staging = await self.storage.write(session.instance)
    return session.worker_save_result(staging)

async def commit_save(self, session):
    locator = await self.storage.commit(
        session.worker_results,
        session.commit_authority("embedding"),
    )
    state = session.checkpoint_object_ref(
        "state", locator.version_id, locator.to_bytes()
    )
    return session.component_checkpoint((state,))
```

`storage_locator` is opaque bytes. Fluxway stores it and passes it back at
restore or cleanup, but never decodes it. A user-side integration adapter may
translate it to a filesystem location, a catalog version, or a storage-specific
locator. The returned `CheckpointObjectRef` is the manifest's logical record of
that saved object. Fluxway does not import the integrated component's package, and the
integrated package does not import Fluxway.

For small manifest-only state such as a reader cursor, use the convenience
factory. It has no physical checkpoint object and therefore no physical cleanup
policy.

```python
from fluxway.checkpoint import state_checkpoint_component

reader = state_checkpoint_component(
    "reader",
    save=lambda reader, session: reader.capture_checkpoint_state(),
    restore=lambda reader, state, session: reader.restore_checkpoint_state(state),
)
```

Use `PickleCheckpointComponent`, `SafeTensorsCheckpointComponent`, or
`OrbaxCheckpointComponent` when state belongs in physical storage. Orbax writes
a shared global value once; rank-local values are separate records only when the
component actually owns rank-local state.

## Component registration and cleanup

A registration declares the component's exact capabilities. Persistence and
lifecycle are always present. A component that exposes holds or a finite
`KeepPolicy(keep_last=N)` must also register both `CheckpointObjectLifecycle`
and `CheckpointObjectCleaner`. Registration fails otherwise; there is no
silent no-op cleanup mode.

```python
ComponentSpec(
    component_id="dense",
    checkpoint_objects=(
        CheckpointObjectSpec("model_state", keep=KeepPolicy(keep_last=3)),
        CheckpointObjectSpec("optimizer_state", keep=KeepPolicy(keep_last=3)),
    ),
)
```

Each declared checkpoint object is independently kept, held, and cleaned. The
component returns one logical `version_id` for all physical partitions of that
object. `partition_key` distinguishes those partitions; it is a component-owned
stable restore key and is not a worker rank.

A `CheckpointHold` protects one live checkpoint object while another consumer
reads it. Holds are shared read protection, not locks. They are persistent,
generation-checked, and expire unless renewed. A named reference such as
`best_model` is kept forever until the name moves to a newer checkpoint.

When a finite policy has excess versions, Fluxway selects candidates by
checkpoint sequence. The component atomically reserves eligible versions,
checks holds, and then cleans physical state after the new manifest is
published. A held candidate prevents publication with `CheckpointCleanupBlocked`.
A cleanup failure remains pending and is retried by
`repair_pending_cleanups()`; it is never treated as successful.

## Commit permits

Each save has three different identities:

- `checkpoint_id`: the published checkpoint;
- `attempt_id`: this request to save;
- `CheckpointCommitPermit`: authority for the current coordinator to publish
  that attempt.

Fluxway persists coordinator generations. Taking over a repository invalidates
older permits. Components receive the permit in every save-side session and can
pass `session.commit_authority(scope)` to a storage system with its own CAS or
commit authority. A permit token is never written to the manifest. Restore and
`close()` do not need a permit.

## Topology and eager save

The manifest records source topology. Restore receives both source and current
topologies. A component explicitly chooses one topology mode:

- `EXACT_TOPOLOGY` for rank-local iterator or process state;
- `LOGICAL_PARTITION_REMAP` for stable logical partitions;
- `GLOBAL_RESHARD` for global arrays that the component can reshard.

Fluxway never guesses a partition function or silently copies rank-local state.
The user-side adapter supplies any component-specific restore policy.

`CheckpointCaptureMode.EAGER` requires a component that declares eager capture
support and captures an immutable view. Unsupported components fail explicitly;
Fluxway does not silently turn eager capture into a fenced save.
