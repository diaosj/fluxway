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

### Durable position and transient execution

The runtime meaning of the cut is a `position`: the single boundary at
which every operator's persisted cursor is defined. `CheckpointCut.position` is
the manifest's canonical value; each SPMD runtime projection stores the same
value as `drain_boundary.committed_cut`, and publication rejects any mismatch.
Reader cursors,
virtual-interleave units, and model update counts may use different coordinate
systems; each component maps the durable position to its own checkpoint state.

Worker execution cursors, mailbox entries, coherence leases, ready queues, and
credit observations are transient. Drain makes the durable position quiescent;
save records that position, and restore starts a new empty execution runtime at
it. No mailbox/coherence generation or credit-reset restore API is part of the
checkpoint contract.

### Source elasticity and exactly-once

Source checkpointing is defined over logical output positions, not over a
reader class, physical rank, or iterator implementation. A source component
must persist state that reconstructs its output at the durable
`CheckpointCut.position`; the checkpointer only applies the component SPI and
does not inspect or special-case the reader. For a `GLOBAL_RESHARD` component,
the restored state is topology-independent and the current virtual owner is
derived from the current `(position, world_size, interleave_unit_size)`.

This gives the checkpoint-relative invariant
`Committed = [0, position)` and `Replay = [position, ...)`, with an empty
intersection, even when the number of virtual-interleave workers changes.
Internal reader records or batches are outside Fluxway's inference boundary;
their mapping to logical output positions belongs to the source component's
state contract.

## Component SPI

`CheckpointComponent` only supplies the descriptor. Runtime behavior is
composed from independent SPI contracts; a component does not need to expose
every contract.

```python
class MyComponent(
    CheckpointComponent,
    CheckpointComponentLifecycle,
    CheckpointComponentSave,
    CheckpointComponentRestore,
):
    @property
    def descriptor(self) -> CheckpointComponentSpec: ...

    async def open(self, runtime): ...
    async def close(self): ...
    async def prepare_save(self, session): ...
    async def save(self, session) -> WorkerSaveResult: ...
    async def finish_save(self, session): ...
    async def commit_save(self, session) -> CheckpointComponentPublication: ...
    async def rollback_save(self, session): ...
    async def publish_save(self, session): ...
    async def restore(self, session): ...
```

The five capability names map directly to the five SPI contracts:
`COMPONENT_LIFECYCLE` (`open`/`close`), `SAVE`, `RESTORE`,
`OBJECT_LIFECYCLE` (holds and cleanup reservations), and `OBJECT_CLEANUP`
(`cleanup`). `SAVE` is one atomic transaction contract; its phases are not
independent capabilities. `commit_save` creates the component publication;
`publish_save` is called only after the global manifest is durable and visible.

Register each implemented SPI explicitly. The registration's non-null SPI
fields are the only source of truth for `registration.capabilities`:

```python
registration = CheckpointComponentRegistration(
    component=component,
    component_lifecycle=component,
    save=component,
    restore=component,
)
```

All stable authoring imports live in `fluxway.checkpoint`.

`SaveSession` is passed to worker-side save methods. It contains the component
instance, `checkpoint_id`, save attempt, commit permit, progress cut, SPMD,
rank, source topology, and component identity. `CommitSaveSession` adds every
worker's `WorkerSaveResult`. `RestoreSession` contains the selected checkpoint,
the saved topology, current topology, and the `CheckpointComponentPublication` to restore.

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
    save=lambda reader, session: reader.save_checkpoint_state(session),
    restore=lambda reader, state, session: reader.restore_checkpoint_state(state),
)
```

Use `PickleCheckpointComponent`, `SafeTensorsCheckpointComponent`, or
`OrbaxCheckpointComponent` when state belongs in physical storage. Orbax writes
a shared global value once; rank-local values are separate records only when the
component actually owns rank-local state.

## Component registration and cleanup

A component with finite
`CheckpointRetentionPolicy(retain_last=N)` must register both
`CheckpointObjectLifecycle` and `CheckpointObjectCleanup`. Registration fails
otherwise; there is no silent no-op cleanup mode.

```python
CheckpointComponentSpec(
    component_id="dense",
    save=CheckpointSaveSpec(),
    restore=CheckpointRestoreSpec(),
    checkpoint_objects=(
        CheckpointObjectSpec("model_state", retention=CheckpointRetentionPolicy(retain_last=3)),
        CheckpointObjectSpec("optimizer_state", retention=CheckpointRetentionPolicy(retain_last=3)),
    ),
)
```

Each declared checkpoint object is independently retained, held, and cleaned. The
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

## Topology

The manifest records source topology. Restore receives both source and current
topologies. A component explicitly chooses one topology mode:

- `EXACT_TOPOLOGY` for rank-local iterator or process state;
- `LOGICAL_PARTITION_REMAP` for stable logical partitions;
- `GLOBAL_RESHARD` for global arrays that the component can reshard.

Fluxway never guesses a partition function or silently copies rank-local state.
The user-side adapter supplies any component-specific restore policy.
