# User Profiler

Fluxway automatically records every operator call as `user_callable.execute`.
Use `fw.profiler.scope()` and `fw.profiler.event()` only for smaller regions
inside your operator.

```python
import fluxway as fw


def lookup(self, ids):
    if fw.profiler.enabled():
        id_count = len(ids)
    else:
        id_count = 0

    with fw.profiler.scope("embedding_lookup", table="item", id_count=id_count):
        values = self.table.lookup(ids)

    fw.profiler.event("cache_hit", hit=True)
    return values
```

## API

`fw.profiler.scope(name: str, **attrs)` records a duration slice named
`user.<name>`.

`fw.profiler.event(name: str, **attrs)` records an instant marker named
`user.<name>`.

`fw.profiler.enabled()` returns `True` only while profiler is enabled and the
current code is running inside a Fluxway user callable.

Attributes may be `str`, `int`, `float`, `bool`, or `None`. Complex objects
raise `TypeError` when profiler is enabled. When profiler is disabled, scope and
event are no-ops and attributes are not validated, interned, or serialized.

Use `enabled()` before constructing expensive attributes:

```python
attrs = {}
if fw.profiler.enabled():
    attrs["ids_preview"] = ",".join(str(x) for x in ids[:8])

with fw.profiler.scope("prepare_ids", **attrs):
    ids = normalize(ids)
```

## Trace Display

Worker duration slices use this Perfetto hierarchy:

```text
SPMDs/<spmd_id>/rank:<rank>/<subsystem>/<leaf>/lane:<lane>
```

Expand a rank to inspect its compute, transport, mailbox, and scheduler
subsystems. The leaf identifies the recorded behavior. `lane:<lane>` separates
overlapping slices on the same leaf so Perfetto preserves each slice's original
duration. Lane allocation is deterministic and uses the minimum number of
lanes required by the leaf's maximum concurrency; it does not create one track
per step.

Scheduler activity is grouped by responsibility under `scheduler/task`,
`scheduler/queue`, `scheduler/step`, and `scheduler/loop`. Checkpoint backend
operations use `checkpoint/storage`, while snapshot coordination uses
`checkpoint/hook`. A `worker` level is not repeated below `rank:<rank>`.

Custom user scopes and events use the leaf:

```text
compute/user_custom/<execution_mode>
```

The `<execution_mode>` component is a runtime call mode. It is distinct from
the final Perfetto concurrency track `lane:<lane>`.

Payload fields are correlated with the surrounding call:

- `op_id`
- `step`
- `rank`
- `execution_mode`

User attributes are expanded as readable Perfetto debug annotations. The
internal `attrs_id` is not shown.

The surrounding `user_callable.execute` slice carries the immutable execution
plan identity fields:

- `graph_id`
- `spmd_id`
- `op_id`
- `function_id`
- `input_signature_id`
- `output_signature_id`

The same values are searchable in Model Explorer. `step` and `execution_mode`
are runtime facts and do not belong to the compiled graph.

Perfetto also derives readable graph annotations from the installed
`CompiledGraph` when the trace is exported. Events carrying `op_id` show the
operator function and every declared data, control, and static input/output
name. When the compiled `OperatorSpec` contains `input_param_map` or
`output_param_map`, Perfetto also shows those Python parameter-to-field paths.
An empty parameter map is displayed as no parameter-path annotation; Fluxway
does not guess parameter names from ports. Events carrying `transfer_code` show
the complete source and target operator, function, and port. `logical_tuuid`
identifies the producer output.

One producer output may feed several target ports. Those events share one
`logical_tuuid` but have different `transfer_code` values, so each target port
is displayed independently instead of being overwritten by a single tensor
name map.

The recorder hot path continues to store only fixed-width integer records.
Graph names are not copied into each event and no graph lookup happens while an
operator runs. The frozen profiler symbol projection is built once from the
current `CompiledGraph` during trace export. A missing operator, transfer, or
logical tensor identity raises `MissingGraphTraceSymbol`; Fluxway does not emit
an incomplete name as a fallback.

## Rules

- Do not add a decorator around the whole operator. Fluxway already records that
  as `user_callable.execute`.
- Use `scope` for code regions that take time.
- Use `event` for point-in-time facts such as a cache hit, sampled branch, or
  algorithm phase marker.
- Attribute values must be scalar. Put large tensors, arrays, dicts, lists, or
  object reprs in your own logs instead of the profiler.
