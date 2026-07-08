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

Custom user scopes and events appear under:

```text
compute/user_custom/<lane>
```

Payload fields are correlated with the surrounding call:

- `op_id`
- `step`
- `rank`
- `lane_id`

User attributes are expanded as readable Perfetto debug annotations. The
internal `attrs_id` is not shown.

## Rules

- Do not add a decorator around the whole operator. Fluxway already records that
  as `user_callable.execute`.
- Use `scope` for code regions that take time.
- Use `event` for point-in-time facts such as a cache hit, sampled branch, or
  algorithm phase marker.
- Attribute values must be scalar. Put large tensors, arrays, dicts, lists, or
  object reprs in your own logs instead of the profiler.
