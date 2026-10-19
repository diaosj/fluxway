# Placement Recipe

This English companion mirrors [placement_recipe_zh.md](placement_recipe_zh.md).
A recipe states hard placement constraints and the order in which valid
solutions are optimized. Worker resource demand remains owned by each SPMD's
`ResourceProfile`.

## Using a published recipe

Pass the graph and recipe to `fluxway.start`. Graph and recipe are transferred
as native Python objects; no JSON reconstruction protocol is involved. Fluxway
validates worker counts, resources, topology, constraints, and objectives. An
invalid problem raises instead of silently selecting a fallback strategy.

<!-- test:placement-controller-entry -->
```python
from inspect import signature
from fluxway import start


async def start_training(graph, recipe):
    return await start(graph, placement_recipe=recipe)


assert "graph" in signature(start).parameters
assert "placement_recipe" in signature(start).parameters
```

## Recipe ownership rules

Use `workers`, `support`, `equal`, `lexicographic`, `max_min`, and
`occupancy` to express relations. Do not duplicate resource profiles or rank
mapping in a recipe. The controller owns the placement group; managers and
drivers borrow the resulting rank-to-bundle mapping.

Each resolved NodeID receives a composite placement-group bundle pinned by Ray's
`ray.io/node-id` label selector. IP addresses do not substitute for node identity.
The materializer preserves the planner's CPU, GPU and custom accelerator resources;
Workers attest their actual node against the plan. Describe hardware differences
through resource profiles and recipe constraints. Use compatible software on every
node and shared checkpoint storage for recoverable multi-node jobs.

See [placement_mathematics.md](placement_mathematics.md) for solver semantics
and the Chinese companion for the complete parallel explanation.

## Dynamic resources and ResourceRegistry

While static hardware quotas (e.g., CPU, GPU, accelerators, memory) are declared in each SPMD's
`ResourceProfile`, runtime-bound resources (such as dynamic routing tables, device meshes, or rank
providers) are managed via `ResourceRegistry` (`fluxway.common.resource_registry`).
Operators and partition strategies reference these resources using `ResourceHandle(namespace=..., resource_id=...)`,
which the runtime resolves dynamically within the active registry context established by `bind_resource_registry`
or `set_active_resource_registry`, keeping placement recipes completely decoupled from physical runtime instances.
