# Placement mathematics

This English companion mirrors [placement_mathematics_zh.md](placement_mathematics_zh.md).
It defines the solver semantics implemented by Fluxway placement modules.

`workers(S)` denotes the worker set for an SPMD, `support(S)` its node support,
and `occupancy(S)` its node occupancy. Constraints are hard predicates. An
objective is evaluated only after all constraints hold, and lexicographic
objectives are optimized in declaration order.

The placement compiler lowers the frozen expression tree into CP-SAT variables
and validates the materialized assignment independently. Resource profiles
come from compiled declarations; recipes do not maintain a second class-to-
resource mapping. Unsupported expressions, missing declarations, and invalid
topology relations raise directly.

The following sections provide executable API examples for common placement workflows. See [placement_recipe.md](placement_recipe.md) for detailed step-by-step guidance.

<!-- test:placement-controller-entry -->
```python
from inspect import signature
from fluxway import start


async def start_training(graph, recipe):
    return await start(graph, placement_recipe=recipe)


assert "graph" in signature(start).parameters
assert "placement_recipe" in signature(start).parameters
```
