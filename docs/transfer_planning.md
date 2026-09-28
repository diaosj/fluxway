# Transfer planning

## Planes and ownership

`CompiledGraph` owns operator and transfer declarations. `TensorTransfer.unit_routing`
optionally declares a complete-unit relation; it is independent of `is_source`.
`CompiledDependency` binds that declaration to the active producer and consumer
rank domains. `CompiledSendPlan` lowers the result into local sender instructions.

The data plane stages tensor components, creates transport frames and delivers them
with the existing receipt and coherence protocol. It does not allocate source work,
advance checkpoint cuts or choose which execution instance an operator runs.
`LogicalWorkLedger`, Controller admission and Dispatcher execution selection own
those respective decisions.

## Mathematical contract

For one execution instance `e`, a producer `p` emits a unit `U(e,p)`. Whole-unit
routing specifies a function `f: C -> P`. Consumer `c` receives exactly `U(e,f(c))`.
The compiled relation is `R = {(f(c),c) | c in C}`. Its inverse incidence tables are:

- `producers_by_target[c] = (f(c),)`;
- `consumers_by_source[p] = {c | f(c) = p}`.

The compiler checks both rank domains, single-producer completeness per target,
and coverage of every active producer. A bijection expresses a permutation;
repeated producer IDs express replication or broadcast. Missing producer coverage
is an error, not an implicit selection/drop policy. Multiple complete units cannot
silently become one tensor input at the same execution instance. Use distinct
execution instances or explicit input ports when units must remain separate.

`WholeUnitRouting((1, 0, 1, 0))` maps two producer units to four consumers. Consumers
0 and 2 receive producer 1; consumers 1 and 3 receive producer 0. Every destination
receives one complete payload. Its `input_sharding` does not request a second
implicit split after the explicit unit relation has been selected.

Without `unit_routing`, the existing sharding algebra computes fragment overlap
between `output_sharding` and `input_sharding`. Multiple producer fragments may
assemble into one target tensor. Content-addressed routing and receipt-based reverse
routing retain their existing semantics. Whole-unit routing and receipt routing on
the same transfer are rejected.

Receipt routing takes precedence over static shard geometry. Equal source and
target layouts do not authorize a direct route when the transfer follows a receipt.

## Execution groups

An `ExecutionGroup` contains ordered physical ranks and a normalized logical mesh.
Layout algebra uses logical group coordinates. Frame headers and coherence subjects
retain the physical sender rank. Invalid rank lookup raises.

Each `CompiledOperator.execution_layout` owns the immutable cycle and inverse rank
indexes. The graph compiler binds `OperatorSpec.execution_groups` to the physical
mesh once. Groups must have equal size, cover every physical rank exactly once,
and preserve the logical trailing mesh dimensions. Without an explicit group
declaration, the compiler constructs contiguous groups. Dispatcher, source planning,
control-message leaders, send plans and placement all consume this compiled layout;
runtime rank arithmetic does not define a second grouping.

For example, in a typical training workload with 16 Readers and 4 Inputs, the declaration forms the groups `(0,4,8,12)`,
`(1,5,9,13)`, `(2,6,10,14)` and `(3,7,11,15)`. Thus Readers 0 through 3 supply
Input 0 in successive steps, Readers 4 through 7 supply Input 1, and so on.
The placement engine requires each Reader to share its target Input's node. Explicit user
placement constraints must agree with the declared grouping or compilation/placement
fails; the runtime never silently changes the binding or adds cross-node routing.

Eight physical Readers can supply two Inputs using four groups of two Readers. Each
execution instance activates one group and routes its two complete microbatches to
the two Inputs. Over the cycle, each Input receives from four different Readers;
it does not concatenate those four microbatches into one execution instance.
The same complete-unit contract can connect ordinary non-source operators.

The topology graph declares the unit relation explicitly and derives
Reader/Input placement constraints from the compiled relation and execution cycle.
Affinity between downstream nodes (e.g., Input/Dense) is a separate placement constraint.

## Execution and memory

Compiled direct targets use the existing distributor and frame ABI. Complete-unit
relations and provably equivalent static chunk geometry can both lower to direct
targets.

Direct delivery avoids sorting, packing gather and multi-shard value concatenation.
For fragment routing, `sort_indices=None` means no reorder; an actual permutation
uses backend-native gather. Dynamic routing cannot be declared identity merely
because one observed batch happened to be ordered.


## Public declaration

```python
from fluxway.common.transfer import WholeUnitRouting

result = consumer.process(
    produced,
    unit_routing=WholeUnitRouting((1, 0)),
)
```

`Builder.add_func_edge` accepts the same `unit_routing` argument. It applies to each
mapped tensor port on that edge. Ordering-only and `local_only` edges reject it.
Custom unit-routing declarations implement `UnitRouting.bind`; the returned
`TransferRelation` is checked by the graph compiler before execution.
