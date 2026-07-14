# Placement Recipe

本文是 placement 的用户入口，说明如何选择或编写 `PlacementRecipe`，以及如何读取运行结果。AST、objective 和求解器的形式语义见 [Placement Mathematics](placement_mathematics.md)；模块所有权见 [Architecture](ARCHITECTURE.md#placement)。

## 使用 Recipe

<!-- test:placement-controller-entry -->
```python
from inspect import signature

from fluxway import create_controller


async def create_training_controller(graph):
    return await create_controller(
        graph,
        placement_recipe=standard_training,
    )


assert "placement_recipe" in signature(create_controller).parameters
```

运行时会校验 Graph 中的 SPMD class、world size、rank 和 Ray 资源。配置不可行、求解超时、Ray 资源快照变化或实际节点与解不一致都会抛出异常，不会改用另一种调度策略。

## 完整示例

<!-- test:placement-recipe -->
```python
from decimal import Decimal

from fluxway import (
    AFFINITY_FIRST_V1,
    PlacementRecipe,
    ResourceProfile,
    ResourceScale,
    cardinality,
    disjoint,
    equal,
    rank,
    subseteq,
    support,
    workers,
)


class SPMD0: ...
class SPMD1: ...
class SPMD2: ...
class SPMD3: ...
class SPMD4: ...
class SPMD5: ...


S0 = workers(SPMD0)
S1 = workers(SPMD1)
S2 = workers(SPMD2)
S3 = workers(SPMD3)
S4 = workers(SPMD4)
S5 = workers(SPMD5)

standard_training = PlacementRecipe(
    recipe_id="standard-training",
    version="1.0.0",
    profiles={
        SPMD0: ResourceProfile(base={"GPU": 1, "CPU": 1}),
        SPMD1: ResourceProfile(base={"GPU": 1, "CPU": 1}),
        SPMD2: ResourceProfile(
            base={"CPU": 1},
            scale=ResourceScale(
                step={"CPU": 1},
                max_steps=100,
                weight=1,
            ),
        ),
        SPMD3: ResourceProfile(
            base={"CPU": 1},
            scale=ResourceScale(
                step={"CPU": 1},
                max_steps=2,
                weight=1,
            ),
        ),
        SPMD4: ResourceProfile(base={"CPU": 1}),
        SPMD5: ResourceProfile(base={"CPU": 1}),
    },
    constraints=(
        subseteq(support(S3), support(S0)),
        subseteq(support(S4), support(S1)),
        subseteq(support(rank(S5, 0)), support(S0)),
        subseteq(support(rank(S5, 1)), support(S1)),
    ),
    objective=AFFINITY_FIRST_V1,
)

assert standard_training.recipe_id == "standard-training"
assert len(standard_training.constraints) == 4
assert len(standard_training.digest) == 64
```

## 选择 SPMD

`workers(S)` 选择 Graph 中 class 为 `S` 的全部实例，class 级约束因此会作用于每个实例。`instance("spmd_id")` 只选择一个命名实例；`rank(A, i)` 再从集合 `A` 中选择指定 rank。选择器的集合定义见 [Placement Mathematics](placement_mathematics.md#基本对象)。

<!-- test:placement-selectors -->
```python
from fluxway import instance


all_spmd5_workers = workers(SPMD5)
all_spmd5_rank_zero = rank(all_spmd5_workers, 0)
one_shared_instance = instance("shared-a")

assert all_spmd5_workers == S5
assert all_spmd5_rank_zero.kind == "rank"
assert one_shared_instance.spmd_id == "shared-a"
```

## 亲和与均分

<!-- test:placement-constraints -->
```python
independent = disjoint(support(S2), support(S0))
four_nodes = equal(cardinality(support(S2)), 4)
same_node = equal(support(rank(S0, 0)), support(rank(S1, 0)))

assert independent.kind == "disjoint"
assert four_nodes.kind == "equal"
assert same_node.kind == "equal"
```

- `subseteq(support(A), support(B))`：`A` 只能使用已经承载 `B` 的节点。
- `disjoint(support(A), support(B))`：`A` 与 `B` 不能使用相同节点。
- `equal(cardinality(support(A)), k)`：`A` 必须占用恰好 `k` 个节点。
- `equal(support(rank(A, i)), support(rank(B, j)))`：两个指定 rank 必须同机。

`support(A)` 是 `A` 在求解结果中实际占用的节点集合，不是候选节点列表。hard constraint 只规定合法位置；worker 在合法节点间如何分布由 objective 决定。下面的 Recipe 让 `S3` 使用与 `S0` 相同的节点，并优先最大化最小 occupancy，从而均分 `S3` workers：

<!-- test:placement-balanced-affinity -->
```python
from fluxway import lexicographic, max_min, occupancy


balanced_affinity = PlacementRecipe(
    recipe_id="balanced-affinity",
    version="1.0.0",
    profiles={
        SPMD0: ResourceProfile(base={"GPU": 1, "CPU": 1}),
        SPMD3: ResourceProfile(base={"CPU": 1}),
    },
    constraints=(
        subseteq(support(S3), support(S0)),
        equal(cardinality(support(S3)), cardinality(support(S0))),
    ),
    objective=lexicographic(max_min(occupancy(S3))),
)

assert balanced_affinity.objective.objectives[0].kind == "max_min"
```

`AFFINITY_FIRST_V1` 还会依次优化节点数、target occupancy 集中度、coverage、co-location mass 和 fragmentation，适合集中优先的通用 Recipe；它不表示均分。完整顺序见 [Placement Mathematics](placement_mathematics.md#字典序-objective)。

## 资源

资源名直接使用 Ray resource key。普通 Ray 集群无需平台 catalog：

<!-- test:placement-resource-vector -->
```python
ordinary = ResourceProfile(
    base={"CPU": 2, "GPU": 1, "memory": 4 * 1024**3}
)
assert ordinary.base["GPU"] == 1
```

区分多种 accelerator 时使用 Ray 已暴露的 custom resource：

<!-- test:placement-custom-accelerator -->
```python
h100 = ResourceProfile(
    base={"GPU": 1, "accelerator_type:H100": Decimal("0.001")}
)
assert h100.base["accelerator_type:H100"] == Decimal("0.001")
```

如果 Ray 只暴露 `GPU`，Fluxway 无法推导它是 H100、L4 或其他型号。请求不可观察的维度会失败。

`ResourceScale` 只增加每个 worker 的资源，不改变 DeviceMesh world size。`step` 是不可分割的分配单位，`weight` 是同一分配阶段的 weighted dominant-share 权重。没有 `scale` 的 SPMD 不进入 DRF 向量，但仍然参与基础资源与拓扑约束。

精确 Recipe 调度要求它请求的 resource 在求解期间由该作业独占。普通 `ray start` 集群可以直接使用，不需要平台 label 或 catalog；但若同一资源已经被其他 Ray task/actor 占用，Fluxway 会因缺少逐节点空闲账本而抛出 `PlacementContractError`，不会猜测占用位置或等待后改用另一种放置。

## 分配状态

Controller 依次记录：

```text
placement.recipe.selected
placement.capacity.snapshot
placement.program.normalized
placement.program.complexity
placement.constraint.validated
placement.objective.started
placement.objective.completed
placement.solution.resolved
placement.pg.waiting
placement.pg.ready
placement.worker.attested
placement.summary
```

`Controller.get_resource_allocation_summary()` 返回当前 `AllocationReport`。`PLANNED` 表示精确解已经产生，`PG_READY` 表示 Ray 已原子预留 composite bundles，`READY` 表示全部 worker 的实际 node ID 已通过 attestation。报告包含 normalized Recipe AST、每条 constraint 的 AST 与真值、objective 最优值及 `value_scale`、每个 worker 的节点和资源、occupancy、support、elastic quanta、PG ID、fragmentation 及未使用资源。对 fragmentation，`resource_fragmentation_score / resource_fragmentation_scale` 是跨 resource 的无量纲 allocation-quantum 数，`resource_fragmentation` 字段仍保存各 resource 的实际 Decimal 数量。

`PlacementRuntimeConfig` 在构造 CP-SAT 模型前执行复杂度准入，默认最多允许 1024 个 objective 分量、32768 个 Batcher sorting comparator、2048 次字典序 solve、500000 个模型变量和 1000000 条模型约束。五个上限都可显式调高，但不能关闭；超过任一上限会记录 `placement.program.complexity` 并抛出 `PlacementContractError`。变量/约束估算覆盖 assignment matrix、同构节点 symmetry、elastic products、fragmentation、AST lowering、coverage/mass、WDRF 和 sorting network。

<!-- test:placement-complexity-limits -->
```python
from fluxway import PlacementRuntimeConfig


runtime = PlacementRuntimeConfig(
    max_objective_components=2048,
    max_sorting_comparators=70000,
    max_lexicographic_solves=4096,
    max_model_variables=750000,
    max_model_constraints=1500000,
)
assert runtime.max_objective_components == 2048
```

## 错误

- `PlacementContractError`：Recipe、Graph、资源精度、复杂度准入或 Ray snapshot 契约不一致。
- `PlacementInfeasibleError`：不存在满足全部硬约束和资源容量的解。
- `PlacementSolveTimeoutError`：CP-SAT 未在时限内证明最优或不可行。
- `PlacementMaterializationError`：Ray 不能表达或物化求解结果。
- `PlacementGroupTimeoutError`：Ray 无法在时限内原子预留所有 composite bundles。
- `PlacementAttestationError`：worker 实际 node ID 与求解结果不同。

这些异常不会触发 PACK/SPREAD 切换、约束放松、重新解释资源名或贪心放置。
worker 或 driver 错误也不会局部重启一个 rank 或 SPMD；在没有 checkpointed whole-job gang reconstruction 协议前，Controller 会直接进入失败清理。
