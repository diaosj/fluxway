# Placement Recipe

Placement Recipe 只告诉 Fluxway 两件事：哪些调度结果必须成立，以及多个合法结果中先优化什么。每个 worker 需要多少资源由对应 SPMD 的 `ResourceProfile` 声明，不在 Recipe 中重复保存。

可以把 Recipe 想成一张经过专家审核的“调度说明书”。普通算法工程师选择说明书，不需要自己设计求解算法。

如果只使用专家准备好的 Recipe，阅读“直接使用现成 Recipe”和“怎么看运行结果”即可。其余章节供需要编写 Recipe 的用户阅读。

## 开始前先认识几个词

SPMD 是一组运行同一段程序的 worker。每个 worker 有 rank，可以把 rank 理解为组内编号 0、1、2……

node 是一台 Ray 机器。Recipe 不写机器 IP，而是写 worker 与机器之间必须满足的关系，让 Fluxway 根据当前集群求解。

profile 是一个 worker 的资源需求。例如 `{"CPU": 2, "GPU": 1}` 表示每个 worker 需要 2 个 CPU 和 1 张 GPU。

Graph 是作业中所有 SPMD 及其连接关系的清单。亲和性表示“两组 worker 希望使用有关联的机器”，不是数据流量大小。

constraint 是必须满足的规则。objective 是多个合法方案之间的比较顺序。Fluxway 不会为了改善 objective 而违反 constraint。

## 直接使用现成 Recipe

普通算法工程师把 Graph 和已发布的 Recipe 一起交给 Controller。两者在同一次 Ray actor 创建中传输，不需要先转换成 JSON 或其他中间格式：

<!-- test:placement-controller-entry -->
```python
from inspect import signature

from fluxway import start


async def start_training(graph, recipe):
    return await start(graph, placement_recipe=recipe)


assert "graph" in signature(start).parameters
assert "placement_recipe" in signature(start).parameters
```

Fluxway 会检查 Graph、worker 数量、资源和亲和规则。没有合法方案时直接抛出异常，不会偷偷改用另一种调度策略。

因此普通使用者只需要确认两件事：Recipe 名称和版本是否正确，运行日志中的最终状态是否为 `READY`。

## 第一个小 Recipe

下面只有两个 SPMD。先只看它们的机器关系；资源在创建这两个 SPMD 时分别声明。

<!-- test:placement-first-recipe -->
```python
from fluxway import (
    PlacementRecipe,
    equal,
    lexicographic,
    max_min,
    occupancy,
    support,
    workers,
)


class GpuTraining: ...
class CpuPreprocess: ...


gpu = workers(GpuTraining)
cpu = workers(CpuPreprocess)

first_recipe = PlacementRecipe(
    recipe_id="first-recipe",
    version="1.0.0",
    constraints=(
        equal(support(cpu), support(gpu)),
    ),
    objective=lexicographic(
        max_min(occupancy(cpu)),
    ),
)

assert first_recipe.constraints[0].kind == "equal"
```

`equal(support(cpu), support(gpu))` 表示两组 worker 必须使用完全相同的机器。

`max_min(occupancy(cpu))` 表示在这些机器之间尽量均匀分布 CPU worker。constraint 决定什么方案合法，objective 决定合法方案中更喜欢哪一个。

Recipe 中的两个主要字段可以这样读：

| 字段 | 问题 | 本例答案 |
|---|---|---|
| `constraints` | 哪些规则绝不能违反？ | 两组 worker 使用相同机器 |
| `objective` | 多个合法方案中选哪个？ | CPU worker 尽量均匀分布 |

## 怎样选择 worker

`workers(S)` 选择 Graph 中类型为 `S` 的全部实例。这里的类型就是 Python class。若同一个 Reader class 在 Graph 中出现多次，它们都会被选择。

`instance("name")` 只选择 Graph 中叫这个名字的一个实例。`rank(A, 0)` 从 A 中选择每个实例的 0 号 worker。

<!-- test:placement-selectors -->
```python
from fluxway import instance, rank


all_gpu_workers = workers(GpuTraining)
one_gpu_instance = instance("Dense")
all_gpu_rank_zero = rank(all_gpu_workers, 0)

assert all_gpu_workers.kind == "workers"
assert one_gpu_instance.spmd_id == "Dense"
assert all_gpu_rank_zero.kind == "rank"
```

`union(A, B, ...)` 可以把多个 worker 选择器合并。重复选择到的 worker 只保留一次。

## support 为什么会出现

worker set 是 worker 名单，node set 是机器名单。亲和性比较的是机器，所以要用 `support` 把 worker 名单转换成实际使用的机器名单。

假设 A 的 worker 分布是 `{机器 0: 2, 机器 1: 1}`：

```text
occupancy(A) = {机器 0: 2, 机器 1: 1}
support(A) = {机器 0, 机器 1}
```

support 不保留每台机器上的 worker 数。因此“使用相同机器”和“按相同数量分布”是两件不同的事。

在底层数学语言中，`support` 不能省略，因为 worker 名单和机器名单是两种不同的值。现成 Recipe 已经写好了这一层，普通使用者不需要重复填写。

更完整的解释见 [Placement Mathematics: support 的准确含义](placement_mathematics.md#support-的准确含义)。

## 常用亲和规则

<!-- test:placement-constraints -->
```python
from fluxway import cardinality, disjoint, subseteq


same_machines = equal(support(cpu), support(gpu))
cpu_only_on_gpu_machines = subseteq(support(cpu), support(gpu))
separate_machines = disjoint(support(cpu), support(gpu))
exactly_two_machines = equal(cardinality(support(cpu)), 2)

assert same_machines.kind == "equal"
assert cpu_only_on_gpu_machines.kind == "subseteq"
assert separate_machines.kind == "disjoint"
assert exactly_two_machines.kind == "equal"
```

`equal(support(A), support(B))`：A 和 B 使用完全相同的机器。

`subseteq(support(A), support(B))`：A 使用的机器必须都承载 B，但 B 可以额外使用其他机器。

`disjoint(support(A), support(B))`：A 和 B 不能共用机器。

`equal(cardinality(support(A)), 2)`：A 必须使用恰好两台机器。

例如 A 使用 `{机器 0}`，B 使用 `{机器 0, 机器 1}`。A 是 B 的子集，但两者并不相等。

如果业务代码同时要求“A 的机器都在 B 中”和“A、B 的机器数量相同”，可以直接写 `equal(support(A), support(B))`。集群机器总是有限的，所以两种写法含义完全相同。

## 均分还是集中

同一个亲和 constraint 可以搭配不同 objective。

<!-- test:placement-balance-and-pack -->
```python
from fluxway import maximize_majorization


balanced = lexicographic(max_min(occupancy(cpu)))
concentrated = lexicographic(maximize_majorization(occupancy(cpu)))

assert balanced.objectives[0].kind == "max_min"
assert concentrated.objectives[0].kind == "maximize_majorization"
```

`max_min(occupancy(A))` 会先照顾 worker 最少的机器，倾向均分。

`maximize_majorization(occupancy(A))` 会先增加 worker 最多的机器，倾向集中。

例如 4 个 worker 放到两台机器：均分更喜欢 `[2, 2]`，集中更喜欢 `[4, 0]`。方括号里的两个数字分别是两台机器上的 worker 数。

`AFFINITY_FIRST_V1` 是集中优先的通用目标。它还会依次处理弹性资源公平分配、共同覆盖机器数、同机 worker 配对数和无法继续使用的零碎资源。

## 完整六 SPMD 示例

下面对应两台 8 卡 GPU 机器和若干 CPU 机器的训练场景。

<!-- test:placement-recipe -->
```python
from fluxway import AFFINITY_FIRST_V1, ResourceScale


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
    constraints=(
        subseteq(support(S3), support(S0)),
        subseteq(support(S4), support(S1)),
        subseteq(support(rank(S5, 0)), support(S0)),
        subseteq(support(rank(S5, 1)), support(S1)),
    ),
    objective=AFFINITY_FIRST_V1,
)

assert len(standard_training.constraints) == 4
assert S0.spmd.spmd_class is SPMD0
```

每个 `ResourceProfile` 在创建对应 Graph SPMD 实例时传入。Recipe 不再保存第二份资源映射。

SPMD0 和 SPMD1 是 GPU SPMD。SPMD3、SPMD4 和 SPMD5 通过 support 规则使用相应 GPU SPMD 已经占用的机器。

SPMD2 没有亲和规则。标准 objective 会先完成亲和 SPMD 的弹性分配，再把剩余 CPU 分给 SPMD2。

两台 GPU 机器分别有 8 张卡时，一个集中结果是：

```text
GPU 机器 A: SPMD1 × 8
GPU 机器 B: SPMD1 × 4 + SPMD0 × 4
```

因此 SPMD3 只能去机器 B。SPMD4 有一个 worker 时更喜欢机器 A；有两个 worker 时会优先让 A、B 各有一个。SPMD5 的 rank 0 跟随 SPMD0，rank 1 跟随 SPMD1。

## 资源怎么写

`ResourceProfile` 是一个 SPMD 实例中每个 worker 的资源声明，也是调度器、Ray actor 和运行报告共用的唯一资源事实。资源名直接使用 Ray resource key，普通 Ray 集群不需要平台 catalog。

Builder 在 `build()` 时把 SPMD class、mesh 和 `ResourceProfile` 一起冻结为 `CompiledSPMDDeclaration`，并放入 `CompiledGraph`。placement、worker graph view 和运行报告引用同一个 declaration；build 后再改原始 Graph node，不会改变已经编译的资源或亲和身份。

<!-- test:placement-resource-vector -->
```python
from decimal import Decimal
from fluxway import ResourceProfile


ordinary = ResourceProfile(
    base={
        "CPU": 2,
        "GPU": 1,
        "memory": 4 * 1024**3,
        "object_store_memory": 1024**3,
        "gpu_memory:H100": 40 * 1024**3,
    }
)
assert ordinary.base["GPU"] == 1
assert ordinary.minimum == ordinary.base
```

`CPU`、`GPU`、`memory` 和 `object_store_memory` 是 Fluxway 直接映射的 Ray actor 资源。显存没有通用的 Ray 原生 key；这里的 `gpu_memory:H100` 是 custom resource，只有在 `ray start --resources` 已经暴露同名资源时才能使用。网卡、TPU、其他 accelerator 或拓扑能力也用 Ray 已暴露的任意 custom resource 名表示。

`base` 是每个 worker 必须获得的资源，也就是 `minimum`。固定资源没有 `scale`，所以 `maximum` 与 `minimum` 相同。

`ray_component_limits` 是该 worker 触发的 Ray 原生组件的软资源上限，例如 Ray Data 的读、shuffle、join 或普通 Ray task。它参与 Fluxway placement 的容量核算，避免 Graph 声明超过集群可承载范围；但它不进入 Ray actor PlacementGroup bundle，也不表示 Fluxway 已经为这些组件预留了 node-local 资源。实际执行仍由 Ray 原生 scheduler 调度到集群自由资源上。

```python
metrics = ResourceProfile(
    base={"CPU": 1},
    ray_component_limits={"CPU": 2},
)
assert metrics.base["CPU"] == 1
assert metrics.ray_component_limits["CPU"] == 2
```

`ResourceScale.step` 是每次额外分配给每个 worker 的资源。`max_steps` 是最多增加多少次，`weight` 越大，公平分配时优先级越高。

每个资源的上限都按同一个公式计算：`maximum = base + step × max_steps`。没有写在 `step` 中的资源不会随 scale 增加。一个 step 中可以同时写 CPU、GPU、内存、显存和 custom resource，它们会按这个固定比例一起增长。

<!-- test:placement-resource-scale -->
```python
from fluxway import ResourceScale


elastic = ResourceProfile(
    base={"CPU": 1, "memory": 2 * 1024**3},
    scale=ResourceScale(
        step={"CPU": Decimal("0.5"), "memory": 1024**3},
        max_steps=2,
    ),
)
assert elastic.minimum["CPU"] == 1
assert elastic.maximum["CPU"] == 2
assert elastic.maximum["memory"] == 4 * 1024**3
assert elastic.scale is not None and elastic.scale.weight == 1
```

这个例子表示每个 worker 至少使用 1 CPU 和 2 GiB 内存，最多增加两次；上限是 2 CPU 和 4 GiB 内存。`weight` 默认是 1，专家才需要修改它。资源数量接受 `int` 或精确的 `Decimal`，不接受不精确的 Python `float`。

有多种 accelerator 时，使用 Ray 已经暴露的 custom resource：

<!-- test:placement-custom-accelerator -->
```python
h100 = ResourceProfile(
    base={"GPU": 1, "accelerator_type:H100": Decimal("0.001")}
)
assert h100.base["accelerator_type:H100"] == Decimal("0.001")
```

如果 Ray 只暴露 `GPU`，Fluxway 无法知道它是 H100、L4 还是其他型号。Recipe 不能约束系统看不见的信息。

精确调度还要求该作业独占 SPMD profiles 请求的资源。普通 Ray API 只给出全局空闲量，不提供每台机器的空闲明细。

如果已有其他任务占用同类资源，Fluxway 会抛出 `PlacementContractError`，不会猜测资源在哪台机器，也不会退回贪心调度。

## 怎么看运行结果

日志按下面顺序出现：

```text
placement.recipe.selected       选中了哪个 Recipe
placement.capacity.snapshot     看见了多少机器和资源
placement.problem.created       已从 SPMD 定义和 Recipe 建立求解问题
placement.program.complexity    预计模型规模
placement.constraint.validated  规则已经编译
placement.objective.started     开始一层优化
placement.objective.completed   该层得到精确最优值
placement.solution.resolved     得到 worker 到机器的方案
placement.pg.waiting            等待 Ray 预留资源
placement.pg.ready              Ray 已预留资源
placement.worker.attested       worker 已报告真实机器
placement.summary               唯一的最终摘要
```

`AllocationReport.status` 常见值如下：

| 状态 | 含义 |
|---|---|
| `PLANNED` | 已求出方案，还没有完成 Ray 资源预留 |
| `PG_READY` | PlacementGroup 已原子预留成功 |
| `READY` | 所有 worker 的真实机器与方案一致 |
| `FAILED` | 任一步骤失败，报告中有 error |

报告还保存每个 worker 的机器、每个 SPMD 的 occupancy/support、每条 constraint 的真假和每层 objective 的最优值。

## 模型过大时（专家内容）

Fluxway 在创建求解模型前估算规模。默认上限是 1024 个 objective 分量、32768 个排序 comparator、2048 次 solve、500000 个变量和 1000000 条约束。

普通用户可以跳过本节，不需要调整这些数。专家确认机器内存和求解时间足够后，可以显式提高：

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
assert runtime.max_model_variables == 750000
```

这些上限不能关闭。超过上限会抛出 `PlacementContractError`，而不是先创建巨大模型再耗尽内存。

## 出错时看哪里

| 异常 | 含义 | 首先检查 |
|---|---|---|
| `PlacementContractError` | 输入含义不一致或规模超限 | Graph、资源名、数字精度、复杂度日志 |
| `PlacementInfeasibleError` | 没有任何方案能满足全部规则 | support 关系、节点数、资源总量 |
| `PlacementSolveTimeoutError` | 没能在时限内证明最优 | 模型规模和 timeout |
| `PlacementGroupTimeoutError` | Ray 没能及时预留全部资源 | 集群空闲资源 |
| `PlacementAttestationError` | worker 实际机器与方案不同 | Ray 节点和 PlacementGroup |

Fluxway 不会自动放松 constraint，不会切换 PACK/SPREAD，也不会只重启 collective SPMD 的一个 rank。

如果业务需要故障恢复，必须由未来的 checkpointed whole-job gang reconstruction 协议重建完整通信组、拓扑和状态。
