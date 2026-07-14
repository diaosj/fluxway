# Placement Mathematics

本文只定义 `PlacementRecipe` 的形式语义和求解边界。配置示例、运行状态与错误处理见 [Placement Recipe](placement_recipe.md)。

## 基本对象

设 `W` 为有限 worker 集合，`N` 为 `ray.nodes()` 返回的存活节点集合：

```text
x[w,n] ∈ {0,1}  worker w 是否分配到节点 n
q[w,r]           worker w 对资源 r 的需求
C[n,r]           节点 n 的资源 r 容量
```

每个数学程序自动包含：

```text
∀w∈W: Σ[n∈N] x[w,n] = 1
∀n∈N,r: Σ[w∈W] x[w,n]q[w,r] ≤ C[n,r]
```

对任意 `A ⊆ W`：

```text
occupancy(A)[n] = Σ[w∈A] x[w,n]
support(A) = {n∈N | occupancy(A)[n] > 0}
```

因此 `subseteq(support(A), support(B))` 等价于 `supp(occupancy(A)) ⊆ supp(occupancy(B))`。

若 Graph 中多个实例具有同一 SPMD class `S`，则：

```text
workers(S) = ⋃[class(j)=S] workers(j)
rank(workers(S), i) = ⋃[class(j)=S] {w∈workers(j) | rank(w)=i}
instance(j) = workers(j)
```

class 表达式用于跨实例统一约束，`instance(j)` 用于只选择一个 Graph 实例。

这是公开语义。编译器会把没有被 `rank()`/`ranks()` 引用、且没有独立 driver resource 的同一 SPMD 同质 workers 聚合为一个整数 cohort count；被 rank 表达式观察的 worker 保持独立变量。该变换保持上述 `x`、occupancy 和 support 的解集合不变，把常见模型的 assignment 变量数从 `|W||N|` 降为 `|cohort||N|`。

## AST 与序列化

所有节点均为 frozen Pydantic discriminated AST。序列化结果保存 `kind` 字段，可以稳定生成 digest，也可以在 Ray actor 之间传输。下表中的 `A/B` 表示嵌套 AST，`R` 表示 `{"kind":"affinity_constraints"}`；字段没有省略的隐式默认语义。

| Python | 数学类型 | JSON shape |
|---|---|---|
| `workers(S)` | worker set | `{"kind":"workers","spmd":{"module":m,"qualname":q}}` |
| `instance(id)` | worker set | `{"kind":"instance","spmd_id":id}` |
| `rank(A,i)` | worker set | `{"kind":"rank","source":A,"index":i}` |
| `ranks(A,I)` | worker set | `{"kind":"ranks","source":A,"indices":I}` |
| `union(A,...)` | worker/node set | `{"kind":"worker_union|node_union","items":[A,...]}` |
| `all_workers()` | worker set | `{"kind":"all_workers"}` |
| `incident_spmds(R)` | worker set | `{"kind":"incident_spmds","relations":R}` |
| `isolated_spmds(R)` | worker set | `{"kind":"isolated_spmds","relations":R}` |
| `occupancy(A)` | vector | `{"kind":"occupancy","workers":A}` |
| `target_occupancy(R)` | vector | `{"kind":"target_occupancy","relations":R}` |
| `support(A)` | node set | `{"kind":"support","workers":A}` |
| `intersection(A,...)` | node set | `{"kind":"node_intersection","items":[A,...]}` |
| `difference(A,B)` | node set | `{"kind":"node_difference","left":A,"right":B}` |
| `cardinality(A)` | scalar | `{"kind":"cardinality","nodes":A}` |
| `resource_usage(A,r)` | scalar | `{"kind":"resource_usage","workers":A,"resource":r}` |
| `resource_capacity(A,r)` | scalar | `{"kind":"resource_capacity","nodes":A,"resource":r}` |
| `allocated_quanta(A)` | scalar | `{"kind":"allocated_quanta","workers":A}` |
| `weighted_dominant_share(A)` | vector | `{"kind":"weighted_dominant_share","workers":A}` |
| `affinity_coverage(R)` | scalar | `{"kind":"affinity_coverage","relations":R}` |
| `co_location_mass(R)` | scalar | `{"kind":"co_location_mass","relations":R}` |
| `resource_fragmentation()` | scalar | `{"kind":"resource_fragmentation"}` |
| `subseteq(A,B)` | constraint | `{"kind":"subseteq","left":A,"right":B}` |
| `disjoint(A,B)` | constraint | `{"kind":"disjoint","left":A,"right":B}` |
| `equal(A,B)` | constraint | `{"kind":"equal","left":A,"right":B}` |
| `less_equal(a,b)` | constraint | `{"kind":"less_equal","left":a,"right":b}` |
| `all_of(c,...)` | constraint | `{"kind":"all_of","items":[c,...]}` |
| `any_of(c,...)` | constraint | `{"kind":"any_of","items":[c,...]}` |
| `negate(c)` | constraint | `{"kind":"negate","item":c}` |
| `minimize(a)` | objective | `{"kind":"minimize","expression":a}` |
| `maximize(a)` | objective | `{"kind":"maximize","expression":a}` |
| `max_min(a)` | objective | `{"kind":"max_min","expression":a}` |
| `maximize_majorization(v)` | objective | `{"kind":"maximize_majorization","expression":v}` |
| `lexicographic(o,...)` | objective list | `{"kind":"lexicographic","objectives":[o,...]}` |

<!-- test:placement-ast-json -->
```python
from fluxway import support, workers


class ExampleSPMD: ...


ast = support(workers(ExampleSPMD))
payload = ast.model_dump(mode="json")
assert payload == {
    "kind": "support",
    "workers": {
        "kind": "workers",
        "spmd": {
            "module": "builtins",
            "qualname": "ExampleSPMD",
        },
    },
}
```

Normalizer 递归处理 AST。`union`、`intersection`、`all_of`、`any_of` 的参数以及顶层 constraint 合取会排序、展开同类嵌套并去重；`equal` 和 `disjoint` 的两侧按 canonical JSON 排序。`lexicographic` 的 objective 顺序永远不会重排。digest 只从 normalized JSON 计算。

## 字典序 Objective

`lexicographic(o0, o1, ...)` 逐阶段求精确最优。阶段 `i` 完成后，`oi` 的最优值会成为后续阶段的等式约束。因此后续目标不能牺牲任何前序目标。

`AFFINITY_FIRST_V1` 的顺序是：

1. affinity graph 中 incident 且含 `ResourceScale` 的 SPMDs 做 topology-aware weighted DRF。
2. isolated 且含 `ResourceScale` 的 SPMDs对第一阶段真实剩余资源做 weighted DRF。
3. 最小化 `cardinality(support(all_workers()))`。
4. 按 occupancy 降序向量逐项最大化 affinity target occupancy 的 majorization refinement。
5. 最大化 affinity coverage。
6. 最大化 co-location mass。
7. 最小化 resource fragmentation。

多个目标不会通过 Big-M 合成。

`affinity_coverage(R)`、`co_location_mass(R)` 与 `resource_fragmentation()` 是真正的 scalar：既可用于 `minimize`/`maximize`，也可用于 `equal`/`less_equal` hard constraint，并在 worker node-ID attestation 后重新求值。`weighted_dominant_share(A)` 对每个可伸缩 SPMD 产生一个分量，因此是 vector。所有公开 vector 都可用于 `max_min` 或 `maximize_majorization`；两种 objective 分别按升序 leximin 和降序 majorization refinement 锁定分量。

resource scalar 不使用模型内部 unit 直接与整数比较。对资源 `r`，若模型 quantum 为 `g[r]` 个 Ray resource unit，则编译器把值表示为精确分数 `model_value × g[r] / 10000`，约分后交叉相乘；attestation 使用等价的 `Decimal` 物理数量。整数常量是无量纲字面量，可以作为某一资源的阈值；两个带量纲 scalar 只有维度相同才能比较，例如 CPU usage 可与 CPU capacity 比较，CPU usage 与 GPU usage 会抛出 `PlacementContractError`。

对 SPMD `s` 的一个 scale quantum，设资源增量为 `d[s,r]`，当前阶段可分配容量为 `E[r]`，权重为 `p[s]`：

```text
share[s] = max_r d[s,r] / (E[r] p[s]) · quanta[s]
```

所有 Decimal 先转成整数 resource unit，再取有理数分母的最小公倍数。求解器对升序 `share` 向量逐项最大化并锁定，因此是离散 leximin，不是浮点近似。`ObjectiveResult.optimum_value / value_scale` 还原精确 share 向量。

整数化不是对所有 resource 固定乘 `10000`。对每个 resource `r`，编译器收集 Recipe、driver 和 snapshot 中的精确 Decimal，以 Ray 的 `0.0001` 精度转成整数后求最大公约数 `g[r]`，模型使用 `quantity/g[r]`。整数 byte 的 memory 因而不会额外放大 `10000`。所有变量上界、线性系数、乘积与 DRF 公分母在建模前检查 CP-SAT int64 范围，溢出直接抛出 `PlacementContractError`。

对 `maximize_majorization(v)`，编译器通过 min/max comparator network 得到 `sort(v, descending=True)`，再逐项最大化并锁定。这个确定性全序 refinement 保留 majorization 的集中方向，也让后续 objective 得到稳定的可复现可行域。

令 `U` 为 Recipe 中所有不同的 positive base/scale resource vector。对每个节点引入整数 `y[n,u]`，表示残余资源还能组成多少个完整的合法单位 `u`。fragmentation 定义为：

```text
Σ[u∈U] y[n,u]u[r] ≤ C[n,r] - usage[n,r]
remainder[n,r]
  = C[n,r] - usage[n,r] - Σ[u∈U] y[n,u]u[r]
q[r] = gcd({u[r] | u∈U, u[r]>0})
resource_fragmentation() = Σ[n,r] remainder[n,r] / q[r]
```

`y` 与 placement 位于同一个 CP-SAT 模型。各 resource 的 remainder 先按自己的最小 allocation quantum `q[r]` 归一化，再用这些分母的最小公倍数精确整数化。因此一个无法使用的 GPU 与一个完整 memory allocation quantum 量纲相同，不会因为 memory 使用 byte 表示而支配 accelerator。`ObjectiveResult.value_scale` 保存该公分母；报告使用每个 resource 自己的 `g[r]` 恢复精确 Decimal quantity。

## 六 SPMD 推导

两台节点各有八张 GPU，SPMD0 需要四张，SPMD1 需要十二张。最小节点数是二。对每个 affinity target 最大化 occupancy 集中度后，同构解为：

```text
GPU A: SPMD1 × 8
GPU B: SPMD1 × 4 + SPMD0 × 4
```

对关系 `R=(A,B)`：

```text
affinity_coverage(R)
  = Σ[n∈N] 1[occupancy(A)[n]>0 ∧ occupancy(B)[n]>0]

co_location_mass(R)
  = Σ[n∈N] occupancy(A)[n]occupancy(B)[n]
```

SPMD4 有一个 worker 时 coverage 恒为一，mass 选择 SPMD1 occupancy 为八的节点。SPMD4 有两个 workers 时 coverage 先达到二，因此两台节点各一个。SPMD5 的两个 ranks 由两个独立 subset constraint 分别关联 SPMD0 和 SPMD1。没有位置约束的 SPMD2 是 isolated vertex，在 incident SPMDs 完成 scale 后使用剩余 CPU。

当六个 CPU 节点各有 16 CPU、SPMD3 的 `max_steps=2`、SPMD2 的 `max_steps≥15` 时，离散 DRF 与单节点容量共同得到：

```text
allocated_quanta(SPMD3) = 2
allocated_quanta(SPMD2) = 15

support(SPMD3) = support(SPMD0)
support(SPMD4) ⊆ support(SPMD1)
support(rank(SPMD5,0)) ⊆ support(SPMD0)
support(rank(SPMD5,1)) ⊆ support(SPMD1)
```

`AllocationReport` 同时保存这组 normalized constraints 的逐条真值、计划 assignments、attestation 得到的实际 node IDs，以及按 SPMD 汇总的 occupancy/support。

## 求解边界

一般化 placement、bin packing 和带资源约束的图放置是 NP-hard。数学语言定义有限问题和严格目标，不承诺多项式求解时间。

Fluxway 使用 OR-Tools CP-SAT，固定单 search worker、seed、变量顺序和逐阶段 objective。quanta、worker-node assignment、资源容量与 topology constraint 位于同一个有限整数模型。majorization/leximin 使用 `O(K log²K)` Batcher sorting network。令 `P=2^ceil(log2 K)`，实际 comparator 数为 `P log2(P)(log2(P)+1)/4`；向量还需要 `max(1,K)` 次精确 solve。模型构造前会计算全部 objective 分量、comparator、solve、模型变量与模型约束，并按 `PlacementRuntimeConfig` 的显式上限准入。例如 100 个 target×100 个节点会在创建 860160 个 comparator 前被拒绝；3000 个 placement units×3000 个节点会在创建九百万 assignment variables 之前被拒绝。

整个 `resolve()` 从编译、可行性检查到全部 objective 共享一个 deadline，每次 solve 只得到剩余预算。每个标量或向量分量只有在 `OPTIMAL` 后才作为等式锁定；`UNKNOWN` 和 timeout 都是错误，不能解释为不可行，也不能返回启发式结果。

Ray adapter 只能表达 Ray 已暴露的 node、CPU、GPU、memory 和 custom resources。未暴露的 NUMA、NVLink、rack 或 accelerator 型号不是可观察变量。普通 Ray API 只提供逐节点总容量和全局空闲量，不能提供逐节点空闲账本；精确 placement 因而要求 Recipe 请求的 resources 在求解期间独占，若 `ray.available_resources()` 与存活节点总容量不一致就直接拒绝求解。
