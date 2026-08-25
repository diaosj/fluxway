# Placement Mathematics

本文解释 Fluxway placement 语言。读者只需要会读 Python，并理解“任务需要 CPU/GPU，机器有容量”。不要求学过集合论、线性规划或组合优化。

可以把 placement 想成“把带资源需求的工人安排到机器上”。Recipe 写的是规则，Fluxway 负责寻找满足规则的安排。

阅读顺序建议是：先看小例子，再看 operator 表，最后再看公式。公式只是精确记录前面的意思，不是使用 Recipe 的前置知识。

只想使用现成 Recipe，可以先读 [Placement Recipe](placement_recipe_zh.md)。本文用于回答“这条规则到底是什么意思”。

## 从一个小例子开始

假设一个 SPMD 有 3 个 worker，集群有机器 A 和机器 B。调度结果如下：

```text
机器 A: worker 0, worker 1
机器 B: worker 2
```

`occupancy` 记录每台机器上有几个 worker：

```text
occupancy = {机器 A: 2, 机器 B: 1}
```

`support` 只记录哪些机器被使用，不记录数量：

```text
support = {机器 A, 机器 B}
```

因此 occupancy 不同，support 仍可能相同。例如 `{A: 1, B: 2}` 和 `{A: 2, B: 1}` 的 support 都是 `{A, B}`。

## 先认识几个词

SPMD 是“一组运行同一段程序的 worker”。worker 是真正执行代码的进程或 actor；rank 是它在组内的编号，从 0 开始。node 就是一台 Ray 机器。

Graph 是这些 SPMD 及其数据关系的清单。每个 SPMD 声明自己的 `ResourceProfile`；Recipe 只记录哪些机器关系必须成立，以及怎样比较合法方案。创建 Controller 时，Graph 和 Recipe 在同一次 Ray actor 调用中直接传输。

“集合”可以先理解为一份不重复的名单。`{A, B}` 和 `{B, A}` 是同一份名单，名单中没有“第一个”元素。

| 类型 | 普通语言 | 例子 |
|---|---|---|
| worker set | 一组 worker 的不重复名单 | 某类 SPMD 的全部 worker |
| node set | 一组机器的不重复名单 | 实际承载这些 worker 的机器 |
| vector | 按机器或 SPMD 排列的一组数字 | `{A: 2, B: 1}` |
| scalar | 一个数字 | 使用了 2 台机器 |
| constraint | 必须为真的规则 | A 组不能使用 B 组之外的机器 |

`objective` 不是必须满足的规则，而是在所有合法方案中选择“更好方案”的比较方法。例如“机器数越少越好”。

`operator` 就是表里的一个函数名。它读取一些值，产生一个新的值或规则。每个 operator 都有固定的输入类型和输出类型。

## support 的准确含义

`support(A)` 的输入 `A` 是 worker set，输出是 node set。它表示“最终方案中，至少放了一个 A worker 的机器”。

```text
support(A) = {机器 n | 机器 n 上至少有一个 A worker}
```

如果用稍正式的写法，`occupancy(A)[n]` 表示 A 在机器 n 上的 worker 数：

```text
support(A) = {n | occupancy(A)[n] > 0}
```

`support` 是一个必要的类型转换。A 本身是 worker 名单，不能直接与机器名单比较。亲和性讨论的是“用了哪些机器”，所以必须先从 worker set 转成 node set。

`support` 不是候选机器列表，也不是平台 label。它是求解完成后由 placement 结果决定的集合。

下面三条规则容易混淆：

```text
subseteq(support(A), support(B))
```

含义：A 使用的每台机器上，至少有一个 B worker。它不要求 A 与 B 的 worker 数相同，也不要求一一配对。

```text
equal(support(A), support(B))
```

含义：A 和 B 使用完全相同的机器。它仍然不要求每台机器上的 worker 数相同。

```text
disjoint(support(A), support(B))
```

含义：A 和 B 没有共同使用任何机器。

对有限集合，下面两种写法严格等价：

```text
equal(support(A), support(B))
```

```text
all_of(
    subseteq(support(A), support(B)),
    equal(cardinality(support(A)), cardinality(support(B))),
)
```

第一种更短，也更直接。

## 调度器内部的基本规则

设 `W` 是全部 worker，`N` 是全部存活机器。`x[w,n]` 只有 0 或 1：1 表示 worker `w` 放在机器 `n`，0 表示没有放在那里。

每个 worker 必须且只能放到一台机器：

```text
对每个 worker w：所有机器上的 x[w,n] 相加等于 1
```

每台机器上所有 worker 的资源需求之和不能超过机器容量：

```text
对每台机器 n 和资源 r：已使用资源不超过机器容量
```

这两条规则由 Fluxway 自动加入，Recipe 作者不需要重复编写。

## AST 与运行时对象

Recipe 不是立即执行的 Python 条件。每个函数先创建一个不可修改的 AST 节点。AST 是 Abstract Syntax Tree 的缩写，可以理解为“按层次记录规则的数据对象”。

每个节点都有 `kind`，求解器根据它判断这是什么操作。Recipe 保留真实的 Python class 引用，并由 Ray 直接传给 Controller；Fluxway 不把 Recipe 转成 JSON，也不从字符串重新导入 class。下面逐个解释全部公开 operator。

表中的 `A/B` 表示前面已经构造好的表达式，`R` 表示 `affinity_constraints()`。

### 每一种 AST 节点保存什么

下面列出 placement 语言的全部 AST `kind`。`kind` 是对象中的类型标签，后面的字段是它保存的子表达式或值。看到一个运行日志里的 AST 时，可以直接按这张表逐层向内读。

| Python 写法 | AST 标签 | 保存的字段 |
|---|---|---|
| `workers(S)` | `kind="workers"` | `spmd`：Python class 引用 |
| `instance(id)` | `kind="instance"` | `spmd_id`：Graph 中的实例名 |
| `all_workers()` | `kind="all_workers"` | 无额外字段 |
| `rank(A,i)` | `kind="rank"` | `source=A`，`index=i` |
| `ranks(A,I)` | `kind="ranks"` | `source=A`，`indices=I` |
| worker `union(A,...)` | `kind="worker_union"` | `items`：worker set 名单 |
| `affinity_constraints()` | `kind="affinity_constraints"` | 无额外字段 |
| `incident_spmds(R)` | `kind="incident_spmds"` | `relations=R` |
| `isolated_spmds(R)` | `kind="isolated_spmds"` | `relations=R` |
| `occupancy(A)` | `kind="occupancy"` | `workers=A` |
| `artifact_occupancy(R)` | `kind="artifact_occupancy"` | `relations=R` |
| `weighted_dominant_share(A)` | `kind="weighted_dominant_share"` | `workers=A` |
| `support(A)` | `kind="support"` | `workers=A` |
| node `union(A,...)` | `kind="node_union"` | `items`：node set 名单 |
| `intersection(A,...)` | `kind="node_intersection"` | `items`：node set 名单 |
| `difference(A,B)` | `kind="node_difference"` | `left=A`，`right=B` |
| 整数 `k` | `kind="integer"` | `value=k` |
| `cardinality(A)` | `kind="cardinality"` | `nodes=A` |
| `resource_usage(A,r)` | `kind="resource_usage"` | `workers=A`，`resource=r` |
| `resource_capacity(A,r)` | `kind="resource_capacity"` | `nodes=A`，`resource=r` |
| `allocated_quanta(A)` | `kind="allocated_quanta"` | `workers=A` |
| `affinity_coverage(R)` | `kind="affinity_coverage"` | `relations=R` |
| `co_location_mass(R)` | `kind="co_location_mass"` | `relations=R` |
| `resource_fragmentation()` | `kind="resource_fragmentation"` | 无额外字段 |
| `subseteq(A,B)` | `kind="subseteq"` | `left=A`，`right=B` |
| `disjoint(A,B)` | `kind="disjoint"` | `left=A`，`right=B` |
| `equal(A,B)` | `kind="equal"` | `left=A`，`right=B` |
| `less_equal(a,b)` | `kind="less_equal"` | `left=a`，`right=b` |
| `all_of(c,...)` | `kind="all_of"` | `items`：全部规则 |
| `any_of(c,...)` | `kind="any_of"` | `items`：候选规则 |
| `negate(c)` | `kind="negate"` | `item=c` |
| `minimize(a)` | `kind="minimize"` | `expression=a` |
| `maximize(a)` | `kind="maximize"` | `expression=a` |
| `max_min(v)` | `kind="max_min"` | `expression=v` |
| `maximize_majorization(v)` | `kind="maximize_majorization"` | `expression=v` |
| `lexicographic(o,...)` | `kind="lexicographic"` | `objectives`：按优先级排列的目标 |

`PlacementRecipe` 本身保存 `recipe_id`、`version`、`constraints` 和 `objective`。它没有 `kind`，因为它是 AST 的顶层容器，不会作为另一个表达式的子节点。

这些对象由 Ray 按 Python 对象图直接传给 Controller。class、局部函数、lambda 和闭包仍是原来的 Python 对象；Fluxway 不把它们改写成导入字符串，也没有 placement JSON 重建协议。日志可以把不含 callable 的字段展开用于阅读，但该日志文本不是运行时事实来源。

### 选择 worker

这一组 operator 的结果都是 worker set，也就是“选中了哪些 worker”。

| Python | 返回值 | 普通语言 |
|---|---|---|
| `workers(S)` | worker set | 选择 Graph 中 class 为 `S` 的所有实例的全部 worker |
| `instance(id)` | worker set | 只选择名称为 `id` 的一个 Graph 实例 |
| `all_workers()` | worker set | 选择当前 Graph 的全部 worker |
| `rank(A,i)` | worker set | 从 A 中选择 rank 等于 `i` 的 worker |
| `ranks(A,I)` | worker set | 从 A 中选择 rank 位于名单 `I` 的 worker |
| `union(A,...)` | worker set | 合并多组 worker，重复项只保留一次 |
| `incident_spmds(R)` | worker set | 选择至少出现在一条亲和关系中的 SPMD |
| `isolated_spmds(R)` | worker set | 选择没有出现在任何亲和关系中的 SPMD |

若 Graph 中 `Reader` class 有普通实例和两个 virtual 实例，`workers(Reader)` 会同时选择三个实例。`rank(workers(Reader), 0)` 会选择三个实例各自的 rank 0。

亲和关系 `(A, B)` 可以读成“A 希望使用 B 已使用的机器”。因此 `incident_spmds` 会选出关系两边出现过的 SPMD，`isolated_spmds` 则选出没有连线的 SPMD。

### 从 worker 得到机器或数字

这一组 operator 把 worker 名单转换成“每台机器多少人”或“用了哪些机器”。

| Python | 返回值 | 普通语言 |
|---|---|---|
| `occupancy(A)` | vector | 记录每台机器上有多少个 A worker |
| `support(A)` | node set | 记录至少承载一个 A worker 的机器 |
| `artifact_occupancy(R)` | vector | 取亲和关系中被参考一方的 occupancy，用于集中度优化 |
| `weighted_dominant_share(A)` | vector | 为 A 中每个可伸缩 SPMD 计算加权资源占比 |

`affinity_constraints()` 返回当前 Recipe 中 Fluxway 能确认“一定成立”的亲和关系集合：

它返回一个带有 `kind="affinity_constraints"` 的关系表达式。

它主要供标准 objective 使用。业务代码通常不需要读取其中的单条关系。

例如 `occupancy(A) = {机器 A: 2, 机器 B: 1}` 时，`cardinality(support(A))` 就是 2。

### 组合机器集合

| Python | 返回值 | 普通语言 |
|---|---|---|
| `union(A,...)` | node set | 出现在任意一个输入中的机器 |
| `intersection(A,...)` | node set | 同时出现在所有输入中的机器 |
| `difference(A,B)` | node set | 在 A 中但不在 B 中的机器 |

同一个 `union` 会根据输入类型产生 `worker_union` 或 `node_union`。不能把 worker set 和 node set 混在一次 union 中。

### 得到一个数字

这些 operator 的结果都是 scalar，也就是一个可以比较的数字。资源数字必须使用同一种单位；1 CPU 不能直接和 1 GPU 比大小。

| Python | 返回值 | 普通语言 |
|---|---|---|
| 整数 `k` | scalar | Python 整数会自动转成 AST 整数 |
| `cardinality(A)` | scalar | node set A 中有多少台机器 |
| `resource_usage(A,r)` | scalar | A 的全部 worker 共使用多少资源 r |
| `resource_capacity(A,r)` | scalar | node set A 的资源 r 总容量 |
| `allocated_quanta(A)` | scalar | A 获得了多少个额外资源分配步长 |
| `affinity_coverage(R)` | scalar | 亲和关系双方同时出现的“关系-机器”数量 |
| `co_location_mass(R)` | scalar | 同机的左右 worker 配对数量 |
| `resource_fragmentation()` | scalar | 剩余资源中无法再拼成合法分配单位的数量 |

CPU usage 只能与 CPU capacity 或整数阈值比较。CPU usage 与 GPU usage 的单位不同，Fluxway 会直接拒绝比较。

### 写必须满足的规则

constraint 只有“成立”和“不成立”两种结果。只要有一条必须成立的 constraint 不成立，这个调度方案就不能使用。

| Python | 普通语言 |
|---|---|
| `subseteq(A,B)` | A 中每台机器也必须在 B 中 |
| `disjoint(A,B)` | A 与 B 不能有共同机器 |
| `equal(A,B)` | 两个同类型集合或数字必须相等 |
| `less_equal(a,b)` | 数字 a 必须小于或等于 b |
| `all_of(c,...)` | 所有输入规则都必须成立 |
| `any_of(c,...)` | 至少一条输入规则成立 |
| `negate(c)` | 规则 c 必须不成立 |

`any_of` 不表示“任选一条然后忽略其他条”。它表示最终方案只要让其中至少一条为真即可。

### 写方案优先级

objective 只在合法方案之间比较好坏。它不能让 Fluxway 忽略 constraint。

| Python | 普通语言 |
|---|---|
| `minimize(a)` | 让数字 a 尽量小 |
| `maximize(a)` | 让数字 a 尽量大 |
| `max_min(v)` | 先改善 vector 中最小的值，再改善第二小的值 |
| `maximize_majorization(v)` | 先让最大的值更大，再依次比较后续值，偏向集中 |
| `lexicographic(o,...)` | 按书写顺序处理 objective，前一项绝不为后一项让步 |

假设 4 个 worker 放到两台机器。`max_min(occupancy(A))` 更喜欢 `[2, 2]` 而不是 `[3, 1]`，因为最少的那台机器从 1 增加到了 2。

`maximize_majorization(occupancy(A))` 的方向相反：它更喜欢 `[4, 0]` 而不是 `[3, 1]`，因为 worker 更集中。这里的 vector 可以先理解为“按固定机器顺序写成的一排数字”。

### 一个完整 AST 例子

<!-- test:placement-ast-python -->
```python
from fluxway import support, workers


class ExampleSPMD: ...


ast = support(workers(ExampleSPMD))
assert ast.kind == "support"
assert ast.workers.kind == "workers"
assert ast.workers.spmd.spmd_class is ExampleSPMD
```

AST 是 frozen 对象，创建后不能修改。Fluxway 保留专家写下的 constraint 和 objective 顺序，不做 JSON round-trip，也不静默去重。

## 字典序怎样工作

“字典序”类似比较两个单词：先比较第一位，第一位相同才比较第二位。

`lexicographic(o0, o1, ...)` 先求 `o0` 的精确最优值并锁定，再求 `o1`。后面的 objective 不能牺牲前面的结果。

它不是把多个 objective 乘权重后相加。用户不需要猜一个足够大的权重，也不会因为单位大小不同而意外改变优先级。

`AFFINITY_FIRST_V1` 依次执行：

1. 给参与亲和关系、且允许扩展资源的 SPMD 公平分配资源。
2. 再用真实剩余资源分配没有亲和关系的 SPMD。
3. 尽量减少整个作业使用的机器数。
4. 让亲和目标的 worker 分布更集中。
5. 增加亲和双方同时出现的机器数。
6. 增加同机 worker 配对数。
7. 减少无法继续使用的零碎资源。

## 三个较难的数字

### weighted dominant share

一个 SPMD 可能同时需要 CPU、GPU 和 memory。分别计算它使用每种资源的比例，其中最大的比例叫 dominant share，也就是“最紧张资源的占比”。

例如一个 SPMD 使用全场 20% 的 CPU 和 50% 的 GPU，它的 dominant share 是 50%。公平分配时会重点比较这个 50%。

`weight` 越大，同样资源占用产生的加权 share 越小，因此该 SPMD 会获得更高的分配优先级。普通用户使用 SPMD `ResourceProfile` 预设的 weight 即可。

求解器使用精确整数和分数，不使用会积累误差的浮点近似。

### affinity coverage 与 co-location mass

这两个数字只用于比较已经满足 constraint 的方案，不会放松亲和规则。

假设 A 和 B 在机器 n 上分别有 `a`、`b` 个 worker。

只要 `a>0` 且 `b>0`，这台机器为 coverage 贡献 1。它为 co-location mass 贡献 `a×b`。

coverage 鼓励关系覆盖更多机器；mass 在 coverage 相同时，更偏好把 worker 放到目标 occupancy 更高的机器。

### resource fragmentation

假设一个合法 worker 单位需要 `2 CPU + 1 GPU`。机器还剩 `1 CPU + 1 GPU` 时，虽然资源不为零，却无法再放入一个完整 worker。

`resource_fragmentation()` 衡量这类“看起来还有资源，但已经放不下完整 worker”的剩余量。不同资源先按各自分配单位换算，memory 的字节数不会因为数字大而支配 GPU。

## 六 SPMD 例子

两台 GPU 机器各有 8 张卡。SPMD0 需要 4 张，SPMD1 需要 12 张。合法且集中的结果是：

```text
GPU 机器 A: SPMD1 × 8
GPU 机器 B: SPMD1 × 4 + SPMD0 × 4
```

SPMD3 若要求 `support(SPMD3) = support(SPMD0)`，就只能使用机器 B。

SPMD4 若要求 `support(SPMD4) ⊆ support(SPMD1)`，一个 worker 时会由 mass 偏向 SPMD1 有 8 个 worker 的机器 A；两个 workers 时，coverage 会先让两台机器各放一个。

SPMD5 的两个 rank 可以分别关联 SPMD0 和 SPMD1：

```text
support(rank(SPMD5, 0)) ⊆ support(SPMD0)
support(rank(SPMD5, 1)) ⊆ support(SPMD1)
```

没有亲和规则的 SPMD2 会在前面的优先级完成后，使用剩余 CPU。

## 求解能力与边界

placement 同时包含装箱、资源容量和亲和关系。机器和 worker 增多后，可能需要尝试大量组合。这类问题通常没有已知的快速通用算法。

Fluxway 使用 OR-Tools CP-SAT 求解器寻找精确解。可以把求解器理解为“系统地搜索许多安排，并证明没有更好安排的程序”。

精确解表示所有 constraint 都成立，并且每一层 objective 都已经证明最优。只找到一个看起来不错的方案不算完成。

整个求解共享一个 deadline。超时、只找到尚未证明最优的方案、资源快照变化或 Ray 无法创建 PlacementGroup 都会抛出异常。

模型创建前还会估算变量、约束、排序 comparator 和 solve 次数。规模超过 `PlacementRuntimeConfig` 上限时会提前拒绝，避免先耗尽内存。

Ray 只能让 Fluxway 看见 CPU、GPU、memory 和 custom resource。未暴露的 NUMA、NVLink、rack 或 accelerator 型号不能参与精确约束。

普通 Ray API 没有逐机器空闲资源账本。因此精确 placement 要求 SPMD profiles 请求的资源在求解期间由该作业独占；否则 Fluxway 直接拒绝，不猜测其他任务占在哪台机器。

## 公式附录

下面是实现使用的精确定义。理解前文不需要阅读本节。

```text
occupancy(A)[n] = Σ[w∈A] x[w,n]
support(A) = {n∈N | occupancy(A)[n] > 0}
```

```text
affinity_coverage(R)
  = Σ[(A,B)∈R,n∈N] 1[occupancy(A)[n]>0 且 occupancy(B)[n]>0]

co_location_mass(R)
  = Σ[(A,B)∈R,n∈N] occupancy(A)[n] × occupancy(B)[n]
```

令 `P` 是不小于 vector 长度 `K` 的最小 2 的幂。Batcher sorting network 的 comparator 数为：

```text
P × log2(P) × (log2(P)+1) / 4
```

资源数量先按 Ray 的 `0.0001` 精度转成整数，再按每种资源自己的最大公约数缩小。所有系数和上界都检查 CP-SAT int64 范围。
