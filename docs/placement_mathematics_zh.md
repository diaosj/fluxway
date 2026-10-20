# 放置数学 (Placement mathematics)

本文档是 [placement_mathematics.md](placement_mathematics.md) 的中文对应版本。
它定义了 Fluxway placement 模块实现的求解器语义。

`workers(S)` 表示 SPMD 的 worker 集合，`support(S)` 表示其节点的 support，
`occupancy(S)` 表示其在节点上的占用数量。约束（Constraints）是硬断言。只有在所有约束都成立的情况下，才会计算目标（objective），并且按声明顺序优化字典序目标。

放置编译器（placement compiler）将冻结的表达式树降维（lowers）为 CP-SAT 变量，并独立验证具体化的分配。资源分布（Resource profiles）来自编译后的声明；Recipe 不保留第二份类到资源的映射。不支持的表达式、缺失的声明和无效的拓扑关系会直接引发异常。

以下提供了完整的 API 使用示例。有关详细的用户工作流与配置指南，请参阅 [placement_recipe_zh.md](placement_recipe_zh.md)。

<!-- test:placement-controller-entry -->
```python
from inspect import signature
from fluxway import start


async def start_training(graph, recipe):
    return await start(graph, placement_recipe=recipe)


assert "graph" in signature(start).parameters
assert "placement_recipe" in signature(start).parameters
```
