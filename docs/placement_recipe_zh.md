# 放置配方 (Placement Recipe)

本文档是 [placement_recipe.md](placement_recipe.md) 的中文对应版本。
Recipe 规定了硬性的 placement 约束条件以及优化有效解的顺序。Worker 资源需求的所有权仍属于各个 SPMD 的 `ResourceProfile`。

## 使用已发布的 recipe

将计算图和 recipe 传递给 `fluxway.start`。计算图和 recipe 会作为原生的 Python 对象进行传输；不涉及 JSON 重建协议。Fluxway 会验证 worker 数量、资源、拓扑、约束和目标。无效的问题会直接引发异常，而不会静默选择备用策略。

<!-- test:placement-controller-entry -->
```python
from inspect import signature
from fluxway import start


async def start_training(graph, recipe):
    return await start(graph, placement_recipe=recipe)


assert "graph" in signature(start).parameters
assert "placement_recipe" in signature(start).parameters
```

## Recipe 所有权规则

使用 `workers`、`support`、`equal`、`lexicographic`、`max_min` 和 `occupancy` 来表达关系。不要在 recipe 中重复定义资源分布或 rank 映射。Controller（控制器）拥有 placement group；Manager 和 Driver 会借用生成的 rank-to-bundle 映射。

每一个解析出的 NodeID 都会收到一个复合的 placement-group bundle，由 Ray 的 `ray.io/node-id` 标签选择器固定。IP 地址不能代替节点的身份。具体化过程（materializer）保留规划器分配的 CPU、GPU 和自定义加速器资源；Worker 验证它们实际所在的节点是否符合规划。请通过资源分布和 recipe 约束来描述硬件差异。对于可恢复的多节点作业，需在所有节点上使用兼容的软件和共享的 checkpoint 存储。

有关求解器语义，请参阅 [placement_mathematics_zh.md](placement_mathematics_zh.md)。

## 动态运行时资源与 ResourceRegistry

静态硬件配额（如 CPU、GPU、加速器、内存容量）由各 SPMD 的 `ResourceProfile` 声明；而运行期绑定的动态资源（如动态路由表、设备网格实例或运行时 Rank Provider）由 `ResourceRegistry`（`fluxway.common.resource_registry`）统一管理。
算子与分片策略通过 `ResourceHandle(namespace=..., resource_id=...)` 声明对这些资源的引用，在通过 `bind_resource_registry` 或 `set_active_resource_registry` 激活的注册表上下文中动态解析，避免在 Placement 规划阶段硬编码底层物理实例。
