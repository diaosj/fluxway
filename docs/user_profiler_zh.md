# 用户 Profiler

Fluxway 会自动将每个算子调用记录为 `user_callable.execute`。
只有在算子内部的更小代码区域，才需要使用 `fw.profiler.scope()` 和 `fw.profiler.event()`。

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

`fw.profiler.scope(name: str, **attrs)` 记录一个名为 `user.<name>` 的持续时间切片（duration slice）。

`fw.profiler.event(name: str, **attrs)` 记录一个名为 `user.<name>` 的即时标记（instant marker）。

`fw.profiler.enabled()` 仅在 profiler 启用且当前代码运行在 Fluxway 用户可调用对象内时返回 `True`。

属性（Attributes）可以是 `str`、`int`、`float`、`bool` 或 `None`。当 profiler 启用时，复杂的对象会引发 `TypeError`。当 profiler 禁用时，scope 和 event 都是无操作（no-ops），且属性不会被验证、驻留（interned）或序列化。

在构造昂贵的属性之前，请使用 `enabled()`：

```python
attrs = {}
if fw.profiler.enabled():
    attrs["ids_preview"] = ",".join(str(x) for x in ids[:8])

with fw.profiler.scope("prepare_ids", **attrs):
    ids = normalize(ids)
```

## Trace 显示

Worker 的持续时间切片使用以下 Perfetto 层次结构：

```text
SPMDs/<spmd_id>/rank:<rank>/<subsystem>/<leaf>/lane:<lane>
```

展开 rank 可以检查其 compute、transport、mailbox 和 scheduler 子系统。leaf 标识被记录的行为。`lane:<lane>` 区分同一个 leaf 上重叠的切片，因此 Perfetto 会保留每个切片的原始持续时间。Lane 分配是确定性的，并使用 leaf 最大并发度所需的最小 lane 数；它不会为每个 step 创建一条 track。

Scheduler 的活动按职责在 `scheduler/task`、`scheduler/queue`、`scheduler/step` 和 `scheduler/loop` 下分组。Checkpoint 组件操作使用 `checkpoint/storage`，快照协调使用 `checkpoint/hook`。`worker` 级别不在 `rank:<rank>` 下方重复出现。

自定义的 user scopes 和 events 使用的 leaf 是：

```text
compute/user_custom/<execution_mode>
```

`<execution_mode>` 组件是一种运行时调用模式。它与最后的 Perfetto 并发 track `lane:<lane>` 不同。

Payload 字段与外围的调用关联：

- `op_id`
- `step`
- `rank`
- `execution_mode`

用户属性被展开为可读的 Perfetto debug annotations。内部的 `attrs_id` 不会显示。

外围的 `user_callable.execute` 切片携带有不可变的执行计划标识字段：

- `graph_id`
- `spmd_id`
- `op_id`
- `function_id`
- `input_signature_id`
- `output_signature_id`

这些值在 Model Explorer 中是可以搜索的。`step` 和 `execution_mode` 是运行时事实，不属于编译后的计算图。

当导出 trace 时，Perfetto 还会从已安装的 `CompiledGraph` 中派生可读的图注解。携带 `op_id` 的事件会显示该算子函数以及所有声明的数据、控制和静态输入/输出的名称。当编译后的 `OperatorSpec` 包含 `input_param_map` 或 `output_param_map` 时，Perfetto 还会显示这些 Python 参数到字段的路径。空的参数映射会显示为无参数路径注解；Fluxway 不会从端口猜测参数名。携带 `transfer_code` 的事件显示完整的源和目标算子、函数和端口。`logical_tuuid` 标识生产者输出。

一个生产者输出可能提供给多个目标端口。这些事件共享一个 `logical_tuuid`，但具有不同的 `transfer_code` 值，因此每个目标端口都会独立显示，而不是被单一的张量名映射覆盖。

记录器热路径继续只存储固定宽度的整数记录。图名称不会复制到每个事件中，算子运行时也不会发生图查找。冻结的 profiler 符号投影在 trace 导出期间从当前的 `CompiledGraph` 构建一次。缺失的算子、传输或逻辑张量身份会引发 `MissingGraphTraceSymbol`；Fluxway 不会生成不完整的名称作为回退。

## 规则

- 不要将整个算子包裹在装饰器中。Fluxway 已经将其记录为 `user_callable.execute`。
- 对耗时的代码区域使用 `scope`。
- 对时间点事实（如缓存命中、采样的分支或算法 phase 标记）使用 `event`。
- 属性值必须是标量。将大型张量、数组、字典、列表或对象表示（reprs）放在你自己的日志中，而不是 profiler 中。
