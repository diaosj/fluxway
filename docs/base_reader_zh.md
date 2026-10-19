# BaseReader 与数据源读取 API

`BaseReader` 将来源记录转换为携带坐标的 `TransportTensor`。受控来源必须读取
`self.ctx.source_work_items` 中的精确区间。规划组件负责有序来源区间和训练批次，
`LogicalWorkLedger` 负责租约与进度；分发器只选择活跃工作进程并路由数据。

每次受控 source 调用前，OperatorManager 校验非空的 `SourceWorkItem` tuple：
所有条目必须属于当前执行 step，source、schema 和 global batch size 必须一致，
重复的 assignment interval 直接报错。Dispatcher 不再根据物理 rank 或组内 rank
推导 source 行偏移。

## 读取生命周期

1. Controller 发布 canonical plan，Driver 分发瞬态 Worker layout。
2. Worker 从授权缓存取出当前 step 的 lease，未命中时向 Controller 批量申请有界的活跃 step。
3. Worker 将这些 `SourceWorkItem` 注入当前 `OperatorContext`。
4. 自定义 Reader 按固定 revision、locator 和 logical interval 读取。
5. Tensor helper 绑定 work item 的全局行坐标，Worker 通过有界异步队列提交 source ACK。

ACK 表示 source 生产完成，不代表模型消费完成或 durable checkpoint commit。
外部副作用需要独立的事务或幂等契约。

## Tensor Helper

| API | 契约 |
| --- | --- |
| `create_tensor(data, dtype=None, uniform_dense_partition=False)` | 封装 dense 或嵌套数据，校验并绑定 source 行坐标。 |
| `create_row_sparse_tensor(rows, *, width, dtype=None)` | 构造具有 canonical global shape 的行分片稀疏值。 |
| `self.ctx.source_work_items` | 当前 logical step 的 acquired interval，不能跨调用缓存。 |
| `self.rank`, `self.world_size` | 物理 placement 坐标，不能据此推导 ownership。 |

数据外层行数必须等于 acquired cardinality 总和，全部 work item 必须属于当前 step
和同一个 logical batch。Dense uniform 与 row-sparse helper 使用 assignment 的
`global_batch_size` 作为全局行数。Origin indices 拼接各 item 的 `global_start`
与 cardinality，可以表示不连续区间。缺少 context 或 metadata 冲突直接抛异常。

对于没有 work item、但由调用方显式协调的 source，调用方必须提供
`global_source_offset`；BaseReader 不按物理 rank 猜测偏移。这种模式本身不证明
checkpoint replay 或跨拓扑 ownership。

## 自定义 Source

实现 `IngestionSourceAdapter.scan_next(cursor, budget)`，返回有界来源扫描结果和不透明
位置。位置内部由来源编码，Fluxway 不解析。`IngestionReadSession.read` 提供精确
区间的数据，`IngestionLayoutSession` 在线程中迭代并检查实际读取数量。

示例见[读取进程](../fluxway/example/recommendation/reader.py)和
 [来源描述](../fluxway/example/recommendation/source_adapter.py)。当前描述提供一个
 `build_source_scan_binding` 和 `build_component`；binding 统一拥有读取拓扑、扫描适配器、
 尾部策略和初始不透明位置。控制器在来源会话内只创建并缓存这一个 binding。
物理读取位置不能成为独立的
全局检查点。

## 流式与恢复边界

来源扫描结果和尾部由控制面规划。实现读取接口本身不能单独证明事件驱动的补充窗口和全局恢复可用。
规划组件可以为窗口内部的模型保存点计算消费书签，恢复从书签重新规划。
读取进程不另存独立位置，也不恢复旧窗口。持续推进以及模型、优化器和来源的共同
故障恢复必须遵循[来源协调](source_coordination_zh.md)中的检查点契约。
