# BaseReader 与数据源读取 API

`BaseReader` 将 source record 转换为携带坐标的 `TransportTensor`。受控 source
必须读取 `self.ctx.source_work_items` 中的精确区间。Planner 拥有 immutable revision
和 canonical batch，`LogicalWorkLedger` 拥有 lease 与进度；Dispatcher 只选择
活跃 Worker 并路由数据。

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

实现 `IngestionSourceAdapter.capture_view`、`encode_cursor` 与 `decode_cursor`
进行 immutable metadata discovery。Source 自有 metadata 和二进制 cursor 使用共享
MessagePack codec。`IngestionReadSession.read` 可以为 `read_ingestion_layout`
提供同步区间 chunk；`IngestionLayoutSession` 在线程中迭代并校验实际读取 cardinality。

有限受控 source 示例见
[RecommendationReader](../fluxway/example/recommendation/reader.py) 及其
[source descriptor](../fluxway/example/recommendation/source_adapter.py)。
Descriptor 提供 `build_reader_topology`、`build_ingestion_plan` 和已有 Driver
activation component。受控 source 恢复必须先重建 canonical plan，再恢复 logical cut；
物理 Reader cursor 不能成为第二个 ownership 权威。

## 当前流式边界

Planner component 接受空的非终止 observation，不创建空 epoch；不完整 batch 可以跨
revision 拼接，checkpoint 保存 pending tail 和 cursor。系统原生支持在改变 Reader 配额的弹性缩放场景下，依然能够对固定 Revision 进行强一致性读取。

Controller 启动激活一个 layout；公开启动 API 不提供 adapter 自动 discovery/refill。Worker 接受连续的
不可变窗口并持有安装身份；BaseReader 提供安装 hook。仅实现 adapter 尚不能让训练图自动进行
无界流式读取。支持范围见 [Source 协调协议](source_coordination_zh.md)。
