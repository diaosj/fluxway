# 数据面运行时

本文与[英文版本](dataplane_runtime.md)描述同一份传输实现。

`TransportTensor` 持有 payload、布局、路由索引、origin lineage、receipt、访问意图
和运行时有效性。元数据长度必须与路由实体数一致；不足一个完整批次的数据必须根据
实际样本数确定 shape 和 origin descriptor。

数据面负责构建传输 frame、分组投递、有限背压和异步交付，不决定 Worker 的 step
准入、checkpoint 持久位置或 source ownership。Local-only 边可以直接传递引用，
同时遵守同一份 tensor ABI。

`CompiledSendPlan` 使用编译后的传输关系。完整 unit 传输为每个目标指定一个完整
producer unit；fragment 传输按源和目标分片的交集装配。完整 microbatch 直接投递
不需要 packing gather 或多 shard values concat。执行组将物理 sender rank 映射到
组内逻辑坐标，frame 保留物理身份。语义和声明方法见[传输计划](transfer_planning_zh.md)。

对于受控 source，Worker 按 step 索引编译后的 segment，并通过批量 Controller RPC
获取完整 step 前缀。`SourceWorkItem` 的坐标决定 Reader provenance；Dispatcher rank
不是 ownership cursor。ACK 进入有界的 `SourceAcknowledgementRuntime` 队列，并在
checkpoint drain 中 flush。队列容量允许时不必同步等待每个 ACK；grant cache miss
仍需等待 acquire RPC。Worker source scope 在安装后接收连续不可变窗口。启动 API
只启动来源生命周期；Controller 的事件驱动 `SourceProgressSupervisor` 负责 adapter
扫描和窗口补充，adapter 本身不拥有补窗。支持范围见[Source 协调](source_coordination_zh.md)。

Grant 的批量大小同时受到 source credit 窗口和 pipeline depth 限制。如果一个 Reader
每隔 `L` 个逻辑 step 执行一次，覆盖 `C` 个连续逻辑 step 的 credit 窗口最多包含
该 Reader 的 `ceil(C / L)` 个活动 step。授予数量还受 pipeline depth 和计划剩余步数
限制。因此，在逻辑并行度和 credit 固定时，增加物理 Reader 数量可能降低批量效率。
部分前缀回复留下的未授予后缀可以在后续请求中获取。配置 credit 时需要同时考虑
checkpoint 间隔和执行周期；配置的 depth 不代表每次请求都能获得相同步数。

协议失败后关闭 frame 准入。交付完成与 coherence retirement 都有明确的异步生命周期；
错误通过现有失败信道传播，不通过 fallback 路径重试。

一次 `CoherenceBatchRetireRuntime.retire_publication()` 调用独占一个 local step
产生的完整 target fanout。它求出所有 batch 的 `CoherenceInstance` 并集，为每个唯一的
`(instance, child-slot)` 登记一次 directory receipt，执行一次 admission，再并发启动
所有 target enqueue。按 target 切分的 `TransportFrameBatch` 只是高效的物理 carrier，
不是独立的 coherence 排序单元。

`_ChildDeliveryReceipts` 是 publication 唯一的 receipt owner，按相反顺序恰好一次地
执行终止转换。委派失败、任一 target enqueue 失败、取消未完成 sibling 和成功的
receiver acceptance 都经过这个 owner。原始失败会直接传播；Fluxway 不重试 target、
不等待 timeout，也不会把 sibling RPC 的返回当成成功。

对 CPU/Ray carrier，只有在 receiver 完成 frame admission 和 mailbox deposit 后，才生成
并校验 `ReceiverAcceptanceReceipt(publication_id, instance, child_slot, target_rank,
batch_seq)`。receipt 必须匹配不可变的 `PublicationBatch`；单独的 RPC 返回值不能完成
child。receiver 随后拥有 materialize 后的 carrier。`PayloadLifetime` 持有 producer frame
引用，并在全部 receipt 和 local lease 完成后恰好一次调用 `PayloadRecycler.release()`。
`AllocationRegistry` 只追踪身份和 exact handoff，watermark 不能保持 buffer 永久存活，也
不能提前释放。该所有权递交路径避免为变长 sparse batch 引入会造成复制的通用 ring
allocator；它不宣称 Ray 传输为零拷贝。

### Worker 接入方式

Worker 对一个完整 data-plane frame plan 提供一个 publication callback：

```python
batch_submit = partial(
    coherence_batch_retire_runtime.retire_publication,
    compiled_send_plan,
)
```

`BatchingDataPlaneExecutor` 使用全部 target batch 的不可变 tuple 和 enqueue callable
恰好调用该 callback 一次。调用方不得对每一个 target batch 直接调用
`CoherenceSequencer.commit_payload()`。Local-only consumer 继续使用现有 lease 路径，
不实现第二套 publication API。

其他所有权边界见[Coherence 协议](coherence_protocol.md)和[架构](ARCHITECTURE_zh.md)。

## 交付身份与数值可复现性

将 canonical sample 路由到正确的消费者，不会决定该次计算读取哪一个可变算子状态
版本。Recommendation 示例的 sparse lookup 和 update 使用
`AsyncPolicy.bounded_stale`，配置为 `max_staleness=7`。Worker 限制执行进度差，
lookup 仍可能在某个较早步骤的 update 之前或之后运行。两种情况下，数据面都遵循
相同路由契约交付计算结果。

恢复相同 checkpoint 并消费相同的已提交样本序列，不保证恢复后的数值轨迹完全相同，
即使 Worker 数量没有变化。应分别验证保存与恢复状态、样本 provenance 和数值行为。
确定性的模型重放需要显式的计算顺序或状态版本契约；完整 unit 交付和 checkpoint
恢复本身不提供这项保证。

示例的 GPU sparse-gradient scatter 对重复索引也不保证逐位可复现。相同输入数组
可能在进入数据面之前就产生不同的浮点累加结果。验证传输身份时，应比较一次计算的
输出与其接收值；跨次独立计算的输出比较属于另一项数值验证。

## JAX 与 CuPy 导出边界

共享 GPU 指针不代表 producer 已完成。Recommendation Dense 将 JAX 输出投影到唯一
可寻址的本地 shard，通过 `HybridBackend.asarray` 显式指定设备和 `copy=False`
接收。参数梯度必须完整复制后才能 flatten。Backend 使用 DLPack 处理生命周期交接和
consumer stream 协商；不支持的设备、缺失的 CUDA 支持及跨设备域的免拷贝请求直接抛错。

Frame staging 同样使用 `copy=False` 在配置的 backend 设备上接收 values。跨设备
接收直接失败，不隐式经 CPU 复制。即使 values 共享内存，ragged length 元数据也可能
需要单独的 buffer。

物理亲和性与 buffer alias 是不同属性。同进程 alias、同节点 Ray object-store 交付
和跨节点传输必须分别测量。Plan 层证明不能确立跨进程零拷贝或 Controller 吞吐。
接收端 producer cache 按执行周期 slot 索引，重复执行组不会为每个绝对 step 新建条目。

Reader/Input 同节点亲和性不等于选择了共享内存 Ray 传输。`RayActorDataPlane` 将
CPU payload 提交给 Ray，由 Ray 负责跨进程交付和接收内存分配。只读 NumPy payload
也可以位于进程私有内存中。Whole-unit Input assembly 可以直接引用收到的 payload，
但这不能证明 Reader 到 Input 的交付过程没有发生复制。

## 本地提交与交付锁边界

交付等待位于算子 commit resource lock 之外：`SEND` 登记并启动交付；`AWAIT_SEND`
不持有 commit resource 等待完成；`BARRIER/COMMIT` 仅在发布连续 watermark 时获取
有序 commit lock。Drain、checkpoint、export 和 stop 仍通过 delivery fence 等待
全部已登记交付。

```mermaid
flowchart LR
  S[SEND: 登记 token] --> W[AWAIT_SEND: 无锁等待]
  W --> C[COMMIT: 有序发布锁]
  C --> M[连续提交 watermark]
```
