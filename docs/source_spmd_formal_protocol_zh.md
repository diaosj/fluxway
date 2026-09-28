# Source SPMD 正式协议

## Canonical Ownership

`LayoutUnitDescriptor` 固定 revision、物理 locator、cardinality metric、区间及
source 自有 metadata。compiler 先将有序数据流划分为 logical step，再分配物理 Worker 配额。

`SourceWorkAssignment` 描述某个 logical step 内的 descriptor 区间。
`SourceWorkKey(epoch, step, assignment_id)` 是 canonical identity；MessagePack
摘要仅由 assignment 模型计算。对同一逻辑计划更换 Worker 数量不会改变 assignment 或 key。

重绑定必须保持 logical batch cardinality。改变物理 Worker 数量不意味着可以改变
每个 logical step 的数据集合；planner restore 会拒绝 logical batch cardinality 变化。

## 物理区间 Lease

`CompiledSegment` 将 assignment 的一部分绑定到 Worker，包含 `step` 和
`assignment_id`。`WorkerLayoutDescriptor` 第 1 版通过 canonical MessagePack
传输 segment，不再增加另一套 assignment binding 协议。

一个 assignment 可以跨多个 Worker。Ledger 在发布时验证全部 segment 对每个
assignment 的覆盖完整、无重叠。批量 acquire 必须精确匹配已发布的 segment；
重复 acquire 已租用 segment 直接失败，不会替换原 lease。

`SourceWorkItem` 包含 assignment 和本次读取的精确区间；`SourceWorkLease` 增加
运行时 `lease_id`。`SourceWorkAcknowledgement(key, lease_id)` 只确认该 lease
对应的区间，不能代替其他 Worker 确认整个 parent descriptor。

ACK 整批验证后再更新状态。在 lease 保留期间，完全相同的重复 ACK 幂等；提交后释放
已完成的 lease/ACK 记录，已释放 lease 的 ACK 按过期拒绝。未知或过期 lease、冲突 key
直接失败。Restore 后生成新的 lease ID，旧 attempt 延迟到达的 ACK 无法确认新读取。

## Frontier 与 Checkpoint

每个 step 维护剩余 segment 数量，ACK 只更新涉及的 segment，不扫描整个计划。
`candidate_cut` 只越过连续完整确认的 step。ACK 表示 source 生产完成，不代表模型已消费。

`committed_cut` 通过已有的协调 checkpoint 事务推进。Source、模型和 optimizer 必须
对齐同一个边界；source ACK 不构成另一套持久提交点。

Logical ledger checkpoint 第 3 版保存 committed cut 和 immutable epoch digest，
不保存 lease、ACK cache 或物理 owner。Planner checkpoint 第 2 版保存 canonical
request、固定 logical batch cardinality、待拼接尾部及 cursor，不保存物理 topology。

`max_in_flight_steps` 按 committed cut 限制 acquire。窗口耗尽时异步等待，取消不会
持有 ledger 锁。Checkpoint 周期必须允许窗口推进，否则 acquire 会持续受到背压。

## 执行边界

Worker 按 operator pipeline depth 批量申请活跃 step。Controller 可以返回信用窗口
内非空、完整的 step 前缀。Worker 验证 epoch、lease ID 唯一性、精确区间覆盖和
前缀完整性后才写入瞬态缓存；同一批次共享 fetch，调用者取消不终止其监管。
ACK 使用有界本地队列，后台批量发送，错误传播到 Worker，并纳入 checkpoint drain。
缓存未命中仍有 acquire RPC 延迟，批量化本身不能证明 Controller 不会成为吞吐瓶颈。

Compiled execution layout 定义物理成员与组内逻辑 rank，Dispatcher 与 send-plan
lowering 读取同一不可变绑定，不负责 revision discovery、lease 或 checkpoint cut。
同机零拷贝依赖 placement 和 transport 条件，与 checkpoint 正确性独立。

任意用户代码的外部副作用不能默认安全重试。恢复应以协调 checkpoint 为准；外部效果
需要用户事务或能够处理物理区间边界变化的幂等设计。

## 协议一致性保障 (Protocol Consistency Guarantees)

该协议提供了以下严格的一致性与弹性保证：
- 分片 Assignment、乱序 ACK、批量原子性、过期 ACK 隔离、安全的取消等待以及 Revision 冲突的处理。
- **拓扑变更弹性**：在更换拓扑并进行 Ledger Restore 后，系统能够保证从精确的 Committed Cut 继续无缝读取（例如从三个 Reader 安全缩减为两个 Reader）。
- **稳定的数据血缘**：Worker 按 Canonical Assignment 区间进行读取，任何重绑定（Rebinding）操作都严格保持原有的 Logical Batch 大小与数据出处（Provenance）。

**架构流转**：
Driver Activation 时 Checkpoint 状态保持为空。Controller 在 Restore 时，总是首先根据当前的 Reader Topology 重建 Canonical Plan，随后再恢复 Logical Committed Cut。

Controller 启动发布一个 layout。Worker source scope 在 Reader 安装成功后接受连续
不可变窗口。公开启动 API 不提供无界 adapter 的自动 discovery/refill。外部存储集成、
训练收敛和零拷贝性能需要面向具体工作负载验证。

参见 [source coordination](source_coordination_zh.md) 与 [checkpoint](checkpoint_zh.md)。
