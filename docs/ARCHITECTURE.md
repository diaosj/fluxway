# 1. 概述
Fluxway 是一个面向大规模分布式训练的深度学习框架，采用直接 SPMD-to-SPMD 数据传输机制，避免了传统的 "SPMD-to-One + One-to-SPMD" 中间聚合步骤，从而提升训练效率。框架在重构后，进一步加强了面向有向无环图（DAG）的数据驱动编排模型、完全异步化的状态机控制以及极高并行度的数据传输分离。

## 1.1 核心特性
**双平面传播协议**：将数据流分为基于高性能网络协议的 Data Plane（处理张量数据本身）与依赖 Ray RPC/Control Agent 广播的 Control Plane（处理标量、状态量和控制流），根据数据切片进行自动路由分发。

**基于 Mailbox 的纯异步汇聚**：Worker 抛弃了传统事件阻塞，采用 `WorkerMailbox` 管理接收到的数据片。配合 `StepBarrier` 检测条件，一旦所期望的张量和状态量就绪，操作将自动进入 `asyncio.PriorityQueue` 执行。

**细粒度 Async 异步控制和反压机制**：新增了 `AsyncStepManager` 来约束不同 SPMD 中执行异步步数差异；并结合了基于信用的 `_busy_worker_ratio` 进行反向压力追踪，自动使得 Reader 节点级联挂起。

**层级架构与灵活资源管理**：使用全局级别的 `Controller` 调度图信息和 Placement Group；使用局部的 `SPMDDriver` 追踪 Worker 心跳；各个 `SPMDWorker` 则专注于高性能异步 Kernel 计算，真正做到职责隔离。

## 1.2 架构概览
```
+----------------------------------------------------------------------------------+
|                           Control Plane (Ray Actor)                              |
|                                                                                  |
|   +------------------+     +------------------+    +-------------------------+   |
|   |  DAG Parser /    |     |    Scheduler     |    |   AsyncStepManager &    |   |
|   |   Graph Builder  |     |   (PG Manager)   |    |    HeartbeatManager     |   |
|   +------------------+     +------------------+    +-------------------------+   |
|                                                                                  |
|                +------------------------------------------------+                |
|                |              ControllerStateMachine            |                |
|                +------------------------------------------------+                |
+----------------------------------------------------------------------------------+
                                        |
+----------------------------------------------------------------------------------+
|                            SPMD Level (Driver Layer)                             |
|                                                                                  |
|   +----------------------+      +--------------------+      +----------------+   |
|   |   Driver FSM         |      | WorkerSynchronizer |      |  SPMD Monitor  |   |
|   +----------------------+      +--------------------+      +----------------+   |
|                                                                                  |
|                +------------------------------------------------+                |
|                |              Stateful Executor                 |                |
|                +------------------------------------------------+                |
+----------------------------------------------------------------------------------+
                                        |
+----------------------------------------------------------------------------------+
|                           Worker Level (Async Actors)                            |
|                                                                                  |
|   +------------------+     +--------------------------+    +-----------------+   |
|   |  Worker Mailbox  | --> | asyncio.PriorityQueue    | -> | OperatorManager |   |
|   |  (StepBarrier)   |     |                          |    | (Dispatcher)    |   |
|   +------------------+     +--------------------------+    +-----------------+   |
|                                                                                  |
|                +------------------------------------------------+                |
|                |       AsyncDataDistributor & Transport         |                |
|                |               (ULP Protocol)                   |                |
|                +------------------------------------------------+                |
+----------------------------------------------------------------------------------+
```

---

# 2. 控制平面 (Control Plane)
控制平面是整个训练框架的核心协调层，对应代码目录为 `fluxway/control_plane`。

## 2.1 主控制器 (Controller)
`Controller` 是由 Ray Actor 实现的主控入口，负责管理任务的整体生命周期和资源调度。

### 2.1.1 拓扑构建与解析 (Graph Builder & Parser)
`GraphParser` 负责处理和生成有向无环图，将用户编写的层级连线提取成了内部任务视图。并在运行时提供节点的前置、后置关联映射。`FunctionalGraph` 和 `NodeProxy` 为前端 API 提供透明抽象。

### 2.1.2 调度层 (Scheduler & PGManager)
**职责**：基于有向无环图的配置结构进行调度。依赖 `PGManager` 包装了基于 Ray Placement Group 的底层创建，实现了 CPU、GPU 内存的静态计算和按需资源划分分配。
支持创建阶段构建整个网络的节点拓扑连接方式。

### 2.1.3 心跳和限流 (HeartbeatManager & AsyncStepManager)
**双重监控与限流**：
1. **HeartbeatManager**：接收各个 SPMD 的 Driver 带来的心跳汇报信息。并在全部 Driver 准备好后发送拓扑注册请求，在探测到出错时切换系统异常状态抛出错误。
2. **AsyncStepManager**：计算操作的最大步数容忍漂移差，执行针对特定 Op（操作）层级的阻止和释放阻塞通知。
3. **反向压力与 Backpressure**：通过探测下游 Driver 发生大量 mailbox full 或 busy 的概率 (`busy_worker_ratio`)，对入图无前驱节点的 Reader 节点触发 pause 停止拉取新批次。

### 2.1.4 状态管理器与 SPMD 配置 (StateManager & SPMDManager)
负责记录图上每个 SPMD 的句柄，以及对节点启动、驱逐做句柄跟踪。保存并规划整个拓扑的状态持久化策略和集群统计。

---

# 3. SPMD 层面
SPMD 在运行时切分为两层，一层是对内做监督的 `SPMDDriver`，另一层是专注于流式运算的极简 `SPMDWorker`。

## 3.1 SPMD 主控端 (SPMDDriver)
位于 `fluxway/spmd/driver` 目录下，作为该 SPMD 系列资源的总管（Master）。每个 SPMD 有一个特定的 Driver，被 SPMDManager 控制启动。
- **DriverStateMachine (FSM)**：控制当前 SPMD 子循环的局部状态（如初始化等待节点汇聚、暂停、完成等）。
- **WorkerSynchronizer**：执行特定细粒度的 Operation Block。能针对该 SPMD 中每个 Operator（操作符）设定统一的 Step Watermark 水位线限制。
- **StatefulExecutor**：有状态执行器，用来响应并维护模型状态向主存储介质或远程主节点的收发。
- **SPMDMonitor**：周期性收集所有所辖内部的 `SPMDWorker` 上报的心跳。

## 3.2 强异步底层执行流 (SPMDWorker)
重构后的 `SPMDWorker` 使用 Python 原生 `asyncio` 并发替代了原先复杂的异步执行树（Event Loop 模式更加单纯可靠）。

### 3.2.1 双流汇聚与邮箱架构 (Dual-Stream Convergence via WorkerMailbox)
为了支持极端的 "Control + Data" 并行发送到达随机性：
Worker 并不会使用阻塞拉取来推进。相反，采用 `WorkerMailbox` 建立异步回调墙（`StepBarrier`）。
- 控制流数据（如 LR 等由 Controller 发出的标量）在经过 Driver 使用 `receive_control` 发进来。
- 数据流张量（如前向输出特征，ULP 通信包）经过 Data Plane 由 `receive_shards` 发进来。
- 一旦某一对 (`op_name`, `step`) 符合了 `StepBarrier` 的预期收到所有的 tensors / states 数目，它将通过回调函数唤起，把该任务放入 `asyncio.PriorityQueue` 执行队列。

### 3.2.2 优先队列调度 (Execution PriorityQueue)
调度执行循环 `_run()` 仅盯着 PriorityQueue 收取内容，通过对图上的操作权重排序，执行更急迫先获得资源的运算，确保关键链路阻塞最少。

### 3.2.3 操作派发与操作管理 (Operator Manager & Dispatcher)
用户函数已被进一步简化剥离为 `OperatorManager` 管理的具体可派发结构。
`ExecutionDispatcher` 分离了 **逻辑节点** 和 **物理节点**。提供了两类主要派发策略：
- **StandardDispatcher**：正常的 DataParallel 和 SequenceParallel。物理 rank == 逻辑 rank，步步执行。
- **InterleavedDispatcher**：处理特殊的虚拟流水线模式，依据 `stride` 数值对计算流程执行空间复用隔绝。

---

# 4. 数据传输层
底层数据传输位于 `fluxway/transport/communication.py` 等文件中。重构后的传输协议极大解耦了网络拓扑，将原本的 "SPMD To One, One to SPMD" 变为了 JIT 编译加速的 ULP (Universal Layout Protocol) 零拷贝网。

## 4.1 数据切片形式 (Transport Layouts)
通信策略依托于 `PhysicalLayout` 中的定义，包括 `is_content_addressable` （依值路由的 Hash 或 Lookup 等散列）与 `routing_dim`（依行列路由的切片 Dense 等）。并根据目标是否开启 `is_referential`（对称流转回写，如 Mirror 反向梯度流），灵活产生网卡路由寻址图。

## 4.2 AsyncDataDistributor
新的双流架构引入了真正的异步发送管理器 `AsyncDataDistributor`。
**分发工作流**：
1. **打包填充 (Plan and Shard)**：通过 backend 中立的 `_jit_plan_and_pack`（在 `JaxBackend` 下走 `jax.jit`，`HybridBackend`/`TorchBackend` eager 下退化为原函数）对即将发送的批次执行切分和数据重组。其唯一路由契约由 `RoutingDomain` 承载，**不再传入任何张量数据形状参数**（旧 `routing_global_shape` / `dst_partitions` / `dst_strategy` 等已彻底移除）。
2. **异步发送 (Distribute)**：获取 `dest_agg` 内所有下游节点切分后的张量、长度计算块和 Global Shape 等，组成多个长程 RPC（使用 Ray Remote 调用）统一挂在 `asyncio.gather` 同时投递，达成全满带宽高传导。

这种完全避免集合通信堵塞，又兼顾了极高的 Python 层扩展性（支持 Sparse 和 Dense 动态组装）。

## 4.3 RoutingDomain 路由层（L1↔L2↔L3 严格分层）

传输层按 SPMD-to-SPMD 通信的四阶段拆出**严格单向调用**的四个职责层；每层只看下一层暴露的契约对象，不跨层窥视：

| 层 | 模块 | 知道什么 | 不知道什么 |
|---|---|---|---|
| L0 数据层 | `common/types/tensor.py` `TransportTensor` | flat_values 物理形状 (N, D, ...)、ragged splits、origin_indices、routing_axes | 路由策略、mesh、容量分桶 |
| L1 分发层 | `transport/communication.py` `BaseDataDistributor`、`_plan_single_tensor_group` | TransportTensor、PhysicalLayout、容量分桶、padding | sort 数学细节、Cartesian broadcast |
| L2 计划层 | `transport/planner.py` `TransportPlanner` | RoutingDomain、DeviceMesh、ReplicaSpec | 数据形状 (N, D)、payload 类型、容量 |
| L3 策略层 | `common/types/sharding.py` `PartitionStrategy` 子类 | 1-D `(indices, dim_size, num_workers, index_desc) → ranks` | 任何 N-D / 张量数据形状概念 |

**核心契约**：L1→L2 唯一入参是 `RoutingDomain`（不含张量数据形状）；L2→L3 唯一入参是 `RoutingAxis`（不含张量维度概念）。这样历史 BUG（"sort_indices.len = N×D"）在 API 层面被结构性消除。

### 4.3.1 IndicesSpec 四态化（strict typing）

`RoutingAxis.indices` 的类型为 `IndicesSpec`（**必填，禁止 `None` / 裸 tensor**），共四种实现：

| Spec | 语义 | 物化需求 |
|---|---|---|
| `LiteralIndices(values, valid_length=None)` | 1-D 张量显式 ID（hash key、ragged value、ExplicitIndex） | 已为张量 |
| `RangeIndices.{contiguous, cyclic, strided}` | 闭式等差/周期 ID | 可 symbolic fast-path 推导，**不展开** N 长 tensor |
| `IdentityIndices(length)` | identity over `[0, length)` | 同上 |
| `ConstantIndices(value, length=-1, valid_length=None)` | **单值轴**：所有 N 个实体共享同一 routing key | shape=(1,) tensor 通过 `coordinate_to_device` 的 advanced indexing 自动 broadcast，**永不展开**长 N 的常量数组 |

### 4.3.2 SrcSideView 与 src 侧切分映射

`_build_routing_domain` 是 **L1 唯一**从 `PhysicalLayout` 派生 `RoutingDomain` 的工厂；它按 src layout 在 routing dim 的 partition 类型，封装为对应 `IndicesSpec`：

| src layout 在 routing dim 的 partition | 输出 IndicesSpec |
|---|---|
| `ChunkPartition`（与 dst 同切分粒度，**isomorphic chunk**） | `ConstantIndices(value=shard_rank * chunk, length=-1)` |
| `ChunkPartition`（异构切分） | `RangeIndices.contiguous(start=r·chunk_size, length)` |
| `CyclicPartition` | `RangeIndices.cyclic(start=r, length, num_workers=W)` |
| Mirror / Replicated | `IdentityIndices(dim_size)` |
| `HashPartition`（content-addressable） | `LiteralIndices(content_keys)` |
| ExplicitIndex / RaggedIndex | `LiteralIndices(values)` |

**isomorphic chunk** 的关键洞察：当 `src.shard_size(dim) == dst.shard_size(dim)` 时，本 rank 持有的整个数据 chunk 必然全属于 dst 的同一个 sub-rank；因此该轴退化为单值轴，输出 `ConstantIndices`。

### 4.3.3 RoutingDomain.combine 双模式

`RoutingDomain` 通过 `combine: str` 字段分发两种多轴语义：

- **`combine="cartesian"`**（planner 默认）：保留通用 routing 引擎能力 —— 多个 axes 做 Cartesian broadcast，`sort_indices.len = ∏ axis.indices.length`；适用于 planner 直接被外部调用方驱动的场景。
- **`combine="zip"`**（`_build_routing_domain` 显式声明，transport 层专用）：所有 axes 共享同一 primary length，**element-wise zip 对齐**；`sort_indices.len = N`（primary entity count），与 `backend.take(data, sort_indices, axes=(0,))` 的 row-binding 严格匹配。

**zip-invariant 校验**：所有非 `ConstantIndices` 轴必须长度一致；不一致时 planner 抛 `ValueError` 阻断异构二轴误用。

### 4.3.4 高性能要点

- **零 materialise**：`ConstantIndices` 的 fast path 仅产 shape=(1,) tensor，长度为 N 的常量数组永不展开（依赖 `mesh.coordinate_to_device(device_ids[coords])` 的 advanced indexing 自动广播到 (N,)）
- **Symbolic fast-path**：`ChunkPartition.routes_for_range(RangeIndices)` 可闭式推导 `send_counts`，HybridBackend 在 N_local=1e5 量级下相对全物化路径 ≥30% step 时间下降
- **JIT-friendly invariant**：`RangeIndices` / `IdentityIndices` / `ConstantIndices` 全部 frozen dataclass + 不含 tensor，进入 `jax.jit` 的 `static_argnums`；仅 `LiteralIndices.values` 为动态 tensor 输入