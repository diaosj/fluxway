# Worker 进程算子

Fluxway 通常会将用户代码与 `SPMDWorker` 的 actor event loop（事件循环）隔离开来。
在用户算子运行时，scheduler（调度器）、mailbox receiver（信箱接收器）、prepare runtime（准备运行时）和 delivery runtime（交付运行时）必须能够继续推进。

每个有状态（stateful）的算子类都会在其规划的 SPMD worker 进程内被实例化一次。当算子的构造依赖于 worker 拓扑或需要初始化进程本地的运行时状态时，请使用 `@fluxway.requires_worker_process`。

```python
import jax
import fluxway


@fluxway.requires_worker_process
class Trainer:
    def __init__(self):
        jax.distributed.initialize(
            coordinator_address=self.rank0_address,
            num_processes=self.world_size,
            process_id=self.rank,
        )
```

常见原因：

- `jax.distributed.initialize`
- 进程本地的 GPU 或加速器运行时状态
- 必须与 SPMD worker 在同一进程中初始化的库
- 构造时需要最终 rank 和节点信息的原生句柄

普通有状态算子不需要标记，因为它们的所属实例已经属于 SPMD worker。该标记使构造要求显式化，并能验证进程本地初始化。

## 运行时语义

有状态的所有权并不意味着在 Ray actor 事件循环上内联执行。
从第一次调用开始，Fluxway 就会将每个普通的 Python `CALL` 节点异步提交到 `WorkerProcessCallRuntime` 中其命名的 lane（通道）上。有状态且具有进程亲和性的可调用对象也使用相同的路径，即使它们返回一个可等待对象（awaitable）：

- actor 事件循环提交工作并等待 future
- 普通的 Python 可调用对象在同一进程的 worker 本地线程上运行
- 如果可调用对象返回一个可等待对象，运行时将在 worker 线程内完成它
- 诸如 rank/world/step 等 `contextvars` 会被传播
- 异常会传播回 LocalStepPlan，并正常地毒化（poison）worker

Fluxway 每次调用只选择一次执行 lane。在开启 profiling 的情况下，`user_callable.execute` 记录端到端延迟，包括队列等待时间，而 `user_callable.function_run` 记录实际的算子函数调用，不包括 lane 排队和 worker 上下文设置。对于 worker 执行，`worker_process_call.thread_run` 记录精确的 worker 线程执行区间。Profiling 会观察选定的 lane，但不会再次分发它。

每个协程函数使用 `COROUTINE_ASYNC` 执行模式，并且可以在 actor 事件循环上协作运行。状态和进程亲和性不会改变这种选择：`StatePolicy` 控制对可变状态的访问，而 `@requires_worker_process` 控制对象的构造和生命周期。协程在等待期间必须交出控制权（yield），并且不能在 `async def` 内部隐藏阻塞式工作；从第一次调用开始，普通的 Python 函数就会被异步提交到其命名的 worker 进程 lane 上。

这保留了以下组件的事件循环推进：

- `receive_transport_frames`
- 信箱存入（deposit）和就绪（ready）回调
- `LocalInputPrepareRuntime`
- `OutboundDeliveryRuntime`
- 调度器唤醒和 drain fences（排空屏障）

计算密集型的纯 Python 用户代码仍然会竞争 CPython GIL（全局解释器锁）。该运行时是一种事件循环隔离机制，而不是 CPU 并行加速的保证。JAX、CuPy、原生内核、Ray RPC 等待以及许多 I/O 路径仍然可以重叠，因为它们会释放 GIL 或者是组件驱动的。

在 Python 3.14 及更高版本的自由线程（free-threaded）CPython 上，同名的 worker 进程 lane 可以并行运行 Python 字节码。Fluxway 没有为解释器的这种能力暴露第二种执行模式；在各个 CPython 版本中，lane 隔离和调度语义保持一致。

## 状态和资源规则

`AsyncPolicy` 控制并发：

- `StatePolicy.exclusive()` 通过 LocalStepPlan 资源 ID 串行化访问。
- `StatePolicy.thread_safe()` 移除类实例锁，并要求底层的状态/存储对于并发调用是安全的。
- `local_pipeline_depth` 来自 `AsyncPolicy.depth`。

如果一个进程亲和性的类具有可变的 Python 状态，且没有被显式声明为线程安全，请使用默认的 `AsyncPolicy.sync()` 或 `StatePolicy.exclusive()`。

进程亲和性的构造并不禁止本地重叠。一个标有 `@requires_worker_process` 的类可以暴露使用 `StatePolicy.thread_safe()` 的无状态辅助算子，同时保持改变模型/状态的算子是独占的。建议方案 `DenseSPMD` 就使用了这种拆分：

- `prepare_dense_inputs` 和 `scatter_sparse_grads` 在线程安全的本地异步 lane 上运行，因为它们不读取或写入稠密模型状态。
- `train_core` 和 `update` 保持独占的状态访问，因为它们读取或改变稠密模型/梯度累加器。
- `SchedulePolicy.phase_graph` 允许 `scatter_sparse_grads(step=N)` 和 `update(step=N)` 重叠，同时防止 `train_core(step=N+1)` 在 `update(step=N)` 提交之前开始。

这些 phase 之间的内部交接可以使用函数式图的 `local=True`。那是一条同一 worker 的传输边，而不是远程调用。

## 失败模式

当一个类调用了 `jax.distributed.initialize` 但没有被装饰时，Fluxway 会快速失败：

```text
Fluxway Worker Process Error: ... calls jax.distributed.initialize but is not marked with @fluxway.requires_worker_process.
```

内部协议字段 `__fluxway_process_affine__` 被故意隐藏在装饰器之后。用户代码应使用 `@fluxway.requires_worker_process`。

## Checkpoint 组件

算子不实现 checkpoint hooks。Checkpoint 格式通过 `@spmd(checkpoint_components=(...))` 注册一次，并组合可选的 `CheckpointComponentRegistration` SPI 契约。这将状态序列化、集体发布、物理 URI 生成、恢复、对象保持（object holds）和清理置于一个显式的扩展边界之后。

Worker 协调器（coordinator）调用已注册的 `SAVE` 组件的 `prepare_save`、`save`、`finish_save`、`commit_save` 和 `publish_save` 阶段；`rollback_save` 处理失败的事务。已注册的 `RESTORE` 组件处理 `restore`。Controller（控制器）拥有事务排序和清单（Manifest）发布权；组件拥有其 checkpoint 对象 URI、暂存布局、版本和物理保留操作。对象保持和清理保留属于 `OBJECT_LIFECYCLE`；物理删除属于 `OBJECT_CLEANUP`。

`CheckpointContext` 仅携带逻辑身份：事务、快照、分片（cut）、SPMD/rank、拓扑和组件 ID。它故意没有物理存储定位器。请使用 `context.checkpoint_object_key(...)` 获取逻辑的 checkpoint 对象标识，并从 `save`/`commit_save` 返回组件拥有的、不透明的 `CheckpointObjectRef`。

每个修改阶段都会收到 Fluxway 的 `CheckpointCommitPermit`（提交许可证）；在协调器代际接管后，过期的许可证会被拒绝。有关能力契约、清单模式、拓扑恢复规则、保留租约和内置组件示例，请参阅 [checkpoint.md](checkpoint.md)。
