# Source Coordination

Fluxway 将持续数据发现建模为 Driver-local control-plane component，而不是 Reader worker 的
特殊执行分支。普通有限 source 不声明 descriptor，因此不会创建 coordinator。

## 声明与宿主

持续 source 在 operator metadata 上声明 descriptor：

```python
@operator("read", is_source=True, source=descriptor)
async def read(...):
    ...
```

compiled graph 是 descriptor 的唯一事实来源。对应 SPMD Driver 从 `OperatorSpec.source` 构造
component，运行时 id 为 `spmd_id:operator_name`。Controller 不导入 source adapter，也不构造
source component。

```text
Controller
└── ControlPlaneComponentRegistry
    └── global control components

SPMD Driver
└── DriverComponentRegistry
    └── source coordinator
```

两个 registry 复用同一个内部实现。component 不定义独立 lifecycle enum；Controller component
响应 `ControllerState`，Driver component 响应 `DriverState`。

### ControlPlaneComponent 协议与运行机制

所有控制面组件需实现 `ControlPlaneComponent` Protocol（定义于 `fluxway.control_plane.components`）：

```python
@runtime_checkable
class ControlPlaneComponent(Protocol):
    component_id: str

    async def on_state(self, state: ComponentState) -> None: ...

    def checkpoint_state(self) -> Mapping[str, Any]: ...

    def restore_state(self, state: Mapping[str, Any]) -> None: ...
```

1. **唯一标识 (`component_id`)**：注册到 `_ComponentRegistry` 的唯一字符串标识符（如 Driver 端 `spmd_id:operator_name` 或 Controller 端全局标识）。
2. **生命周期状态广播 (`on_state`)**：
   - 宿主（`Controller` 或 `Driver`）状态机发生转换时，通过 `registry.notify_state(state)` 顺序异步调用每个组件的 `on_state`。
   - `ComponentState` 为联合类型 `ControllerState | DriverState`：
     - **Controller 宿主**：传递 `ControllerState`（如 `INITIALIZING`, `PENDING`, `TOPOLOGY_REGISTERING`, `READY`, `RUNNING`, `COMPLETING`, `FAILING`, `COMPLETED`, `FAILED`）。
     - **Driver 宿主**：传递 `DriverState`（如 `INITIALIZING`, `READY`, `RUNNING`, `BLOCKED`, `DRAINING`, `DRAINED`, `SNAPSHOTTING`, `ERROR`, `COMPLETED`）。
   - 组件在 `on_state` 回调中响应宿主生命周期变化（例如开启数据源发现、暂停调度或释放局部资源）。
3. **状态快照与恢复 (`checkpoint_state` / `restore_state`)**：
   - **保存**：宿主触发表干/快照时，调用 `checkpoint_state()` 导出组件内部状态（如 source cursor、控制记录等）。
   - **恢复**：故障重启或恢复时，宿主通过 `restore_state(checkpoint_dict)` 将状态灌回对应组件。

#### 组件示例

```python
class ExampleSourceCoordinator:
    def __init__(self, component_id: str):
        self.component_id = component_id
        self.cursor = 0

    async def on_state(self, state: ComponentState) -> None:
        if state == DriverState.RUNNING:
            # 宿主进入 RUNNING，启动数据发现与拉取
            pass
        elif state == DriverState.DRAINING:
            # 宿主进入 DRAINING，暂停新数据发现
            pass

    def checkpoint_state(self) -> dict[str, Any]:
        return {"cursor": self.cursor}

    def restore_state(self, state: Mapping[str, Any]) -> None:
        self.cursor = int(state["cursor"])
```

## Ordered Control Stream

`ControlLedger` 是通用 ordered metadata ledger，只理解稳定身份：

```text
(stream_id, position)
```

- `position` 与 Fluxway operator logical step 属于同一序列；
- virtual owner 由当前 interleave topology 推导，不进入 record 或 checkpoint；
- payload 对 Fluxway 不透明；
- ledger 不理解 Reader、Iceberg、source revision、文件 split_id（即全局物理切片标识符 `file_path:offset+length`，用于对非确定性 manifest/FileScanTask 进行确定性全序排序与状态唯一标识）或 executor lane。

外部接口只有：

```python
append(records)
try_append(records)
receive(stream_id, position)
seal(stream_id)
progress(stream_id)
```

`append` 对整个参数 batch 原子且幂等。`try_append` 使用同一验证与提交路径，但容量不足时立即返回
`False`，不新增 ownership 或状态语义；Driver component 用它在 ledger 满时停止 discovery，从而不让
drain 等待一个背压中的 append。`receive` 在内部异步等待精确 position，并对同一 active
virtual group 的多个 rank 返回同一 metadata。调用方不实现 `status -> acquire -> wait` 循环，也不
调用 ack。

Controller 的 `seal_control_stream()` 将 sealed end 原子固定为当前 published end。完成的
source 不向任一 physical worker 返回 `None`，而是缩短现有 Driver drain，并在 terminal
checkpoint 后触发既有 `COMPLETING -> COMPLETED` 流程。sealed end 进入 ledger checkpoint；
恢复 terminal checkpoint 时保持所有 SPMD blocked，恢复完 model/Driver state 后直接复用同一
completion 路径，不重启 discovery。

全图 completion 仍使用一个 global logical step，因此所有 coordinated source 的 sealed end
必须相同。不同 source 的 checkpoint cut 可以是向量；不同 sealed end 不能自动取最小值或最大值，
否则分别会丢弃较长 source 的记录或要求较短 source 产生不存在的 batch。当前实现对此直接报错。
独立图分支若要在不同位置结束，必须先定义 source position 到各 Driver clock 的显式映射。

ledger 为每个 stream 独立保留 `[durable_cut, published_end)`。per-stream capacity 等于 checkpoint
interval；容量耗尽时 append 等待 checkpoint commit notification，try_append 立即返回。独立容量避免快 stream 饿死其他
stream。没有 checkpoint 时无法同时支持无限 control stream 与有界 retention，因此使用 control
stream 会直接报错。

完整代数模型与 topology independence 证明见
[Source SPMD Formal Protocol](source_spmd_formal_protocol.md)。

## Exactly-Once Invariants

对任意 stream：

```text
0 <= durable_cut <= published_end
retained positions = [durable_cut, published_end)
```

`report_op_completed()` 只服务 Fluxway 原生调度和流控，不推进 durable cut。全图 drain cut 为 `C`
时，Controller checkpoint 保存所有 `position >= C` 的 retained metadata。完整 checkpoint envelope
原子发布成功后才令 `durable_cut := C`。

```text
applied(C) = {position | position < C}
replay(C)  = {position | C <= position < published_end}
applied(C) intersection replay(C) = empty
```

故 checkpoint 前的 logical position 不会重放，checkpoint 后已预取或读取的 position 可以重放。
这保证 checkpoint-relative logical exactly-once，不声称对象存储物理读取或用户外部副作用
exactly-once。

## Drain 与 Checkpoint

Source coordinator 复用 Driver 原有 drain/snapshot：

```text
request_drain
  -> component DRAINING
  -> metadata discovery quiesced
  -> workers drain
  -> Driver DRAINED
  -> component checkpoint_state
  -> worker checkpoint participants
  -> Controller ControlLedger snapshot at cut C
  -> atomic checkpoint envelope publish
  -> finalize in-memory durable cut
```

Driver component checkpoint 只包含 source cursor、builder tail、至多一个尚未 append 的 control record、
尚未展开成 batch 的紧凑 revision ranges 与 next chunk、
source-domain lineage 和 terminal。Arrow、NumPy、`TransportTensor`、Ray handle 与 local prefetch
payload 不进入 checkpoint。

恢复时 Controller 先恢复 `ControlLedger`，Driver 再恢复 component metadata，最后恢复
worker/operator state。Dense/Orbax、sparse distributed prepare/commit、source metadata 和用户
per-worker pickle 都使用同一个 `CheckpointContext` 与现有 snapshot barrier。

artifact transaction 按三层组合：Worker 回滚本进程内所有已尝试 prepare 的 participant；Driver
在 prepare 或 rank save 失败时执行 worker artifact abort；全局 checkpointer 在任一 Driver、payload
prepare、distributed commit 或 publish 失败时 abort 所有 Driver。cleanup 自身失败使用异常组同时
保留主错误与 rollback 错误。只有完整 envelope publish 后才 finalize control ledger cut。

## 同步与异步 Source

同步 adapter 与异步 adapter 共用一个调用路径。`async def` 直接 await；阻塞的同步 catalog 或
filesystem 方法通过 executor 执行。Reader hot path 只等待 exact-position record 和 materialization，
不调用 catalog、`time.sleep()` 或 `Future.result()`。
