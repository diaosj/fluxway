# Checkpoints（快照与恢复）

快照（Checkpoint）是 Fluxway 在某一确定进度切面（Progress Cut）上的完整、可恢复状态。其公开标识（Public Identity）保持极简设计：

```text
checkpoint_id -> component_id -> checkpoint_object_id
```

* `checkpoint_id`：全局唯一标识一次已发布的快照事务；
* `component_id`：标识拥有该状态的子系统或组件，例如 `dense`、`reader` 或 `embedding`；
* `checkpoint_object_id`：当一个组件需要写入多个独立恢复的对象时用于区分。单对象组件默认使用 `"state"`。

Fluxway 拥有并裁决快照事务、进度切面（Cut）、拓扑结构、快照引用以及物理清理决策；而具体组件则负责序列化、物理存储、恢复映射以及物理清理执行。这种解耦使得 Fluxway 核心完全独立于具体的文件系统、对象存储、Lance 存储引擎、元数据 Catalog 及数据格式。

---

## 1. 保存与恢复（Saving & Restoring）

对于常规的同步训练任务，使用 `SaveCheckpointRequest.for_step(step)` 发起快照。它会自动创建唯一的保存尝试（Attempt ID）、Step 切面以及对应的 Step 引用。当某个快照需要同时被标记为 `best_model`、`final` 等别名时，可通过 `names` 指定：

```python
manifest = await checkpointer.save(
    SaveCheckpointRequest.for_step(1200, names=("best_model",))
)

await checkpointer.restore_step(1200)
await checkpointer.restore_named("best_model")
await checkpointer.restore_latest()
```

Step 引用是一种便捷的检索索引，而非强制的物理路径约定。文件系统组件通常会生成易读的 `steps/<step>/...` 目录结构，而其他组件亦可将同一快照持久化到 Catalog 或对象存储中。恢复时始终通过解析已发布的全局 Manifest 进行，应用层代码无需提供任何物理存储路径。

对于非纯 Step 驱动的流式计算任务，可构造带有 `CheckpointCut`（包含流切面与命名引用）的 `SaveCheckpointRequest`。切面是 Manifest 中唯一的全局进度事实。

### 持久化位点与瞬态运行状态（Durable Position & Transient Execution）

Cut 在运行期的核心语义是 `position`（全局持久化位点）：即所有算子持久化游标所对齐的单一确定边界。`CheckpointCut.position` 是 Manifest 的权威值；每个 SPMD 运行期投影将其记录为 `drain_boundary.committed_cut`，发布阶段将拒绝任何不一致的位点。Reader 消费游标、虚拟交错（Virtual-Interleave）单元以及模型更新步数可能采用不同的局部坐标系，各组件负责将持久化位点映射为自身的快照状态。

Worker 的运行期游标、Mailbox 邮箱、一致性租约（Coherence Leases）、Ready 就绪队列以及 Credit 观察值均属于**瞬态状态（Transient State）**。Drain 操作负责使持久化位点达到静止；Save 记录该静止位点；Restore 则在恢复时于该位点启动全新的空运行期环境。没有任何瞬态邮箱/一致性代际或信用重置属于快照契约的暴露接口。

### 数据源弹性与 Exactly-Once 语义

数据源（Source）快照定义在逻辑输出位点之上，而非绑定于具体的 Reader 类、物理 Rank 或迭代器实现。Source 组件持久化的状态必须能够精确重构其在 `CheckpointCut.position` 处的输出；Checkpointer 仅调用通用的组件 SPI，不对 Reader 进行任何特殊感知。对于 `GLOBAL_RESHARD` 组件，恢复状态与物理拓扑无关，当前的虚拟拥有者由 `(position, world_size, interleave_unit_size)` 动态推导。

这保证了快照相关的严格不变式：
$$\text{Committed} = [0, \text{position}) \quad \text{与} \quad \text{Replay} = [\text{position}, \infty)$$
两者交集严格为空，即使虚拟交错 Worker 的数量在恢复后发生改变依然成立。内部 Reader 的分批逻辑位于 Fluxway 的黑盒边界之外，其到逻辑输出位点的映射属于 Source 组件自身的状态契约。

---

## 2. 组件 SPI 与能力正交组合（Capability Composition）

`CheckpointComponent` 仅提供元数据描述符（Descriptor）。运行期行为由相互独立的正交 SPI 契约组合而成；组件无需暴露所有契约。

```python
class MyComponent(
    CheckpointComponent,
    CheckpointComponentLifecycle,
    CheckpointComponentFinalize,
    CheckpointComponentSave,
    CheckpointComponentRestore,
):
    @property
    def descriptor(self) -> CheckpointComponentSpec: ...

    async def open(self, runtime): ...
    async def close(self): ...
    async def finalize_save(self, session): ...
    async def prepare_save(self, session): ...
    async def save(self, session) -> WorkerSaveResult: ...
    async def finish_save(self, session): ...
    async def commit_save(self, session) -> CheckpointComponentPublication: ...
    async def rollback_save(self, session): ...
    async def publish_save(self, session): ...
    async def restore(self, session): ...
```

六大能力名称与六大 SPI 契约严格一一对应：
1. `COMPONENT_LIFECYCLE` (`open` / `close`)：组件资源生命周期；
2. `FINALIZE` (`finalize_save`)：本地运行态向规范静态的结清收敛；
3. `SAVE`：原子保存事务（`prepare` -> `save` -> `finish` -> `commit` -> `rollback` -> `publish`）；
4. `RESTORE` (`restore`)：状态加载与重构；
5. `OBJECT_LIFECYCLE`：快照对象的读取租约保护（Holds）与清理预留；
6. `OBJECT_CLEANUP` (`cleanup`)：物理快照废弃与垃圾回收。

---

## 3. Worker 事务生命周期：FINALIZING 与 PREPARED 的本质区别

在 Worker 内部，`WorkerCheckpointCoordinator.prepare_save()` 严格将 **内存算法状态规范化** 与 **外部存储资源预留** 分离：

```text
WorkerCheckpointCoordinator.prepare_save()
  │
  ├─ 1. FINALIZING 阶段 (CheckpointComponentFinalize.finalize_save)
  │      └─ 本地内存/显存模型规范化 (例如: 结清未满窗口的残余累积梯度)
  │         [执行范围: ALL_RANKS, 故障语义: Fail-Stop 强停, 存储 I/O: 无]
  │
  ├─ 2. 存储 Staging 预留阶段 (CheckpointComponentSave.prepare_save)
  │      └─ 创建临时 Staging 目录 / 预留云存储租约 (例如: mkdir, 排他锁)
  │         [执行范围: LEADER 或 ALL_RANKS, 故障语义: 可回滚, 存储 I/O: 有]
  │
  ├─ 3. 状态转为 PREPARED
  │
  ▼  [Driver SPMD-Local 屏障: 等待本 SPMD 内所有 Rank 全部达到 PREPARED]
SAVING 阶段 (CheckpointComponentSave.save)
  └─ 并发序列化并上传分片 (此时所有 Worker 显存/内存已 100% 冻结且自洽)
```

### 关键维度深度对比

| 评估维度 | FINALIZING 阶段 (`finalize_save`) | PREPARED 阶段 (`prepare_save`) |
| :--- | :--- | :--- |
| **操作对象** | 本地显存 / 内存 / 优化器一二阶动量 | 磁盘 Staging 目录 / 对象存储 Bucket / 租约 |
| **执行范围 (Scope)** | **必须为 `ALL_RANKS`**（全节点参与） | 可配置为 `LEADER`（仅 Rank 0）或 `ALL_RANKS` |
| **故障语义 (Failure)** | **Fail-Stop 强停**（`SNAPSHOT_FAILED -> ERROR`） | **可安全回滚**（Abort 并删除临时 Staging 目录） |
| **核心职责** | 将动态运算态（在途梯度）结清为静止规范态 | 为落盘数据预留物理存储资源与坑位 |
| **存储 I/O** | **绝对无存储 I/O** | 包含文件系统与远端网络 I/O |

* `FINALIZE` 必须在任何存储组件创建目录或写入字节前执行。它是已 Drain 的训练运行期在序列化模型之前应用**部分累加窗口（Partial Accumulation Window）**的标准扩展点；
* 完整 Worker 事务顺序为：
  ```text
  FINALIZING -> 所有本地 FINALIZE -> SAVE.prepare -> PREPARED -> SAVE.save -> SAVE.finish -> SAVE.commit -> SAVE.publish
  ```
* Driver 自然等待所有 Rank 达到 `PREPARED` 后才触发 `SAVE.save`，天然构成分布式本地屏障，无需 Controller 引入额外的全图调度阶段。

---

## 4. 分层 API 架构（Layered Developer Experience）

为了避免底层状态机与事务概念侵入业务层，Fluxway 采用严格的分层关注点分离：

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ Level 1: 业务算法与模型开发者（95% 的用户）                                   │
│ 体验：零状态机概念。只传入普通的 Python Callables。                            │
│ 示例：OrbaxCheckpointComponent(..., finalize_state=flush_fn)              │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │ 框架自动能力派发
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Level 2: 存储插件开发者（5% 的用户，如扩展 Lance/Parquet 引擎）                │
│ 体验：按需实现 CheckpointComponentSave / CheckpointComponentFinalize SPI   │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │ 内部事务保护与防御断言
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Level 3: Fluxway 引擎内核（Coordinator / Orchestrator）                    │
│ 体验：FINALIZING、PREPARED、SAVING、COMMITTED 强防御性状态机断言              │
└──────────────────────────────────────────────────────────────────────────┘
```

### 业务端注册示例（以 Dense 模型为例）

业务方仅需按需声明回调，完全不感知底层事务状态：

```python
@spmd(
    resource_profile=ResourceProfile(base={"CPU": 1, "memory": 1024**3}),
    mesh_axis_names=("dp",),
    checkpoint_components=(
        OrbaxCheckpointComponent(
            "dense",
            save_state=lambda owner, ctx: owner.save_checkpoint_state(ctx),
            restore_state=lambda owner, state, ctx: owner.restore_checkpoint_state(
                state, ctx
            ),
            restore_layout=lambda owner, ctx: owner.restore_checkpoint_layout(ctx),
            finalize_state=lambda owner, ctx: owner.finalize_checkpoint_state(ctx),
        ).registration(),
    ),
)
class DenseSPMD:
    def finalize_checkpoint_state(self, context: SaveSession) -> None:
        if self._pending_accum_steps > 0:
            scale = 1.0 / self._pending_accum_steps
            params = nnx.state(self.model)
            # 按实际累积微步数 k 进行无偏均值更新
            updated = jtu.tree_map(
                lambda p, g: p - 0.01 * (g * scale), params, self.grad_accum
            )
            nnx.update(self.model, updated)
            self.grad_accum = jtu.tree_map(jnp.zeros_like, self.grad_accum)
            self._pending_accum_steps = 0
```

---

## 5. 组件注册、租约与生命周期清理

声明了有限保留策略 `CheckpointRetentionPolicy(retain_last=N)` 的组件，必须同时注册 `CheckpointObjectLifecycle` 与 `CheckpointObjectCleanup` 能力，否则注册将拒绝通过（禁止静默忽略清理）：

```python
CheckpointComponentSpec(
    component_id="dense",
    save=CheckpointSaveSpec(),
    restore=CheckpointRestoreSpec(),
    checkpoint_objects=(
        CheckpointObjectSpec("model_state", retention=CheckpointRetentionPolicy(retain_last=3)),
        CheckpointObjectSpec("optimizer_state", retention=CheckpointRetentionPolicy(retain_last=3)),
    ),
)
```

* 每个声明的快照对象独立进行保留计数、租约保护与物理清理；
* `CheckpointHold` 用于在其他消费者读取时提供共享读保护，带代际校验与自动续约；
* 当超出保留数量时，Fluxway 自动发起原子清理预留，检查 Hold，并在新 Manifest 发布后执行物理垃圾回收。被 Hold 阻塞的清理将抛出 `CheckpointCleanupBlocked` 并挂起重试，绝不静默假成功。

---

## 6. 提交许可（Commit Permits）与防脑裂

每次保存拥有三重独立身份标识：
* `checkpoint_id`：最终发布的逻辑快照 ID；
* `attempt_id`：当前单次保存尝试的 ID；
* `CheckpointCommitPermit`：当前 Coordinator 发布该尝试的权威代际 Token。

Fluxway 会持久化 Coordinator 代际。新主接管仓库将直接废除旧 Coordinator 的 Permit。组件在 Save Session 中获取 Permit，并可调用 `session.commit_authority(scope)` 将其传递给具备 CAS（Compare-And-Swap）或租约能力的底层存储后端。Permit Token 永远不会被持久化到 Manifest 中，Restore 与 `close()` 操作也无需 Permit。

---

## 7. 拓扑模式（Topology Modes）

Manifest 记录源拓扑信息。恢复时组件显式选择以下三种拓扑模式之一：
* `EXACT_TOPOLOGY`：要求恢复时的 World Size 与切分严格等于保存时的拓扑，适用于 Rank 本地迭代器或硬绑定的进程状态；
* `LOGICAL_PARTITION_REMAP`：适用于逻辑分片稳定的系统（如 Embedding 查表按 Modulo 映射重组）；
* `GLOBAL_RESHARD`：适用于全局分布式张量（如 JAX / PyTorch DTensor），由组件自身在恢复时自动 Reshard 到新网格。

Fluxway 绝不对分区函数做隐式猜测，也绝不静默拷贝 Rank 本地状态。所有的弹性恢复语义均由组件适配层严格保证。
