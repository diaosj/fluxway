# 端点本地 MESI 目录协议 (Endpoint-Local MESI Directory Protocol)

本文档是规范性的 Fluxway 连贯性（coherence）规范，它是 [coherence_protocol.md](coherence_protocol.md) 的中文对应版本。

## 端点本地 MESI 目录连贯性树 (Endpoint-Local MESI Directory Coherence Tree)

MESI（Modified, Exclusive, Shared, Invalid，修改、独占、共享、无效）描述一个张量（tensor）实例在端点本地目录中的访问权限，不描述 Python 对象类型。

TUUID（Tensor Universally Unique Identifier，张量通用唯一标识）是逻辑张量的身份（identity）。DLPack（Deep Learning Pack）是跨组件的张量数据负载接口。BusRd（Bus Read）对应网络协议动作 `BUS_RD`；嗅探无效化（Snoop Invalidate）对应 `SNOOP_INVALIDATE`；授权归还（Grant Return）对应 `GRANT_RETURN`。

Fluxway 使用端点本地的分层目录。每个端点仅维护本地状态、直接父级的授权（direct parent authority）和直接子级的共享位图（direct-child sharing bitmap）；禁止使用全局持有者列表（global holder list）、跨层广播以及全局连贯性锁。任何状态转换都只访问一个条目及其直接的关联边，延迟只会对相关子树产生反压。

用户接口仅包含 `TensorAccessIntent.READONLY`、`WRITABLE` 和 `DONATE`，并根据算子的数据输入端口进行声明：

```python
@operator(tensor_access={
    "input_ids": TensorAccessIntent.READONLY,
    "cache": TensorAccessIntent.WRITABLE,
    "activation": TensorAccessIntent.DONATE,
})
def forward(self, input_ids, cache, activation): ...
```

未声明的数据输入端口默认使用 `READONLY`；mirror/follow 只继承接收/源谱系，不改变访问意图。图形编译、仅限于本地的快速路径和 worker `CALL` 均消费这一份冻结的 `OperatorSpec.tensor_access`，不再从 `AsyncPolicy` 或 `StatePolicy` 中推导第二套事实。运行时在 `CALL` 之前将意图转换为本地授权（grant）或 RFO（请求所有权）；在写授权尚未就绪时协程将被挂起，用户函数绝对不会被提前执行。

当前线路协议仅实现 `BUS_RD`、`RFO`、`SNOOP_INVALIDATE`、`GRANT_RETURN`、`OWNERSHIP_GRANT` 以及在 CALL 之前的 `OWNERSHIP_CANCEL`。`BUS_UPGR` 是目录内部 `S -> M` 的概念转换，不生成数据帧。在 MESI 中 `FLUSH` 表示脏数据的回写，但是复制的 Ray/Axon 数据面无法使用仅带头文件的数据帧来恢复父级的负载，因此执行该协议时将其动作码保留为非法值，不实现远程的回写操作。

## 生命周期与归还控制

父级授权（parent grant）的归还由 worker 本地的 `ParentGrantReturnRuntime` 统一管理。worker 只提交 `ParentReturnRoute`；该模块在发送 `GRANT_RETURN` 之前以原子方式安装显式的返回事务、清除稳定的父级授权，根据直接父级和 epoch 聚合数据帧，并使用一次性的令牌（token）完成事务。交叉到达的同 epoch `SNOOP_INVALIDATE` 会被目录合并到已有的返回事务中，未来 epoch 的 `BUS_RD` 可将该事务作为父端已观察归还的凭证。该路径保持事件循环（event-loop）无锁且操作具有 O(1) 的目录开销。

输入的生命周期由 `InputCoherenceReleasePlan` 和 `InputCoherenceRelease` 组成。释放操作按“释放本地 token -> 授权/别名/后代 quiescent -> 张量失效 -> 父级归还 -> 负载释放 -> 退休证明”的顺序完成。

Epoch 转换也是如此，只有一个事实来源。`CoherenceInstance(subject, epoch)` 是唯一的连贯性身份；`CoherenceDirectory.admit()` 是唯一的同步准入入口。乱序的数据帧构建不进入目录，而是通过 `CoherenceSequencer` 的每个 subject 的重排序缓冲区（ROB）进行收敛。目录仅保留同步协议事实，不持有 worker 任务或传输队列；整个路径不涉及轮询，也不阻塞事件循环。

有关状态定义、跃迁推导、线路动作与完整语义边界的细节，均以英文版本的详细说明为准。
