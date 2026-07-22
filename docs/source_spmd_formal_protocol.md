# Source SPMD 形式化协议

## 1. 结论

Source SPMD 不需要 source 专属的调度、barrier、ack 或 checkpoint 接口。Fluxway 只需提供一个
领域无关的有序控制流模块，Source Coordinator 和 source operator 分别作为生产端与消费端使用它。

该模块对调用者暴露五个操作：

```python
append(records: Sequence[ControlRecord]) -> Awaitable[None]
try_append(records: Sequence[ControlRecord]) -> Awaitable[bool]
receive(stream_id: str, position: int) -> Awaitable[ControlRecord | None]
close(stream_id: str) -> Awaitable[None]
progress(stream_id: str) -> Awaitable[StreamProgress]
```

- `append` 原子、幂等并提供异步背压。
- `try_append` 与 `append` 共用同一个状态转换，容量不足时不等待并返回 `False`。
- `receive` 等待精确位置；有记录时返回记录。普通有限 stream 可在终点返回 `None`，interleaved
  source 的 sealed terminal 不向单个 physical worker 返回 `None`。
- `close` 原子固定 stream 的有限终点；Controller 将 source close 转成 sealed terminal 与原生
  completion，空轮询永远不等于 close。
- `progress` 只返回 durable checkpoint cut、terminal end 和观测信息，不暴露内部 queue 状态。

`receive` 内部封装 condition notification，因此不再把 `status -> try_claim -> wait -> retry` 状态机
泄漏给 source 实现。checkpoint 的 prepare、commit 和 restore 复用 Fluxway 现有 checkpoint 控制链，
不是 Source 用户接口。

## 2. 代数对象

设：

```text
S                  stream identity 集合
N                  非负整数集合
K = S x N          稳定记录身份空间
V                  对 Fluxway 不透明、可 checkpoint 的 metadata payload 集合
L: K -> V          有限偏函数，即有序控制记录
```

每个 stream `s` 有三个单调水位：

```text
d_s    durable cut，所有 n < d_s 已包含在原子 checkpoint
h_s    published end，所有 n < h_s 已发布
z_s    terminal end；未 close 时无定义，close 后 z_s = h_s
sigma_s terminal 是否由宿主 completion 消化的 sealed bit
```

第一次 `append`、`receive`、`close` 或 `progress` 会建立 stream identity。因而尚未产生第一条记录的
open stream 也能以 `d_s = h_s = 0` 参与 checkpoint，不需要额外的 source-specific register 操作。

在任意状态都要求：

```text
0 <= d_s <= h_s
domain(L_s) = [d_s, h_s)
z_s 有定义 => z_s = h_s
sigma_s => z_s 有定义
```

这里的 cut 使用半开区间计数，不使用 `last_position = cut - 1`。这消除了空 stream 需要 `-1`
哨兵值的重复语义。`revision` 仅属于具体 source adapter 的领域 cursor；Fluxway 控制流不再引入
`generation`。新 campaign 使用新的 `stream_id`。

## 3. 状态转换

### Append

`append([r_0, ..., r_k])` 在一个线性化点原子发生：

```text
key(r_i) = (s, n)
n in [d_s, h_s) 且 L(s,n) = r_i  => 幂等 no-op
n = h_s                           => L(s,n) := r_i; h_s := h_s + 1
n < d_s                           => stale error
n > h_s                           => gap error
z_s 有定义                         => terminal error
```

整个 batch 必须先验证容量与连续性，再整体可见。不能逐条接受后在等待容量时被取消，否则
Coordinator 无法判断哪一部分已经发布。容量对每个 stream 独立保留：

```text
forall s: |[d_s, h_s)| <= B
```

达到 `B` 时 `append` 通过异步 notification 等待 checkpoint 释放 metadata，不轮询、不阻塞 event
loop。当前运行的 `B` 等于 checkpoint interval；独立容量避免快 stream 饿死其他 stream。`B` 是运行时
背压参数，不是 durable state，也不进入 checkpoint。恢复只要求每个 replay 区间满足
`h_s-d_s <= B_runtime`，因此 checkpoint interval 可以随重启改变。没有 checkpoint 时无法同时保证无限
control stream 和有界 retention，因此使用 control stream 会直接报错。

`try_append` 在相同线性化点执行同一验证。若候选状态超过 `B`，状态保持不变并返回 `False`；否则与
`append` 的成功结果完全相同。因而它不增加新的状态空间，只让 Driver component 能在 drain 前保留
一个紧凑 pending revision 并停止 discovery。

### Receive

对精确键 `(s,n)`：

```text
(s,n) in domain(L)                         => 返回唯一的 L(s,n)
z_s 有定义且 not sigma_s 且 n >= z_s       => 返回 None
其他                                       => await 状态版本变化后重试
```

`receive` 是共享幂等读取，不是排他 lease。一个 active virtual group 内的多个 rank 获取相同 metadata，
所以接口不使用容易暗示排他所有权和超时的 `claim/lease/ack` 术语。

### Close

```text
close(s): z_s := h_s
seal(s):  z_s := h_s; sigma_s := true
```

重复 close 幂等；close 后 append 非法。terminal 是 producer 的显式事实，不能由 timeout、空 catalog
结果或 Reader 的 `None` 猜测。

## 4. Virtual SPMD 映射

给定当前物理 world size `w` 和 interleave unit size `u`，要求 `u | w`。令：

```text
q = w / u
group(n) = n mod q
O_(w,u)(n) = {group(n)u, ..., group(n)u + u - 1}
```

于是一个周期 `[0,q)` 内：

```text
forall i != j: O(i) intersection O(j) = empty
union_i O(i) = {0, ..., w-1}
|O(i)| = u
```

`O_(w,u)` 是 Fluxway dispatcher 根据当前拓扑计算的执行函数，不属于 `ControlRecord`、Coordinator
state 或 checkpoint。持久身份始终只有 `(stream_id, position)`。

设 payload `L(s,n)` 表示一个完整 global batch `B_n`。当前 active group 的 materializer 必须满足：

```text
B_n = disjoint_union_(r=0..u-1) shard(B_n, r, u)
```

因此重启前后使用 `u_0` 和 `u_1` 时，物理 partition 可以改变，但 global batch 集合与顺序不变。
把旧 `owner_unit`、worker rank 或 world size 写入 payload 会违反该性质。

## 5. Checkpoint 算法

Fluxway drain 后从每个 source operator 的 canonical `completed_step_count` 构造 cut 向量：

```text
kappa: S -> N
kappa(s) = 对应 operator 在全图 drain 中完成的 logical position count
```

使用向量而不是单个 scalar，使多个 source operator、periodic operator 和未来的不同 step 映射仍共享
同一算法。source worker 的文件读取完成不是提交点；只有全图 drain cut 和模型状态一起原子落盘才是
逻辑提交点。

在所有 Driver component 已进入 `DRAINED` 且 discovery 已 quiesce 后：

```text
prepare(kappa):
    require forall s: d_s <= kappa(s) <= h_s
    snapshot.records = {(s,n,L(s,n)) | kappa(s) <= n < h_s}
    snapshot.durable_cuts = kappa
    snapshot.published_ends = h
    snapshot.terminal_ends = z
    snapshot.sealed_streams = sigma

commit(snapshot):
    仅在完整 checkpoint envelope 原子发布成功后令 d := kappa
    删除 n < d_s 的旧 metadata

restore(snapshot):
    d := snapshot.durable_cuts
    h := snapshot.published_ends
    require forall s: h_s - d_s <= B_runtime
    所有 snapshot.records 恢复为可 receive
    sealed terminal checkpoint 恢复后直接进入宿主 completion，不 resume source
    不恢复物理 owner、local prefetch 或 payload materialization
```

Driver-owned Source Coordinator 独立保存：

```text
source cursor
未组成完整 batch 的 range tail
至多一个尚未 append 的完整 record
尚未展开的 revision ranges 与单调 next chunk
next position
terminal discovery fact
```

设 revision 有 `m` 个固定大小 record chunk。可选 shuffle 使用仿射双射
`p(i)=(a*i+b) mod m`，其中 `gcd(a,m)=1`。checkpoint 只保存 `next_chunk=i`，`a,b` 由稳定 seed 与
revision identity 重算，因此无需保存 permutation，且 `{p(i) | 0<=i<m}=[0,m)` 保证无重无漏。

令 ledger key 集合为 `K_L = [d_s,h_s)`，Coordinator 尚未 append 的 key 集合为
`K_U = [h_s,g_s)`，必须满足 `K_L intersection K_U = empty`。Arrow、NumPy、
`TransportTensor`、Ray handle 和本地 ready queue 都不进入 checkpoint。

### Whole-job terminal domain

Checkpoint cut `kappa` is a per-stream vector, but the current whole-job drain has
one global logical clock. Its terminal transition is therefore defined only on:

```text
Z_global = {z in N^S | exists T in N, forall s in S: z_s = T}
```

For `z in Z_global`, every Driver drains to the same `T`. For `z` outside this set,
using `min(z)` would discard valid records from a longer source and using `max(z)`
would require nonexistent batches from a shorter source. Fluxway rejects that state
instead of guessing a completion policy. Independently terminating graph branches
require an explicit source-position-to-Driver-clock morphism and are not part of the
current scalar whole-job completion protocol.

## 6. Safety 定理

对任意合法 checkpoint cut `kappa`，定义：

```text
Produced_s  = [0, h_s)
Committed_s = [0, kappa(s))
Replay_s    = [kappa(s), h_s)
```

整数全序与半开区间直接给出：

```text
Committed_s intersection Replay_s = empty
Committed_s union Replay_s = Produced_s
```

所以：

1. checkpoint 前的 logical positions 不会恢复；
2. checkpoint 后已经预取或读取但未提交的 positions 全部恢复；
3. 没有 position 同时属于提交集合与重放集合；
4. checkpoint 与模型、dense、sparse state 在同一 envelope 原子提交，因此模型状态和 source cut 一致。

这证明的是 checkpoint-relative logical exactly-once。用户 operator 的外部副作用仍需事务或业务幂等，
对象存储的物理读取也允许重复。

## 7. Elasticity 定理

设 checkpoint 前后拓扑为 `t_0=(w_0,u_0)` 与 `t_1=(w_1,u_1)`。因为：

```text
domain(L) subset S x N
O_t 不属于 L、d、h、z 或 Coordinator state
```

恢复集合 `Replay_s` 与 `t_0`、`t_1` 无关。恢复后 position `n` 由 `O_t1(n)` 执行，所有 active
rank 通过共享幂等 `receive(s,n)` 得到同一 payload，并按 `u_1` 重新 partition。故 virtual source
SPMD 数量或 unit size 改变不会遗留旧 owner，也不会改变 global batch。

## 8. Liveness 与性能条件

活性不是无条件安全性质。需要：

1. producer 最终 append 目标 position 或显式 close；
2. Fluxway async scheduler 公平；
3. condition notification 不丢失唤醒；
4. source discovery/materialization 最终返回或进入现有 failure channel；
5. 同一 stream 同一时刻只有一个有效 producer；
6. checkpoint cadence 与容量满足 `forall s: h_s-d_s <= B`；
7. drain 先 quiesce Driver component，再生成 ledger snapshot。

在这些条件下，`receive` 最终返回记录；普通 finite stream 返回 terminal，sealed source 则由宿主
完成整个图；失败进入现有 failure channel。等待只挂起当前 coroutine；其他 operator、transport 和
control task 继续运行。控制路径复杂度为：

```text
append k records     O(k)
receive exact key    O(1)
progress             O(1)
checkpoint           O(number of replay metadata)
```

catalog discovery 只由 source Driver 的一个 Coordinator task 执行；worker 热路径只 materialize 已收到的
metadata，不轮询 catalog。

## 9. 反例

### 持久化 owner

position `2` 在 `(w=2,u=1)` 时属于 rank `0`，在 `(w=3,u=1)` 时属于 rank `2`。保存旧 owner 会使
恢复后的合法 dispatcher 与持久状态冲突。

### 用 source read frontier 提交

若 source 已读取到 cut `4`，但全图模型只原子 checkpoint 到 cut `2`，按 source frontier 删除
`{0,1,2,3}` 会丢失恢复时必须重放的 `{2,3}`。因此 `report_op_completed(source)` 可以用于流控，
不能独立推进 durable source cut。

## 10. 机器校验

可使用 Fluxway 自动化测试套件执行形式化模型与分发逻辑校验：

```bash
pytest tests/control_plane/test_control_ledger.py tests/control_plane/test_components.py
```

形式化校验说明：
1. **代数状态空间穷举**：穷举双 stream 的 published/cut/terminal 状态，验证集合分割、幂等 receive、原子 append、checkpoint restore、interleave partition 和 topology change，并验证“持久 owner”与“source 局部完成即提交”两个反例的判定。
2. **Dispatcher 精化验证**：直接导入 Fluxway `InterleavedDispatcher`，验证真实 dispatcher 与本规范的 $O_{(w,u)}(n)$ 在小规模拓扑和 step 上满足精确等价契约。
