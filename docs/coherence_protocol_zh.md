# 端点本地 MESI 目录协议

本文是 Fluxway 连贯性协议的规范说明。MESI 在这里表示软件访问权限状态机，不表示
Python 或加速器内存具备硬件缓存一致性。

## 状态与唯一事实来源

每个逻辑张量的主体是 owner SPMD、owner rank 和逻辑张量身份；连贯性实例再加上
epoch。逻辑张量身份和路由只来自编译后的 TensorTransfer。传输张量必须携带已绑定的
连贯性实例，source step 只记录来源，不创建或推进目录 epoch。目录不存在第二个可变
epoch 计数器、全局目录、兄弟持有者列表或跨层级广播。

端点目录项包含 epoch、MESI 状态、父级授权、读取授权、写授权、独占子端点、子端点
位图、传输回执、待归还事务、挂起请求和分配绑定。状态含义如下：

| 状态 | 含义 |
| --- | --- |
| M | 当前端点拥有唯一写权限，载荷可能已经修改。 |
| E | 当前端点是唯一有效持有者，尚未开始修改。 |
| S | 受控本地读取者或直接子端点可以访问载荷。 |
| I | 当前端点没有可访问的有效载荷。 |

系统每步状态转换必须严格保持以下形式化不变量：

- **I1（写者排他性）**：`writer != NONE => readers = 0 且 ownership_child = NONE 且 child_bitmap = 0`；
- **I2（独占子端点排他性）**：`ownership_child != NONE => writer = NONE 且 readers = 0 且 child_bitmap = 0 且 mesi = I`；
- **I3（父级所有权委派）**：`parent_grant.authority = OWNERSHIP => mesi in {M, I} 且直接父端点处于 I`；
- **I4（失效态保护）**：`mesi = I => PrRd、PrWr 和 BusRd 委托均不可成为 GrantReady`；
- **I5（Epoch 静默推进）**：推进 epoch 前必须完全排空本地读者、写者、独占子端点、下游子端点位图、在途/已确认交付凭证、父级授权和挂起请求；
- **I6（防陈旧推进）**：`incoming_epoch < epoch => 抛出 StaleEpochError 且目录项 D 保持不变`；
- **I7（子端点位图闭包）**：`delivery_committed`、`return_pending` 以及具有在途交付的子端点均严格属于 `child_bitmap`；
- **I8（令牌线性消费）**：所有读/写/交付/归还令牌必须且只能精确消费一次（Linear Consumption）。

CoherenceDirectory.admit 是唯一的同步准入实现。它重新读取目录项，只有完整静默条件
满足时才推进 epoch。异步等待由运行时观察目录进展并挂起请求协程，不使用轮询，也不
在目录锁内等待。CoherenceSequencer 是唯一的异步重排缓冲区。

## Consumer Lease Reservation 所有权

`ConsumerLeaseReservationRef` 是 Tensor 持有的非拥有引用，真正的 reservation 由
Sequencer 唯一拥有。可变状态和 active consumer 计数绑定只存在于 Sequencer：

~~~mermaid
stateDiagram-v2
    [*] --> RESERVED
    RESERVED --> CLAIMED: claim()
    RESERVED --> RELEASED: cancel()
    RESERVED --> RELEASED: release_for_cleanup()
    CLAIMED --> RELEASED: ConsumerLease.release()
    CLAIMED --> RELEASED: release_for_cleanup()
    RELEASED --> [*]
~~~

| 阶段 | 责任者 | 合法操作 | 结果 |
| --- | --- | --- | --- |
| `RESERVED` | Sequencer reservation 状态 | `claim()`、`cancel()`、cleanup | claim 转移唯一 lease owner；cancel/cleanup 释放预留计数。 |
| `CLAIMED` | 保存的 `ConsumerLease` | lease `release()`；tensor handoff 前的 cleanup | 保存的 lease exactly-once 释放 active 计数；`cancel()` 抛异常。 |
| `RELEASED` | 无 | 正常 lease release 非法；重复 shutdown cleanup 已完成 | 重复显式 `ConsumerLease.release()` 抛错；重复 shutdown cleanup 幂等完成。 |

`claim()` 在 Sequencer 唯一状态机内建立 lease。`TransportTensor` 只保存
`ConsumerLeaseReservationRef`；完成 `acquire_consumer_lease()` 后清空该引用，由
`ConsumerLease` 负责正常释放。如果 shutdown 与 claim 交错，两者访问同一个 Sequencer
状态机，只有一个状态转换真正释放计数，另一个只观察到已完成。Worker shutdown 不能把
claimed reservation 当成可成功 `cancel()` 的对象。

reservation 索引另有线程边界保护；sequencer fence 在该边界内取得快照，再对快照逐个调用
`release_for_cleanup()`。因此 claim 与 fence 交错时不会遗留 active 计数。来自 worker
线程的 `ConsumerLease.release()` 会把计数变更排入 sequencer owner event loop，重复
release 直接抛异常。

## 状态转换

~~~mermaid
stateDiagram-v2
  [*] --> I
  I --> E: 本地发布或新交接
  I --> S: 带载荷的 BusRd
  E --> S: 本地读取或下游共享
  S --> M: 内部升级或获准 RFO
  E --> M: 本地写入
  M --> E: 干净写者归还
  M --> I: 旧主体交接完成
  E --> I: 交接独占权限
  S --> I: 共享权限排空后失效
  I --> I: 静默 epoch 推进
~~~

图中只表示合法状态类别；每条转换仍需满足目录项的完整前置条件。

## 线路动作

| 动作 | 载荷 | 方向 |
| --- | --- | --- |
| BUS_RD | 必须有 | 父到子 |
| RFO | 禁止 | 子到父 |
| SNOOP_INVALIDATE | 禁止 | 父到子 |
| GRANT_RETURN | 禁止 | 子到父 |
| OWNERSHIP_GRANT | 禁止 | 父到子 |
| OWNERSHIP_CANCEL | 禁止 | 子到父 |

张量传输帧只携带 BUS_RD。连贯性帧不携带数据载荷，在邮箱写入前完成处理。不同
artifact 可以并发发送，但同一帧内的动作和 epoch 必须一致。

## 载荷交付、归还与所有权

同一语义张量可以通过本地引用、本地邮箱或远程 DLPack 载荷交付。载荷承载的主体、
逻辑张量身份、epoch 和授权语义不因载体变化而改变。操作系统线程必须把意图提交到
Worker 事件循环，不能直接修改目录项。

BusRd 的顺序是：发送端登记子端点位和交付计数；接收端校验版本、路由、动作和 epoch
后安装共享授权；交付成功不等于归还；子端点完成算子、别名和后代读取后才开始归还；
归还事务清除本地稳定授权并发送一次 GRANT_RETURN；交付失败只撤销本次委派。

RFO 只允许已有共享者请求。父端点先阻止新的 BusRd，再等待交付完成并撤销其他共享
端点；共享清空后进入 I 并发送 OWNERSHIP_GRANT。OWNERSHIP_CANCEL 只在 CALL 前有效；
CALL 开始后发生错误必须封锁整个执行，不能假设旧数据仍然干净。

输入释放是固定生命周期：释放本地访问令牌，等待直接子端点、后代、别名和交付回执
静默，失效张量访问，完成 GRANT_RETURN，释放物理载荷，最后消费 epoch 级归还证明或
当前目录退休证明。传输完成证明和连贯性退休证明是两个独立事实，调用方必须同时取得。
PayloadLifetime 统一释放物理载荷一次；指针整数、DLPack capsule 或 RPC 超时都不是
释放证明。

## 分配交接与回收

分配绑定包含后端、设备、根地址和字节范围，并强引用实际存储所有者。只有根对象和
字节范围完全相同的交接才是 EXACT；重叠或无法识别的身份直接失败。旧 wrapper 失效，
外部裸别名属于未定义行为。跨端点复制得到新分配，不能复用旧分配身份。

AllocationRegistry 只负责身份和精确交接，不能代替物理载荷生命周期。只有目录退休
证明完成后才能解除绑定；缺失回执、重复终态或回收失败都会封锁执行。

## 安全、进展与封锁

写者进入前必须清空读取和子端点，独占委派会撤销父端点有效性，epoch 只在完整静默
后推进。进展依赖受控读取者归还和直接子端点响应；等待使用事件通知，没有全局扫描、
轮询或全局锁。挂起写者会阻止新的读取，避免饥饿。

出现不变量破坏、传输失败或执行错误时，Fluxway 关闭帧准入，取消准备、CALL、发送、
交付和挂起任务，关闭传输端点，清理目录和分配绑定，并等待所有 Worker actor 明确终止。
完成这道屏障后，Controller 才能从检查点创建新执行。协议不在同一执行中替换单个 rank。

## 术语边界

用户层使用 readonly、writable、donate；运行时使用 ShareGrant、OwnershipGrant、
GrantReturn、ParentAuthority、Epoch 和 Allocation Handoff；线路层使用 BusRd、RFO、
SnoopInval、OwnershipGrant 和 OwnershipCancel。BUS_UPGR 只表示本地 S 到 M 的转换，
不在线路上传输。Fluxway 不实现硬件意义上的 FLUSH；远程写回直接失败，状态变化通过
新的 SSA 张量流表示。
