# 传输计划

## 平面与所有权

`CompiledGraph` 拥有算子与传输声明。`TensorTransfer.unit_routing` 可声明完整 unit
传输关系，不依赖 `is_source`。`CompiledDependency` 将声明绑定到活跃 producer
与 consumer 的 rank 域，`CompiledSendPlan` 将结果降低为本地 sender 执行指令。

数据面沿现有 receipt 和 coherence 协议组织 tensor components、构建 frame 并投递。
它不分配 source work、不推进 checkpoint cut，也不决定算子执行哪个实例。
这些职责分别属于 `LogicalWorkLedger`、Controller admission 和 Dispatcher。

## 数学契约

在执行实例 `e` 中，producer `p` 产生完整单元 `U(e,p)`。整批路由声明函数
`f: C -> P`，consumer `c` 恰好接收 `U(e,f(c))`。编译关系为
`R = {(f(c),c) | c in C}`，两个互逆邻接表为：

- `producers_by_target[c] = (f(c),)`；
- `consumers_by_source[p] = {c | f(c) = p}`。

编译器校验两个 rank 域、每个 target 的单 producer 完整性，以及全部活跃 producer
的覆盖。双射表达 permutation；重复 producer ID 表达 replication 或 broadcast。
遗漏 producer 是错误，不能隐式丢弃。多个完整 unit 不能在同一执行实例中悄悄变成
一个 tensor input；需使用不同执行实例或显式 input ports 保持它们独立。

`WholeUnitRouting((1, 0, 1, 0))` 将两个 producer unit 交给四个 consumer：
consumer 0、2 接收 producer 1，consumer 1、3 接收 producer 0。每个目标接收
完整 payload；选择显式 unit relation 后，`input_sharding` 不再隐式请求二次拆分。

未声明 `unit_routing` 时，沿现有 `output_sharding` 与 `input_sharding` 分片交集
计算 fragment 路由。多个 producer fragment 可以装配为一个 target tensor。
内容寻址和基于 receipt 的反向路由保留各自语义；同一 transfer 同时声明整批路由
与 receipt 路由会被拒绝。

Receipt 路由优先于静态 shard 几何。源和目标布局相同，也不能让依赖 receipt 的
transfer 走静态 direct route；目的端仍由 receipt 决定。

## 执行组

`ExecutionGroup` 包含有序物理 rank 与规范化逻辑 mesh。布局代数使用组内逻辑坐标；
frame header 与 coherence subject 保留物理 sender rank。非法 rank 查询直接抛错。

`CompiledOperator.execution_layout` 唯一保存不可变周期和 rank 反向索引。图编译器
一次性将 `OperatorSpec.execution_groups` 绑定到物理 mesh；各组大小必须相等、
完整且不重复地覆盖物理 rank，并保留组内逻辑 mesh 的尾部维度。未显式声明时由
编译器生成连续组。Dispatcher、source planning、控制消息 leader、发送计划和
placement 都使用该结果，不再在运行时用 rank 算术定义另一套分组。

例如，在具有 16 个 Reader 和 4 个 Input 的典型训练负载中，声明可以分为 `(0,4,8,12)`、`(1,5,9,13)`、
`(2,6,10,14)`、`(3,7,11,15)` 四个执行组。因此 Reader 0～3 依次供应 Input 0，
Reader 4～7 供应 Input 1，依此类推。Placement 引擎要求 Reader 与其目标 Input 同节点。
用户的固定布局约束若与声明分组冲突，编译或 placement 明确失败；运行期不会
擅自重绑定或改成跨机发送。

8 个物理 Reader 可以组成 4 个各含 2 个 Reader 的执行组，供应 2 个 Input。
每个执行实例激活一组，将两个完整 microbatch 分别交给两个 Input。一个周期内，每个
Input 由四个不同 Reader 供应；这四个 microbatch 不会被合并成同一执行实例。
同一整批契约也适用于非 source 算子。

拓扑图显式声明 unit relation，并根据编译后的关系和执行周期生成
Reader/Input placement 约束；下游节点间的亲和性（如 Input/Dense）则使用独立约束。

## 执行与内存

编译后的 direct targets 使用现有 distributor 和 frame ABI。显式整批关系，以及
可以证明等价的静态 chunk 几何，都可以降低为 direct targets。

直接投递跳过排序、packing gather 和多 shard values concat。分片路由中，
`sort_indices=None` 表示不重排；真实 permutation 才执行 backend 原生 gather。
不能因为某次动态路由恰好有序，就把全部动态路由声明为恒等排列。


## 公共声明

```python
from fluxway.common.transfer import WholeUnitRouting

result = consumer.process(
    produced,
    unit_routing=WholeUnitRouting((1, 0)),
)
```

`Builder.add_func_edge` 接受相同的 `unit_routing` 参数，将其应用到该边映射的每个
tensor port。Ordering-only 和 `local_only` 边拒绝此声明。自定义策略实现
`UnitRouting.bind`；返回的 `TransferRelation` 在执行前由图编译器校验。
