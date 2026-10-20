# 计算图可视化

Fluxway 将编译后的计算图投射到 Google Model Explorer 中。它不生成 SVG、DOT 图，也不提供独立的可视化服务。

## 安装

```bash
pip install "fluxway[visualization]"
```

可视化扩展组件将安装 `ai-edge-model-explorer`。

## 导出与查看

```python
from fluxway import view_graph, visualize_graph

bundle = visualize_graph(graph, "./output/training-graph")
view_graph(bundle)
```

`visualize_graph` 读取两个现有的 Python 对象：

- `Graph` 提供人类可读的 SPMD 描述。
- `CompiledGraph` 提供冻结的 SPMD 声明、编译后的算子、依赖关系、依赖代码、端口、TUUIDs、可调用标识和 `AsyncPolicy`。

该函数仅写入一个文件：

```text
./output/training-graph/model_explorer.json
```

这个 JSON 文件是 Model Explorer 导出产物。Fluxway 不使用它来重建算子、依赖关系、可调用对象或 worker 状态。Ray 会继续传输原生的 Python 编译图及其用户可调用对象。

`view_graph` 会启动 Model Explorer 服务器并阻塞，直到它被停止。导出和查看是分离的，因此无头（headless）训练进程绝不会隐式打开浏览器或启动 HTTP 服务器。

如果输出目录已存在，则抛出 `FileExistsError`。若传入 `overwrite=True`，则要求现有目录包含有效的 Fluxway Model Explorer 投影，随后将其原子替换。如果目录交换失败，先前的投影会被还原并传播异常。

## 两种视图

导出的 Model Explorer 集合包含相同编译算子和依赖项的两种视图。

| 视图 | 回答的问题 |
|---|---|
| `Dependencies` | 哪个算子必须位于另一个算子之前，以及在它们之间传递什么数据？ |
| `Potential overlap` | 哪些算子可能同时就绪，哪些因果关系约束了这种重叠？ |

这里并不是两套图定义。这两种视图都是同一个 `CompiledGraph` 和编译后的 `AsyncPolicy` 调度的投影。这些视图刻意使用了不同的边缘（edge）表示，因为它们回答的问题不同。

SPMDs 对应 Model Explorer 命名空间，编译算子对应运算节点。选择一个算子将暴露它的 `graph_id`、SPMD 标识、函数标识、输入/输出签名 IDs、mesh、资源范围、依赖秩（dependency rank）和调度策略。

可搜索的关联字段包括：

- `graph_id`
- `graph_generation`
- `spmd_id`
- `op_id`
- `function_id`
- `input_signature_id`
- `output_signature_id`
- `dependency_code`

Model Explorer 可能会从总览中移除仅包含一个运算节点的命名空间。因此，Fluxway 也会将 SPMD 元数据放在每个算子上。当必须保留每个 SPMD 作为可视层时，请在 Model Explorer 中启用 `Retention layers with a single op node child` 设置。

## 依赖与传输

依赖关系表示目标算子在源算子满足该依赖之前，无法开始相应的 step。传输表示张量（tensor）也沿该依赖移动。

```text
dependency: 源算子 -> 目标算子
transfers:  零个或多个 (源端口 -> 目标端口 -> TUUID) 映射
```

零传输是合法的。它是一个仅包含排序逻辑的依赖，由传输编译器实现为 header-only 数据帧，而非张量数据负载（payload）。可视化组件会为这种控制关系创建输入/输出元数据，使其在同一个 Model Explorer 图中与张量传输一样可见。

`Dependencies` 视图对每个编译依赖包含正好一个原生边。当一个依赖携带多个张量时，源和产物元数据包含一个结构化的 `transfers` 列表，其中包括每个传输代码、逻辑 TUUID、源端口和目标端口。它没有叠加层（overlays）。

`Potential overlap` 视图为每对相同 step 因果算子包含一条细长且未标记的原生布局边。这些边使用 Model Explorer 的默认端口，并且没有张量元数据。该视图按有序算子对汇总编译依赖项和 phase 调度关系。张量依赖和排序依赖不会创建彩色叠加层，因为它们在同一 step 内的因果关系已经由原生布局边表示，且其完整元数据属于 `Dependencies` 视图。只有非零 step 的 phase 关系才会创建蓝绿色（teal）叠加层。它们不能成为原生边，因为它们不能改变同一 step 的布局秩（layout rank）。

## 视觉语言

算子方框的颜色描述了其在编译后依赖图中的结构角色。

| 颜色 | 含义 |
|---|---|
| Green | Source (源节点): 无入度依赖 |
| Blue | Transform (转换节点): 同时拥有入度和出度依赖 |
| Amber | Sink (汇节点): 无出度依赖 |
| Purple | 依赖循环的一部分 |
| Gray | 孤立算子 |

`Dependencies` 视图没有叠加层。它的原生 Model Explorer 边构成了完整的依赖投影并携带所有张量端口元数据。

在 `Potential overlap` 中，淡淡的原生边是完整的同 step 因果布局。唯一的彩色叠加层是一个不参与该布局的跨 step phase 约束。

| 颜色 | 含义 |
|---|---|
| Teal `#007B83` | 来自 `AsyncPolicy` 的跨 step phase 依赖 |

Model Explorer 的边缘架构支持颜色、宽度和标签，但没有虚线（dashed/dotted）字段。Fluxway 不会生成会被查看器静默忽略的不受支持的 JSON 字段。

## 异步调度事实

`AsyncPolicy` 是异步执行事实的唯一来源。每个算子节点公开针对该算子编译的策略值，包括：

- `async_mode`, `async_depth`, `async_commit`, 以及 `async_max_staleness`
- `state_access` 以及 `state_resource`
- `schedule_kind`, `schedule_base_priority`, 以及 `schedule_phase`
- `schedule_phase_order` 以及 `schedule_phase_dependencies`
- `schedule_lane`, `schedule_lane_order`, 以及 `schedule_lane_priorities`
- `schedule_phase_priorities`

phase 到 lane 的映射通过每个 phase 算子的 `schedule_phase` 和 `schedule_lane` 表示。有序的 phase 和 lane 集合也会被显示出来，因此 Model Explorer 详细信息面板可以重建完整策略，而无需在可视化模块中使用第二组常量。

`compile_phase_schedule` 是共享的调度语义模块。Worker 在执行热路径上使用其预计算的门（gates），而可视化模块则使用相同的编译关系。因此，可视化适配器中不会重新实现 phase 排序逻辑。

这些字段描述的是调度规则，而不是测量的执行情况。对于实际队列排队、传输和设备时序，Perfetto tracing 仍然是权威依据。

## 潜在重叠

Fluxway 在不递归遍历 Python 对象的情况下计算重叠组：

1. 将编译依赖项与同 step 的 phase 关系结合。
2. 将所有同 step 的因果循环折叠为一个组件。
3. 在所得无环图上计算最长的依赖秩。
4. 将具有相同秩的算子放入同一个外部命名空间。
5. 将调度 lane 放入每个秩内，如果存在 `lane_order` 则遵循其排序。
6. 将每个 SPMD 保留为最内部的命名空间。

不同组件之间的每个同 step 因果关系都会增加目标秩。跨 step 的 phase 关系显示为蓝绿色叠加层，但不增加同 step 的秩。处于同一秩的算子被显示为一个组，而不是两两相连；如果成对连接则重叠线会呈二次方增长。同一个组内的成员可能并发就绪，但此视图并不断言它们在运行时肯定会重叠。资源竞争、条件控制、不同的 step 序号和反压仍可能将它们序列化。

## 远程使用

将查看器绑定到本地回环并使用 SSH 转发：

```python
view_graph(
    "./output/training-graph",
    host="127.0.0.1",
    port=8080,
    open_browser=False,
)
```

```bash
ssh -L 8080:127.0.0.1:8080 user@training-host
```

导出的文件也可以被复制到另一台机器并直接打开：

```bash
scp user@training-host:./output/training-graph/model_explorer.json \
  "$HOME/Downloads/training-graph.json"
model-explorer "$HOME/Downloads/training-graph.json"
```

若要将计算图与 tracing 对比，请复制 `.pftrace` 文件并使用 Perfetto 打开它：

```bash
scp user@training-host:./output/recommendation-traces/run.pftrace \
  "$HOME/Downloads/run.pftrace"
```

使用 `graph_id`、`spmd_id`、`op_id` 和 `function_id` 将 Model Explorer 节点与 trace 事件关联。

## 规模边界

迭代式组件和依赖秩算法避免了 Python 递归，并在导出期间支持具有超过 10,000 个算子的图。浏览器布局是一个独立的 Model Explorer 开销：非常密集的图可能需要更长时间打开，即便 Fluxway 导出它们的速度很快。
