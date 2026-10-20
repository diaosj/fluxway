# fluxway

[English](README.md)

Fluxway 打破了传统 SPMD 的全局同步壁垒与刚性集群配比，以**异步状态机驱动的 MPMD** 重塑大规模分布式计算范式：不仅实现 **CPU 大内存 I/O 与 GPU/TPU 高密算力**解耦，更能解耦**不同代际、不同显存规格的异构 GPU 集群**。通过算力与 I/O 的 **M:N 弹性拓扑匹配**与**全程无锁全异步流水（Zero-Bubble Overlap）**，将异构集群的硬件利用率推向极致。

### 核心落地场景

* **超大规模搜推系统 (Gigantic Recommenders)**：TB/PB 级稀疏 Embedding 的 CPU 大内存存储、多级去重通信与 GPU 高密 Dense 计算全异步 Overlap。
* **大规模条件记忆与稀疏 Engram 架构 (Engram & External Memory)**：解耦静态记忆查找（常数级 N-gram 哈希查表/外部知识库）与动态神经推理，海量参数常驻 CPU 主存与跨级存储，确定性预取完全隐藏于 GPU Transformer 计算之后。
* **解耦 MoE 专家训推 (Decoupled MoE)**：Attention 稠密计算与 MoE 专家池硬件物理分离，不同规格 GPU 按需配比，动态路由无气泡调度。
* **大模型长上下文与 PD 分离 (Disaggregated Prefill & Decode)**：计算密集型 Prefill 节点与显存/带宽敏感型 Decode 节点弹性配比，实现 KV Cache 极速传输。
* **多模态世界模型与端到端复杂网络 (World Models & Omni)**：视觉/时空物理模拟器、语音 Encoder 与自回归 LLM 主干跨异构算力非对称流水部署，全流程互不阻塞。
* **强化学习异步闭环 (RL / Self-Alignment)**：Rollout 采样生成集群与 Learner 梯度更新集群异步并发，彻底消除环境交互与策略迭代的相互等待。

### 相比 Google Pathways 的核心优势

与 [Google Pathways](https://arxiv.org/abs/2203.12533) 这类深度依赖受限静态计算图与重量级后端的分布式系统相比，Fluxway 提供：
* **更自由的原生 Python 组件编排**：支持以标准 Python 算子、装饰器与自定义类灵活组合，算子内部无缝支持 JAX、PyTorch、DLPack 与自定义 C++/CUDA 扩展，摆脱死板约束。
* **Eager 动态执行模式**：支持即时交互式调试与动态控制流，开发者无需等待冗长的全图编译即可快速验证算法与原型，兼具极致性能与 Pythonic 的极简开发体验。

## 文档导航 (Documentation)

- [贡献指南 (Contributing)](CONTRIBUTING_zh.md) ([English](CONTRIBUTING.md))：本地检查与 CI 覆盖。
- [版本发布 (Releasing)](docs/releasing_zh.md) ([English](docs/releasing.md))：版本 Tag、wheel 构建与 PyPI 发布指南。

### 核心架构与执行机制 (Core Architecture & Execution)
- [架构设计 (Architecture)](docs/ARCHITECTURE_zh.md) ([English](docs/ARCHITECTURE.md))：控制面与运行时所有权。
- [执行生命周期与使用 (Execution lifecycle and usage)](docs/execution_lifecycle_zh.md) ([English](docs/execution_lifecycle.md))：启动、等待、取消、checkpoint 续跑与就绪/激活契约。
- [检查点持久化 (Checkpoints)](docs/checkpoint_zh.md) ([English](docs/checkpoint.md))：可持久化进度截断、保存/恢复与生命周期管理。
- [数据面运行时 (Data-plane runtime)](docs/dataplane_runtime_zh.md) ([English](docs/dataplane_runtime.md))：异步传输、张量 ABI、DLPack 边界与无锁分发。
- [一致性协议 (Coherence protocol)](docs/coherence_protocol_zh.md) ([English](docs/coherence_protocol.md))：端点本地 MESI 目录树、访问意图与形式化不变式。

### 数据读取与搬运 (Data Ingestion & Transfer)
- [BaseReader 与数据源读取 API](docs/base_reader_zh.md) ([English](docs/base_reader.md))：Reader 坐标绑定、行稀疏张量与自定义适配器。
- [数据源协同机制 (Source coordination)](docs/source_coordination_zh.md) ([English](docs/source_coordination.md))：规范任务所有权与支持的数据摄入边界。
- [数据源 SPMD 形式化协议 (Source SPMD protocol)](docs/source_spmd_formal_protocol_zh.md) ([English](docs/source_spmd_formal_protocol.md))：代数规划、Lease 租约与 checkpoint fencing。
- [编译后传输计划 (Compiled transfer plans)](docs/transfer_planning_zh.md) ([English](docs/transfer_planning.md))：Shard 交集、执行组与整单元路由。

### 放置策略与算子 (Placement & Operators)
- [Placement Recipe (放置配方)](docs/placement_recipe_zh.md) ([English](docs/placement_recipe.md))：声明每 SPMD 资源并配置亲和性、目标、运行时限制与诊断。
- [Placement 数学建模 (Placement Mathematics)](docs/placement_mathematics_zh.md) ([English](docs/placement_mathematics.md))：面向 Recipe 编写者的形式化语义与求解器边界。
- [Worker 进程算子 (Worker-process operators)](docs/worker_process_operators_zh.md) ([English](docs/worker_process_operators.md))：`@fluxway.requires_worker_process`、进程亲和运行时与 JAX 初始化。

### 可观测性与工具链 (Observability & Tooling)
- [用户性能分析器 (User profiler)](docs/user_profiler_zh.md) ([English](docs/user_profiler.md))：自定义 Scope、瞬时 Marker 与 Perfetto Trace 关联分析。
- [计算图可视化 (Computation Graph Visualization)](docs/graph_visualization_zh.md) ([English](docs/graph_visualization.md))：Google Model Explorer 导出、依赖元数据、潜在重叠布局、远程查看、Trace 关联与 Bundle 语义。

## 计算图可视化 (Computation Graph Visualization)

```bash
pip install "fluxway[visualization]"
```

```python
import fluxway

bundle = fluxway.visualize_graph(graph, "/tmp/training-graph")
fluxway.view_graph(bundle)
```

详见 [计算图可视化文档](docs/graph_visualization_zh.md) 获取关于 Model Explorer 使用方式、依赖元数据、重叠布局、远程查看、Trace 关联及 Bundle 语义的完整说明。
