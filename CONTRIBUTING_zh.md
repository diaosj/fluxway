# Fluxway 贡献指南

[English](CONTRIBUTING.md)

感谢你对 Fluxway 的关注与贡献！本文档简明扼要地介绍如何搭建本地开发环境、执行代码规范检查、运行单元测试以及提交高质量 Pull Request。

---

## 1. 环境准备

* **系统与 Python 版本**：Linux，**Python 3.12 或 3.13**。
* 建议创建独立的 Python 虚拟环境，并安装开发所需的测试与构建依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[jax,profiler,visualization]' --group test --group lint --group build
```

---

## 2. 本地自检与预提交验证

在提交代码或开 PR 前，请确保本地通过以下三道校验（与 GitHub CI 流程完全一致）：

### 第一步：代码规范与格式化 (Lint & Format)
```bash
bash scripts/format.sh --check
# 应用 lint 修复与格式化：
bash scripts/format.sh
```

### 第二步：运行单元测试 (CPU 矩阵)
```bash
JAX_PLATFORMS=cpu python -m pytest -ra --strict-markers --strict-config \
  --ignore=tests/checkpointer/test_retention.py \
  --ignore=tests/example/test_sparse_checkpoint_e2e.py
```
> *注*：上述两个忽略的文件涉及可选的私有原生 `sparse_embedding` 库及 Lance 驱动，通用 CPU 自动化测试默认跳过。

### 第三步：打包与包完整性校验
```bash
# 清空旧产物
rm -rf dist/
python -m build
python -m twine check --strict dist/*
python scripts/distribution.py check dist

# 可选：在完全隔离的干净环境中验证安装与导入
python -m venv tmp/wheel-env
tmp/wheel-env/bin/python -m pip install dist/*.whl
tmp/wheel-env/bin/python -m pip check
tmp/wheel-env/bin/python -I scripts/check_installed.py
```

---

## 3. Pull Request 提交规范

1. **聚焦单一职责**：每个 PR 仅解决一个具体功能或 Bug，切勿掺杂不相关的无关改动。
2. **测试与文档齐备**：任何行为变更或代码修复，均须附带对应的单元测试，并同步更新中英文档。
3. **明示破坏性变更**：如有 API 破坏性改动（Breaking Changes），必须在 PR 说明中详细附上升级迁移指南。
4. **CI 必须全绿**：合并入主分支前，GitHub Actions 所有 CI 任务必须在最新 commit 上完全通过。

关于版本 Tag、PyPI 发布机制与 Trusted Publisher 权限，请参阅 [版本发布指南](docs/releasing_zh.md) ([English](docs/releasing.md))。

定时的 [Nightly wheel](.github/workflows/nightly.yml) 工作流复用完整 CI 流程，并将验证过的 wheel 与源码包保存在 `python-distributions` Actions artifact 中 14 天。工作流进入 `main` 后可以手动运行；修改 CI 或 nightly 工作流时，也会在 PR 中运行 nightly 验证。
