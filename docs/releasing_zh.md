# Fluxway 版本发布指南

> **核心速览**
>
> * **唯一版本源**：版本号仅由 `pyproject.toml` 中的 `project.version` 决定。默认的 `0+unreleased` 仅供本地开发使用，发布流水线会直接拒绝。
> * **发布触发**：推送 `v*` 格式的 Git Tag 会自动触发 GitHub Actions 运行完整 CI、构建 sdist 和 wheel、推送到 PyPI，并自动生成 GitHub Release。
> * **免密安全发布**：采用 PyPI Trusted Publishing (OIDC 身份互信)，无需维护长期有效的 PyPI API Token 或仓库密钥。

---

## 1. 首次配置（仅需一次）

1. 在 GitHub 仓库的 **Settings -> Environments** 中新建名为 `pypi` 的环境，限定部署分支仅允许匹配 `v*` 的 Tag。
2. 在 PyPI 项目设置中（若为新项目，则在 PyPI 账号的 Publishing 设置中添加 Pending Publisher）配置 **GitHub Trusted Publisher**：
   * **PyPI project**: `fluxway`
   * **GitHub owner**: `fluxway-project`
   * **Repository**: `fluxway`
   * **Workflow filename**: `release.yml`
   * **Environment**: `pypi`

*(无需添加任何 API Token 或 Secret。详见：[PyPI Trusted Publishing 官方文档](https://docs.pypi.org/trusted-publishers/))*

---

## 2. 标准发布步骤

1. **修改版本号（提 PR）**：
   * 创建 PR 将 `pyproject.toml` 中的 `project.version` 修改为标准规范版本号（如 `0.1.0` 或预发布版 `0.1.0rc1`）。
   * 若有破坏性改动（Breaking Changes），在 PR 说明中附上升级迁移指南。
2. **合并 PR**：
   * 确保 CI 校验全部通过且 Review 完成后，将 PR 合并入 `main` 主分支。
3. **打 Tag 并推送**：
   * 拉取主分支最新代码，执行本地检查，打 Tag 并推送到远程：
     ```bash
     git checkout main && git pull
     python scripts/distribution.py release v<version>
     git tag -a v<version> -m "Release v<version>"
     git push origin v<version>
     ```
4. **自动化流水线**：
   * 推送 Tag 后，GitHub Actions 中的 `Release` 工作流将全自动执行：
     1. 验证 Tag 与 `pyproject.toml` 中的版本一致，且当前 Commit 属于 `main` 分支。
     2. 运行全量 CI（代码格式化、多 Python 版本测试矩阵、打包合规性校验）。
     3. 构建源码包 (sdist) 与二进制包 (wheel)，并在干净隔离的环境中验证安装。
     4. 通过 OIDC 权限免密推送包到 PyPI。
     5. 自动创建 GitHub Release，附带构建产物并生成更新日志。

---

## 3. 质量与打包保障机制

* **代码规范严格要求**：Ruff Lint 和 Format 零告警通过。
* **跨环境测试矩阵**：覆盖 Python 3.12 与 3.13，涵盖 JAX、Orbax、Grain、Flax、SafeTensors、Profiler 与可视化工具链路。
* **打包完整性检查**：严格执行 Twine 检查、`py.typed` 类型标记确认、开源 License 包含以及干净环境下的隔离导入测试。
* **模块化依赖解耦**：核心 wheel 包安装极简轻量；缺失可选依赖（如特定后端）时抛出明确指导安装的异常，而非静默崩溃。

---

## 4. 常见异常与排查处理

* **严禁覆盖版本或移动 Tag**：PyPI 严格禁止覆盖已上传的同名同版本文件。若发布的包存在问题，切勿强行推 Tag，请更新 `project.version`（例如发 patch 版本）走新发布流程。
* **PyPI 上传失败**：先查阅 Action 失败日志。若为 OIDC 权限或配置未生效导致且 PyPI 尚未产生该文件，调整 GitHub/PyPI 配置后在 Actions 界面重试失败任务即可。
* **PyPI 成功但 GitHub Release 失败**：切勿全量重新触发流水线（因为 PyPI 会拒绝重复文件上传）。可直接在 GitHub Release 页面手动上传本次 Action 构建好的 wheel/sdist 产物，或单独重试 GitHub Release 步骤。
