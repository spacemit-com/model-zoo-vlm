# model-zoo-vlm

SpacemiT VLM (Vision Language Model) 推理服务组件，面向进迭时空 K3 等 RISC-V 平台，提供视觉语言模型的加载、推理与集成能力。

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [快速使用](docs/quick-start.md) | 安装依赖、下载模型、编译、基础验证和配置参考 |
| [API 文档](docs/production/vlm_api_reference.md) | C++、Python、HTTP Gateway 接口说明 |
| [Demo 使用文档](docs/demo-usage.md) | C++ Demo、Python Demo、Gateway curl 示例 |
| [测试文档](docs/testing.md) | 单元测试、功能测试、真实模型矩阵和 Gateway curl 测试入口 |
| [架构文档](docs/architecture.md) | 系统架构、Server 模式数据流、TCM 硬件加速说明 |
| [生产交付说明](docs/production/vlm_external_api_delivery.md) | 对外交付范围、生产建议和版本结论 |



> 注意：在 `/home/bianbu/model-zoo` 目录中，ai-gateway 复制位于 `model-zoo/gateway`，其中 VLM 域目前通过远程 OpenAI 兼容服务代理方式接入本组件。Gateway VLM 域已支持本地 llama-server 模式（通过 VlmLlamaAdapter 启动 llama-server 子进程）和远程代理模式。本地模式与 LLM 域的接入方式完全统一，均通过 llama-server HTTP API 调用。

## 1. 项目简介

本组件封装了 VLM 模型的推理能力，支持通过 llama-server HTTP API 调用 TCM (Tensor Compute Module) 硬件加速，实现图像描述、视觉问答等应用。

| 能力 | 说明 |
|------|------|
| 推理后端 | Server 模式：通过 llama-server HTTP API，支持 SMT/TCM 硬件加速 |
| 推理接口 | `Generate()` 单轮推理、`Chat()` 多轮对话、`GenerateStream()`/`ChatStream()` 流式输出 |
| 图像输入 | 支持图片路径、内存字节、Base64 编码三种方式 |
| 模型配置 | YAML 配置文件 + 模型目录 config.json，支持运行时切换模型 |
| 硬件加速 | SMT 后端自动启用 TCM，CPU 亲和性绑定，OpenMP 线程控制 |
| 可观测性 | 返回输出文本、reasoning 内容、token 用量、延迟、TTFT、tokens/s |
| 多语言接口 | C++ 原生接口 + Python HTTP 封装（纯 Python，无需 CABI/ctypes） |
| Gateway 集成 | FastAPI 适配层，OpenAI 兼容 API，支持本地/远程模式，可挂载至 ai-gateway |

## 2. 快速入口

```bash
cd vlm
sudo apt-get update
sudo apt-get install -y build-essential cmake curl libyaml-cpp-dev nlohmann-json3-dev
sudo apt install llama.cpp-tools-spacemit
bash scripts/download_model.sh Qwen3.5-0.8B
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
ctest --output-on-failure
```

更多部署、配置和验证步骤见 [快速使用](docs/quick-start.md)。

## 3. 常用入口

| 场景 | 入口 |
| --- | --- |
| 查看服务架构 | [架构文档](docs/architecture.md) |
| 查询 C++ / Python / HTTP 接口 | [API 文档](docs/production/vlm_api_reference.md) |
| 运行示例程序 | [Demo 使用文档](docs/demo-usage.md) |
| 执行测试与验收 | [测试文档](docs/testing.md) |
| 查看生产交付结论 | [生产交付说明](docs/production/vlm_external_api_delivery.md) |

## 4. 常见问题

### Q: 为什么默认使用 server 后端？

当前部署的 VLM vision 模型是 SpacemiT SMT ONNX 格式。公开的 `libmtmd` C API 面向 GGUF mmproj vision 模型，无法直接加载 ONNX vision 文件。因此通过 `llama-server --vision-backend smt` 加载 vision 后端，再调用 HTTP API。

### Q: direct 后端的用途是什么？

Direct 后端保留给未来 GGUF mmproj vision 模型使用。遇到当前 ONNX vision 文件时，direct 后端会给出明确错误并提示使用 server 后端。

### Q: 如何切换模型？

```python
# Python
service.shutdown()
service = VlmService("path/to/other_model.yaml")

# Gateway API
curl -X POST http://localhost:8000/models/switch \
  -d '{"config_path": "path/to/other_model.yaml", "model_id": "new_model"}'
```

### Q: 图片路径如何传递？

- **C++ API**: `input.image_path = "/absolute/path/to/image.jpg"`
- **Python API**: `service.generate("prompt", image_path="/path/to/img.jpg")`
- **llama-server 直接调用**: 使用 `file://relative_path.jpg`（相对于 `--media-path`）

### Q: 推理速度慢怎么办？

1. 确认 TCM 已启用：`spacemit-tcm-smi` 显示 busy 块
2. 增加线程数：`generation.threads: 8`（K3 推荐 8）
3. 减小 `max_tokens` 和 `context_size`
4. 使用更小的模型（如 FastVLM 0.5B）

## 5. 版本与发布

| 版本 | 说明 |
|------|------|
| 0.1.0 | 初始版本：C++ 推理接口、Server 后端、Python 绑定、Gateway 适配 |
| 0.2.0 | TCM 硬件加速支持：SMT 后端、CPU 亲和性、TCM 内存管理 |
| 0.3.0 | 移除 CABI 接口，Python 改为纯 HTTP 封装；Gateway VLM 域支持本地 llama-server 模式 |

## 6. 贡献方式

欢迎参与贡献：提交 Issue 反馈问题，或通过 Pull Request 提交代码。

- 编码规范：C++ 遵循 Google C++ 风格指南，Doxygen 注释
- 提交前检查：

```bash
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
bash scripts/lint/cpplint.sh
```

## 7. License

本组件以 Apache-2.0 协议发布，详见 `LICENSE` 文件。
