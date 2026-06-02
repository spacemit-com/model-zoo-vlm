# VLM 生产环境对外接口交付说明

版本日期：2026-05-29  
适用代码库：`/home/bianbu/model-zoo/vlm`  
交付范围：C++ 公共头文件、Python 封装、OpenAI 兼容 HTTP Gateway

## 1. 文档入口

| 文档 | 内容 |
| --- | --- |
| [API 文档](vlm_api_reference.md) | C++、Python、HTTP Gateway 的完整接口说明 |
| [Gateway curl 测试报告](vlm_gateway_curl_test_report.md) | Gateway 端到端 curl 实测记录、生命周期验证和指标结果 |
| [测试文档](../testing.md) | CTest、Python/Gateway 边界测试、真实模型矩阵和生产验证命令 |
| [快速使用](../quick-start.md) | 依赖安装、模型下载、编译、基础验证和配置参考 |
| [Demo 使用文档](../demo-usage.md) | C++ Demo、Python Demo、Gateway curl 示例 |
| [架构文档](../architecture.md) | 系统架构、Server 模式数据流、TCM 硬件加速说明 |

## 2. 交付物清单

| 类型 | 文件/入口 | 用途 |
| --- | --- | --- |
| C++ 头文件 | `include/vlm/vlm_service.h` | 原生 C++ 服务接口，提供同步、异步、流式、单轮、多轮、配置、指标能力 |
| Python API | `src/python/vlm.py` | Python HTTP 封装，纯 Python 实现，通过 OpenAI 兼容 HTTP API 与 llama-server 通信 |
| HTTP API | `gateway/api.py`（通过 AI Gateway 统一管理） | OpenAI 兼容 `/chat/completions`，以及模型管理、健康检查、指标接口 |
| 验证脚本 | `tests/functional/test_vlm_service.py` | 生命周期、异常、Gateway、边界测试 |
| 实机矩阵脚本 | `tests/functional/full_model_matrix.py` | fastvlm/qwen3.5-0.8b/qwen3.5-2b 单轮、多轮、流式实模验证 |

## 3. 能力边界总览

| 能力 | C++ | Python | HTTP Gateway |
| --- | --- | --- | --- |
| 单轮文本生成 | 支持 | 支持 | 通过 `/chat/completions` 支持 |
| 单轮图文生成 | `image_path` / `image_bytes` | `image_path` / `image_bytes` | multimodal content `image_url.url` |
| 多轮文本对话 | 支持 | 支持 | 支持 |
| 多轮图文对话 | 支持 | 支持 | 支持，当前取 OpenAI content 数组中的图片 URL |
| 同步输出 | 支持 | 支持 | 支持 |
| 流式输出 | 支持 | 支持 | SSE 支持 |
| 单次请求生成参数覆盖 | 通过更新服务配置或实现层调用 | `generate()` / `chat()` 支持 | 非流式支持 |
| 流式单次参数覆盖 | 服务配置级别 | 当前使用服务配置 | 当前不透传请求级 options |
| 生命周期管理 | `Shutdown()` | `close()` / `shutdown()` | `close()` / context manager | `/models/load` / `/models/unload` / `/models/switch` |
| 指标 | `GetMetrics()` | `get_metrics()` | `/stats` / `/info` |

## 4. 已验证结果

| 验证项 | 结果 |
| --- | --- |
| CMake build | 通过 |
| CTest | 2/2 通过 |
| Python/Gateway 边界测试 | 15/15 通过 |
| fastvlm 实机矩阵 | 通过，覆盖单轮、多轮、generate stream、chat stream |
| qwen3.5-0.8b 实机矩阵 | 通过，`FULL_MODEL_MATRIX_OK` |
| qwen3.5-2b 实机矩阵 | 通过，`FULL_MODEL_MATRIX_OK` |
| Gateway curl 端到端测试 | 通过，详见 [Gateway curl 测试报告](vlm_gateway_curl_test_report.md) |

## 5. 生产使用建议

- 生产推荐使用 server 后端，由 `llama-server --vision-backend smt` 负责加载 SMT/TCM vision 模型。
- 部署前先执行 [测试文档](../testing.md) 中的常规验证和生产验证命令。
- Gateway 对外接入优先使用 OpenAI 兼容 `/chat/completions`，模型生命周期通过 `/models/load`、`/models/switch`、`/models/unload` 管理。
- 指标读取使用 `/stats` 或 `/info.metrics`；非流式 server 后端 `last_ttft_ms` 保持 `0.0`，流式请求会记录首个非空 chunk 的 TTFT。
- `last_output_tokens` 在非流式请求中表示 completion token 数，在流式请求中表示非空输出 chunk 数。

## 6. 版本交付结论

当前版本已具备 C++、Python 和 HTTP Gateway 三层对外接口；Python 层为纯 HTTP 封装，无需 C ABI/ctypes 依赖。接口已通过本地测试、功能边界测试、真实模型矩阵和 Gateway curl 端到端验证。完整接口细节已拆分至 [API 文档](vlm_api_reference.md)，测试细节已拆分至 [测试文档](../testing.md) 和 [Gateway curl 测试报告](vlm_gateway_curl_test_report.md)，避免单一交付文档继续膨胀。
