# VLM API 文档

版本日期：2026-05-29  
适用代码库：`/home/bianbu/model-zoo/vlm`  
交付范围：C++ 公共头文件、Python 封装、OpenAI 兼容 HTTP Gateway

## 1. 交付物清单

| 类型 | 文件/入口 | 用途 |
| --- | --- | --- |
| C++ 头文件 | `include/vlm_service.h` | 原生 C++ 服务接口，提供同步、异步、流式、单轮、多轮、配置、指标能力 |
| Python API | `src/python/vlm.py` | Python HTTP 封装，纯 Python 实现，通过 OpenAI 兼容 HTTP API 与 llama-server 通信 |
| HTTP API | `gateway/api.py` | OpenAI 兼容 `/chat/completions`，以及模型管理、健康检查、指标接口 |
| 验证脚本 | `tests/functional/test_vlm_service.py` | 生命周期、异常、Gateway、边界测试 |
| 实机矩阵脚本 | `tests/functional/full_model_matrix.py` | fastvlm/qwen3.5-0.8b/qwen3.5-2b 单轮、多轮、流式实模验证 |

## 2. 能力边界总览

| 能力 | C++ | Python | HTTP Gateway |
| --- | --- | --- | --- | --- |
| 单轮文本生成 | 支持 | 支持 | 通过 chat/completions 支持 |
| 单轮图文生成 | `image_path` / `image_bytes` | `image_path` / `image_bytes` | `image_path` / `image_bytes` | multimodal content `image_url.url` |
| 多轮文本对话 | 支持 | 支持 | 支持 |
| 多轮图文对话 | 支持 | 支持 | 支持，当前取 OpenAI content 数组中的图片 URL |
| 同步输出 | 支持 | 支持 | 支持 |
| 流式输出 | 支持 | 支持 | SSE 支持 |
| 单次请求生成参数覆盖 | 通过更新服务配置或实现层调用 | `*_with_options` 支持同步接口 | `generate()` / `chat()` 支持 | 非流式支持 |
| 流式单次参数覆盖 | 服务配置级别 | 当前 C 流式接口不接收 options | 当前使用服务配置 | 当前不透传请求级 options |
| 生命周期管理 | `Shutdown()` | `close()` / `shutdown()` | `close()` / context manager | `/models/load` / `/models/unload` |
| 指标 | `GetMetrics()` | `get_metrics()` | `/stats` / `/info` |

## 3. C++ 公共接口

头文件：`include/vlm_service.h`

### 3.1 命名空间与枚举

```cpp
namespace vlm {

enum class BackendType {
    kDirect,
    kServerCompat,
};

enum class VlmMediaBackend {
    kAuto,
    kMtmd,
    kSmt,
};

enum class VlmHistoryPolicy {
    kKeepAll,
    kLastOnly,
    kNone,
};
```

说明：

- `BackendType::kServerCompat`：当前生产建议使用的 server 兼容模式，服务会启动/管理 `llama-server`。
- `BackendType::kDirect`：保留的直接加载模式。
- `VlmMediaBackend::kSmt`：当前 SpacemiT SMT ONNX vision 后端。
- `VlmHistoryPolicy`：控制服务级历史保留策略。

### 3.2 配置结构体

```cpp
struct VlmGenerationConfig {
    int max_tokens = 150;
    int context_size = 4096;
    int top_k = 40;
    float top_p = 0.95F;
    float temperature = 0.2F;
    float repeat_penalty = 1.2F;
    int threads = 8;
    int batch_threads = 8;
    bool enable_thinking = false;
    int reasoning_budget = -1;
    VlmHistoryPolicy history_policy = VlmHistoryPolicy::kLastOnly;
    std::vector<std::string> stop_sequences;
};

struct VlmModelConfig {
    std::string model_dir;
    std::string model_name;
    std::string text_model_path;
    std::vector<std::string> vision_model_paths;
    std::string architecture;
    BackendType backend_type = BackendType::kDirect;
    VlmMediaBackend media_backend = VlmMediaBackend::kSmt;
    std::string smt_config_dir;
    std::string media_path;
    std::string cpu_affinity;
    int n_gpu_layers = 0;
    bool no_warmup = true;
    std::string host = "127.0.0.1";
    int port = 8086;
    VlmGenerationConfig generation;
};
```

关键字段：

| 字段 | 含义 |
| --- | --- |
| `max_tokens` | 最大输出 token 数 |
| `context_size` | 上下文长度 |
| `temperature` / `top_p` / `top_k` | 采样参数 |
| `repeat_penalty` | 重复惩罚 |
| `enable_thinking` / `reasoning_budget` | 推理/思考模式控制 |
| `history_policy` | 多轮历史保留策略 |
| `model_dir` / `text_model_path` / `vision_model_paths` | 模型路径配置 |
| `host` / `port` | server 模式监听地址 |
| `smt_config_dir` | SMT vision 配置目录 |
| `media_path` | 媒体/vision 相关资源路径 |
| `cpu_affinity` | CPU 亲和性配置 |

### 3.3 请求与响应结构体

```cpp
struct VlmInput {
    std::string prompt;
    std::string image_path;
    std::vector<std::uint8_t> image_bytes;
};

struct VlmUsage {
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int total_tokens = 0;
};

struct VlmResult {
    std::string text;
    std::string reasoning_content;
    VlmUsage usage;
    double latency_ms = 0.0;
    double ttft_ms = 0.0;
    double tokens_per_second = 0.0;
};
```

图片输入规则：

- `image_path` 和 `image_bytes` 二选一即可。
- 不传图片时即文本生成。
- 生产环境建议使用绝对路径，避免服务进程工作目录变化导致找不到图片。

### 3.4 多轮消息结构体

```cpp
struct VlmChatMessage {
    enum class Role { SYSTEM, USER, ASSISTANT };
    Role role = Role::USER;
    std::string content;
    std::string image_path;
    std::vector<std::uint8_t> image_bytes;

    static VlmChatMessage System(const std::string& content);
    static VlmChatMessage User(const std::string& content);
    static VlmChatMessage UserWithImage(const std::string& content,
                                        const std::string& path);
    static VlmChatMessage UserWithImageBytes(
        const std::string& content,
        const std::vector<std::uint8_t>& bytes);
    static VlmChatMessage Assistant(const std::string& content);
};

struct VlmChatResult {
    std::string content;
    std::string reasoning_content;
    VlmUsage usage;
    double latency_ms = 0.0;
    double ttft_ms = 0.0;
    double tokens_per_second = 0.0;
    std::string error;
};
```

### 3.5 回调与指标

```cpp
using VlmStreamCallback = std::function<bool(
    const std::string& chunk,
    bool is_done,
    const std::string& error)>;

using VlmAsyncCallback = std::function<void(
    const VlmResult& result,
    const std::string& error)>;

struct VlmMetrics {
    int total_requests = 0;
    bool is_processing = false;
    double last_latency_ms = 0.0;
    double avg_latency_ms = 0.0;
    double last_ttft_ms = 0.0;
    int64_t last_output_tokens = -1;
    double last_tokens_per_second = 0.0;
};
```

  指标含义：

  | 字段 | 含义 |
  | --- | --- |
  | `total_requests` | 当前服务实例已完成请求数 |
  | `is_processing` | 当前是否正在处理请求 |
  | `last_latency_ms` | 最近一次请求端到端延迟 |
  | `avg_latency_ms` | 当前服务实例内完成请求的平均端到端延迟 |
  | `last_ttft_ms` | 最近一次流式请求首个非空输出 chunk 到达耗时；非流式 server 后端保持 `0.0` |
  | `last_output_tokens` | 非流式请求为 completion token 数；流式请求为非空输出 chunk 数 |
  | `last_tokens_per_second` | 非流式请求为 token/s；流式请求为 chunk/s |

流式回调规则：

- `chunk`：当前返回文本片段。
- `is_done`：非零/true 表示流完成。
- `error`：非空表示错误。
- 回调返回 `false` 表示调用方主动停止接收后续 chunk。

### 3.6 服务类接口

```cpp
class VlmService {
public:
    virtual ~VlmService() = default;

    virtual bool Initialize(const VlmModelConfig& config,
                            std::string* error) = 0;
    virtual void Shutdown() = 0;
    virtual bool IsReady() const = 0;

    virtual bool Generate(const VlmInput& input,
                          VlmResult* result,
                          std::string* error) = 0;
    virtual bool GenerateAsync(const VlmInput& input,
                               VlmAsyncCallback callback,
                               std::string* error) = 0;
    virtual bool GenerateStream(const VlmInput& input,
                                VlmStreamCallback callback,
                                VlmResult* result,
                                std::string* error) = 0;

    virtual bool Chat(const std::vector<VlmChatMessage>& messages,
                      VlmChatResult* result,
                      std::string* error) = 0;
    virtual bool ChatStream(const std::vector<VlmChatMessage>& messages,
                            VlmStreamCallback callback,
                            VlmChatResult* result,
                            std::string* error) = 0;

    virtual bool Reset(std::string* error) = 0;
    virtual bool UpdateGenerationConfig(const VlmGenerationConfig& config,
                                        std::string* error) = 0;
    virtual VlmGenerationConfig GetGenerationConfig() const = 0;
    virtual VlmMetrics GetMetrics() const = 0;
};
```

### 3.7 工厂函数

```cpp
std::unique_ptr<VlmService> CreateVlmService(
    const VlmModelConfig& config,
    std::string* error);

std::unique_ptr<VlmService> CreateVlmServiceFromConfig(
    const std::string& config_path,
    std::string* error);
```

生产推荐使用 `CreateVlmServiceFromConfig()` 从 YAML 配置创建服务。

### 3.8 C++ 使用示例

```cpp
#include "vlm_service.h"
#include <iostream>

int main() {
    std::string error;
    auto service = vlm::CreateVlmServiceFromConfig(
        "examples/fastvlm/config/qwen3_5_0.8b.yaml", &error);
    if (!service) {
        std::cerr << error << std::endl;
        return 1;
    }

    vlm::VlmInput input;
    input.prompt = "请描述这张图片";
    input.image_path = "/home/bianbu/pic/猫咪.png";

    vlm::VlmResult result;
    if (!service->Generate(input, &result, &error)) {
        std::cerr << error << std::endl;
        return 1;
    }

    std::cout << result.text << std::endl;
    service->Shutdown();
    return 0;
}
```

## 4. Python HTTP 封装实现

Python 层通过 OpenAI 兼容 HTTP API 直接与 llama-server 通信，无需 C ABI/ctypes 依赖。

### 4.1 架构

```
VlmService (Python)
  -> 启动 llama-server 子进程 (端口 8093)
  -> HTTP POST /v1/chat/completions
  -> 解析 JSON/SSE 响应
  -> 返回结果字典
```

### 4.2 通信协议

Python 封装使用标准 OpenAI Chat Completions API 格式：

- 非流式：POST /v1/chat/completions，stream=false
- 流式：POST /v1/chat/completions，stream=true，SSE 格式
- 健康检查：GET /health

### 4.3 图片输入

图片通过 OpenAI multimodal content 格式传递，Python 接口支持三种方式：
- `image_path`：文件路径（自动读取并 Base64 编码）
- `image_bytes`：原始字节数据（自动 Base64 编码）
- 多轮对话中的 `image_path`/`image_bytes` 字段

### 4.4 生命周期管理

```python
with VlmService("config.yaml") as svc:
    result = svc.generate("Hello")
# 自动关闭 llama-server 进程
```

也可手动管理：
```python
svc = VlmService("config.yaml")
result = svc.generate("Hello")
svc.close()
```

## 5. Python 对外接口

模块：`src/python/vlm.py`

### 5.1 导入与初始化

```python
from vlm import VlmService

service = VlmService("examples/fastvlm/config/qwen3_5_0.8b.yaml")
```

推荐使用上下文管理器：

```python
from vlm import VlmService

with VlmService("examples/fastvlm/config/qwen3_5_0.8b.yaml") as service:
    result = service.generate("请描述这张图片", image_path="/home/bianbu/pic/猫咪.png")
    print(result["text"])
```

### 5.2 环境变量

| 环境变量 | 用途 |
| --- | --- |
| `VLM_LIB_PATH` | 已废弃（v0.3.0 移除 C ABI，不再需要） |
| `PYTHONPATH` | 需要包含 `src/python` 和仓库根目录 |

示例：

```bash
# VLM_LIB_PATH 已废弃，Python 接口不再依赖 libvlm.so
export PYTHONPATH=$PWD/src/python:$PWD
```

### 5.3 方法清单

```python
class VlmService:
    def __init__(self, config_path: str) -> None: ...
    def close(self) -> None: ...

    def generate(self,
                 prompt: str,
                 image_path: Optional[str] = None,
                 image_bytes: Optional[bytes] = None,
                 **options: Any) -> Dict[str, Any]: ...

    def generate_stream(self,
                        prompt: str,
                        image_path: Optional[str] = None) -> Generator[str, None, None]: ...

    def chat(self,
             messages: List[Dict[str, Any]],
             **options: Any) -> Dict[str, Any]: ...

    def chat_stream(self,
                    messages: List[Dict[str, Any]]) -> Generator[str, None, None]: ...

    def is_ready(self) -> bool: ...
    def reset(self) -> None: ...
    def shutdown(self) -> None: ...
    def get_metrics(self) -> Dict[str, Any]: ...
```

### 5.4 `generate()`

```python
result = service.generate(
    "请描述这张图片",
    image_path="/home/bianbu/pic/猫咪.png",
    max_tokens=64,
    temperature=0.2,
    top_p=0.95,
    top_k=40,
    repeat_penalty=1.2,
    enable_thinking=False,
    reasoning_budget=-1,
)
```

返回字典字段：

| 字段 | 含义 |
| --- | --- |
| `text` | 输出文本 |
| `reasoning_text` | 推理内容 |
| `prompt_tokens` | prompt token 数 |
| `completion_tokens` | completion token 数 |
| `total_tokens` | 总 token 数 |
| `latency_ms` | 端到端延迟 |
| `ttft_ms` | 首 token 延迟 |
| `tokens_per_second` | 输出吞吐 |

server 后端非流式 `generate()` / `chat()` 的 `latency_ms`、`completion_tokens`、`tokens_per_second` 来自真实服务端响应和本地计时，`ttft_ms` 暂为 `0.0`。流式 `generate_stream()` / `chat_stream()` 的服务级指标会在流结束后更新，其中 TTFT 为首个非空 chunk 耗时，输出规模为非空 chunk 数。

### 5.5 `chat()`

```python
messages = [
    {"role": "system", "content": "你是一个图像理解助手"},
    {"role": "user", "content": "请描述图片", "image_path": "/home/bianbu/pic/猫咪.png"},
    {"role": "assistant", "content": "这是一只猫。"},
    {"role": "user", "content": "它在做什么？"},
]

result = service.chat(messages, max_tokens=64)
print(result["text"])
```

消息字段：

| 字段 | 必填 | 含义 |
| --- | --- | --- |
| `role` | 否 | `system` / `user` / `assistant`，默认 `user` |
| `content` | 否 | 文本内容，默认空字符串 |
| `image_path` | 否 | 图片路径 |
| `image_bytes` | 否 | 图片 bytes |

### 5.6 流式输出

```python
for chunk in service.generate_stream("请描述这张图片", image_path="/home/bianbu/pic/猫咪.png"):
    print(chunk, end="", flush=True)

for chunk in service.chat_stream([
    {"role": "user", "content": "请描述这张图片", "image_path": "/home/bianbu/pic/猫咪.png"}
]):
    print(chunk, end="", flush=True)
```

限制：`generate_stream()` 和 `chat_stream()` 当前不支持 `max_tokens` 等单次请求参数，使用服务默认生成配置。

### 5.7 生命周期与异常

```python
service = VlmService("examples/fastvlm/config/qwen3_5_0.8b.yaml")
try:
    print(service.is_ready())
finally:
    service.close()
```

规则：

- `close()` 幂等，可重复调用。
- `close()` 后再调用推理、指标或状态方法会抛出 `RuntimeError("VlmService is closed")`。
- `shutdown()` 用于优雅关闭底层服务，但最终仍建议 `close()`。

## 6. HTTP Gateway 对外接口

默认服务风格：OpenAI-compatible Chat Completions + 模型管理接口。具体 base URL 由部署方式决定，下文以 `http://127.0.0.1:8000` 为例。

### 6.1 健康检查

```http
GET /healthz
```

响应：

```json
{
  "status": "ok",
  "model": "vlm",
  "backend": "vlm"
}
```

`status` 可能为：

- `ok`：服务 ready。
- `not_ready`：服务已创建但模型未就绪。

若 Gateway 服务实例未初始化，返回 HTTP `503`。

### 6.2 OpenAI 兼容非流式对话

```http
POST /chat/completions
Content-Type: application/json
```

请求：

```json
{
  "model": "qwen0.8b",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "请描述这张图片"},
        {"type": "image_url", "image_url": {"url": "/home/bianbu/pic/猫咪.png"}}
      ]
    }
  ],
  "max_tokens": 64,
  "temperature": 0.2,
  "top_p": 0.95,
  "enable_thinking": false,
  "vision_history": "last",
  "stream": false
}
```

响应：

```json
{
  "id": "chatcmpl-xxxxxxxx",
  "object": "chat.completion",
  "created": 1779936000,
  "model": "qwen0.8b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 171,
    "completion_tokens": 48,
    "total_tokens": 219
  }
}
```

`usage.prompt_tokens`、`usage.completion_tokens`、`usage.total_tokens` 为本次非流式请求的 token 用量。该用量来自底层 VLM 服务响应，Gateway 只做 OpenAI 兼容格式透传。

请求字段说明：

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `model` | string | `vlm` | 模型 ID，响应原样返回 |
| `messages` | array | `[]` | OpenAI 兼容消息数组 |
| `messages[].role` | string | `user` | `system` / `user` / `assistant` |
| `messages[].content` | string 或 array | `""` | 文本或 multimodal content 数组 |
| `max_tokens` | int | `150` | 非流式请求透传到 Python `chat()` |
| `temperature` | float | `0.2` | 非流式请求透传 |
| `top_p` | float | `0.95` | 非流式请求透传 |
| `enable_thinking` | bool | `false` | 非流式请求透传 |
| `vision_history` | string | `last` | Gateway 接收字段，传给服务层 |
| `stream` | bool | `false` | 是否使用 SSE 流式输出 |

Multimodal content 解析规则：

- `type=text` 的 `text` 会按空格拼接为消息文本。
- `type=image_url` 的 `image_url.url` 会作为 `image_path` 传入后端。
- 未知 `type` 当前会被忽略。

### 6.3 OpenAI 兼容流式对话

```http
POST /chat/completions
Content-Type: application/json
```

请求：

```json
{
  "model": "qwen0.8b",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "请描述这张图片"},
        {"type": "image_url", "image_url": {"url": "/home/bianbu/pic/猫咪.png"}}
      ]
    }
  ],
  "stream": true
}
```

响应为 SSE：

```text
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1779936000,"model":"qwen0.8b","choices":[{"index":0,"delta":{"content":"..."},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1779936000,"model":"qwen0.8b","choices":[{"index":0,"delta":{"content":""},"finish_reason":"stop"}]}

data: [DONE]
```

限制：当前 Gateway 流式请求不透传 `max_tokens`、`temperature`、`top_p`、`enable_thinking` 等单次请求覆盖参数，使用服务配置默认值。

### 6.4 模型列表

```http
GET /models
```

响应：

```json
{
  "object": "list",
  "data": [
    {
      "id": "vlm",
      "object": "model",
      "owned_by": "spacemit",
      "status": "ready"
    }
  ]
}
```

实际列表由 `VlmGatewayService.list_models()` 返回。

### 6.5 加载模型

```http
POST /models/load
Content-Type: application/json
```

请求：

```json
{
  "model_id": "qwen0.8b",
  "config_path": "examples/fastvlm/config/qwen3_5_0.8b.yaml"
}
```

响应：

```json
{
  "status": "ok",
  "model_id": "qwen0.8b"
}
```

错误：

- `400`：缺少 `config_path`。
- `500`/`503`：加载失败或服务未初始化。

### 6.6 卸载模型

```http
POST /models/unload
```

响应：

```json
{
  "status": "ok"
}
```

### 6.7 切换模型

```http
POST /models/switch
Content-Type: application/json
```

请求格式与 `/models/load` 相同。该接口会设置新的 `model_id` 并加载对应配置。

### 6.8 指标

```http
GET /stats
```

`/stats` 返回当前 Gateway 持有的 VLM 服务实例指标，适合获取平均延时、最近一次延时、TTFT、输出规模和吞吐等运行指标。也可以通过 `/info` 获取同一份指标快照：

```http
GET /info
```

其中 `/info.metrics` 与 `/stats` 字段和语义一致。

响应示例：

```json
{
  "total_requests": 4,
  "is_processing": false,
  "last_latency_ms": 1423.47,
  "avg_latency_ms": 1521.60,
  "last_ttft_ms": 236.49,
  "last_output_tokens": 30,
  "last_tokens_per_second": 21.08
}
```

字段语义：

| 字段 | 含义 |
| --- | --- |
| `total_requests` | 当前 Gateway 服务实例已完成请求数 |
| `is_processing` | 当前是否正在处理请求 |
| `last_latency_ms` | 最近一次请求端到端延迟 |
| `avg_latency_ms` | 当前服务实例内完成请求的平均端到端延迟 |
| `last_ttft_ms` | 最近一次流式请求首个非空 SSE chunk 到达耗时；非流式 server 后端保持 `0.0` |
| `last_output_tokens` | 非流式请求为 completion token 数；流式请求为非空输出 chunk 数 |
| `last_tokens_per_second` | 非流式请求为 token/s；流式请求为 chunk/s |

统计口径：

- `avg_latency_ms` 按当前服务实例生命周期内已完成请求累计计算，模型卸载或服务重建后重新开始统计。
- 非流式 `/chat/completions` 成功后会更新请求数、`last_latency_ms`、`avg_latency_ms`、`last_output_tokens` 和 `last_tokens_per_second`；非流式 token 用量同时在响应体 `usage` 中返回。
- 流式 `/chat/completions` 成功后会更新请求数、`last_latency_ms`、`avg_latency_ms`、`last_ttft_ms`、`last_output_tokens` 和 `last_tokens_per_second`。
- 当前流式 SSE 事件不提供可靠 token usage，因此流式场景下 `last_output_tokens` 表示非空输出 chunk 数，不表示 token 数。
- 当前非流式 server 后端不通过内部流式实现，非流式请求的 `last_ttft_ms` 保持 `0.0`。

非流式请求后的示例：

```json
{
  "total_requests": 1,
  "is_processing": false,
  "last_latency_ms": 1619.72,
  "avg_latency_ms": 1619.72,
  "last_ttft_ms": 0.0,
  "last_output_tokens": 36,
  "last_tokens_per_second": 22.23
}
```

流式请求后的示例中，`last_ttft_ms` 为真实首 chunk 时间，`last_output_tokens` 为非空 chunk 数。

### 6.9 运行信息

```http
GET /info
```

响应：

```json
{
  "domain": "vlm",
  "model_id": "qwen0.8b",
  "ready": true,
  "metrics": {
    "total_requests": 4,
    "is_processing": false,
    "last_latency_ms": 1423.47,
    "avg_latency_ms": 1521.60,
    "last_ttft_ms": 236.49,
    "last_output_tokens": 30,
    "last_tokens_per_second": 21.08
  }
}
```

curl 示例：

```bash
curl -s http://127.0.0.1:8000/stats
curl -s http://127.0.0.1:8000/info
```

## 7. 错误码与异常行为

| 场景 | C++ | Python | HTTP |
| --- | --- | --- | --- |
| 配置文件不存在/模型加载失败 | 返回 `false`，写入 `error` | 抛出 `RuntimeError` | `/models/load` 返回异常 |
| 服务未初始化 | `IsReady()==false` 或调用失败 | 初始化失败抛异常 | `503 VLM service not initialized` |
| 推理失败 | 返回 `false`，写入 `error` | 抛出 `RuntimeError` | `500` 或 `503` |
| `close()` 后继续调用 | 不适用 | `RuntimeError("VlmService is closed")` | 不适用 |
| 流式过程异常 | callback error / 返回 false | generator 抛 `RuntimeError` | SSE 发送 `finish_reason=error` 后 `[DONE]` |
| 缺少 `config_path` | 不适用 | 不适用 | `400 config_path is required` |

## 2. 生产部署前验证命令

在远端仓库根目录执行：

```bash
cd /home/bianbu/model-zoo-vlm/vlm
cmake --build build_perf -j2
ctest --test-dir build_perf --output-on-failure
PYTHONPATH=$PWD/src/python:$PWD \
  python3 -m unittest discover -s tests/functional -p "test_vlm_service.py" -v
```

实机模型矩阵：

```bash
PYTHONPATH=$PWD/src/python:$PWD \
  python3 tests/functional/full_model_matrix.py \
  --image /home/bianbu/pic/猫咪.png \
  --max-tokens 24
```

本次交付已验证结果：

| 验证项 | 结果 |
| --- | --- |
| CMake build | 通过 |
| CTest | 2/2 通过 |
| Python/Gateway 边界测试 | 15/15 通过 |
| fastvlm 实机矩阵 | 通过，覆盖单轮、多轮、generate stream、chat stream |
| qwen3.5-0.8b 实机矩阵 | 通过，`FULL_MODEL_MATRIX_OK` |
| qwen3.5-2b 实机矩阵 | 通过，`FULL_MODEL_MATRIX_OK` |

本次验证中出现但不影响通过的日志：

- GPU 不可用提示：当前部署使用 CPU/RISCV/SMT 路径。
- `/dev/tcm_sync_mem` 不存在导致 fallback heap：运行继续。
- SMT audio backend config 缺失：当前图文任务不依赖 audio。
- ONNX Runtime constant folding warning：不影响 vision 推理通过。
- FastAPI `on_event` deprecation warning：不影响当前接口行为。

## 9. 生产使用建议

1. 优先使用 YAML 配置文件创建服务，避免调用方手动拼装模型路径。
2. 图片路径使用绝对路径，并确保服务进程有读取权限。
3. Python 调用优先使用 `with VlmService(...) as service` 自动释放资源。
4. C/C++ 调用必须在异常路径和正常路径都释放服务资源。
5. HTTP 流式接口按 SSE 消费，直到收到 `data: [DONE]`。
6. 如需流式请求级 `max_tokens` 等覆盖，当前版本需先修改服务默认生成配置或 YAML；HTTP/Python 流式接口暂不支持单次覆盖。
7. 生产发布前保留 `full_model_matrix.py` 的单轮、多轮、双流式矩阵作为回归门禁。

## 10. 版本交付结论

当前对外接口已覆盖：

- C++ 原生服务能力。
- Python HTTP 封装（纯 Python，无需 C ABI/ctypes）。
- Python 高层调用能力。
- OpenAI-compatible HTTP chat completions 能力。
- 健康检查、模型加载/卸载/切换、指标读取。
- 单轮、多轮、不同模型、图文输入、同步输出、流式输出。
- 生命周期误用、未初始化、流式限制、未知 multimodal content 等边界异常测试。

基于本轮远端验证结果，可作为当前生产环境接口交付文档使用。
