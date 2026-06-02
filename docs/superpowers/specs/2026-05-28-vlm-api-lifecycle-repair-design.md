# VLM API、打包与生命周期修复设计

审查对象：`~/model-zoo-vlm/vlm`  
本地工作镜像：`/home/hsh/model_eval/_remote_audit/vlm`  
日期：2026-05-28

## 1. 目标

本次修复要把 VLM 的对外接口从“基础可用”提升到“生产可集成”：Python 包安装后能自动加载随 wheel 打包的 `libvlm.so`；OpenAI 兼容多模态请求能正确解析；用户显式传入的请求级生成参数要生效；异步接口要么真正安全可用，要么从头文件中移除；server backend 的子进程和对象生命周期要避免僵尸进程、悬空指针和资源泄漏；最终确认头文件暴露所有需要给 C++/C/Python 使用者的接口。

## 2. 现状判断

### 2.1 Python so 加载

`src/python/vlm.py` 当前优先读取 `VLM_LIB_PATH`，再搜索相对路径 `src/build`、`src/build_perf`、系统库目录。源码树中真实构建产物在项目根 `build_perf/libvlm.so`，因此不设置 `VLM_LIB_PATH` 时测试失败。

`~/model-zoo` 其他模块的模式是：

- `asr/python/setup.py`、`tts/python/setup.py`：通过 `setuptools.Extension + build_ext` 调用 CMake 构建扩展模块，并复制 `.so` 到 Python package 输出目录。
- `vad/python/setup.py`：支持 `SPACEMIT_PREBUILT_EXTENSION` 和 `SPACEMIT_CMAKE_BUILD_DIR`，更适合预构建二进制和 CI wheel。

VLM 当前是 ctypes 包装 `libvlm.so`，不是 pybind11 extension，因此更适合采用“Python package 内携带 `libvlm.so` + loader 优先从 package 目录静默加载”的方案。

### 2.2 OpenAI 多模态解析

`gateway/schemas.py::ChatCompletionRequest.from_dict()` 已能解析 OpenAI 风格：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]
    }
  ]
}
```

但 `gateway/api.py::chat_completions()` 直接声明 `request: ChatCompletionRequest`，没有调用 `from_dict()`，复杂 content-list 解析逻辑实际不会稳定生效。

### 2.3 请求级参数

Gateway schema 有 `max_tokens`、`temperature`、`top_p`、`stream`、`enable_thinking`、`reasoning_budget`、`vision_history`，但服务层没有把它们转换为临时 generation config；最终仍使用 YAML 中的静态配置。

C++ 层已有 `UpdateGenerationConfig()`，但它是全局修改，不适合 HTTP per-request 参数，因为并发请求会互相污染配置。

### 2.4 异步接口

C++ 头文件 `VlmService::GenerateAsync()` 已暴露异步接口，两个 backend 也有实现，但当前实现都通过 detached thread 捕获 `this` 调用成员函数，存在对象析构后悬空访问风险。C API 只定义了 `VlmAsyncCallbackC`，没有暴露 `vlm_generate_async()`；Python 也没有 async 方法。

判断：异步能力在 C++ 层“有接口但实现不安全”，在 C/Python 层“不完整”。建议本轮不删除异步接口，而是补齐安全实现和 C/Python 暴露，因为用户明确关注“如果支持，添加对应代码”，且 C++ public API 已经承诺该能力。

### 2.5 生命周期与子进程

`ServerVlmModel` 会 fork `llama-server`，`Shutdown()` 发送 SIGTERM 并 `waitpid()`。风险点：

- 如果进程不响应 SIGTERM，`waitpid()` 可能长时间阻塞。
- detached async 任务可能在析构后继续访问对象。
- `popen(curl)` 产生的 curl 子进程在中断流式请求时没有强制终止路径。
- 临时 payload 文件没有 RAII 包装，当前大部分路径会 unlink，但后续维护容易遗漏。
- `metrics_` 多线程读写没有统一加锁。

## 3. 推荐方案

采用“补齐能力并收紧生命周期”的方案：

1. 新增 Python package 化结构，wheel 内包含 `libvlm.so`，loader 优先从 package 目录静默加载，再 fallback 到 `VLM_LIB_PATH`、源码根 `build`/`build_perf`、系统目录。
2. Gateway 路由显式读取原始 JSON，并调用 `ChatCompletionRequest.from_dict()`，保证 OpenAI 多模态 content-list 被正确转换。
3. 引入 per-request generation override，不污染全局 YAML 配置：Gateway 根据请求参数构建 override，传给 C API/Python/C++。C++ 层新增带 config 的 generate/chat 重载或 request options；C API/Python 同步暴露。
4. 保留异步接口并修复：C++ backend 不再 detach 捕获裸 `this`；改为受控 async task 列表，析构时设置 stopping 标志并 join；C API 新增 `vlm_generate_async()`；Python 新增 `generate_async()` 返回 `concurrent.futures.Future`。
5. 完善生命周期：显式 `close()`/destroy、context manager 释放 handle、server 进程 TERM 后超时 KILL、阻止 shutdown 后新请求、析构等待 async 任务、metrics 加锁。
6. 保持 ABI 兼容优先：原有 C 函数不破坏，新增 `_ex` 或新增结构体函数承载扩展能力。

## 4. 备选方案

### 方案 A：删除异步接口

删除 `GenerateAsync`、`VlmAsyncCallbackC`，只保留 sync/stream。实现量较小，但会破坏 C++ public API，且当前头文件已经将 async 作为正式能力公开，不推荐。

### 方案 B：只修 Python/Gateway，不碰 C++ async

能快速解决 HTTP/Python 集成问题，但留下悬空指针风险；用户明确要求确认和修改生命周期问题，不推荐。

### 方案 C：全面补齐 SDK 能力

即推荐方案。工作量最大，但与对外 SDK 完整性目标一致。

## 5. 接口设计

### 5.1 Python 打包与加载

新增包结构：

```text
src/python/spacemit_vlm/
  __init__.py
  vlm.py
  libvlm.so        # wheel 构建时复制或 CMake 输出
```

新增/调整：

- `src/python/setup.py`：参考 `vad/python/setup.py`，支持 `SPACEMIT_PREBUILT_VLM_LIB`，默认调用 CMake target `vlm` 并把 `libvlm.so` 放入 package。
- `src/python/pyproject.toml`：声明 `setuptools`、`wheel`、`cmake`。
- `src/python/spacemit_vlm/vlm.py::_find_library()`：搜索顺序为 package 目录、`VLM_LIB_PATH`、源码根 `build`/`build_perf`、系统目录。
- `src/python/vlm.py`：保留兼容 shim，从 `spacemit_vlm.vlm` re-export。

### 5.2 OpenAI 请求解析

`gateway/api.py` 改为：

- endpoint 接收 `fastapi.Request`。
- `body = await raw_request.json()`。
- `request = ChatCompletionRequest.from_dict(body)`。
- 对非法 JSON 或非法 messages 返回 400。

`gateway/schemas.py` 增强：

- 支持 `content` 为字符串、list、空 list。
- 支持 `image_url.url` 为 data URL 或普通 URL。
- 对多个 text part 进行拼接。
- 保留第一个 image URL，若后续需要多图，再扩展 message 结构。
- 解析 `stop` 字段为 `list[str]`。

### 5.3 请求级参数覆盖

新增 C++ 结构：

```cpp
struct VlmRequestOptions {
    bool has_generation = false;
    VlmGenerationConfig generation;
};
```

在 `VlmService` 增加默认虚函数或纯虚函数：

```cpp
virtual bool GenerateWithOptions(const VlmInput& input,
                                 const VlmRequestOptions& options,
                                 VlmResult* result,
                                 std::string* error);
virtual bool ChatWithOptions(const std::vector<VlmChatMessage>& messages,
                             const VlmRequestOptions& options,
                             VlmChatResult* result,
                             std::string* error);
virtual bool ChatStreamWithOptions(const std::vector<VlmChatMessage>& messages,
                                   const VlmRequestOptions& options,
                                   VlmStreamCallback callback,
                                   VlmChatResult* result,
                                   std::string* error);
```

Direct backend 实现方式：生成时选择 `options.generation` 创建临时 sampler 或临时更新受锁保护的 sampler，调用结束恢复原配置。为了并发安全，优先使用“复制 config + 临时 sampler”的实现，不修改 `config_`。

Server backend 实现方式：构建 payload 时使用 effective config：`options.has_generation ? options.generation : config_.generation`，不修改 `config_`。注意 server 启动参数中的 `ctx-size`、threads、reasoning server flag 仍来自 YAML，因为这些需要重启 server；per-request 只覆盖 max_tokens、temperature、top_p、stop、vision_history、reasoning_budget 等请求 payload 参数。

C API 新增：

```c
typedef struct VlmCGenerationConfig {
    int max_tokens;
    float temperature;
    float top_p;
    int top_k;
    float repeat_penalty;
    int enable_thinking;
    int reasoning_budget;
    int history_policy;
    const char** stop_sequences;
    int stop_sequence_count;
} VlmCGenerationConfig;

int vlm_generate_with_config(..., const VlmCGenerationConfig* config, ...);
int vlm_chat_with_config(..., const VlmCGenerationConfig* config, ...);
int vlm_chat_stream_with_config(..., const VlmCGenerationConfig* config, ...);
int vlm_update_generation_config(VlmHandle handle, const VlmCGenerationConfig* config, char* error, size_t error_len);
```

Python 新增：

- `generate(..., max_tokens=None, temperature=None, top_p=None, top_k=None, repeat_penalty=None, enable_thinking=None, reasoning_budget=None, vision_history=None, stop=None)`
- `chat(messages, **generation_options)`
- `chat_stream(messages, **generation_options)`
- `update_generation_config(**generation_options)`

### 5.4 异步接口

C++ 保留 `GenerateAsync`，实现改为：

- backend 内保存 `std::vector<std::thread> async_threads_`。
- `std::atomic<bool> stopping_` 标记 shutdown/destructor。
- `GenerateAsync` 在持锁下检查 stopping，创建线程并记录。
- 线程执行前后不访问已析构对象：析构和 `Shutdown()` 先设置 stopping，再 join 所有 async 线程。
- callback 必须只调用一次，成功返回 result，失败返回 error。

C API 新增：

```c
int vlm_generate_async(VlmHandle handle,
                       const char* prompt,
                       const char* image_path,
                       const uint8_t* image_data,
                       size_t image_size,
                       VlmAsyncCallbackC callback,
                       void* user_data,
                       char* error,
                       size_t error_len);
```

Python 新增：

- `generate_async(...) -> concurrent.futures.Future`
- 内部持有 callback 引用，直到 Future 完成，避免 ctypes callback 被 GC。
- `close()`/`shutdown()` 前等待或取消未完成 Future；无法取消已进入 C++ 的任务时等待 callback 完成。

### 5.5 生命周期

C++：

- `ServerVlmModel::Shutdown()`：设置 stopping，join async；SIGTERM 子进程；轮询 `waitpid(pid, &status, WNOHANG)` 最多 5 秒；仍未退出则 SIGKILL 并 waitpid。
- `ServerVlmModel::StartServerIfNeeded()`：若启动超时，kill 并 wait，避免残留进程。
- `SyncRequest/StreamRequest`：用 RAII 临时文件对象确保 unlink；stream callback 返回 false 时尽快关闭 pipe。
- `GetMetrics()`、metrics 更新统一持锁或使用 atomic 成员，避免数据竞争。

Python：

- 新增 `close()`，调用 `vlm_shutdown` + `vlm_destroy`，handle 置空。
- `shutdown()` 只关闭服务但不释放 handle；`close()` 释放 handle。
- `__exit__()` 调用 `close()`。
- `__del__()` 捕获所有异常，只做 best-effort close。
- 所有 public method 在 handle 为空时抛 `RuntimeError("VlmService is closed")`。

Gateway：

- `VlmGatewayService.shutdown()` 调用 Python wrapper 的 `close()` 或 C++ service shutdown。
- model unload/switch 时持锁，阻止请求中途把 service 销毁。
- streaming 请求期间保留 service 局部引用，避免 unload 导致悬空。

## 6. 头文件暴露要求

最终对外头文件应包含：

### `include/vlm/vlm_service.h`

- 保留现有枚举、配置、输入输出、消息、指标。
- 新增 `VlmRequestOptions`。
- 保留 `GenerateAsync`。
- 新增 `GenerateWithOptions`、`ChatWithOptions`、`ChatStreamWithOptions`。
- 保留 `UpdateGenerationConfig` 作为全局配置更新接口。

### `src/core/cpp/vlm_c_api.h`

- 保留所有现有函数，避免破坏已有调用方。
- 新增 `VlmCGenerationConfig`。
- 新增 `vlm_generate_with_config`、`vlm_chat_with_config`、`vlm_chat_stream_with_config`。
- 新增 `vlm_update_generation_config`。
- 新增 `vlm_generate_async`，使 `VlmAsyncCallbackC` 不再是孤立定义。
- 视实现成本新增 image bytes chat 扩展结构；若不新增，文档必须明确 C/Python chat 仅支持 image path/data URL。

## 7. 测试计划

### Python

- `test_library_discovery_finds_packaged_lib`：模拟 package 目录有 `libvlm.so`，无需 `VLM_LIB_PATH`。
- `test_context_manager_closes_handle`：`with VlmService(...)` 退出后 `_handle is None`，再次调用抛 `RuntimeError`。
- `test_generate_accepts_request_options`：mock C 函数，确认传入 `VlmCGenerationConfig`。
- `test_generate_async_returns_future`：mock callback，确认 Future 完成且 callback 引用未提前释放。

### Gateway

- `test_chat_completion_parses_openai_multimodal_content`：POST content-list，service 收到 `image_url` 和文本。
- `test_chat_completion_forwards_generation_options`：POST `max_tokens/temperature/top_p/stop/enable_thinking`，service 收到 options。
- `test_streaming_keeps_service_reference`：streaming generator 创建后 unload 不破坏当前流。

### C++/C API

- `test_vlm_c_api_update_generation_config`：确认 C config 转 C++ config。
- `test_vlm_c_api_generate_async`：mock service 或轻量 fake backend 验证 callback 触发。
- `test_server_shutdown_kills_unresponsive_child`：对可控子进程验证 TERM 后 KILL fallback。
- `test_build_payload_uses_request_options`：server backend payload 包含请求覆盖参数。

## 8. 实施顺序

1. 先补测试，锁定 Python loader、Gateway 多模态解析、请求级参数传递、生命周期 close 行为。
2. 修 Python package/loader 和 close 语义。
3. 修 Gateway 请求解析和 generation options 传递。
4. 扩展 C++ request options 和 C API generation config。
5. 修异步线程与子进程生命周期。
6. 补齐头文件与 README/API 文档。
7. 远程运行 CTest、Python unittest、必要时构建 wheel 验证 so 被打包。

## 9. 审查点

请重点确认以下设计决策：

1. 异步接口：是否同意保留并补齐 C/Python 暴露，而不是删除？
2. per-request 参数：是否同意只覆盖请求 payload 级参数，不动态重启 server 改 threads/context size？
3. Python 包名：是否使用 `spacemit_vlm`，并保留旧 `vlm.py` shim 兼容历史导入？
4. C API：是否接受新增 `_with_config` 函数保持 ABI 兼容，而不是修改现有函数签名？
5. image bytes chat：本轮是否必须补到 C/Python，还是先把 OpenAI data URL 在 Gateway 层解析为 `image_url`/路径并明确限制？