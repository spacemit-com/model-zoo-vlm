# VLM 用户适配指导

本文档为 SpacemiT VLM 服务的用户适配指导，涵盖功能模式选择、性能优化、常见问题解决、高级功能技巧及系统资源配置。

---

## 1. 功能模式及选择建议

VLM 服务支持三种使用模式，根据场景选择最合适的方式：

### 1.1 独立服务模式（推荐用于单应用场景）

通过配置文件直接启动 VLM 服务，适合单应用直接调用。

```python
from vlm import VlmService

with VlmService("config/fastvlm.yaml") as svc:
    result = svc.generate("Describe this image", image_path="photo.jpg")
    print(result["text"])
```

**适用场景：**
- 单一应用直接使用 VLM 推理
- 嵌入式部署，不需要多模型管理
- 开发调试阶段

### 1.2 Gateway 集成模式（推荐用于多模型/多服务场景）

通过 AI Gateway 统一管理 VLM 模型，支持模型加载/卸载/切换，适合多模型并发服务。

```bash
# 启动 Gateway
cd /home/bianbu/model-zoo/gateway
PYTHONPATH=src python3 -m uvicorn spacemit_ai_gateway.app.main:app \
  --host 0.0.0.0 --port 18790

# 注册并加载模型
curl -X POST http://127.0.0.1:18790/v1/vlm/models/register \
  -H "Content-Type: application/json" \
  -d {model: fastvlm-0.5b, source_type: local_path, local_path: /path/to/model}

curl -X POST http://127.0.0.1:18790/v1/vlm/models/load \
  -H "Content-Type: application/json" \
  -d {model: fastvlm-0.5b}

# 推理
curl -X POST http://127.0.0.1:18790/v1/vlm/chat/completions \
  -H "Content-Type: application/json" \
  -d {model: fastvlm-0.5b, messages: [{role: user, content: Hello}], max_tokens: 50}
```

**适用场景：**
- 多模型管理（同时服务 LLM + VLM）
- 需要动态加载/卸载模型
- 需要远程代理模式（连接外部 VLM 服务）
- 生产环境部署

### 1.3 C++ 直接调用模式

通过 C++ API 直接调用 VLM 推理，适合对延迟敏感的嵌入式场景。

```cpp
#include "vlm_service.h"

int main() {
    vlm::VlmService service("config/fastvlm.yaml");
    service.start();
    service.stop();
    return 0;
}
```

**适用场景：**
- C/C++ 应用集成
- 对启动延迟有严格要求
- 不需要 Python 环境

### 模式对比

| 特性 | 独立服务模式 | Gateway 模式 | C++ 直接调用 |
|------|-------------|-------------|-------------|
| 多模型管理 | 否 | 是 | 否 |
| 动态加载/卸载 | 否 | 是 | 否 |
| 远程代理 | 否 | 是 | 否 |
| Python 接口 | 是 | 是(HTTP) | 否 |
| C++ 接口 | 否 | 否 | 是 |
| 流式推理 | 是 | 是 | 是 |
| 多轮对话 | 是 | 是 | 是 |
| 部署复杂度 | 低 | 中 | 低 |

---

## 2. 性能优化配置参数说明

### 2.1 核心推理参数

| 参数 | 默认值 | 说明 | 优化建议 |
|------|--------|------|---------|
| `ctx_size` | 4096 | 上下文窗口大小 | 长文档场景增大到 8192，短对话可减小到 2048 |
| `threads` | 8 | 推理线程数 | 设为物理核心数，RISC-V 8核建议 8 |
| `batch_threads` | 8 | 批处理线程数 | 与 threads 保持一致 |
| `n_gpu_layers` | 0 | GPU 加速层数 | 有 GPU 时设为 -1（全部离载） |
| `max_tokens` | 150 | 最大生成 token 数 | 按需设置，过长增加延迟 |
| `temperature` | 0.2 | 采样温度 | 精确任务用 0-0.1，创意任务用 0.7-1.0 |
| `top_p` | 0.95 | 核采样概率 | 与 temperature 配合调整 |
| `top_k` | 40 | 候选 token 数 | 减小可加速但降低多样性 |
| `repeat_penalty` | 1.2 | 重复惩罚 | 增大可减少重复输出 |

### 2.2 VLM 视觉参数

| 参数 | 默认值 | 说明 | 优化建议 |
|------|--------|------|---------|
| `media_backend` | smt | 视觉后端类型 | SpacemiT 硬件用 smt，其他用 auto |
| `smt_config_dir` | 模型目录 | SMT 配置路径 | 指向模型目录即可 |
| `vision_model_paths` | 自动检测 | 视觉模型路径 | 通常自动从 config.json 读取 |
| `no_warmup` | true | 跳过预热 | 生产环境建议 false（首次推理更快） |

### 2.3 推理模式参数

| 参数 | 默认值 | 说明 | 优化建议 |
|------|--------|------|---------|
| `enable_thinking` | false | 启用思维链推理 | 复杂推理任务开启 |
| `reasoning_budget` | -1 | 思维链 token 预算 | 设为正数限制推理长度 |
| `port` | 8093 | 服务监听端口 | 确保端口不冲突 |

### 2.4 CPU 亲和性优化

在 RISC-V 多核平台上，通过 CPU 亲和性绑定可提升性能：

```yaml
cpu_affinity: "0-7"   # 绑定到 0-7 号核心
```

对于大小核架构：
```yaml
cpu_affinity: "4-7"   # 仅绑定到性能核心
```

---

## 3. 常见使用问题解决方案

### 3.1 服务启动失败

**问题：** VLM 服务启动超时或失败

**排查步骤：**
1. 检查端口是否被占用：`ss -tlnp | grep 8093`
2. 检查模型文件是否完整：`ls -la /path/to/model/*.gguf /path/to/model/config.json`
3. 查看服务日志：`cat /tmp/vlm_8093.log`

**常见原因：**
- 端口冲突：修改配置文件中的 port
- 模型路径错误：检查 model_dir 和 text_model_path
- 内存不足：减小 ctx_size 或 threads

### 3.2 推理结果为空或截断

**问题：** 推理返回空文本或内容不完整

**解决方案：**
- 增大 `max_tokens`（默认 150 可能不够）
- 检查 `temperature` 是否为 0（可能导致确定性截断）
- 确认 `ctx_size` 足够容纳输入+输出

### 3.3 推理速度慢

**问题：** 推理延迟过高

**优化方案：**
1. 减小 `ctx_size`（如 2048）
2. 减小 `max_tokens`
3. 设置 `cpu_affinity` 绑定性能核心
4. 增大 `threads`（不超过物理核心数）
5. 开启 `no_warmup: true` 跳过预热

### 3.4 多轮对话上下文丢失

**问题：** 多轮对话中模型无法记住之前的内容

**原因分析：**
- 0.5B 小模型上下文保持能力有限
- `ctx_size` 设置过小，历史被截断

**解决方案：**
- 增大 `ctx_size`（如 8192）
- 使用更大的模型（如 2B）
- 在应用层管理对话历史，只保留最近 N 轮

### 3.5 Gateway 模型加载失败

**问题：** Gateway 加载模型报错

**排查步骤：**
1. 确认模型已注册：`curl http://127.0.0.1:18790/v1/vlm/models`
2. 确认 llama-server 可执行：`which llama-server`
3. 查看 Gateway 日志：`tail -50 /tmp/gateway.log`

### 3.6 图片推理不生效

**问题：** 传入图片但模型没有视觉理解

**排查步骤：**
1. 确认视觉模型文件存在：`ls -la /path/to/model/*.onnx`
2. 确认 `--mmproj` 参数正确传递
3. 确认 `--vision-backend` 设置正确（SpacemiT 硬件用 smt）

---

## 4. 高级功能使用技巧及最佳实践

### 4.1 流式推理最佳实践

流式推理适合交互式场景（如聊天界面），可显著降低首字延迟：

```python
from vlm import VlmService

with VlmService("config/fastvlm.yaml") as svc:
    for chunk in svc.generate_stream("Tell me a story"):
        print(chunk, end="", flush=True)
```

**最佳实践：**
- 交互式 UI 必须使用流式模式
- 设置合理的 `max_tokens` 避免无限生成
- 流式中断后服务会自动恢复，无需重启

### 4.2 多轮对话最佳实践

```python
from vlm import VlmService

with VlmService("config/fastvlm.yaml") as svc:
    messages = []
    messages.append({"role": "user", "content": "Hello, I am Alice."})
    r1 = svc.chat(messages)
    messages.append({"role": "assistant", "content": r1["text"]})
    messages.append({"role": "user", "content": "What is my name?"})
    r2 = svc.chat(messages)
    messages.append({"role": "assistant", "content": r2["text"]})
```

**最佳实践：**
- 始终维护完整的 messages 列表
- 当历史过长时，保留最近 N 轮对话
- 使用 `history_policy: "last"` 自动管理历史

### 4.3 图片推理最佳实践

```python
from vlm import VlmService

with VlmService("config/fastvlm.yaml") as svc:
    # 方式1：通过文件路径
    result = svc.generate("Describe this image", image_path="photo.jpg")
    # 方式2：通过字节数据
    with open("photo.jpg", "rb") as f:
        image_bytes = f.read()
    result = svc.generate("Describe this image", image_bytes=image_bytes)
    # 方式3：多轮对话中带图片
    messages = [
        {"role": "user", "content": "What is in this image?", "image_path": "photo.jpg"}
    ]
    result = svc.chat(messages)
```

**最佳实践：**
- 图片建议压缩到 512x512 以内
- 支持 JPEG/PNG 格式
- 单次请求建议不超过 1 张图片

### 4.4 Gateway 远程代理模式

当 VLM 服务部署在其他机器上时，可使用远程代理模式：

```bash
curl -X POST http://127.0.0.1:18790/v1/vlm/models/register \
  -H "Content-Type: application/json" \
  -d {model: remote-vlm, source_type: remote, api_base_url: http://192.168.1.100:8093/v1}

curl -X POST http://127.0.0.1:18790/v1/vlm/models/switch \
  -H "Content-Type: application/json" \
  -d {model: remote-vlm}
```

### 4.5 性能监控

```python
from vlm import VlmService

with VlmService("config/fastvlm.yaml") as svc:
    result = svc.generate("Hello")
    metrics = svc.get_metrics()
    print(f"Total requests: {metrics[total_requests]}")
    print(f"Last latency: {metrics[last_latency_ms]:.1f} ms")
```

---

## 5. 系统资源配置推荐及环境要求

### 5.1 硬件要求

| 模型规模 | 最低内存 | 推荐内存 | 推荐核心数 | 磁盘空间 |
|---------|---------|---------|-----------|---------|
| 0.5B (Q4_1) | 1 GB | 2 GB | 4 核 | 1 GB |
| 2B (Q4_1) | 2 GB | 4 GB | 8 核 | 2 GB |
| 7B (Q4_1) | 6 GB | 8 GB | 8 核 | 5 GB |

### 5.2 软件依赖

| 组件 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.10+ | Python 接口和 Gateway |
| llama-server | 最新版 | 核心推理引擎 |
| llama.cpp (libllama) | 最新版 | C++ 接口依赖 |
| FastAPI | 0.100+ | Gateway 框架 |
| httpx | 0.24+ | Gateway HTTP 客户端 |
| PyYAML | 6.0+ | 配置文件解析 |

### 5.3 支持的架构

| 架构 | 状态 | 说明 |
|------|------|------|
| RISC-V 64 | 完全支持 | SpacemiT K1/X1 等 |
| x86_64 | 完全支持 | 标准 Linux |
| AArch64 | 完全支持 | ARM 服务器 |

### 5.4 网络配置

| 服务 | 默认端口 | 说明 |
|------|---------|------|
| VLM llama-server | 8093 | 独立模式推理端口 |
| AI Gateway | 18790 | Gateway 管理端口 |

**防火墙配置：**
```bash
sudo iptables -A INPUT -p tcp --dport 8093 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 18790 -j ACCEPT
```

### 5.5 生产环境部署建议

1. **进程管理：** 使用 systemd 或 supervisor 管理 llama-server 和 Gateway 进程
2. **日志管理：** 配置日志轮转，避免磁盘写满
3. **健康检查：** 定期调用 `/health` 端点监控服务状态
4. **资源隔离：** 使用 cgroup 限制内存和 CPU 使用
5. **安全加固：** Gateway 启用 API Key 认证，VLM 服务绑定 127.0.0.1

---

## 附录 A：配置文件模板

```yaml
model_dir: ~/.cache/models/vlm/fastvlm-mm-0.5b-q4_1
host: 127.0.0.1
port: 8093
media_backend: smt
cpu_affinity: ""
n_gpu_layers: 0
no_warmup: true

generation:
  max_tokens: 150
  context_size: 4096
  top_k: 40
  top_p: 0.95
  temperature: 0.2
  repeat_penalty: 1.2
  threads: 8
  batch_threads: 8
  enable_thinking: false
  reasoning_budget: -1
  history_policy: last
```

## 附录 B：Gateway API 速查

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/v1/vlm/models` | 列出所有已注册模型 |
| POST | `/v1/vlm/models/register` | 注册模型 |
| POST | `/v1/vlm/models/deregister` | 注销模型 |
| POST | `/v1/vlm/models/load` | 加载模型 |
| POST | `/v1/vlm/models/unload` | 卸载模型 |
| POST | `/v1/vlm/models/switch` | 切换活跃模型 |
| GET | `/v1/vlm/healthz` | 健康检查 |
| POST | `/v1/vlm/chat/completions` | VLM 推理（OpenAI 兼容） |
| POST | `/v1/chat/completions` | 兼容路径 |
