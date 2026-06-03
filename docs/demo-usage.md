# VLM Demo 使用文档

本文件集中整理 C++、Python 和 Gateway demo 的运行方式，便于按入口快速验证模型能力。

## 1. C++ Demo

### 运行 C++ Demo

```bash
cd build

# 基础图像描述
./fastvlm_demo \
  --config ../examples/fastvlm/config/fastvlm.yaml \
  --image /path/to/image.jpg \
  --prompt "用一句话描述这张图片"

# 多轮对话
./chat_demo \
  --config ../examples/fastvlm/config/qwen3_5_2b.yaml \
  --image /path/to/image.jpg \
  --prompt "图片中有什么？"

# 流式输出
./stream_demo \
  --config ../examples/fastvlm/config/fastvlm.yaml \
  --image /path/to/image.jpg \
  --prompt "详细描述这张图片"
```

## 2. Python Demo

### 运行 Python Demo

```bash
cd vlm

# 基础图像描述
python3 examples/fastvlm/python/vlm_demo.py \
  --config examples/fastvlm/config/fastvlm.yaml \
  --image /path/to/image.jpg \
  --prompt "Describe this image"

# 多轮对话
python3 examples/fastvlm/python/chat_demo.py \
  --config examples/fastvlm/config/qwen3_5_2b.yaml \
  --image /path/to/image.jpg

# 流式输出
python3 examples/fastvlm/python/stream_demo.py \
  --config examples/fastvlm/config/fastvlm.yaml \
  --image /path/to/image.jpg
```

## 3. Gateway Demo

HTTP Gateway 使用 OpenAI 兼容接口，完整端点定义见 [API 文档](production/vlm_api_reference.md)。

### 3.1 启动 Gateway

```bash
cd /home/bianbu/model-zoo/gateway
PYTHONPATH=src python3 -m uvicorn spacemit_ai_gateway.app.main:app \
  --host 0.0.0.0 \
  --port 18790
```

### 3.2 加载模型

```bash
# 加载预设模型
curl -X POST http://127.0.0.1:18790/v1/vlm/models/load \
  -H 'Content-Type: application/json' \
  -d '{"model":"fastvlm-mm-0.5b-q4_1"}'
```

### 3.3 发送非流式请求

```bash
curl -sS http://127.0.0.1:18790/v1/vlm/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"fastvlm-mm-0.5b-q4_1",
    "messages":[{"role":"user","content":"用一句话描述这张图片"}],
    "max_tokens":64,
    "stream":false
  }'
```

### 3.4 发送流式请求

```bash
curl -sS http://127.0.0.1:18790/v1/vlm/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"fastvlm-mm-0.5b-q4_1",
    "messages":[{"role":"user","content":"Count from one to three."}],
    "max_tokens":32,
    "stream":true
  }'
```
