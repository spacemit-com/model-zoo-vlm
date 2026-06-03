# VLM 快速使用指南

本指南面向首次部署和本地快速验证，保留最短路径：安装依赖、下载模型、编译、运行测试和启动 demo。完整 API、测试报告和架构说明见 README 导航。

## 1. 快速开始

### 1.1 安装依赖

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake curl libyaml-cpp-dev nlohmann-json3-dev
sudo apt install llama.cpp-tools-spacemit
```

### 1.2 下载模型

```bash
cd vlm

# 下载 FastVLM-MM 0.5B 模型
bash scripts/download_model.sh fastvlm-mm-0.5b-q4_1

# 或下载 Qwen3.5-2B 模型
bash scripts/download_model.sh Qwen3.5-2B

# 下载脚本支持的全部 VLM 模型
for model in fastvlm-mm-0.5b-q4_1 Qwen3.5-0.8B Qwen3.5-2B; do
  bash scripts/download_model.sh "$model"
done

# 自定义存储目录
VLM_CACHE_DIR=/data/models bash scripts/download_model.sh Qwen3.5-2B
```

模型将下载到 `.cache/models/vlm/<model_name>/` 目录。下载完成后目录结构：

```
.cache/models/vlm/
├── fastvlm-mm-0.5b-q4_1/
│   ├── config.json
│   ├── fastvlm-text-0.5B-Q4_1.gguf
│   └── fastvlm_vision.f16.onnx
└── Qwen3.5-2B/
    ├── config.json
    ├── qwen3_5_2b-text-q41.gguf
    ├── qwen3_5_2b-vision-224-op23.f16.onnx
    ├── qwen3_5_2b-vision-384-op23.f16.onnx
    └── qwen3_5_2b-vision-768-op23.f16.onnx
```

### 1.3 编译

```bash
cd vlm
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
ctest --output-on-failure
```

### 1.4 验证测试

提交或部署前建议执行以下验证，覆盖 C++ 单元测试、Python/Gateway 绑定测试，以及脚本支持的全部 VLM 模型 demo：

```bash
cd vlm

# 构建与 C++ 单元测试
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure

# Python 绑定与 Gateway 路由测试
VLM_LIB_PATH=$PWD/build PYTHONPATH=$PWD/src/python:$PWD \
  python3 -m pytest tests/functional/test_vlm_service.py -q -k "not Gateway"
```

Gateway 绑定与模型切换测试在上级 `model-zoo/gateway` 包中执行，使用 `tests/unit/test_vlm_service.py` 验证远程模型注册、切换和生命周期管理。VLM 本地绑定测试可通过排除 Gateway 相关测试执行。真实模型全量矩阵可用以下脚本验证单轮、多轮、流式生成、流式对话和指标统计：

```bash
VLM_LIB_PATH=$PWD/build/libvlm.so PYTHONPATH=$PWD/src/python:$PWD \
  python3 tests/functional/full_model_matrix.py \
  --image /path/to/image.jpg --max-tokens 24
```

Demo 验证矩阵：

| 模型 | 配置 | C++ Demo | Python Demo | 说明 |
|------|------|----------|-------------|------|
| FastVLM-MM 0.5B Q4_1 | `examples/fastvlm/config/fastvlm.yaml` | `fastvlm_demo`, `chat_demo`, `stream_demo` | `vlm_demo.py`, `chat_demo.py`, `stream_demo.py` | 图像输入、非流式与流式均需通过 |
| Qwen3.5-0.8B | `examples/fastvlm/config/qwen3_5_0.8b.yaml` | `fastvlm_demo`, `chat_demo`, `stream_demo` | `vlm_demo.py`, `chat_demo.py`, `stream_demo.py` | 图像输入、非流式与流式均需通过 |
| Qwen3.5-2B | `examples/fastvlm/config/qwen3_5_2b.yaml` | `fastvlm_demo`, `chat_demo`, `stream_demo` | 可按同样方式验证 | 图像输入、非流式与流式均需通过；文本 smoke 可用无 `--image` 请求 |

每个 demo 退出后应自动清理其启动的 `llama-server` 子进程；若手动中断测试，可用以下命令清理残留进程和释放 TCM：

```bash
pkill -f llama-server || true
spacemit-tcm-smi -c
```

### 1.5 运行 C++ Demo

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

### 1.6 运行 Python Demo

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

## 2. 配置参考

### 2.1 YAML 配置文件

配置文件位于 `examples/fastvlm/config/` 目录，支持以下字段：

```yaml
# 后端类型: server (推荐) 或 direct
backend: server

# 模型目录 (包含 config.json、text GGUF、vision ONNX)
model_dir: .cache/models/vlm/fastvlm-mm-0.5b-q4_1

# 视觉后端: smt (TCM硬件加速), mtmd (标准GGUF mmproj), auto
media_backend: smt

# SMT 配置目录 (默认等于 model_dir)
# smt_config_dir: /path/to/model/dir

# 本地图片文件访问路径 (--media-path 参数)
# media_path: /home/user/pictures

# CPU 亲和性 (避免 NUMA 崩溃)
cpu_affinity: 0-7

# GPU 层数 (0 = 纯 CPU)
n_gpu_layers: 0

# 跳过热身 (避免 NUMA 崩溃)
no_warmup: true

# 服务监听地址和端口
host: 127.0.0.1
port: 8093

# 生成参数
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
  reasoning_budget: 1024
  history_policy: last        # keep_all / last / none
  stop_sequences:             # 可选停止序列
    - "<|im_end|>"
```

### 2.2 可用模型配置

| 配置文件 | 模型 | 端口 | 说明 |
|---------|------|------|------|
| `fastvlm.yaml` | FastVLM-MM 0.5B Q4_1 | 8093 | 轻量级，适合嵌入式部署 |
| `qwen3_5_0.8b.yaml` | Qwen3.5-0.8B | 8094 | 中等规模，平衡精度与速度 |
| `qwen3_5_2b.yaml` | Qwen3.5-2B | 8095 | 高精度，推荐服务器部署 |

### 2.3 llama-server 启动命令

组件会根据配置自动构建并启动 llama-server，等效命令：

```bash
OMP_NUM_THREADS=8 taskset -c 0-7 llama-server \
  -m /path/to/text-model.gguf \
  --vision-backend smt \
  --smt-config-dir /path/to/model/dir \
  --media-path /path/to/images \
  -ngl 0 --ctx-size 4096 -t 8 -tb 8 \
  --host 127.0.0.1 --port 8093 \
  --no-warmup --reasoning off
```

也可手动启动 llama-server 进行测试：

```bash
# 释放 TCM 内存 (如有旧进程占用)
spacemit-tcm-smi -c

# 启动服务
OMP_NUM_THREADS=8 taskset -c 0-7 llama-server \
  -m ~/.cache/models/vlm/Qwen3.5-2B/qwen3_5_2b-text-q41.gguf \
  --vision-backend smt \
  --smt-config-dir ~/.cache/models/vlm/Qwen3.5-2B \
  --media-path /home/user/pictures \
  -ngl 0 -c 4096 --port 8093 -t 8 -tb 8 --no-warmup

# 健康检查
curl -s http://127.0.0.1:8093/health

# 文本推理
curl -s http://127.0.0.1:8093/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":30}'

# 图像推理 (file:// 路径相对于 --media-path)
curl -s http://127.0.0.1:8093/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":[{"type":"text","text":"描述这张图片"},{"type":"image_url","image_url":{"url":"file://image.jpg"}}]}],"max_tokens":100}'

# 检查 TCM 状态
spacemit-tcm-smi
```
