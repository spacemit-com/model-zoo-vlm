# VLM 架构文档

本文件聚焦整体架构、数据流、后端边界和硬件加速机制。

## 1. 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│  Gateway 层 (FastAPI)                                       │
│  adapter.py → api.py → service.py                           │
│  OpenAI 兼容: /chat/completions, /models/load|unload|switch │
├─────────────────────────────────────────────────────────────┤
│  Python HTTP 封装层 (纯 Python)                              │
│  vlm.py → llama-server HTTP API (OpenAI 兼容)                │
├─────────────────────────────────────────────────────────────┤
│  C++ 核心层                                                  │
│  VlmService (抽象基类)                                       │
│  ├── ServerVlmModel (Server 后端) ← 推荐使用                │
│  │   └── llama-server 子进程 + curl HTTP API                │
│  │       └── --vision-backend smt → TCM 硬件加速             │
│  └── LlamaVlmModel (Direct 后端) ← 预留给 GGUF mmproj 模型  │
│      └── llama.cpp + mtmd 直接调用                           │
├─────────────────────────────────────────────────────────────┤
│  基础设施层                                                   │
│  vlm_config (YAML/JSON 解析) │ vlm_utils (工具函数)          │
│  mtmd_image (图像加载)       │ sampler (采样器)               │
│  vlm_model_factory (工厂)                                    │
└─────────────────────────────────────────────────────────────┘
```

**数据流 (Server 模式)**:

```
用户图片 + 文本 → VlmService API → ServerVlmModel
  → Base64 编码 → curl POST /v1/chat/completions
  → llama-server (SMT/TCM 加速) → SSE/JSON 响应
  → 解析结果 → 返回文本 + 统计信息
```

## 2. TCM 硬件加速

### 2.1 工作原理

SpacemiT K3 平台提供 TCM (Tensor Compute Module) 硬件加速单元。VLM 推理通过以下方式启用 TCM：

1. `llama-server` 使用 `--vision-backend smt` 参数加载 SMT 视觉后端
2. SMT 后端调用 SpacemiT ONNX Runtime Execution Provider
3. ONNX EP 自动将计算密集型算子卸载到 TCM 硬件

### 2.2 验证 TCM 是否启用

```bash
# 查看 TCM 块使用状态 (busy = 已启用)
spacemit-tcm-smi

# 期望输出:
# ID   STATE  SIZE       ...
# 0    busy   393216     ...    ← TCM 块已被视觉模型占用
# 1    busy   393216     ...
# ...

# 释放 TCM 内存
spacemit-tcm-smi -c
```

