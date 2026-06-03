# VLM 测试文档

本文件汇总本地测试、功能测试、真实模型矩阵和 Gateway curl 验证入口。

## 1. 常规验证命令

### 验证测试

提交或部署前建议执行以下验证，覆盖 C++ 单元测试、Python/Gateway 绑定测试，以及脚本支持的全部 VLM 模型 demo：

```bash
cd vlm

# 构建与 C++ 单元测试
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure

# Python 绑定与 Gateway 路由测试
PYTHONPATH=$PWD/src/python:$PWD \
  python3 -m pytest tests/functional/test_vlm_service.py -q -k "not Gateway"
```

Gateway 绑定与模型切换测试在上级 `model-zoo/gateway` 包中执行，使用 `tests/unit/test_vlm_service.py` 验证远程模型注册、切换和生命周期管理。VLM 本地绑定测试可通过排除 Gateway 相关测试执行。真实模型全量矩阵可用以下脚本验证单轮、多轮、流式生成、流式对话和指标统计：

```bash
PYTHONPATH=$PWD/src/python:$PWD \
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

## 2. 生产验证命令

## 2. 生产部署前验证命令

在远端仓库根目录执行：

```bash
cd /home/bianbu/model-zoo-vlm/vlm
cmake --build build_perf -j2
ctest --test-dir build_perf --output-on-failure
PYTHONPATH=$PWD/src/python:$PWD \
  python3 -m pytest tests/functional/test_vlm_service.py -q -k "not Gateway"
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


## 3. Gateway curl 测试结论

| 验证项 | 结果 |
| --- | --- |
| Gateway 启动与健康检查 | 通过 |
| 模型列表 | 通过 |
| 模型加载 | 通过 |
| 模型切换 | 通过 |
| 模型卸载 | 通过 |
| 卸载后推理保护 | 通过 |
| 恢复加载与热身推理 | 通过 |
| 非流式参数推理 | 通过 |
| 多 role 多轮推理 | 通过 |
| 流式 SSE 推理 | 通过 |
| `/stats` 指标更新 | 通过 |

当前 Gateway 实现已通过本轮 curl 端到端验证，可满足启动、加载、切换、停止模型以及多参数、多 role、流式和非流式推理的基础交付要求。
