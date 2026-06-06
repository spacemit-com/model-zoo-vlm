#!/usr/bin/env python3
"""Run full real-model VLM validation against configured model files.

This script is intentionally separate from the fast unittest suite because it
starts real llama-server processes and runs hardware-backed inference.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS = [
    ("fastvlm", "examples/fastvlm/config/fastvlm.yaml"),
    ("qwen3.5-0.8b", "examples/fastvlm/config/qwen3_5_0.8b.yaml"),
    ("qwen3.5-2b", "examples/fastvlm/config/qwen3_5_2b.yaml"),
]


def cleanup_runtime() -> None:
    subprocess.run(["pkill", "-f", "llama-server"], stderr=subprocess.DEVNULL)
    subprocess.run(
        ["spacemit-tcm-smi", "-c"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def require_non_empty(label: str, value: str) -> None:
    if not value.strip():
        raise RuntimeError(f"{label} produced empty output")


def run_model(name: str, config: Path, image: str, max_tokens: int) -> None:
    from vlm import VlmService

    print(f"\n=== MODEL {name} ({config}) ===", flush=True)
    started = time.monotonic()

    with VlmService(str(config)) as service:
        single = service.generate(
            "用不超过二十个中文词描述这张图。",
            image_path=image,
            max_tokens=max_tokens,
        )
        print("single_turn_len", len(single["text"]))
        require_non_empty("single-turn generation", single["text"])

        messages = [
            {"role": "system", "content": "你是简洁的视觉问答助手。"},
            {
                "role": "user",
                "content": "这张图片的主体是什么？",
                "image_path": image,
            },
            {"role": "assistant", "content": single["text"]},
            {"role": "user", "content": "再补充一个可见细节。"},
        ]
        chat = service.chat(messages, max_tokens=max_tokens)
        print("multi_turn_len", len(chat["text"]))
        require_non_empty("multi-turn chat", chat["text"])

        stream_chunks = list(
            service.generate_stream("用一句话描述图片。", image_path=image)
        )
        stream_text = "".join(stream_chunks)
        print("stream_chunks", len(stream_chunks), "stream_len", len(stream_text))
        require_non_empty("stream generation", stream_text)

        chat_stream_chunks = list(
            service.chat_stream(
                [
                    {"role": "system", "content": "你是简洁的视觉问答助手。"},
                    {
                        "role": "user",
                        "content": "用一句话说明图片内容。",
                        "image_path": image,
                    },
                ]
            )
        )
        chat_stream_text = "".join(chat_stream_chunks)
        print(
            "chat_stream_chunks",
            len(chat_stream_chunks),
            "chat_stream_len",
            len(chat_stream_text),
        )
        require_non_empty("chat stream", chat_stream_text)

        metrics = service.get_metrics()
        if metrics.get("total_requests", 0) < 4:
            raise RuntimeError(f"metrics did not count all requests: {metrics}")

    elapsed = time.monotonic() - started
    print(f"=== OK {name}: {elapsed:.1f}s ===", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default="/home/bianbu/pic/猫咪.png",
        help="Image path available on the target machine.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=24,
        help="Per-request max token override for non-stream calls.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Model config to test. Can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configs = []
    if args.config:
        for item in args.config:
            if "=" not in item:
                raise SystemExit("--config must use NAME=PATH")
            name, path = item.split("=", 1)
            configs.append((name, path))
    else:
        configs = DEFAULT_CONFIGS

    failures = []
    for name, config_path in configs:
        cleanup_runtime()
        try:
            run_model(name, ROOT / config_path, args.image, args.max_tokens)
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"=== FAILED {name}: {exc} ===", file=sys.stderr, flush=True)
        finally:
            cleanup_runtime()

    if failures:
        print("\nFAILED_MODELS")
        for name, error in failures:
            print(f"- {name}: {error}")
        return 1

    print("\nFULL_MODEL_MATRIX_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
