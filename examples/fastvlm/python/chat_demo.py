#!/usr/bin/env python3
"""Multi-turn chat demo using the VLM Python bindings.

Demonstrates how to use vlm.VlmService.chat() with system and user
messages, including image input. Supports SMT/TCM hardware acceleration.

Usage:
    python3 chat_demo.py --config examples/fastvlm/config/qwen3_5_2b.yaml \
        --image /path/to/image.jpg --prompt "What is in this image?"

Copyright (C) 2026 SpacemiT
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "python"))

from vlm import VlmService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-turn chat demo for VLM service"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent.parent / "config" / "fastvlm.yaml"),
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--image", default=None, help="Path to an image file (optional)"
    )
    parser.add_argument(
        "--prompt",
        default="Describe this image.",
        help="User prompt text",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with VlmService(args.config) as service:
        messages = [
            {"role": "system", "content": "You are a helpful vision-language assistant."},
            {"role": "user", "content": args.prompt},
        ]

        if args.image:
            messages[-1]["image_path"] = args.image

        result = service.chat(messages)

        print(f"\n{'='*50}")
        print("  Chat Result")
        print(f"{'='*50}")
        print(f"  Content:  {result['text']}")

        if result.get("reasoning_text"):
            print(f"  Reasoning: {result['reasoning_text']}")

        print("\n  --- Usage ---")
        print(f"  Prompt tokens:     {result['prompt_tokens']}")
        print(f"  Completion tokens: {result['completion_tokens']}")
        print(f"  Total tokens:      {result['total_tokens']}")

        print("\n  --- Performance ---")
        print(f"  Latency:    {result['latency_ms']:.2f} ms")
        print(f"  TTFT:       {result['ttft_ms']:.2f} ms")
        print(f"  Tokens/sec: {result['tokens_per_second']:.2f}")


if __name__ == "__main__":
    main()
