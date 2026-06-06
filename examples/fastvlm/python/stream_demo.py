#!/usr/bin/env python3
"""Streaming generation demo using the VLM Python bindings.

Demonstrates how to use vlm.VlmService.generate_stream() to receive
tokens incrementally and print them to stdout. Supports SMT/TCM
hardware acceleration.

Usage:
    python3 stream_demo.py --config examples/fastvlm/config/fastvlm.yaml \
        --image /path/to/image.jpg --prompt "Describe this image in detail"

Copyright (C) 2026 SpacemiT
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "python"))

from vlm import VlmService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streaming generation demo for VLM service"
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
        print("Streaming output:")

        t_start = time.monotonic()
        chunk_count = 0

        for chunk in service.generate_stream(
            args.prompt, image_path=args.image
        ):
            print(chunk, end="", flush=True)
            chunk_count += 1

        elapsed = (time.monotonic() - t_start) * 1000.0

        print(f"\n\n{'='*50}")
        print("  Streaming Statistics")
        print(f"{'='*50}")
        print(f"  Chunks received: {chunk_count}")
        print(f"  Wall time:       {elapsed:.2f} ms")

        metrics = service.get_metrics()
        print(f"  Last latency:    {metrics['last_latency_ms']:.2f} ms")
        print(f"  Last TTFT:       {metrics['last_ttft_ms']:.2f} ms")
        print(f"  Last tokens/sec: {metrics['last_tokens_per_second']:.2f}")


if __name__ == "__main__":
    main()
