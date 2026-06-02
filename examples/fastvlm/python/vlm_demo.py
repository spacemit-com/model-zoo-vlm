#!/usr/bin/env python3
"""Python demo for the VLM service with TCM hardware acceleration.

Demonstrates synchronous text/image generation using VlmService,
with automatic llama-server startup and SMT/TCM backend support.

Usage:
    python3 vlm_demo.py --config examples/fastvlm/config/fastvlm.yaml \
        --image /path/to/image.jpg --prompt "Describe this image"

Copyright (C) 2026 SpacemiT
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "python"))

from vlm import VlmService


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VLM inference demo (TCM accelerated via SMT backend)"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent.parent / "config" / "fastvlm.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--image",
        default="",
        help="Path to input image (optional for text-only)",
    )
    parser.add_argument(
        "--prompt", default="Describe this image", help="Text prompt"
    )
    args = parser.parse_args()

    try:
        with VlmService(args.config) as service:
            if args.image:
                result = service.generate(args.prompt, image_path=args.image)
            else:
                result = service.generate(args.prompt)

            print(f"\n{'='*50}")
            print(f"  VLM Generation Result")
            print(f"{'='*50}")
            print(f"  Text:     {result['text']}")
            if result.get("reasoning_text"):
                print(f"  Reasoning: {result['reasoning_text']}")
            print(f"  Latency:  {result['latency_ms']:.1f} ms")
            print(f"  TTFT:     {result['ttft_ms']:.1f} ms")
            print(f"  Speed:    {result['tokens_per_second']:.2f} tok/s")
            print(f"  Tokens:   {result['prompt_tokens']} prompt + "
                  f"{result['completion_tokens']} completion")

            metrics = service.get_metrics()
            if metrics.get("total_requests", 0) > 0:
                print(f"\n  Service Metrics:")
                print(f"  Total requests: {metrics['total_requests']}")
                print(f"  Last speed:    {metrics['last_tokens_per_second']:.2f} tok/s")

    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
