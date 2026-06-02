"""VLM Python bindings — HTTP-based interface to llama-server for VLM inference.

Provides a pure-Python VlmService that starts a llama-server process
and communicates via OpenAI-compatible HTTP API, without requiring
HTTP-based interface to llama-server for VLM inference.

Copyright (C) 2026 SpacemiT
SPDX-License-Identifier: Apache-2.0
"""

import base64
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import yaml

logger = logging.getLogger(__name__)


def _expand_user(path: str) -> str:
    """Expand ~ in path to home directory."""
    if path == "~" or path.startswith("~/"):
        return os.path.expanduser(path)
    return path


class VlmService:
    """High-level Python wrapper around a VLM llama-server HTTP endpoint.

    Starts a llama-server process and communicates via OpenAI-compatible
    HTTP API. Supports synchronous and streaming generation, multi-turn
    chat, and image input.

    Example::

        with VlmService("/path/to/config.yaml") as svc:
            result = svc.generate("Describe this image", image_path="cat.jpg")
            print(result["text"])
    """

    def __init__(self, config_path: str) -> None:
        """Initialize a VLM service from a configuration file.

        Args:
            config_path: Path to the model configuration YAML file.

        Raises:
            RuntimeError: If the service fails to initialize.
        """
        self._config = self._load_config(config_path)
        self._process: Optional[subprocess.Popen] = None
        self._base_url = f"http://{self._config['host']}:{self._config['port']}"
        self._started_server = False
        self._request_count = 0
        self._last_latency_ms = 0.0

        if not self._server_responds():
            self._start_server()
            self._wait_for_server()

    def _load_config(self, config_path: str) -> dict:
        """Load and validate VLM configuration from YAML file."""
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        model_dir = _expand_user(config.get("model_dir", ""))
        config["model_dir"] = model_dir
        config.setdefault("host", "127.0.0.1")
        config.setdefault("port", 8093)
        config.setdefault("media_backend", "smt")
        config.setdefault("cpu_affinity", "")
        config.setdefault("n_gpu_layers", 0)
        config.setdefault("no_warmup", True)

        gen = config.setdefault("generation", {})
        gen.setdefault("max_tokens", 150)
        gen.setdefault("context_size", 4096)
        gen.setdefault("top_k", 40)
        gen.setdefault("top_p", 0.95)
        gen.setdefault("temperature", 0.2)
        gen.setdefault("repeat_penalty", 1.2)
        gen.setdefault("threads", 8)
        gen.setdefault("batch_threads", 8)
        gen.setdefault("enable_thinking", False)
        gen.setdefault("reasoning_budget", -1)
        gen.setdefault("history_policy", "last")

        if model_dir:
            manifest_path = os.path.join(model_dir, "config.json")
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                    if "text_model_path" in manifest and not config.get("text_model_path"):
                        config["text_model_path"] = os.path.join(model_dir, manifest["text_model_path"])
                    if "vision_model_paths" in manifest and not config.get("vision_model_paths"):
                        config["vision_model_paths"] = manifest["vision_model_paths"]
                except Exception:
                    pass

        if not config.get("text_model_path") and model_dir:
            config["text_model_path"] = self._find_gguf(model_dir)

        return config

    def _find_gguf(self, model_dir: str) -> Optional[str]:
        """Find the text model GGUF file in the model directory."""
        for root, _dirs, files in os.walk(model_dir):
            for f in files:
                if f.endswith(".gguf"):
                    return os.path.join(root, f)
        return None

    def _build_command(self) -> list[str]:
        """Build the llama-server command from configuration."""
        config = self._config
        cmd: list[str] = []

        if config.get("cpu_affinity"):
            cmd.extend(["taskset", "-c", config["cpu_affinity"]])

        cmd.extend([
            "llama-server",
            "-m", config["text_model_path"],
            "--port", str(config["port"]),
            "--host", config["host"],
            "--ctx-size", str(config["generation"]["context_size"]),
            "-t", str(config["generation"]["threads"]),
            "-tb", str(config["generation"]["batch_threads"]),
            "-ngl", str(config.get("n_gpu_layers", 0)),
        ])

        media_backend = config.get("media_backend", "smt")
        cmd.extend(["--vision-backend", media_backend])

        if media_backend in ("smt", "auto"):
            smt_dir = config.get("smt_config_dir", config.get("model_dir", ""))
            if smt_dir:
                cmd.extend(["--smt-config-dir", smt_dir])

        vision_paths = config.get("vision_model_paths", [])
        if isinstance(vision_paths, str):
            vision_paths = [vision_paths]
        for vp in vision_paths:
            if vp:
                full_path = vp if os.path.isabs(vp) else os.path.join(config.get("model_dir", ""), vp)
                cmd.extend(["--mmproj", full_path])

        media_path = config.get("media_path", "")
        if media_path:
            cmd.extend(["--media-path", media_path])

        if config.get("no_warmup", True):
            cmd.append("--no-warmup")

        if not config["generation"].get("enable_thinking", False):
            cmd.extend(["--reasoning", "off"])
        else:
            cmd.extend(["--reasoning", "on"])
            budget = config["generation"].get("reasoning_budget", -1)
            if budget and budget > 0:
                cmd.extend(["--reasoning-budget", str(budget)])

        return cmd

    def _start_server(self) -> None:
        """Start the VLM llama-server process."""
        cmd = self._build_command()
        logger.info("Starting VLM llama-server: %s", " ".join(cmd))

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self._config["generation"]["threads"])

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        self._started_server = True

    def _wait_for_server(self, timeout: float = 120.0) -> None:
        """Wait for the llama-server to become ready."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process and self._process.poll() is not None:
                raise RuntimeError("VLM llama-server exited unexpectedly")
            if self._server_responds():
                return
            time.sleep(1.0)
        raise RuntimeError("VLM llama-server failed to start within 120 seconds")

    def _server_responds(self) -> bool:
        """Check if the llama-server is responding to health checks."""
        try:
            import urllib.request
            url = f"{self._base_url}/health"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                return data.get("status") == "ok"
        except Exception:
            return False

    def close(self) -> None:
        """Stop the llama-server process if it was started by this instance."""
        if self._process and self._process.poll() is None:
            logger.info("Stopping VLM llama-server (pid=%d)", self._process.pid)
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._process = None

    def _image_to_base64(self, image_path: str) -> str:
        """Read an image file and return its Base64 encoding."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _build_messages_payload(
        self,
        messages: List[Dict[str, Any]],
        image_base64: Optional[str] = None,
        stream: bool = False,
    ) -> dict:
        """Build the OpenAI-compatible chat completions request payload."""
        config = self._config
        openai_messages = []

        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if image_base64 and role == "user" and i == len(messages) - 1:
                openai_messages.append({
                    "role": role,
                    "content": [
                        {"type": "text", "text": content},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    ],
                })
            else:
                openai_messages.append({"role": role, "content": content})

        payload = {
            "messages": openai_messages,
            "max_tokens": config["generation"]["max_tokens"],
            "temperature": config["generation"]["temperature"],
            "top_p": config["generation"]["top_p"],
            "stream": stream,
        }

        if config["generation"].get("enable_thinking"):
            payload["enable_thinking"] = True
            budget = config["generation"].get("reasoning_budget", -1)
            if budget and budget > 0:
                payload["reasoning_budget"] = budget

        return payload

    def _post_sync(self, payload: dict) -> dict:
        """Send a synchronous HTTP POST request to llama-server."""
        import urllib.request
        url = f"{self._base_url}/v1/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read())

    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        **options: Any,
    ) -> Dict[str, Any]:
        """Run synchronous text/image generation.

        Args:
            prompt: Text prompt.
            image_path: Optional path to an image file.
            image_bytes: Optional raw image bytes.

        Returns:
            Dictionary with keys: text, reasoning_text, prompt_tokens,
            completion_tokens, total_tokens, latency_ms, ttft_ms,
            tokens_per_second.
        """
        image_b64 = None
        if image_path:
            image_b64 = self._image_to_base64(image_path)
        elif image_bytes:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        messages = [{"role": "user", "content": prompt}]
        payload = self._build_messages_payload(messages, image_b64, stream=False)

        start = time.monotonic()
        response = self._post_sync(payload)
        latency_ms = (time.monotonic() - start) * 1000.0

        self._request_count += 1
        self._last_latency_ms = latency_ms

        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = response.get("usage", {})

        content = message.get("content", "")
        reasoning = message.get("reasoning_content", "")
        completion_tokens = usage.get("completion_tokens", 0)
        tokens_per_second = completion_tokens / (latency_ms / 1000.0) if latency_ms > 0 and completion_tokens > 0 else 0.0

        return {
            "text": content,
            "reasoning_text": reasoning,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": completion_tokens,
            "total_tokens": usage.get("total_tokens", 0),
            "latency_ms": latency_ms,
            "ttft_ms": 0.0,
            "tokens_per_second": tokens_per_second,
        }

    def generate_stream(
        self,
        prompt: str,
        image_path: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Run streaming text/image generation.

        Args:
            prompt: Text prompt.
            image_path: Optional path to an image file.

        Yields:
            Text chunks (strings).
        """
        image_b64 = None
        if image_path:
            image_b64 = self._image_to_base64(image_path)

        messages = [{"role": "user", "content": prompt}]
        payload = self._build_messages_payload(messages, image_b64, stream=True)

        url = f"{self._base_url}/v1/chat/completions"
        yield from self._stream_request(url, payload)

    def chat(self, messages: List[Dict[str, Any]], **options: Any) -> Dict[str, Any]:
        """Run synchronous multi-turn chat.

        Args:
            messages: List of message dicts with keys role, content,
                and optionally image_path or image_bytes.

        Returns:
            Dictionary with the same keys as generate().
        """
        image_b64 = None
        for msg in messages:
            if msg.get("image_path") and msg.get("role") == "user":
                image_b64 = self._image_to_base64(msg["image_path"])
                break
            if msg.get("image_bytes") and msg.get("role") == "user":
                image_b64 = base64.b64encode(msg["image_bytes"]).decode("utf-8")
                break

        payload = self._build_messages_payload(messages, image_b64, stream=False)

        start = time.monotonic()
        response = self._post_sync(payload)
        latency_ms = (time.monotonic() - start) * 1000.0

        self._request_count += 1
        self._last_latency_ms = latency_ms

        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = response.get("usage", {})

        content = message.get("content", "")
        reasoning = message.get("reasoning_content", "")
        completion_tokens = usage.get("completion_tokens", 0)
        tokens_per_second = completion_tokens / (latency_ms / 1000.0) if latency_ms > 0 and completion_tokens > 0 else 0.0

        return {
            "text": content,
            "reasoning_text": reasoning,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": completion_tokens,
            "total_tokens": usage.get("total_tokens", 0),
            "latency_ms": latency_ms,
            "ttft_ms": 0.0,
            "tokens_per_second": tokens_per_second,
        }

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
    ) -> Generator[str, None, None]:
        """Run streaming multi-turn chat.

        Args:
            messages: List of message dicts (see chat()).

        Yields:
            Text chunks (strings).
        """
        image_b64 = None
        for msg in messages:
            if msg.get("image_path") and msg.get("role") == "user":
                image_b64 = self._image_to_base64(msg["image_path"])
                break
            if msg.get("image_bytes") and msg.get("role") == "user":
                image_b64 = base64.b64encode(msg["image_bytes"]).decode("utf-8")
                break

        payload = self._build_messages_payload(messages, image_b64, stream=True)

        url = f"{self._base_url}/v1/chat/completions"
        yield from self._stream_request(url, payload)

    def _stream_request(self, url: str, payload: dict) -> Generator[str, None, None]:
        """Send a streaming HTTP POST request and yield text chunks."""
        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            buffer = ""
            for chunk in iter(lambda: resp.read(4096), b""):
                buffer += chunk.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        return
                    try:
                        parsed = json.loads(data_str)
                        delta = parsed.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    def is_ready(self) -> bool:
        """Check whether the VLM service is ready for inference."""
        return self._server_responds()

    def reset(self) -> None:
        """Reset the VLM service by restarting the server."""
        self.close()
        self._start_server()
        self._wait_for_server()

    def shutdown(self) -> None:
        """Shut down the VLM service gracefully."""
        self.close()

    def get_metrics(self) -> Dict[str, Any]:
        """Retrieve operational metrics from the VLM service."""
        return {
            "total_requests": self._request_count,
            "is_processing": False,
            "last_latency_ms": self._last_latency_ms,
            "avg_latency_ms": self._last_latency_ms,
            "last_ttft_ms": 0.0,
            "last_output_tokens": -1,
            "last_tokens_per_second": 0.0,
        }

    def __enter__(self) -> "VlmService":
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager and release the service."""
        self.close()

    def __del__(self) -> None:
        """Clean up the VLM service handle on garbage collection."""
        try:
            self.close()
        except Exception:
            pass
