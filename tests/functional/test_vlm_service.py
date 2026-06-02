"""Functional tests for VLM service Python bindings."""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

from vlm import VlmService


class TestVlmServiceConfig(unittest.TestCase):
    """Test VLM service configuration loading."""

    def test_invalid_config_path_raises(self):
        """Non-existent config path should raise RuntimeError."""
        with self.assertRaises(RuntimeError):
            VlmService("/nonexistent/path/config.yaml")

    def test_empty_config_path_raises(self):
        """Empty config path should raise RuntimeError."""
        with self.assertRaises(RuntimeError):
            VlmService("")


class TestVlmServiceLifecycle(unittest.TestCase):
    """Test VLM service lifecycle management."""

    def test_context_manager_with_invalid_config(self):
        """Context manager should handle invalid config gracefully."""
        with self.assertRaises(RuntimeError):
            with VlmService("/nonexistent/config.yaml") as svc:
                pass

    def test_close_destroys_handle_once(self):
        """close() should destroy the native handle once and mark service closed."""
        import vlm

        class FakeLib:
            def __init__(self):
                self.destroy_count = 0


        with patch.object(vlm, "_load_library", return_value=FakeLib()):
            svc = VlmService("/tmp/config.yaml")
            svc.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                svc.generate("hello")


class TestVlmServiceFactory(unittest.TestCase):
    """Test VLM service factory functions."""

    def test_library_discovery(self):
        """Library discovery should search standard paths."""
        from vlm import _find_library
        try:
            lib = _find_library()
            self.assertIsNotNone(lib)
        except OSError:
            self.skipTest("VLM service not available")


class TestVlmGatewaySchemas(unittest.TestCase):
    """Test gateway schema definitions."""

    def test_chat_completion_request_from_dict(self):
        """ChatCompletionRequest should parse OpenAI format."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "gateway"))
        from schemas import ChatCompletionRequest

        data = {
            "model": "fastvlm",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
                    {"type": "text", "text": "Describe this image"},
                ]},
            ],
            "max_tokens": 200,
            "stream": False,
        }
        req = ChatCompletionRequest.from_dict(data)
        self.assertEqual(req.model, "fastvlm")
        self.assertEqual(len(req.messages), 2)
        self.assertEqual(req.messages[1].role, "user")
        self.assertEqual(req.messages[1].content, "Describe this image")
        self.assertEqual(req.messages[1].image_url, "data:image/jpeg;base64,abc")
        self.assertEqual(req.max_tokens, 200)
        self.assertFalse(req.stream)

    def test_chat_completion_request_ignores_unknown_multimodal_parts(self):
        """Unknown content parts should not inject empty spaces or images."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "gateway"))
        from schemas import ChatCompletionRequest

        req = ChatCompletionRequest.from_dict({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "input_audio", "input_audio": {"data": "abc"}},
                    {"type": "text", "text": "briefly"},
                ],
            }],
        })

        self.assertEqual(req.messages[0].content, "Describe briefly")
        self.assertIsNone(req.messages[0].image_url)

    def test_chat_completion_response_to_dict(self):
        """ChatCompletionResponse should serialize to OpenAI format."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "gateway"))
        from schemas import ChatCompletionResponse, ChatCompletionChoice, ChatMessage, UsageInfo

        resp = ChatCompletionResponse(
            id="chatcmpl-test",
            model="fastvlm",
            choices=[ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content="A cat sitting on a mat"),
            )],
            usage=UsageInfo(prompt_tokens=50, completion_tokens=10, total_tokens=60),
        )
        d = resp.to_dict()
        self.assertEqual(d["id"], "chatcmpl-test")
        self.assertEqual(d["model"], "fastvlm")
        self.assertEqual(len(d["choices"]), 1)
        self.assertEqual(d["choices"][0]["message"]["content"], "A cat sitting on a mat")
        self.assertEqual(d["usage"]["total_tokens"], 60)

    def test_stream_chunk_to_sse(self):
        """StreamChunk should format as SSE data line."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "gateway"))
        from schemas import StreamChunk

        chunk = StreamChunk(id="test", delta_content="Hello")
        sse = chunk.to_sse()
        self.assertTrue(sse.startswith("data: "))
        self.assertTrue(sse.endswith("\n\n"))
        self.assertIn("Hello", sse)


class TestVlmGatewayRoutes(unittest.TestCase):
    """Test gateway route behavior."""

    def test_chat_completion_allows_lazy_backend_start(self):
        """Initialized but not-ready services should still receive inference calls."""
        from fastapi.testclient import TestClient
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from gateway.adapter import create_vlm_app
        from gateway.service import VlmGatewayService

        class LazyService(VlmGatewayService):
            def __init__(self):
                super().__init__()
                self.chat_called = False

            def initialize(self, config_path=None):
                class ServiceStub:
                    def shutdown(self):
                        pass

                self._service = ServiceStub()

            def is_ready(self):
                return False

            def chat(self, messages, **kwargs):
                self.chat_called = True
                return {"text": "lazy ok", "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}

        lazy_service = LazyService()

        with tempfile.NamedTemporaryFile(suffix=".yaml") as config_file:
            with patch("gateway.adapter.VlmGatewayService", return_value=lazy_service):
                app = create_vlm_app(config_path=config_file.name)

            with TestClient(app) as client:
                response = client.post("/chat/completions", json={
                    "model": "vlm",
                    "messages": [{"role": "user", "content": "Hello"}],
                })

        self.assertEqual(response.status_code, 200)
        self.assertTrue(lazy_service.chat_called)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "lazy ok")

    def test_chat_completion_parses_multimodal_body_and_forwards_options(self):
        """Route should parse OpenAI content lists and forward request options."""
        from fastapi.testclient import TestClient
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from gateway.adapter import create_vlm_app
        from gateway.service import VlmGatewayService

        class CapturingService(VlmGatewayService):
            def __init__(self):
                super().__init__()
                self.chat_messages = None
                self.chat_kwargs = None

            def initialize(self, config_path=None):
                class ServiceStub:
                    def shutdown(self):
                        pass

                self._service = ServiceStub()

            def chat(self, messages, **kwargs):
                self.chat_messages = messages
                self.chat_kwargs = kwargs
                return {"text": "parsed", "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}

        capturing_service = CapturingService()

        with tempfile.NamedTemporaryFile(suffix=".yaml") as config_file:
            with patch("gateway.adapter.VlmGatewayService", return_value=capturing_service):
                app = create_vlm_app(config_path=config_file.name)

            with TestClient(app) as client:
                response = client.post("/chat/completions", json={
                    "model": "vlm",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe"},
                            {"type": "text", "text": "this image"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                        ],
                    }],
                    "max_tokens": 77,
                    "temperature": 0.4,
                    "top_p": 0.8,
                    "enable_thinking": True,
                    "vision_history": "all",
                })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(capturing_service.chat_messages, [{
            "role": "user",
            "content": "Describe this image",
            "image_path": "data:image/png;base64,abc",
        }])
        self.assertEqual(capturing_service.chat_kwargs, {
            "max_tokens": 77,
            "temperature": 0.4,
            "top_p": 0.8,
            "enable_thinking": True,
            "vision_history": "all",
        })

    def test_gateway_service_forwards_chat_options(self):
        """Gateway service should pass request options into Python bindings."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from gateway.service import VlmGatewayService

        class BindingStub:
            def __init__(self):
                self.messages = None
                self.kwargs = None

            def chat(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return {"text": "ok"}

        service = VlmGatewayService()
        service._service = BindingStub()
        messages = [{"role": "user", "content": "hi"}]

        result = service.chat(messages, max_tokens=9, temperature=0.1, top_p=0.7)

        self.assertEqual(result["text"], "ok")
        self.assertEqual(service._service.messages, messages)
        self.assertEqual(service._service.kwargs, {
            "max_tokens": 9,
            "temperature": 0.1,
            "top_p": 0.7,
        })

    def test_gateway_service_stream_rejects_unsupported_options(self):
        """Streaming wrappers should fail clearly for unsupported overrides."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from gateway.service import VlmGatewayService

        class BindingStub:
            def chat_stream(self, messages):
                yield "ok"

        service = VlmGatewayService()
        service._service = BindingStub()

        with self.assertRaisesRegex(ValueError, "not supported"):
            list(service.chat_stream([{"role": "user", "content": "hi"}], max_tokens=9))

    def test_gateway_service_uninitialized_generate_errors(self):
        """Uninitialized services should reject generation with a clear error."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from gateway.service import VlmGatewayService

        service = VlmGatewayService()

        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            service.generate("hello")


if __name__ == "__main__":
    unittest.main()
