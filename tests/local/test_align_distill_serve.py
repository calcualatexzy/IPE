"""align-distill's writers (align_distill/conf/generate_data.yaml): the API each one resolves to, and the vLLM server
of a local writer (align_distill/datagen/serve.py), its command and its lifecycle, with a stand-in HTTP server in
place of vLLM. No GPU, no network."""
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from align_distill_tiny import compose  # noqa: F401  (sets the offline environment)
from omegaconf import OmegaConf

from align_distill.datagen import serve
from msm.datagen.llm import LLM

LOCAL = "qwen38_abliterated"

# Answers 200 on /health once it has slept the given seconds, like vLLM once the model is loaded
FAKE_SERVER = """
import http.server, sys, time
time.sleep(float(sys.argv[2]))
class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()
    def log_message(self, *args):
        pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Health).serve_forever()
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def local_cfg(tmp: Path, port: int):
    cfg = compose("generate_data", [f"writer={LOCAL}", f"writers.{LOCAL}.serve.port={port}",
                                    f"writers.{LOCAL}.serve.tensor_parallel_size=1",
                                    f"writers.{LOCAL}.serve.startup_timeout_s=30",
                                    f"writers.{LOCAL}.serve.log={tmp / 'server.log'}"])
    return cfg


class WriterConfigTests(unittest.TestCase):
    def test_each_writer_resolves_to_its_api(self):
        deepseek = compose("generate_data", [])
        self.assertEqual(deepseek.writer, "deepseek")
        self.assertEqual(deepseek.api.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(deepseek.api.model, "deepseek/deepseek-v4.1-flash")
        self.assertEqual(OmegaConf.to_container(deepseek.api.extra_body), {"reasoning": {"enabled": False}})
        self.assertFalse(deepseek.writers.deepseek.get("serve"))

        local = compose("generate_data", [f"writer={LOCAL}", f"writers.{LOCAL}.serve.port=9123"])
        self.assertEqual(local.api.base_url, "http://127.0.0.1:9123/v1")
        self.assertEqual(local.api.model, "huihui-ai/Huihui-Qwen3.8-27B-abliterated")
        self.assertEqual(local.api.key_env, "VLLM_API_KEY")
        self.assertEqual((local.api.max_concurrency, deepseek.api.max_concurrency), (64, 8))
        with mock.patch.dict(os.environ, {"VLLM_API_KEY": "local"}):
            client = LLM(local.api)
        self.assertEqual(client.extra_body,
                         {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"}})
        self.assertEqual((local.max_tokens.catalog, local.api.timeout_s), (60000, 3600))
        self.assertEqual((deepseek.max_tokens.catalog, deepseek.api.timeout_s), (12000, 600))
        self.assertEqual(str(client.client.base_url), "http://127.0.0.1:9123/v1/")

    def test_server_command_serves_the_cached_snapshot_under_the_hub_id(self):
        cfg = compose("generate_data", [f"writer={LOCAL}"])
        command = serve.server_command(cfg.writers[LOCAL], 2)
        self.assertEqual(command[1], "serve")
        self.assertTrue(Path(command[2]).is_dir())  # the snapshot directory, found offline
        flags = dict(zip(command[3::2], command[4::2]))
        self.assertEqual(flags["--served-model-name"], "huihui-ai/Huihui-Qwen3.8-27B-abliterated")
        self.assertEqual(flags["--host"], "127.0.0.1")
        self.assertEqual(flags["--tensor-parallel-size"], "2")
        self.assertEqual(flags["--port"], "8011")
        self.assertIn("--language-model-only", command)
        self.assertEqual(flags["--reasoning-parser"], "qwen3")  # thinking is on

    def test_reasoning_parser_only_with_thinking(self):
        # Without thinking the parser would take the whole answer for thinking
        cfg = compose("generate_data", [f"writer={LOCAL}",
                                        f"writers.{LOCAL}.extra_body.chat_template_kwargs.enable_thinking=false"])
        self.assertNotIn("--reasoning-parser", serve.server_command(cfg.writers[LOCAL], 1))


class ServerLifecycleTests(unittest.TestCase):
    def tearDown(self):
        serve.stop_servers()

    def test_api_writer_starts_nothing(self):
        with mock.patch.object(serve.subprocess, "Popen") as popen:
            serve.ensure_server(compose("generate_data", []))
        popen.assert_not_called()

    def test_local_writer_is_started_once_and_stopped(self):
        port = free_port()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(serve, "server_command",
                                  return_value=[sys.executable, "-c", FAKE_SERVER, str(port), "1"]), \
                mock.patch.object(serve, "POLL_S", 0.2), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_API_KEY", None)
            cfg = local_cfg(Path(tmp), port)
            serve.ensure_server(cfg)
            self.assertTrue(serve.healthy(port))
            self.assertEqual(os.environ["VLLM_API_KEY"], "local")
            (process,) = serve._servers.values()
            serve.ensure_server(cfg)  # already up: no second server
            self.assertEqual(list(serve._servers.values()), [process])
            serve.stop_servers()
            self.assertIsNotNone(process.poll())
            self.assertFalse(serve.port_taken(port))
            self.assertIn(sys.executable, (Path(tmp) / "server.log").read_text())

    def test_server_that_exits_is_reported_with_its_log(self):
        port = free_port()
        crash = [sys.executable, "-c", "print('CUDA out of memory'); raise SystemExit(3)"]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(serve, "server_command", return_value=crash), \
                mock.patch.object(serve, "POLL_S", 0.2):
            with self.assertRaisesRegex(RuntimeError, r"(?s)exited with code 3.*CUDA out of memory"):
                serve.ensure_server(local_cfg(Path(tmp), port))
        self.assertEqual(serve._servers, {})

    def test_taken_port_is_refused(self):
        with socket.socket() as s, tempfile.TemporaryDirectory() as tmp:
            s.bind(("127.0.0.1", 0))
            s.listen()
            with self.assertRaisesRegex(RuntimeError, "is taken"):
                serve.ensure_server(local_cfg(Path(tmp), s.getsockname()[1]))


if __name__ == "__main__":
    unittest.main()
