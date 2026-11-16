"""The vLLM server of a local writer: a writer of align_distill/conf/generate_data.yaml with serve settings (the
abliterated Qwen3.8). It starts the first time a stage calls the writer and stops when the run ends, so a run whose
catalog exists loads no model. The client (msm/datagen/llm.py) talks to it through the same OpenAI-compatible API as
to OpenRouter: the chat template, with the writer's chat_template_kwargs, is applied by the server.

The server listens on 127.0.0.1 inside the job and writes its output to the writer's serve.log.
"""

from __future__ import annotations

import atexit
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig

from align_distill.tokens import resolve_path
from align_distill.write import resolve_model

POLL_S = 5
STOP_S = 60  # after SIGTERM, before SIGKILL
_servers: dict[str, subprocess.Popen] = {}  # base_url: the server this process started


def thinks(writer: DictConfig) -> bool:
    """Whether the writer's chat template is asked to think (Qwen3.8's template thinks unless told not to)."""
    kwargs = (writer.get("extra_body") or {}).get("chat_template_kwargs") or {}
    return kwargs.get("enable_thinking", True) is not False


def server_command(writer: DictConfig, tensor_parallel_size: int) -> list[str]:
    """The vllm serve command line of a local writer, on its cached snapshot, served under its configured name. With
    thinking on, the reasoning parser puts the thinking in its own field, so that the reply is the answer alone."""
    serve = writer.serve
    model_path, _ = resolve_model(writer.model)
    vllm = Path(sys.executable).with_name("vllm")
    parser = ["--reasoning-parser", serve.reasoning_parser] if serve.get("reasoning_parser") and thinks(writer) else []
    return [str(vllm) if vllm.exists() else shutil.which("vllm") or "vllm", "serve", model_path,
            "--served-model-name", writer.model, "--host", "127.0.0.1", "--port", str(serve.port),
            "--dtype", serve.dtype, "--tensor-parallel-size", str(tensor_parallel_size),
            "--max-model-len", str(serve.max_model_len), "--gpu-memory-utilization", str(serve.gpu_memory_utilization),
            "--max-num-seqs", str(serve.max_num_seqs), "--seed", str(serve.seed), *parser,
            *[str(arg) for arg in serve.get("args") or []]]


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=POLL_S) as response:
            return response.status == 200
    except OSError:
        return False


def port_taken(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def tail(path: Path, lines: int = 40) -> str:
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def ensure_server(cfg: DictConfig) -> None:
    """Start the chosen writer's server and wait until it answers, unless the writer is an API (no serve settings) or
    this run started its server already."""
    name, writer = cfg.writer, cfg.writers[cfg.writer]
    if not writer.get("serve") or writer.base_url in _servers:
        return
    serve = writer.serve
    if port_taken(serve.port):
        raise RuntimeError(f"Port {serve.port} is taken; set writers.{name}.serve.port to a free one")
    tensor_parallel_size = serve.tensor_parallel_size
    if not tensor_parallel_size:
        import torch

        tensor_parallel_size = torch.cuda.device_count()
        if not tensor_parallel_size:
            raise RuntimeError(f"The {name} writer runs in vLLM and needs GPUs; none are visible")
    command = server_command(writer, tensor_parallel_size)
    os.environ.setdefault(writer.key_env, "local")  # vLLM requires VLLM_API_KEY of every request when it is set
    log = resolve_path(serve.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as out:
        out.write(f"\n# {datetime.now().isoformat(timespec='seconds')}: {' '.join(command)}\n")
        out.flush()
        process = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    _servers[writer.base_url] = process
    atexit.register(stop_servers)
    logger.info(f"Starting the {name} writer in vLLM on {tensor_parallel_size} GPU(s), port {serve.port}; its log: "
                f"{log}")
    start, deadline = time.monotonic(), time.monotonic() + serve.startup_timeout_s
    while not healthy(serve.port):
        if process.poll() is not None:
            _servers.pop(writer.base_url)
            raise RuntimeError(f"The {name} vLLM server exited with code {process.returncode} before it was up. The end "
                               f"of {log}:\n{tail(log)}")
        if time.monotonic() > deadline:
            stop_servers()
            raise TimeoutError(f"The {name} vLLM server was not up after {serve.startup_timeout_s} s; see {log}")
        time.sleep(POLL_S)
    logger.info(f"The {name} writer is up after {time.monotonic() - start:.0f} s")


def stop_servers() -> None:
    """Stop every server this process started, with the workers of each (its process group)."""
    while _servers:
        _, process = _servers.popitem()
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            try:
                process.wait(timeout=STOP_S)
                break
            except subprocess.TimeoutExpired:
                continue
