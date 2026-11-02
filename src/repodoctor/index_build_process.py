"""Build indexes in a disposable process so model and native memory is released."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import traceback
from contextlib import redirect_stdout

from .errors import error_message
from .index_progress import emit_index_progress
from .runtime_env import worker_environment


class IndexBuildProcessError(RuntimeError):
    """The isolated index build failed or its response was incomplete."""


def _terminate_process_tree(process):
    # Windows venv launchers can own a second Python process running the actual worker.
    if os.name == "nt" and hasattr(process, "pid"):
        try:
            stopped = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=5,
                check=False,
            )
            if stopped.returncode == 0:
                return
        except (OSError, subprocess.TimeoutExpired):
            pass
    process.terminate()


def _stop_process(process):
    """Always reap the child, including when the progress consumer fails."""
    if process.poll() is None:
        _terminate_process_tree(process)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    for stream in (process.stdin, process.stdout):
        if stream is not None and not stream.closed:
            stream.close()


def _diagnostic_tail(diagnostics):
    diagnostics.seek(0, os.SEEK_END)
    diagnostics.seek(max(0, diagnostics.tell() - 8192))
    message = diagnostics.read().decode("utf-8", errors="replace").strip()[-2000:]
    return error_message(RuntimeError(message)) if message else ""


def _parse_response(line):
    try:
        response = json.loads(line)
        if not isinstance(response, dict):
            raise ValueError("invalid response")
        kind = response["type"]
        if kind == "progress":
            event = response["event"]
            if not isinstance(event, dict) or not all(
                isinstance(event.get(key), str) for key in ("stage", "message")
            ):
                raise ValueError("invalid progress")
        elif kind == "result":
            if not isinstance(response.get("metadata"), dict):
                raise ValueError("invalid result")
        elif kind == "error":
            if not isinstance(response.get("message"), str):
                raise ValueError("invalid error")
        else:
            raise ValueError("unknown response")
        return response
    except (ValueError, KeyError, TypeError) as exc:
        raise IndexBuildProcessError("索引构建子进程返回了无效响应，请检查依赖和可用内存。") from exc


def build_knowledge_isolated(
    root, backend="linearrag", embedding_model=None, include_tests=False, docs=None, progress=None,
    force=False,
):
    """Forward real progress while waiting for process exit before returning success."""
    request = {
        "root": os.fspath(root),
        "backend": backend,
        "embedding_model": str(embedding_model) if embedding_model is not None else None,
        "include_tests": include_tests,
        "docs": [os.fspath(path) for path in docs] if docs is not None else None,
        "force": force,
    }
    environment = worker_environment(backend)
    emit_index_progress(progress, "start_process", "正在启动独立索引构建进程…")
    terminal = None
    completion = None
    with tempfile.TemporaryFile(mode="w+b") as diagnostics:
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "repodoctor.index_build_process"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=diagnostics,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=environment,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise IndexBuildProcessError("无法启动独立索引构建进程：" + error_message(exc)) from exc
        try:
            process.stdin.write(json.dumps(request, ensure_ascii=False))
            process.stdin.close()
            for line in process.stdout:
                response = _parse_response(line)
                if terminal is not None:
                    raise IndexBuildProcessError("索引构建子进程返回了重复的完成响应。")
                kind = response["type"]
                if kind == "progress":
                    event = response["event"]
                    if event["stage"] == "complete":
                        completion = event
                    elif progress is not None:
                        progress(event)
                else:
                    terminal = response
                    if kind == "result":
                        emit_index_progress(progress, "release_process", "索引已生成，正在释放构建进程内存…")
            exit_code = process.wait()
            if terminal is not None and terminal["type"] == "error":
                raise IndexBuildProcessError("独立索引构建失败：" + terminal["message"])
            if exit_code != 0 or terminal is None:
                detail = _diagnostic_tail(diagnostics)
                message = f"独立索引构建进程异常退出（退出码 {exit_code}），请检查模型路径、依赖和可用内存。"
                if detail:
                    message += "\n" + detail
                raise IndexBuildProcessError(message)
            if completion is not None and progress is not None:
                progress(completion)
            return terminal["metadata"]
        finally:
            _stop_process(process)


def _worker_main(request_stream, protocol_stream):
    """Keep library logs off the JSON protocol and propagate a readable failure."""
    def send(payload):
        protocol_stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        protocol_stream.flush()

    try:
        request = json.load(request_stream)
        with redirect_stdout(sys.stderr):
            from .knowledge import build_knowledge

            metadata = build_knowledge(
                **request,
                progress=lambda event: send({"type": "progress", "event": event}),
            )
        send({"type": "result", "metadata": metadata})
        return 0
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        send({"type": "error", "message": error_message(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(_worker_main(sys.stdin, sys.stdout))
