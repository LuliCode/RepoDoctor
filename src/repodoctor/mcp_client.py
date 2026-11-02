from __future__ import annotations

import json
import sys
import tempfile
from collections import deque
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .errors import (
    MCPConnectionError,
    error_message,
    is_memory_failure,
    is_windows_commit_failure,
    windows_commit_guidance,
)
from .knowledge import model_key, require_knowledge
from .runtime_env import worker_environment


def _server_diagnostic(stderr):
    """Keep allocation failures and their loading stage even after noisy cleanup."""
    stderr.seek(0)
    tail, allocations, stages = deque(maxlen=6), deque(maxlen=3), deque(maxlen=2)
    commit_failures = deque(maxlen=2)
    memory_failure = False
    continuation = False
    # Stream the log: loading failures can be followed by many model/cleanup lines.
    # Bound every read and the retained excerpts instead of reading a full log.
    while chunk := stderr.readline(4096):
        line = chunk.strip()
        first_chunk = not continuation
        continuation = not chunk.endswith("\n")
        if not line:
            continue
        is_commit = is_windows_commit_failure(line)
        is_memory = is_memory_failure(line)
        memory_failure = memory_failure or is_memory
        # Continuations of an unusually long line can start with half a secret;
        # show only its first bounded, redacted excerpt.
        if not first_chunk:
            continue
        excerpt = error_message(RuntimeError(line))[:400]
        tail.append(excerpt)
        if is_memory:
            memory_failure = True
            if "LinearRAG" in line:
                stages.append(excerpt)
            elif is_commit:
                commit_failures.append(excerpt)
            else:
                allocations.append(excerpt)
    lines = list(dict.fromkeys([*stages, *commit_failures, *allocations, *tail]))
    return error_message(RuntimeError("\n".join(lines))) if lines else "", memory_failure


class MCPTools:
    def __init__(self, session, definitions):
        self.session, self.definitions = session, definitions

    async def call(self, name, arguments):
        result = await self.session.call_tool(name, arguments)
        if result.isError:
            return {"error": "\n".join(c.text for c in result.content if c.type == "text")}
        if result.structuredContent is not None:
            return result.structuredContent
        content = "\n".join(c.text for c in result.content if c.type == "text")
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"content": content}


@asynccontextmanager
async def connect_tools(root, backend="lexical", embedding_model=None, *, exact_rerank=False):
    require_knowledge(root, backend, embedding_model)
    args = ["-m", "repodoctor.mcp_server", "--root", str(Path(root).resolve()), "--backend", backend]
    args.append("--exact-rerank" if exact_rerank else "--no-exact-rerank")
    if embedding_model:
        args += ["--embedding-model", model_key(embedding_model)]
    env = worker_environment(backend)
    params = StdioServerParameters(command=sys.executable, args=args, env=env)
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8", errors="replace") as stderr:
        try:
            async with stdio_client(params, errlog=stderr) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=180)) as session:
                    await session.initialize()
                    discovered = await session.list_tools()
                    definitions = [
                        {
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "description": t.description or "",
                                "parameters": t.inputSchema,
                            },
                        }
                        for t in discovered.tools
                    ]
                    yield MCPTools(session, definitions)
        except Exception as exc:
            diagnostic, memory_failure = _server_diagnostic(stderr)
            commit_failure = is_windows_commit_failure(exc) or is_windows_commit_failure(diagnostic)
            memory_failure = memory_failure or is_memory_failure(exc)
            message = "代码服务连接失败：" + error_message(exc)
            if diagnostic:
                message += "\n服务端原因：" + diagnostic
            if commit_failure:
                message += "\n" + windows_commit_guidance()
            elif memory_failure:
                message += (
                    "\n代码服务内存不足。请关闭占用内存较大的程序后重试；"
                    "需要时可手动切换到离线关键词检索。"
                )
            else:
                message += "\n请检查所选项目的索引状态、模型路径与依赖；准备完成后重新定位。"
            raise MCPConnectionError(message) from exc
