from __future__ import annotations

import asyncio
import time

from .agent import DoctorAgent
from .index import Index
from .errors import MCPConnectionError, error_message
from .knowledge import require_knowledge
from .llm import ChatModel
from .mcp_client import connect_tools
from .models import Budget
from .progress import ProgressReporter


async def diagnose(
    root,
    problem,
    backend="lexical",
    embedding_model=None,
    mode="agent",
    budget=None,
    on_trace=None,
    on_progress=None,
    *, exact_rerank=False,
):
    started = time.monotonic()
    budget = budget or Budget()
    progress = ProgressReporter(on_progress, started=started)
    progress.emit({"event": "preflight", "message": "正在检查项目与知识库状态…"})
    if not problem.strip():
        raise ValueError("请先输入错误描述或 Java 异常堆栈。")
    require_knowledge(root, backend, embedding_model)
    model = ChatModel()
    remaining = max(0, budget.max_seconds - (time.monotonic() - started))
    timeout = asyncio.timeout(remaining)
    try:
        # The timeout must surround the entire MCP context in this task: entering or
        # closing its anyio cancel scopes in a separate wait_for task is unsafe.
        async with progress, timeout:
            progress.emit({"event": "connecting", "message": "正在连接代码服务并加载检索模型…"})
            async with connect_tools(root, backend, embedding_model, exact_rerank=exact_rerank) as tools:
                progress.emit({"event": "connected", "message": "代码服务已就绪，开始调查问题。"})
                report = await DoctorAgent(Index(root), tools, model, budget, on_trace, progress.emit).run(
                    problem, mode, started_at=started
                )
                progress.emit({"event": "cleanup", "message": "正在整理报告并释放代码服务…"})
    except Exception as exc:
        # MCP's anyio readers can report closed pipes as an ExceptionGroup while
        # cancellation unwinds. Normalize those only when our deadline fired.
        if not timeout.expired():
            if isinstance(exc, ExceptionGroup):
                raise MCPConnectionError("代码服务连接失败：" + error_message(exc)) from exc
            raise
        raise TimeoutError(
            f"诊断总耗时超过 {budget.max_seconds:g} 秒预算（包含 MCP 连接和检索模型启动）。"
            "请先构建索引并预加载检索模型，或增大 --max-seconds 后重试。"
        ) from exc
    finally:
        await model.close()
    progress.emit({"event": "complete", "message": "本轮分析已结束，报告已生成。"})
    report["progress"] = progress.events
    report["metrics"]["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["exact_rerank"] = exact_rerank
    return report
