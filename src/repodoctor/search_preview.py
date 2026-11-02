"""Preview code without loading semantic models into the long-lived UI process."""

from __future__ import annotations

from .index import Index
from .knowledge import SEMANTIC_BACKENDS
from .mcp_client import connect_tools
from .tools import CodeTools


async def search_preview(root, query, backend="lexical", embedding_model=None, top_k=5, *, exact_rerank=False):
    if backend == "lexical":
        return CodeTools(Index(root), backend, embedding_model, exact_rerank=exact_rerank).search_code(query, top_k)
    if backend not in SEMANTIC_BACKENDS:
        raise ValueError("backend 必须为 lexical、linearrag 或 vector")
    if not query.strip() or len(query) > 20000 or not 1 <= top_k <= 10:
        raise ValueError("query 不能为空且最多 20000 字符；top_k 为 1–10")
    async with connect_tools(root, backend, embedding_model, exact_rerank=exact_rerank) as tools:
        result = await tools.call("search_code", {"query": query, "top_k": top_k})
    # Return only after the MCP context has shut down its model-owning process.
    if "error" in result:
        raise RuntimeError("检索失败：" + str(result["error"]))
    return result
