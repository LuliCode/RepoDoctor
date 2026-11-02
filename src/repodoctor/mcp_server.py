from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from .index import Index
from .knowledge import BACKENDS, SEMANTIC_BACKENDS, require_knowledge
from .tools import CodeTools
from .retrieval_settings import default_exact_rerank


def create_server(root: str, backend="lexical", embedding_model=None, *, exact_rerank=False) -> FastMCP:
    require_knowledge(root, backend, embedding_model)
    server = FastMCP("RepoDoctor")
    code = CodeTools(Index(root), backend, embedding_model, exact_rerank=exact_rerank)
    # On Windows a first NumPy DLL import after AnyIO's stdio reader threads have
    # started can hang. Prepare the native runtime and model before opening stdio.
    if backend in SEMANTIC_BACKENDS:
        with contextlib.redirect_stdout(sys.stderr):
            code.searcher.prepare()
    annotation = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
    server.tool(annotations=annotation)(code.search_code)
    server.tool(annotations=annotation)(code.read_code)
    server.tool(annotations=annotation)(code.find_callers)
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--backend", choices=BACKENDS, default="lexical")
    parser.add_argument("--embedding-model")
    parser.add_argument("--exact-rerank", action=argparse.BooleanOptionalAction, default=default_exact_rerank(),
                        help="启用/关闭最终堆栈与符号精确重排；默认关闭，适用于所有后端")
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=os.getenv("REPODOCTOR_LOG_LEVEL", "WARNING").upper())
    if os.getenv("REPODOCTOR_DEBUG_STACKS") == "1":
        import faulthandler

        faulthandler.dump_traceback_later(30, file=sys.stderr)
    create_server(args.root, args.backend, args.embedding_model, exact_rerank=args.exact_rerank).run(transport="stdio")


if __name__ == "__main__":
    main()
