from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .index import Index
from .index_progress import run_index_build
from .knowledge import BACKENDS, build_knowledge
from .models import Budget
from .retrieval_settings import default_exact_rerank


def output(value, destination=None):
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if destination:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


def index_progress_printer():
    last = {"stage": None, "elapsed": -1, "completed": None}

    def show(event):
        elapsed = event["elapsed_seconds"]
        completed, total = event.get("completed"), event.get("total")
        finished = total is not None and completed == total and completed != last["completed"]
        if event["stage"] == last["stage"] and elapsed - last["elapsed"] < 1 and not finished:
            return
        text = event["message"]
        if total:
            text += f" · {completed}/{total} {event.get('unit', '')}"
        if event.get("current_file"):
            text += " · " + event["current_file"]
        print(f"[索引进度] {text} · 已用时 {elapsed:.1f} 秒", file=sys.stderr, flush=True)
        last.update(stage=event["stage"], elapsed=elapsed, completed=completed)

    return show


def main():
    load_dotenv()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="RepoDoctor：Python 实现的 Java 代码故障定位 Agent")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("index", "search", "read", "callers", "diagnose"):
        command = sub.add_parser(name)
        command.add_argument("root", help="Java 应用源码目录，可包含多个模块，无需根目录 pom.xml")
        command.add_argument(
            "--backend", choices=BACKENDS, default=os.getenv("REPODOCTOR_BACKEND", "lexical")
        )
        command.add_argument("--embedding-model", default=os.getenv("REPODOCTOR_EMBEDDING_MODEL"))
        command.add_argument("--exact-rerank", action=argparse.BooleanOptionalAction, default=default_exact_rerank(),
                             help="启用/关闭最终堆栈与符号精确重排；默认关闭，适用于所有后端")
        command.add_argument("--output")
        if name == "index":
            command.add_argument("--include-tests", action="store_true")
            command.add_argument("--force", action="store_true", help="忽略已有缓存，重新构建源码和所选后端索引")
            command.add_argument("--docs", nargs="*", help="可选文档相对路径；默认 README.md 和 docs/")
        if name in {"search", "diagnose"}:
            command.add_argument("query")
        if name == "search":
            command.add_argument("--top-k", type=int, default=5)
        if name == "read":
            command.add_argument("file_path")
            command.add_argument("start_line", type=int)
            command.add_argument("end_line", type=int)
        if name == "callers":
            command.add_argument("symbol")
        if name == "diagnose":
            command.add_argument("--mode", choices=["agent", "rag"], default="agent")
            command.add_argument(
                "--max-calls", type=int, default=int(os.getenv("REPODOCTOR_MAX_TOOL_CALLS", "8"))
            )
            command.add_argument(
                "--max-seconds", type=float, default=float(os.getenv("REPODOCTOR_MAX_SECONDS", "120"))
            )
            command.add_argument(
                "--max-tokens", type=int, default=int(os.getenv("REPODOCTOR_MAX_TOKENS", "24000"))
            )
            command.add_argument(
                "--max-context-tokens",
                type=int,
                default=int(os.getenv("REPODOCTOR_MAX_CONTEXT_TOKENS", "12000")),
                help="单轮上下文估算上限；累计用量仍受 --max-tokens 限制",
            )
    args = parser.parse_args()
    index = Index(args.root)
    try:
        if args.command == "index":
            result = run_index_build(
                lambda callback: build_knowledge(
                    args.root, args.backend, args.embedding_model, args.include_tests, args.docs,
                    progress=callback, force=args.force,
                ),
                index_progress_printer(),
            )
            result["retrieval_backend_built"] = args.backend
        elif args.command in {"search", "read", "callers"}:
            from .tools import CodeTools

            tools = CodeTools(index, args.backend, args.embedding_model, exact_rerank=args.exact_rerank)
            if args.command == "search":
                result = tools.search_code(args.query, args.top_k)
            elif args.command == "read":
                result = tools.read_code(args.file_path, args.start_line, args.end_line)
            else:
                result = tools.find_callers(args.symbol)
        else:
            budget = Budget(
                args.max_calls, args.max_seconds, args.max_tokens, max_context_tokens=args.max_context_tokens
            )
            from .service import diagnose

            result = asyncio.run(
                diagnose(args.root, args.query, args.backend, args.embedding_model, args.mode, budget,
                         exact_rerank=args.exact_rerank)
            )
        output(result, args.output)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"RepoDoctor: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
