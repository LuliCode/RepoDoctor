"""Bound model payloads while keeping full raw tool results in the report."""

from __future__ import annotations

import json
import math


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def estimate_tokens(value) -> int:
    """Approximation for mixed code/Chinese, not a model-specific tokenizer.

    Three ASCII bytes or two non-ASCII UTF-8 bytes per token, plus framing.
    API usage remains authoritative; compatible providers use different tokenizers.
    """
    text = value if isinstance(value, str) else dumps(value)
    ascii_count = sum(ord(char) < 128 for char in text)
    return math.ceil(ascii_count / 3 + (len(text.encode("utf-8")) - ascii_count) / 2) + 32


REDUNDANT = {"entities", "numbered_content", "content_hash", "snapshot", "project", "score", "id"}


def compact_result(value):
    if isinstance(value, dict):
        return {key: compact_result(item) for key, item in value.items() if key not in REDUNDANT}
    if isinstance(value, list):
        return [compact_result(item) for item in value]
    return value


def source_chunks(result):
    if "hits" in result:
        return result["hits"]
    if "new_ranges" in result:
        return result["new_ranges"]
    return [result] if "file_path" in result and "content" in result else []


def bound_result(result, limit=3000):
    """Trim on source line boundaries; disclose omissions for subsequent reads."""
    result = compact_result(result)
    if estimate_tokens(result) <= limit:
        return result
    chunks = source_chunks(result)
    # Preserve ranked first hits; smaller tails can be explicitly read later.
    for chunk in reversed(chunks):
        lines = chunk["content"].split("\n")
        original_end = chunk.get("available_end_line", chunk.get("end_line"))
        while lines and estimate_tokens(result) > limit:
            lines = lines[: max(0, len(lines) // 2)]
            chunk["content"] = "\n".join(lines)
            chunk["available_end_line"] = chunk["start_line"] + len(lines) - 1
            chunk["truncated"] = True
            chunk["omitted_end_line"] = original_end
    for key in ("callers", "unresolved_candidates"):
        if key in result:
            while result[key] and estimate_tokens(result) > limit:
                result[key] = result[key][: len(result[key]) // 2]
                result["truncated"] = True
    result["context_notice"] = (
        "上下文已压缩；截断的源码可按文件与行号用 read_code 补读，完整工具结果保留在报告中。"
    )
    return result


def fit_messages(messages, definitions, limit):
    """Keep call/result pairs intact and preserve the user's entire original log."""
    messages = [dict(message) for message in messages]
    changes = 0
    for index, message in enumerate(messages):
        if estimate_tokens([messages, definitions]) <= limit:
            break
        if message.get("role") != "tool":
            continue
        result = json.loads(message["content"])
        reduced = bound_result(result, 500)
        message["content"] = dumps(reduced)
        changes += 1
        # Older results have already been presented to the model. Their raw
        # evidence stays in the report; the latest result retains source excerpts.
        if estimate_tokens([messages, definitions]) > limit and any(
            later.get("role") == "tool" for later in messages[index + 1 :]
        ):
            message["content"] = dumps(
                {
                    "notice": "较早的工具结果已移出本轮上下文；如需核对请补读以下位置。",
                    "locations": [
                        {key: chunk[key] for key in ("file_path", "start_line", "end_line") if key in chunk}
                        for chunk in source_chunks(result)
                    ][:10],
                }
            )
    return messages, changes
