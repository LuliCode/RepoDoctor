"""Lightweight defaults for UI and CLI retrieval controls."""

import os


def default_exact_rerank() -> bool:
    value = os.getenv("REPODOCTOR_EXACT_RERANK", "false").strip().lower()
    if value in {"true", "1", "yes", "on"}:
        return True
    if value in {"false", "0", "no", "off"}:
        return False
    raise ValueError("REPODOCTOR_EXACT_RERANK 必须为 true 或 false。")
