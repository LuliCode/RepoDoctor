"""Readable errors at process and UI boundaries, including nested task groups."""

from __future__ import annotations

import errno
import os
import re


_WINDOWS_COMMIT_FAILURE = re.compile(
    r"\b(?:winerror|os\s+error|errno|windows\s+error)\s*[:=]?\s*1455\b|"
    r"(?:Windows\s*)?错误(?:码)?\s*[:：=]?\s*1455\b|"
    r"(?:页面|分页)文件太小|虚拟内存(?:不足|太小)|"
    r"系统提交(?:内存|限制|额度).{0,24}(?:不足|耗尽|超限|用尽)|"
    r"\b(?:paging\s+file|page\s*file)\s+(?:is\s+)?too\s+small\b|"
    r"\b(?:not\s+enough|insufficient)\s+(?:paging\s+file|page\s*file)\b|"
    r"\b(?:system\s+)?commit(?:ment)?\s+(?:limit|memory|charge)\s+"
    r"(?:is\s+|has\s+been\s+)?(?:exceeded|exhausted|reached|insufficient|too\s+low)\b",
    re.IGNORECASE,
)
_MEMORY_FAILURE = re.compile(
    r"\b\w*MemoryError(?::|$)|std::bad_alloc|out of memory|cannot allocate memory|"
    r"unable to allocate|realloc of size .+ failed|memory\s+allocation\b.{0,80}\bfailed\b|内存不足",
    re.IGNORECASE,
)


def _error_nodes(error: BaseException | str):
    """Visit wrapped failures once, including task groups and chained causes."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        if isinstance(current, BaseException):
            if current.__cause__ is not None:
                pending.append(current.__cause__)
            elif (
                not current.__suppress_context__
                and current.__context__ is not None
                and _os_error_allows_context(current)
            ):
                pending.append(current.__context__)


def _os_error_allows_context(error: BaseException) -> bool:
    """A concrete I/O failure can follow an earlier, already handled allocation."""
    if isinstance(error, (PermissionError, FileNotFoundError)):
        return False
    if not isinstance(error, OSError):
        return True
    winerror = getattr(error, "winerror", None)
    if winerror is not None:
        return winerror in {8, 14, 1455}
    return error.errno is None or error.errno in {errno.ENOMEM, 1455}


def is_windows_commit_failure(error: BaseException | str) -> bool:
    """Recognize Windows commitment/pagefile exhaustion, rather than file damage."""
    return any(
        (
            isinstance(item, OSError)
            and (getattr(item, "winerror", None) == 1455 or item.errno == 1455)
        )
        or bool(_WINDOWS_COMMIT_FAILURE.search(str(item)))
        for item in _error_nodes(error)
    )


def is_memory_failure(error: BaseException | str) -> bool:
    return is_windows_commit_failure(error) or any(
        isinstance(item, MemoryError) or bool(_MEMORY_FAILURE.search(str(item)))
        for item in _error_nodes(error)
    )


def is_cuda_memory_failure(error: BaseException | str) -> bool:
    return any(re.search(r"CUDA.*out of memory|CUDA.*显存不足", str(item), re.IGNORECASE)
               for item in _error_nodes(error))


def cuda_memory_guidance() -> str:
    return ("CUDA 显存不足。请关闭其他占用显存的程序，将 REPODOCTOR_EMBEDDING_BATCH_SIZE 降为 8 或 4 后重试；"
            "如果模型本身仍无法放入显存，可使用更小的模型，或设置 REPODOCTOR_EMBEDDING_DEVICE=cpu。"
            "已保存的向量批次可继续复用，不需要删除索引缓存。")


def windows_commit_guidance() -> str:
    return (
        "Windows 系统提交内存 / 页面文件额度不足（错误 1455）。"
        "请关闭其他占用内存的程序后重试；若仍失败，检查 Windows 高级系统设置的"
        "“性能 → 高级 → 虚拟内存”。已启用系统管理的页面文件时仍可能耗尽提交额度，"
        "可增大页面文件容量或释放所在磁盘空间，并按系统提示重启。"
        "需要时可手动切换到离线关键词检索。"
    )


def error_message(error: BaseException) -> str:
    if isinstance(error, BaseExceptionGroup):
        messages = list(dict.fromkeys(error_message(item) for item in error.exceptions))
        return "；".join(messages)
    message = str(error).strip() or type(error).__name__
    key = os.getenv("OPENAI_API_KEY", "")
    if key:
        message = message.replace(key, "[已隐藏密钥]")
    message = re.sub(r"(?i)Bearer\s+\S+", "Bearer [已隐藏密钥]", message)
    return message[:2000]


class MCPConnectionError(RuntimeError):
    """A local code-server failure with the underlying startup diagnostic."""
