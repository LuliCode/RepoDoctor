"""Load cached sentence encoders without contacting the model hub."""

from __future__ import annotations

import errno
import os
from pathlib import Path

from .errors import _error_nodes, is_memory_failure
from .index_progress import emit_index_progress
from .knowledge import DEFAULT_EMBEDDING


def model_cache_directory():
    # Read configuration at load time: dotenv may run after hub constants were
    # imported by another library in the Streamlit process.
    for name in ("SENTENCE_TRANSFORMERS_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if value := os.getenv(name):
            return str(Path(value).expanduser().resolve())
    if value := os.getenv("HF_HOME"):
        return str((Path(value).expanduser() / "hub").resolve())
    if value := os.getenv("XDG_CACHE_HOME"):
        return str((Path(value).expanduser() / "huggingface" / "hub").resolve())
    return None


def _missing_cache(error):
    from huggingface_hub.errors import LocalEntryNotFoundError

    # Retry only missing local files. Allocation failures, invalid model JSON,
    # incompatible weights and permissions must never trigger a download.
    if is_memory_failure(error):
        return False
    if isinstance(error, ValueError) and not isinstance(error, LocalEntryNotFoundError):
        return False
    if isinstance(error, PermissionError):
        return False
    if isinstance(error, OSError) and error.errno not in {None, errno.ENOENT}:
        return False
    return any(isinstance(item, (LocalEntryNotFoundError, FileNotFoundError)) for item in _error_nodes(error))


def _offline_requested():
    from huggingface_hub import constants

    return constants.HF_HUB_OFFLINE or any(
        os.getenv(name, "").strip().upper() in {"1", "ON", "YES", "TRUE"}
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def _local_path(model):
    path = Path(model).expanduser()
    return path.exists() or path.is_absolute() or model.startswith((".", "~")) or "\\" in model


def load_embedding_model(model=None, progress=None, device=None):
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    from .embedding_settings import resolve_embedding_device

    model = os.fspath(model) if model is not None else DEFAULT_EMBEDDING
    model = model or DEFAULT_EMBEDDING
    path = _local_path(model)
    source = str(Path(model).expanduser().resolve()) if path else model
    device = resolve_embedding_device(device)
    options = {"device": device, "local_files_only": True}
    if cache := model_cache_directory():
        options["cache_folder"] = cache
    # Keep this on before loading any checkpoint, including offline loads.
    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    emit_index_progress(progress, "load_model", f"正在从本地目录或缓存加载向量模型（设备：{device}）…")
    try:
        local_source = source if path else snapshot_download(
            repo_id=source, cache_dir=cache, local_files_only=True,
        )
        # Some transformers versions probe model_info for generic fast
        # tokenizers even with local_files_only=True. A snapshot *path* also
        # makes that tokenizer branch local, so the cached name never reaches
        # the network. The configured name still identifies vector caches.
        return SentenceTransformer(local_source, **options)
    except Exception as exc:
        if path:
            if not Path(source).is_dir() or _missing_cache(exc):
                raise RuntimeError(
                    "向量模型本地目录不存在或文件不完整，请检查 REPODOCTOR_EMBEDDING_MODEL；"
                    "目录需包含权重、tokenizer、模型配置及 Pooling 等组件。"
                ) from exc
            raise
        if not _missing_cache(exc):
            raise
        if _offline_requested():
            raise RuntimeError(
                "已启用离线模式，但当前向量模型的本地缓存不完整。请先准备完整模型，"
                "检查 HF_HOME / HF_HUB_CACHE，或将 REPODOCTOR_EMBEDDING_MODEL 设置为完整本地模型目录。"
            ) from exc
    emit_index_progress(progress, "download_model", "本地模型文件不足，正在从 Hugging Face 下载向量模型…")
    try:
        return SentenceTransformer(source, **{**options, "local_files_only": False})
    except Exception as exc:
        from requests.exceptions import RequestException
        from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

        if any(isinstance(item, (RequestException, HfHubHTTPError, LocalEntryNotFoundError))
               for item in _error_nodes(exc)):
            raise RuntimeError(
                "向量模型下载失败，无法连接 Hugging Face 或获取所需文件。"
                "请检查网络及模型访问权限；也可在有网络的机器下载完整模型后，"
                "将 REPODOCTOR_EMBEDDING_MODEL 设置为完整本地模型目录。"
                "使用已有缓存时请检查 HF_HOME / HF_HUB_CACHE；更换索引范围不能解决模型下载问题。"
            ) from exc
        raise
