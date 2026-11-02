"""Embedding runtime settings; changing device/batches does not change cache keys."""

import os
import re


def embedding_batch_size():
    try:
        size = int(os.getenv("REPODOCTOR_EMBEDDING_BATCH_SIZE", "32"))
    except ValueError as exc:
        raise ValueError("REPODOCTOR_EMBEDDING_BATCH_SIZE 必须为正整数。") from exc
    if size <= 0:
        raise ValueError("REPODOCTOR_EMBEDDING_BATCH_SIZE 必须为正整数。")
    return size


def resolve_embedding_device(device=None):
    import torch

    requested = (device or os.getenv("REPODOCTOR_EMBEDDING_DEVICE", "auto")).strip().lower()
    if requested not in {"auto", "cpu", "cuda"} and not re.fullmatch(r"cuda:\d+", requested):
        raise ValueError("REPODOCTOR_EMBEDDING_DEVICE 仅支持 auto、cpu、cuda 或 cuda:N。")
    if requested == "cpu":
        return "cpu"
    available = torch.cuda.is_available()
    if requested == "auto":
        return "cuda" if available else "cpu"
    if not available:
        raise RuntimeError(
            "已指定 CUDA，但当前 PyTorch 无法使用 GPU。请在实际运行应用的 Python 环境安装 CUDA 版 "
            "PyTorch 并检查 NVIDIA 驱动；torch.cuda.is_available() 应为 True。"
            "也可将 REPODOCTOR_EMBEDDING_DEVICE 改为 auto 或 cpu。"
        )
    if ":" in requested and int(requested.split(":")[1]) >= torch.cuda.device_count():
        raise ValueError(f"CUDA 设备不存在：{requested}。")
    return requested
