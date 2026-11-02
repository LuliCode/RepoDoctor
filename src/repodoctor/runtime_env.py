"""Environment for disposable local workers, before numerical libraries import."""

from __future__ import annotations

import os


def worker_environment(backend):
    environment = dict(os.environ)
    environment.pop("OPENAI_API_KEY", None)
    environment.update(PYTHONUNBUFFERED="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    if backend in {"linearrag", "vector"}:
        # Each BLAS/OpenMP thread can reserve scratch buffers. A small default
        # avoids multiplying that memory in local model workers on Windows.
        # Explicit user thread settings remain authoritative.
        for setting in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment.setdefault(setting, "1")
    return environment
