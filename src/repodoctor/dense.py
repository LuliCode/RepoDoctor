"""Standalone passage-vector retrieval; optional libraries load only at runtime."""

from __future__ import annotations

import contextlib
import sys

from .errors import (
    cuda_memory_guidance, is_cuda_memory_failure, is_memory_failure,
    is_windows_commit_failure, windows_commit_guidance,
)
from .index_progress import emit_index_progress
from .embedding_settings import embedding_batch_size
from .knowledge import dense_cache_source, dense_directory, mark_dense_ready


def passage_mapping(passages):
    """The same stable full-passage text is encoded for both semantic backends."""
    return {f"File: {p.file_path}\nSymbol: {p.symbol or p.class_name}\n{p.content}": p for p in passages}


class DenseBackend:
    def __init__(self, index, embedding_model=None, encoder=None, progress=None, *, reuse_embeddings=True):
        self._loading_stage = "加载向量模型"
        try:
            self._initialize(index, embedding_model, encoder, progress, reuse_embeddings)
        except Exception as exc:
            if is_cuda_memory_failure(exc):
                raise RuntimeError(cuda_memory_guidance()) from exc
            if is_windows_commit_failure(exc):
                raise RuntimeError(f"向量 RAG 在{self._loading_stage}时无法完成内存分配。"
                                   + windows_commit_guidance()) from exc
            if is_memory_failure(exc):
                advice = ("可使用更小的向量模型，或手动选择离线关键词检索。" if self._loading_stage == "加载向量模型"
                          else "可减小索引范围，或手动选择离线关键词检索。")
                raise RuntimeError(f"向量 RAG 在{self._loading_stage}时内存不足。" + advice) from exc
            raise

    def _initialize(self, index, model, encoder, progress, reuse_embeddings):
        try:
            import numpy as np
            from ._vendor.linearrag.embedding_store import EmbeddingStore
            from .embedding import WindowedEncoder
            if encoder is None:
                from .embedding_model import load_embedding_model

                emit_index_progress(progress, "load_model", "正在加载向量模型…")
                with contextlib.redirect_stdout(sys.stderr):
                    encoder = load_embedding_model(model, progress=progress)
            self.encoder = WindowedEncoder(encoder)
        except ImportError as exc:
            raise RuntimeError('向量 RAG 依赖缺失，请运行 pip install -e ".[vector]"') from exc
        self.mapping = passage_mapping(index.passages())
        self.batch_size = embedding_batch_size()
        own_path = dense_directory(index, model) / "passage_embedding.parquet"
        source = dense_cache_source(index, model)
        # A complete local file without its marker can be validated and recovered.
        if source is None and own_path.is_file() and own_path.stat().st_size:
            source = own_path
        self.cache_path = source or own_path
        reuse_cache = None
        if source is None:
            from .embedding_cache import EmbeddingCache

            dimension = getattr(self.encoder, "get_sentence_embedding_dimension", lambda: None)()
            reuse_cache = EmbeddingCache(index, model, progress, reuse=reuse_embeddings, dimension=dimension)
        self._loading_stage = "读取向量缓存"
        with contextlib.redirect_stdout(sys.stderr):
            self.store = EmbeddingStore(self.encoder, str(self.cache_path), self.batch_size, "passage",
                                        progress=progress, reuse_cache=reuse_cache)
            if source is None:
                self._loading_stage = "生成代码片段向量"
                self.store.insert_text(list(self.mapping))
        self._validate_cache()
        if self.cache_path == own_path and dense_cache_source(index, model) != own_path:
            mark_dense_ready(index, model)
        self.embeddings = np.asarray(self.store.embeddings, dtype=np.float32)
        self.passage_ids = [self.mapping[text].id for text in self.store.texts]

    def _validate_cache(self):
        import numpy as np
        from ._vendor.linearrag.utils import compute_mdhash_id

        problem = "向量缓存损坏或与当前代码片段不一致，请重新运行 index --backend vector --force。"
        store = self.store
        count = len(self.mapping)
        if (not count or len(store.hash_ids) != count or len(store.texts) != count
                or len(store.text_to_hash_id) != count or set(store.texts) != self.mapping.keys()):
            raise ValueError(problem)
        vectors = np.asarray(store.embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != count or not vectors.shape[1]:
            raise ValueError(problem)
        for start in range(0, count, 256):
            batch = vectors[start:start + 256]
            if not np.isfinite(batch).all() or not np.allclose(np.linalg.norm(batch, axis=1), 1, atol=1e-3):
                raise ValueError(problem)
        for position, (hash_id, text) in enumerate(zip(store.hash_ids, store.texts)):
            if (hash_id != compute_mdhash_id(text, prefix="passage-")
                    or store.hash_id_to_idx.get(hash_id) != position
                    or store.hash_id_to_text.get(hash_id) != text):
                raise ValueError(problem)

    def rank(self, query, top_k):
        import numpy as np

        with contextlib.redirect_stdout(sys.stderr):
            vector = np.asarray(self.encoder.encode(query, normalize_embeddings=True,
                                                   show_progress_bar=False, batch_size=self.batch_size), dtype=np.float32)
        vector = vector.reshape(-1)
        if not np.isfinite(vector).all() or vector.size != self.embeddings.shape[1]:
            raise ValueError("查询向量无效或维度与索引不一致。")
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError("查询向量长度为零。")
        scores = self.embeddings @ (vector / norm)
        positions = np.argsort(-scores, kind="stable")[:top_k]
        return {self.passage_ids[i]: float(scores[i]) for i in positions}, "vector_dense"
