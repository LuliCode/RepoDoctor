"""Lightweight readiness checks: never import or load the embedding model here."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .errors import error_message
from .index import Index, IndexErrorBase
from .index_progress import emit_index_progress, index_build_lock
from .models import digest

DEFAULT_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ADAPTER_VERSION = "java-v4-statements-windowed"
DENSE_ADAPTER_VERSION = "passage-v1-windowed"
BACKENDS = ("lexical", "linearrag", "vector")
SEMANTIC_BACKENDS = {"linearrag", "vector"}
VECTOR_FILES = (
    "passage_embedding.parquet",
    "entity_embedding.parquet",
    "sentence_embedding.parquet",
    "ner_results.json",
    "LinearRAG.graphml",
)


def model_key(model: str | None) -> str:
    value = model or DEFAULT_EMBEDDING
    path = Path(value).expanduser()
    return str(path.resolve()) if path.is_dir() else value


def vector_directory(index: Index, model: str | None) -> Path:
    snapshot = index.metadata()["snapshot"]
    return (
        index.db_path.parent / "linearrag" / f"{snapshot}-{ADAPTER_VERSION}-{digest(model_key(model))[:12]}"
    )


def dense_directory(index: Index, model: str | None) -> Path:
    return index.db_path.parent / "vector" / (
        f"{index.metadata()['snapshot']}-{DENSE_ADAPTER_VERSION}-{digest(model_key(model))[:12]}"
    )


def _dense_marker(index, model):
    return {"snapshot": index.metadata()["snapshot"], "model": model_key(model),
            "adapter_version": DENSE_ADAPTER_VERSION}


def dense_cache_source(index: Index, model: str | None) -> Path | None:
    """Find completed passage embeddings without importing a model or a graph.

    v3/v4 LinearRAG use the same passage text and windowed encoder. Their entity
    and sentence files are irrelevant to dense retrieval. Earlier truncated
    encodings are deliberately excluded.
    """
    directory = dense_directory(index, model)
    candidates = [(directory, _dense_marker(index, model))]
    for version in dict.fromkeys((ADAPTER_VERSION, "java-v3-windowed")):
        candidates.append((
            index.db_path.parent / "linearrag" / (
                f"{index.metadata()['snapshot']}-{version}-{digest(model_key(model))[:12]}"
            ),
            {"snapshot": index.metadata()["snapshot"], "model": model_key(model), "adapter_version": version},
        ))
    for candidate, expected in candidates:
        path = candidate / "passage_embedding.parquet"
        try:
            if (json.loads((candidate / "ready.json").read_text(encoding="utf-8")) == expected
                    and path.is_file() and path.stat().st_size > 0):
                return path
        except (OSError, ValueError):
            continue
    return None


def dense_vectors_ready(index: Index, model: str | None) -> bool:
    return dense_cache_source(index, model) is not None


def mark_dense_ready(index: Index, model: str | None):
    directory = dense_directory(index, model)
    path = directory / "passage_embedding.parquet"
    if not path.is_file() or not path.stat().st_size:
        raise IndexErrorBase("代码片段向量索引未完整生成，请重新构建。")
    temporary = directory / "ready.json.tmp"
    temporary.write_text(json.dumps(_dense_marker(index, model), ensure_ascii=False), encoding="utf-8")
    temporary.replace(directory / "ready.json")


def vectors_ready(index: Index, model: str | None) -> bool:
    directory = vector_directory(index, model)
    try:
        ready = json.loads((directory / "ready.json").read_text(encoding="utf-8"))
        return ready == {
            "snapshot": index.metadata()["snapshot"],
            "model": model_key(model),
            "adapter_version": ADAPTER_VERSION,
        } and all(
            (directory / name).is_file() and (directory / name).stat().st_size > 0 for name in VECTOR_FILES
        )
    except (OSError, ValueError):
        return False


def mark_vectors_ready(index: Index, model: str | None):
    directory = vector_directory(index, model)
    if not all(
        (directory / name).is_file() and (directory / name).stat().st_size > 0 for name in VECTOR_FILES
    ):
        raise IndexErrorBase("向量索引未完整生成，请重新构建索引。")
    data = {
        "snapshot": index.metadata()["snapshot"],
        "model": model_key(model),
        "adapter_version": ADAPTER_VERSION,
    }
    temporary = directory / "ready.json.tmp"
    temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temporary.replace(directory / "ready.json")


def knowledge_status(root, backend="lexical", embedding_model=None, include_tests=None) -> dict:
    state = {"ready": False, "source_ready": False, "vector_ready": False, "metadata": None, "message": ""}
    try:
        if backend not in BACKENDS:
            raise IndexErrorBase("未知检索方式")
        if not str(root).strip():
            raise IndexErrorBase("请填写本机 Java 应用目录。")
        index = Index(root)
        index.validate_project()
        metadata = index.metadata()
        state["metadata"] = metadata
        index.check_fresh()
        if include_tests is not None and metadata["include_tests"] != include_tests:
            raise IndexErrorBase("索引范围已改变，请构建或刷新所需检索方式的索引使设置生效。")
        state["source_ready"] = True
        if backend in SEMANTIC_BACKENDS:
            state["vector_ready"] = (vectors_ready(index, embedding_model) if backend == "linearrag"
                                     else dense_vectors_ready(index, embedding_model))
            if not state["vector_ready"]:
                if backend == "vector":
                    raise IndexErrorBase("源码索引已存在，代码片段向量尚未就绪。请点击“构建 / 刷新向量 RAG 索引”。")
                raise IndexErrorBase(
                    "源码索引已存在，但当前模型的向量和关系索引尚未就绪。请点击“构建 / 刷新 LinearRAG 语义索引”。"
                )
        state["ready"] = True
        state["message"] = {"linearrag": "源码、向量和关系索引已就绪，可复用。",
                            "vector": "源码和代码片段向量已就绪，可进行向量检索。",
                            "lexical": "源码索引已就绪，可检索。"}[backend]
    except (ValueError, OSError, sqlite3.Error, KeyError) as exc:
        state["message"] = error_message(exc)
        if isinstance(exc, (sqlite3.Error, KeyError)):
            state["message"] = "索引损坏或不完整，请重新构建索引。"
    return state


def require_knowledge(root, backend="lexical", embedding_model=None):
    status = knowledge_status(root, backend, embedding_model)
    if not status["ready"]:
        raise IndexErrorBase(status["message"])
    return status["metadata"]


def build_knowledge(
    root, backend="lexical", embedding_model=None, include_tests=False, docs=None, progress=None, force=False
):
    if backend not in BACKENDS:
        raise IndexErrorBase("未知检索方式")
    index = Index(root)
    with index_build_lock(index):
        emit_index_progress(progress, "check_source", "正在检查源码索引是否可复用…")
        status = knowledge_status(root, "lexical", include_tests=include_tests)
        if not force and status["ready"] and status["metadata"].get("docs") == docs:
            metadata = status["metadata"]
            emit_index_progress(
                progress, "reuse_source", "源码与索引范围未变化，复用已有代码片段和符号索引"
            )
        else:
            metadata = index.build(include_tests, docs, progress=progress)
        if backend == "linearrag":
            from .linear import LinearBackend

            # An explicit refresh verifies the graph even with a ready marker;
            # file existence alone cannot detect a truncated GraphML document.
            options = {"reuse_embeddings": False} if force else {}
            LinearBackend(index, embedding_model, progress=progress, repair_cache=True, **options)
        elif backend == "vector":
            if dense_vectors_ready(index, embedding_model):
                emit_index_progress(progress, "reuse_vectors", "当前模型的代码片段向量已就绪，直接复用")
            else:
                from .dense import DenseBackend

                options = {"reuse_embeddings": False} if force else {}
                DenseBackend(index, embedding_model, progress=progress, **options)
        emit_index_progress(progress, "verify", "正在校验索引完整性…")
        require_knowledge(root, backend, embedding_model)
        emit_index_progress(progress, "complete", "代码知识库构建完成", completed=1, total=1)
        return metadata
