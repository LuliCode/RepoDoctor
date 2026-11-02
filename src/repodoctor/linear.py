"""Java adapter for the pinned, vendored upstream LinearRAG implementation."""

from __future__ import annotations

import contextlib
import logging
import re
import sys
from collections import defaultdict
from types import SimpleNamespace

from .errors import (
    cuda_memory_guidance, is_cuda_memory_failure, is_memory_failure,
    is_windows_commit_failure, windows_commit_guidance,
)
from .dense import passage_mapping
from .embedding_settings import embedding_batch_size
from .index import Index
from .index_progress import emit_index_progress
from .knowledge import VECTOR_FILES, mark_vectors_ready, vector_directory, vectors_ready
from .retrieval import FRAME, parse_query
from .symbols import code_identifiers, entity_occurrences

logger = logging.getLogger(__name__)


_JAVA_SENTENCE_PARTS = re.compile(
    r'//[^\r\n]*|/\*[\s\S]*?\*/|"(?:\\[\s\S]|[^"\\])*"'
    r"|'(?:\\[\s\S]|[^'\\])*'|\s+|.",
)


def _java_sentences(content):
    """Keep a Java statement together across physical lines, within its passage.

    This is lexical segmentation, not inferred control flow: comments and quoted
    literals are atomic; semicolons in for/resource headers are not boundaries.
    Block headers remain separate units so unrelated methods cannot share a
    sentence merely because their declarations occupy the same source line.
    """
    parts = []
    parens = brackets = 0
    braces = []
    previous = ""

    def flush():
        sentence = "".join(parts).strip()
        parts.clear()
        return sentence

    for match in _JAVA_SENTENCE_PARTS.finditer(content):
        part = match.group()
        if part.startswith(("//", "/*")):
            # Standalone documentation is useful retrieval text in its own right.
            if not any(piece.strip() for piece in parts):
                parts.clear()
                yield part.strip()
            else:
                parts.append(part)
            continue
        parts.append(part)
        if part.isspace():
            continue
        if part == "(":
            parens += 1
        elif part == ")":
            parens = max(0, parens - 1)
        elif part == "[":
            brackets += 1
        elif part == "]":
            brackets = max(0, brackets - 1)
        elif part == "{":
            # Arrays and braces nested inside a call/annotation belong to the
            # surrounding expression, including nested array initializers.
            expression = bool(parens or brackets or any(braces) or previous in {"=", "]"})
            braces.append(expression)
            if not expression:
                yield flush()
        elif part == "}":
            expression = braces.pop() if braces else False
            if not expression:
                yield flush()
        elif part == ";" and not parens and not brackets and not any(braces):
            yield flush()
        previous = part
    if sentence := flush():
        yield sentence


def _passage_sentences(text, passage):
    if passage.file_path.lower().endswith(".java") and passage.content and text.endswith(passage.content):
        # File/Symbol metadata provides literal qualified identities even when
        # their short names are all that appear in the source body.
        metadata = text[:-len(passage.content)]
        yield from (line.strip() for line in metadata.splitlines() if line.strip())
        yield from _java_sentences(passage.content)
    else:
        yield from (line.strip() for line in text.splitlines() if line.strip())


class JavaNER:
    def __init__(self, text_to_passage, progress=None):
        self.text_to_passage = text_to_passage
        self.progress = progress
        self.entities = {entity for passage in text_to_passage.values() for entity in passage.entities}
        self.aliases = defaultdict(set)
        self.owner_aliases = defaultdict(set)
        for passage in text_to_passage.values():
            if passage.class_name:
                self.owner_aliases[passage.class_name].add(passage.class_name)
                self.owner_aliases[passage.class_name.rsplit('.', 1)[-1]].add(passage.class_name)
            if not passage.symbol:
                continue
            owner, method = passage.symbol.split("#", 1)
            name = method.split("(", 1)[0]
            for alias in (passage.symbol, f"{owner}.{name}", f"{owner}#{name}",
                          f"{owner.rsplit('.', 1)[-1]}.{name}", f"{owner.rsplit('.', 1)[-1]}#{name}",
                          f"{owner.rsplit('.', 1)[-1]}#{method}"):
                self.aliases[alias].add(passage.symbol)

    def batch_ner(self, hash_id_to_passage, max_workers):
        passages, sentences = {}, defaultdict(list)
        total = len(hash_id_to_passage)
        emit_index_progress(
            self.progress, "extract_entities", "正在提取代码实体与句子…",
            completed=0, total=total, unit="个片段",
        )
        for hash_id, text in hash_id_to_passage.items():
            passage = self.text_to_passage[text]
            # The graph and sentence mentions use the same case-sensitive Java boundaries.
            entities = sorted({e for e in passage.entities if entity_occurrences(text, e)})
            passages[hash_id] = entities
            # Qualified names and complete method keys occur in metadata sentences.
            # Upstream BFS requires every passage entity to have a sentence entry.
            for sentence in _passage_sentences(text, passage):
                matches = [e for e in entities if entity_occurrences(sentence, e)]
                if matches:
                    sentences[sentence] = sorted(set(sentences[sentence]) | set(matches))
            emit_index_progress(
                self.progress, "extract_entities", "正在提取代码实体与句子…",
                completed=len(passages), total=total, unit="个片段", current_file=passage.file_path,
            )
        return passages, dict(sentences)

    def question_ner(self, question):
        parsed = parse_query(question)
        # Stack frames have explicit class/method structure; do not treat their
        # package components, filenames, or the word "at" as separate entities.
        candidates = set(code_identifiers(FRAME.sub("", question)))
        if question.strip() in self.entities:
            candidates.add(question.strip())
        for frame in parsed["frames"]:
            candidates.add(frame["class_name"])
            candidates.add(f"{frame['class_name']}.{frame['method']}")
        for location in parsed["log_locations"]:
            candidates.add(location["class_name"])
        explicit_symbols = {
            symbol for candidate in candidates if "(" in candidate
            for symbol in self.aliases.get(candidate, ())
        }
        explicit_names = {symbol.split("#", 1)[1].split("(", 1)[0] for symbol in explicit_symbols}
        # Space-separated "Class method" is an explicit owner hint too. Align
        # that pair to the same method keys as Class.method/stack frames rather
        # than seeding a bare method shared by unrelated classes.
        owners = {owner for candidate in parsed["identifiers"] for owner in self.owner_aliases.get(candidate, ())}
        paired_names = set()
        paired_symbols = set()
        for candidate in parsed["identifiers"]:
            if not candidate.isidentifier() or candidate in self.owner_aliases or candidate in explicit_names:
                continue
            matches = {symbol for owner in owners for symbol in self.aliases.get(f"{owner}#{candidate}", ())}
            if matches:
                paired_names.add(candidate)
                paired_symbols.update(matches)
        candidates.difference_update(paired_names)
        candidates.update(paired_symbols)
        entities = set()
        for candidate in candidates:
            if candidate in explicit_names:
                continue
            aliases = self.aliases.get(candidate, set())
            if aliases:
                # A complete signature selects its overload. The extractor may
                # also emit a bare alias; it must not reintroduce other overloads.
                entities.update(aliases & explicit_symbols or aliases)
            else:
                entities.add(candidate)
        return sorted(entities)


class LinearBackend:
    def __init__(self, index: Index, embedding_model: str | None = None, encoder=None, progress=None,
                 *, reuse_embeddings=True, repair_cache=False):
        self._loading_stage = "加载向量模型"
        try:
            self._initialize(index, embedding_model, encoder, progress, reuse_embeddings, repair_cache)
        except Exception as exc:
            if is_cuda_memory_failure(exc):
                raise RuntimeError(cuda_memory_guidance()) from exc
            if is_windows_commit_failure(exc):
                raise RuntimeError(
                    f"LinearRAG 在{self._loading_stage}时无法完成内存分配。" + windows_commit_guidance()
                ) from exc
            if is_memory_failure(exc):
                advice = "请关闭占用内存较大的程序后重试；"
                if self._loading_stage == "加载向量模型":
                    advice += "也可使用更小的向量模型，或手动选择离线关键词检索。"
                else:
                    advice += (
                        "也可减小索引范围（例如不收录测试源码或额外文档），"
                        "或手动选择离线关键词检索。"
                    )
                raise RuntimeError(
                    f"LinearRAG 在{self._loading_stage}时内存不足，代码服务未能完成启动。" + advice
                ) from exc
            raise

    def _initialize(self, index, embedding_model, encoder, progress, reuse_embeddings, repair_cache):
        cache_ready = vectors_ready(index, embedding_model)
        logger.info("Loading LinearRAG runtime")
        emit_index_progress(progress, "load_model", "正在加载向量模型，首次可能需要下载模型文件…")
        try:
            from ._vendor.linearrag.core import GraphCacheError, LinearRAG
            from .embedding import WindowedEncoder

            if encoder is None:
                logger.info("Loading SentenceTransformer runtime")
                from .embedding_model import load_embedding_model

                with contextlib.redirect_stdout(sys.stderr):
                    encoder = load_embedding_model(embedding_model, progress=progress)
                logger.info("Embedding model ready")
            encoder = WindowedEncoder(encoder)
        except ImportError as exc:
            raise RuntimeError('LinearRAG 依赖缺失，请运行 pip install -e ".[linearrag]"') from exc
        passages = index.passages()
        # Include stable metadata to distinguish identical methods at different locations.
        self.mapping = passage_mapping(passages)
        self.ner = JavaNER(self.mapping, progress=progress)
        cache_directory = vector_directory(index, embedding_model)
        cache_complete = all(
            (cache_directory / name).is_file() and (cache_directory / name).stat().st_size > 0
            for name in VECTOR_FILES
        )
        embedding_cache = None
        if repair_cache or (not cache_ready and not cache_complete):
            from .embedding_cache import EmbeddingCache

            dimension = getattr(encoder, "get_sentence_embedding_dimension", lambda: None)()
            embedding_cache = EmbeddingCache(index, embedding_model, progress, reuse=reuse_embeddings,
                                             dimension=dimension)
        config = SimpleNamespace(
            dataset_name=cache_directory.name,
            embedding_model=encoder,
            working_dir=str(index.db_path.parent / "linearrag"),
            java_ner=self.ner,
            index_progress=progress,
            embedding_cache=embedding_cache,
            batch_size=embedding_batch_size(),
            max_workers=1,
            retrieval_top_k=10,
            max_iterations=3,
            top_k_sentence=3,
            passage_ratio=1.5,
            passage_node_weight=0.05,
            damping=0.5,
            iteration_threshold=0.5,
            seed_min_similarity=0.8,
            seed_min_margin=0.1,
        )
        # Upstream cache prints must never appear on the MCP stdout protocol channel.
        if not cache_ready:
            (cache_directory / "ready.json").unlink(missing_ok=True)
        self._loading_stage = "读取向量缓存"
        with contextlib.redirect_stdout(sys.stderr):
            self.rag = LinearRAG(config)
            if cache_ready or cache_complete:
                self._loading_stage = "读取关系图缓存"
                emit_index_progress(progress, "load_graph", "正在读取已完成的 LinearRAG 关系图…")
                try:
                    self.rag.load_index()
                except GraphCacheError:
                    if not repair_cache:
                        raise
                    # Only graph failures reach here: load_index validated all
                    # vectors and NER before classifying this failure. Retrieval
                    # startup stays strict/read-only; explicit builds may repair.
                    if self.rag.passage_embedding_store.text_to_hash_id.keys() != self.mapping.keys():
                        raise ValueError("向量缓存与当前源码片段不一致，请使用 --force 重建。")
                    cache_ready = False
                    (cache_directory / "ready.json").unlink(missing_ok=True)
                    self._loading_stage = "修复关系图缓存"
                    emit_index_progress(progress, "repair_graph", "检测到关系图损坏，正在复用已有向量重建关系图…")
                    self.rag.index(list(self.mapping))
                    self.rag.load_index()
                except ValueError:
                    if repair_cache:
                        (cache_directory / "ready.json").unlink(missing_ok=True)
                    raise
                if self.rag.passage_embedding_store.text_to_hash_id.keys() != self.mapping.keys():
                    raise ValueError(
                        "向量缓存与当前源码片段不一致，请运行 "
                        "repodoctor index <项目目录> --backend linearrag --force 重建。"
                    )
            else:
                self._loading_stage = "生成向量与关系索引"
                self.rag.index(list(self.mapping))
                self.rag.load_index()
        if not cache_ready:
            mark_vectors_ready(index, embedding_model)
        logger.info("LinearRAG index ready")

    def rank(self, query, top_k):
        self.rag.config.retrieval_top_k = top_k
        with contextlib.redirect_stdout(sys.stderr):
            result = self.rag.retrieve([{"question": query}])[0]
        scores = {
            self.mapping[text].id: float(score)
            for text, score in zip(result["sorted_passage"], result["sorted_passage_scores"])
        }
        return scores, result["retrieval_path"]
