from __future__ import annotations

import math
import re
from collections import Counter

from .index import Index
from .index_progress import index_build_lock
from .knowledge import BACKENDS, SEMANTIC_BACKENDS, require_knowledge
from .models import Passage

FRAME = re.compile(r"(?:at\s+)?([\w.$]+)\.([\w$<>]+)\(([^():]+\.java):(\d+)\)")
LOGGER = re.compile(r"\b([A-Za-z_$][\w.$]*)\s+\[(\d+)\]\s*-")
IDENTIFIER = re.compile(r"(?<![\w$#.])[A-Za-z_$][\w$]*(?:[.#][A-Za-z_$][\w$]*)*")
SIGNATURE = re.compile(IDENTIFIER.pattern + r"\s*\(([^()\r\n]*)\)(?![\w$#])")


def parse_query(query: str) -> dict:
    return {
        "frames": [
            {"class_name": m[1].replace("$", "."), "method": m[2], "file": m[3], "line": int(m[4])}
            for m in FRAME.finditer(query)
        ],
        "identifiers": {m[0] for m in IDENTIFIER.finditer(query)},
        "signatures": {re.sub(r"\s+", "", m[0]) for m in SIGNATURE.finditer(query)},
        "log_locations": [{"class_name": m[1], "line": int(m[2])} for m in LOGGER.finditer(query)],
    }


def words(text: str) -> list[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-z0-9_$]+|[\u4e00-\u9fff]", text.lower())


def exact_score(
    passage: Passage, query: str, *, parsed: dict | None = None, query_owners: set[str] | None = None
) -> float:
    parsed = parsed if parsed is not None else parse_query(query)
    score = 0.0
    for location in parsed["log_locations"]:
        if (
            passage.class_name.rsplit(".", 1)[-1] == location["class_name"].rsplit(".", 1)[-1]
            and passage.start_line <= location["line"] <= passage.end_line
        ):
            score = max(score, 750)
    for rank, frame in enumerate(parsed["frames"]):
        if passage.class_name == frame["class_name"] and passage.symbol:
            name = passage.symbol.split("#")[1].split("(")[0]
            if name == frame["method"]:
                score = max(
                    score,
                    1000 - rank * 10 + (5 if passage.start_line <= frame["line"] <= passage.end_line else 0),
                )
        elif not passage.structured and passage.file_path.endswith("/" + frame["file"]):
            if passage.start_line <= frame["line"] <= passage.end_line:
                score = max(score, 900 - rank * 10)
    if passage.symbol:
        owner, method = passage.symbol.split("#", 1)
        name = method.split("(")[0]
        short_owner = owner.rsplit(".", 1)[-1]
        aliases = {
            f"{owner}#{name}",
            f"{owner}.{name}",
            f"{short_owner}.{name}",
            f"{short_owner}#{name}",
        }
        identifiers = parsed["identifiers"]
        short_signature = f"{short_owner}#{method}"
        if passage.symbol in parsed["signatures"]:
            score = max(score, 800)
        elif short_signature in parsed["signatures"] or f"{short_owner}.{method}" in parsed["signatures"]:
            score = max(score, 650)
        elif aliases.intersection(identifiers):
            score = max(score, 600)
        elif name in identifiers and name != short_owner:
            if owner in identifiers or short_owner in identifiers:
                score = max(score, 600)
            elif not query_owners or owner in query_owners:
                # A bare method is useful only when no different known owner was requested.
                score = max(score, 300)
    for path, line in re.findall(r"([\w/\\.-]+\.java):(\d+)", query):
        if (
            passage.file_path.endswith(path.replace("\\", "/"))
            and passage.start_line <= int(line) <= passage.end_line
        ):
            score = max(score, 700)
    return score


class Search:
    def __init__(self, index: Index, backend="lexical", embedding_model: str | None = None,
                 *, exact_rerank: bool = False):
        if backend not in BACKENDS:
            raise ValueError("backend 必须为 lexical、linearrag 或 vector")
        self.index, self.backend = index, backend
        self.embedding_model = embedding_model
        self.exact_rerank = exact_rerank
        self._linear = None
        self._vector = None
        self._snapshot = None

    def prepare(self):
        """Load native numerical libraries before starting a stdio event loop on Windows."""
        if self.backend in SEMANTIC_BACKENDS:
            with index_build_lock(self.index):
                require_knowledge(self.index.root, self.backend, self.embedding_model)
                snapshot = self.index.metadata()["snapshot"]
                if self.backend == "linearrag" and (self._linear is None or self._snapshot != snapshot):
                    from .linear import LinearBackend

                    self._linear = LinearBackend(self.index, self.embedding_model)
                    self._snapshot = snapshot
                elif self.backend == "vector" and (self._vector is None or self._snapshot != snapshot):
                    from .dense import DenseBackend

                    self._vector = DenseBackend(self.index, self.embedding_model)
                    self._snapshot = snapshot

    def search(self, query: str, top_k=5) -> dict:
        if not query.strip() or len(query) > 20000 or not 1 <= top_k <= 10:
            raise ValueError("query 不能为空且最多 20000 字符；top_k 为 1–10")
        self.index.check_fresh()
        passages = self.index.passages()
        parsed = parse_query(query)
        owner_references = set(parsed["identifiers"])
        for identifier in parsed["identifiers"]:
            for separator in ("#", "."):
                if separator in identifier:
                    owner_references.add(identifier.rsplit(separator, 1)[0])
        query_owners = {
            p.class_name
            for p in passages
            if p.class_name and (
                p.class_name in owner_references or p.class_name.rsplit(".", 1)[-1] in owner_references
            )
        }
        exact = {p.id: (exact_score(p, query, parsed=parsed, query_owners=query_owners)
                       if self.exact_rerank else 0.0) for p in passages}
        has_exact = any(exact.values())
        routes = []
        scores = {}
        if self.backend == "vector":
            self.prepare()
            # Exact candidates can be outside dense Top K too. Read all scores
            # before reordering so overloads retain their real semantic scores.
            scores, route = self._vector.rank(query, max(10, len(passages)) if has_exact else top_k)
            routes.append(route)
        elif self.backend == "linearrag":
            self.prepare()
            # Exact matches can sit below the backend's top ten. Retain their real
            # scores so overloads and same-name methods do not tie at a fabricated zero.
            scores, route = self._linear.rank(query, max(10, len(passages)) if has_exact else max(10, top_k))
            routes.append(route)
        else:
            query_words = Counter(words(query))
            documents = [Counter(words(p.content + " " + p.symbol)) for p in passages]
            df = Counter(w for doc in documents for w in doc)
            avg_len = sum(sum(d.values()) for d in documents) / max(1, len(documents))
            for p, doc in zip(passages, documents):
                length = sum(doc.values())
                scores[p.id] = sum(
                    math.log(1 + (len(documents) - df[w] + 0.5) / (df[w] + 0.5))
                    * doc[w]
                    * 2.2
                    / (doc[w] + 1.2 * (0.25 + 0.75 * length / max(1, avg_len)))
                    for w in query_words
                    if doc[w]
                )
            routes.append("lexical_bm25")
        if has_exact:
            routes.insert(0, "exact_stack_or_symbol")
        candidates = [p for p in passages if p.id in scores or exact[p.id]] if self.backend == "vector" else passages
        backend_positions = {passage_id: rank for rank, passage_id in enumerate(scores)}
        ranked = sorted(candidates, key=lambda p: (
            exact[p.id], scores.get(p.id, 0), -backend_positions.get(p.id, len(scores)),
        ), reverse=True)
        hits = []
        file_lengths = {}
        for p in ranked:
            if self.backend != "vector" and not exact[p.id] and scores.get(p.id, 0) <= 0:
                continue
            hit = p.to_dict()
            if p.file_path not in file_lengths:
                file_lengths[p.file_path] = self.index.read(p.file_path, 1, 1)["total_lines"]
            hit["total_lines"] = file_lengths[p.file_path]
            hit.update(
                score=round(exact[p.id] + scores.get(p.id, 0), 6),
                exact_score=exact[p.id],
                backend_score=scores.get(p.id, 0),
            )
            # Preserve whole methods in storage, bound the model payload and disclose omitted lines.
            lines = p.content.splitlines()
            locations = [frame["line"] for frame in parsed["frames"] if frame["class_name"] == p.class_name]
            locations += [
                loc["line"]
                for loc in parsed["log_locations"]
                if loc["class_name"].rsplit(".", 1)[-1] == p.class_name.rsplit(".", 1)[-1]
            ]
            locations += [
                int(line)
                for path, line in re.findall(r"([\w/\\.-]+\.java):(\d+)", query)
                if p.file_path.endswith(path.replace("\\", "/"))
            ]
            anchor = next((line for line in locations if p.start_line <= line <= p.end_line), p.start_line)
            offset = max(0, min(anchor - p.start_line - 50, len(lines) - 100))
            hit["content"] = "\n".join(lines[offset : offset + 100])
            hit["source_range"] = {"start_line": p.start_line, "end_line": p.end_line}
            hit["start_line"] = p.start_line + offset
            hit["available_end_line"] = min(p.end_line, hit["start_line"] + 99)
            if parsed["log_locations"] and locations:
                hit["location_hint"] = "日志行号仅定位日志输出语句，不一定是实际异常抛出位置。"
            hit["truncated"] = len(lines) > 100
            hits.append(hit)
            if len(hits) >= top_k:
                break
        return {
            "query": query,
            "top_k": top_k,
            "hits": hits,
            "retrieval_path": routes,
            "backend": self.backend,
            "exact_rerank": self.exact_rerank,
            "snapshot": self.index.metadata()["snapshot"],
            "notice": "离线词法检索不具备跨语言语义匹配；纯中文建议使用 vector 或 linearrag 双语向量模型。"
            if self.backend == "lexical"
            else "",
        }
