from .embedding_store import EmbeddingStore
from .utils import compute_mdhash_id, min_max_normalize
from ...symbols import entity_occurrences
from ...errors import is_memory_failure
import os
import json
from collections import defaultdict
import numpy as np
import math
import tempfile

import igraph as ig
import re
import logging
logger = logging.getLogger(__name__)


class GraphCacheError(ValueError):
    """Vectors/NER validated successfully, but the cached graph is invalid."""


class LinearRAG:
    def __init__(self, global_config):
        self.config = global_config
        logger.info(f"Initializing LinearRAG with config: {self.config}")
        self.dataset_name = global_config.dataset_name
        self.load_embedding_store()
        self.spacy_ner = self.config.java_ner
        self.graph = ig.Graph(directed=False)

    def load_embedding_store(self):
        progress = getattr(self.config, "index_progress", None)
        reuse_cache = getattr(self.config, "embedding_cache", None)
        self.passage_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir,self.dataset_name, "passage_embedding.parquet"), batch_size=self.config.batch_size, namespace="passage", progress=progress, reuse_cache=reuse_cache)
        self.entity_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir,self.dataset_name, "entity_embedding.parquet"), batch_size=self.config.batch_size, namespace="entity", progress=progress, reuse_cache=reuse_cache)
        self.sentence_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir,self.dataset_name, "sentence_embedding.parquet"), batch_size=self.config.batch_size, namespace="sentence", progress=progress, reuse_cache=reuse_cache)

    def _report_index(self, stage, message, completed=None, total=None):
        progress = getattr(self.config, "index_progress", None)
        if progress is not None:
            progress({
                "stage": stage,
                "message": message,
                "completed": completed,
                "total": total,
                "unit": "条",
            })

    def load_existing_data(self,passage_hash_ids):
        self.ner_results_path = os.path.join(self.config.working_dir,self.dataset_name, "ner_results.json")
        if os.path.exists(self.ner_results_path):
            try:
                with open(self.ner_results_path, encoding="utf-8") as source:
                    existing_ner_reuslts = json.load(source)
            except (ValueError, UnicodeError):
                # An older interrupted writer may have left an incomplete JSON.
                # Re-extraction is safe here; load_index() remains strict/read-only.
                return {}, {}, passage_hash_ids
            existing_passage_hash_id_to_entities = existing_ner_reuslts["passage_hash_id_to_entities"]
            existing_sentence_to_entities = existing_ner_reuslts["sentence_to_entities"]
            existing_passage_hash_ids = set(existing_passage_hash_id_to_entities.keys())
            new_passage_hash_ids = set(passage_hash_ids) - existing_passage_hash_ids
            return existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_ids
        else:
            return {}, {}, passage_hash_ids

    def load_index(self):
        """Open a complete cached index without encoding or rewriting its files."""
        problem = "LinearRAG 索引损坏或不完整"
        loading_graph = False
        try:
            dimension = None
            for store in (
                self.passage_embedding_store,
                self.entity_embedding_store,
                self.sentence_embedding_store,
            ):
                count = len(store.hash_ids)
                if (
                    len(store.texts) != count
                    or len(store.embeddings) != count
                    or len(store.hash_id_to_text) != count
                    or len(store.hash_id_to_idx) != count
                    or len(store.text_to_hash_id) != count
                ):
                    raise ValueError("向量记录与哈希映射的条数不一致")
                vectors = np.asarray(store.embeddings, dtype=np.float32)
                if count:
                    if vectors.ndim != 2 or vectors.shape[1] == 0:
                        raise ValueError("向量缓存维度无效")
                    if dimension is not None and vectors.shape[1] != dimension:
                        raise ValueError("片段、实体与句子的向量维度不一致")
                    dimension = vectors.shape[1]
                    for start in range(0, count, 256):
                        if not np.isfinite(vectors[start:start + 256]).all():
                            raise ValueError("向量缓存包含无效数值")
                for position, (hash_id, text) in enumerate(zip(store.hash_ids, store.texts)):
                    if (
                        not isinstance(text, str)
                        or hash_id != compute_mdhash_id(text, prefix=store.namespace + "-")
                        or store.hash_id_to_text.get(hash_id) != text
                        or store.hash_id_to_idx.get(hash_id) != position
                        or store.text_to_hash_id.get(text) != hash_id
                    ):
                        raise ValueError("向量缓存的文本与哈希映射不一致")

            directory = os.path.join(self.config.working_dir, self.dataset_name)
            self.ner_results_path = os.path.join(directory, "ner_results.json")
            with open(self.ner_results_path, encoding="utf-8") as source:
                relations = json.load(source)
            passage_entities = relations["passage_hash_id_to_entities"]
            sentence_entities = relations["sentence_to_entities"]
            if not isinstance(passage_entities, dict) or not isinstance(sentence_entities, dict):
                raise ValueError("实体与句子关系缓存格式无效")
            if len(passage_entities) != len(self.passage_embedding_store.hash_ids):
                raise ValueError("实体关系中的代码片段数量不一致")
            referenced_entities = set()
            for hash_id, entities in passage_entities.items():
                if hash_id not in self.passage_embedding_store.hash_id_to_text:
                    raise ValueError("实体关系引用了不存在的代码片段")
                if not isinstance(entities, list):
                    raise ValueError("代码片段的实体关系格式无效")
                for entity in entities:
                    if not isinstance(entity, str) or entity not in self.entity_embedding_store.text_to_hash_id:
                        raise ValueError("代码片段关系引用了不存在的实体")
                    referenced_entities.add(entity)
            if len(referenced_entities) != len(self.entity_embedding_store.hash_ids):
                raise ValueError("实体缓存与代码片段关系不完整")
            if len(sentence_entities) != len(self.sentence_embedding_store.hash_ids):
                raise ValueError("实体关系中的句子数量不一致")

            entity_to_sentences = {hash_id: [] for hash_id in self.entity_embedding_store.hash_ids}
            sentence_to_entities = {}
            for sentence, entities in sentence_entities.items():
                sentence_id = self.sentence_embedding_store.text_to_hash_id.get(sentence)
                if sentence_id is None or not isinstance(entities, list):
                    raise ValueError("实体关系引用了不存在的句子或关系格式无效")
                entity_ids = []
                for entity in entities:
                    entity_id = self.entity_embedding_store.text_to_hash_id.get(entity)
                    if entity_id is None:
                        raise ValueError("句子关系引用了不存在的实体")
                    entity_ids.append(entity_id)
                    entity_to_sentences[entity_id].append(sentence_id)
                if entity_ids:
                    sentence_to_entities[sentence_id] = entity_ids
            if any(not sentences for sentences in entity_to_sentences.values()):
                raise ValueError("实体缺少对应的句子关系")
            # Keep cache and indexing adjacency order consistent; propagation
            # also uses explicit tie-breaking for equal sentence similarities.
            for sentences in entity_to_sentences.values():
                sentences.sort(key=self.sentence_embedding_store.hash_id_to_text.__getitem__)
            for entities in sentence_to_entities.values():
                entities.sort(key=self.entity_embedding_store.hash_id_to_text.__getitem__)
            # Release JSON's duplicated strings/containers before parsing GraphML.
            del relations, passage_entities, sentence_entities, referenced_entities

            loading_graph = True
            graph = ig.Graph.Read_GraphML(os.path.join(directory, "LinearRAG.graphml"))
            if "name" not in graph.vs.attributes():
                raise ValueError("关系图缺少节点名称")
            names = graph.vs["name"]
            expected_count = len(self.passage_embedding_store.hash_ids) + len(self.entity_embedding_store.hash_ids)
            node_lookup = {name: position for position, name in enumerate(names)}
            if len(names) != expected_count or len(node_lookup) != expected_count:
                raise ValueError("关系图中的实体或代码片段节点不完整或重复")
            if any(
                name not in self.passage_embedding_store.hash_id_to_text
                and name not in self.entity_embedding_store.hash_id_to_text
                for name in names
            ):
                raise ValueError("关系图引用了不存在的实体或代码片段")
            if graph.ecount():
                if "weight" not in graph.es.attributes():
                    raise ValueError("关系图缺少边权重")
                for edge in graph.es:
                    weight = edge["weight"]
                    if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
                        raise ValueError("关系图的边权重无效")
            elif self.entity_embedding_store.hash_ids:
                raise ValueError("关系图缺少实体与代码片段之间的边")
            self.graph = graph
            self.entity_hash_id_to_sentence_hash_ids = entity_to_sentences
            self.sentence_hash_id_to_entity_hash_ids = sentence_to_entities
            self.node_name_to_vertex_idx = node_lookup
            self.vertex_idx_to_node_name = dict(enumerate(names))
            self.passage_node_indices = [
                node_lookup[hash_id] for hash_id in set(self.passage_embedding_store.hash_ids)
            ]
            self._retrieval_graph = graph
        except MemoryError:
            raise
        except Exception as exc:
            if is_memory_failure(exc) or isinstance(exc, PermissionError):
                raise
            error_type = GraphCacheError if loading_graph else ValueError
            advice = (
                "请点击“构建 / 刷新 LinearRAG 语义索引”尝试保留向量并修复关系图。"
                if loading_graph else
                "请运行 repodoctor index <项目目录> --backend linearrag --force 重新构建。"
            )
            raise error_type(f"{problem}：{exc}。{advice}") from exc


    def retrieve(self, questions):
        self.entity_hash_ids = list(self.entity_embedding_store.hash_id_to_text.keys())
        self.entity_embeddings = np.asarray(self.entity_embedding_store.embeddings, dtype=np.float32)
        self.passage_hash_ids = list(self.passage_embedding_store.hash_id_to_text.keys())
        self.passage_embeddings = np.asarray(self.passage_embedding_store.embeddings, dtype=np.float32)
        self.sentence_hash_ids = list(self.sentence_embedding_store.hash_id_to_text.keys())
        self.sentence_embeddings = np.asarray(self.sentence_embedding_store.embeddings, dtype=np.float32)
        if getattr(self, "_retrieval_graph", None) is not self.graph:
            self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()}
            self.vertex_idx_to_node_name = {v.index: v["name"] for v in self.graph.vs if "name" in v.attributes()}
            self._retrieval_graph = self.graph

        retrieval_results = []
        for question_info in questions:
            question = question_info["question"]
            question_embedding = self.config.embedding_model.encode(question,normalize_embeddings=True,show_progress_bar=False,batch_size=self.config.batch_size)
            seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores = self.get_seed_entities(question)
            if len(seed_entities) != 0:
                retrieval_path = "linearrag_entity_graph"
                sorted_passage_hash_ids,sorted_passage_scores = self.graph_search_with_seed_entities(question,question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores)
                final_passage_hash_ids = sorted_passage_hash_ids[:self.config.retrieval_top_k]
                final_passage_scores = sorted_passage_scores[:self.config.retrieval_top_k]
                final_passages = [self.passage_embedding_store.hash_id_to_text[passage_hash_id] for passage_hash_id in final_passage_hash_ids]
            else:
                retrieval_path = "linearrag_vector_fallback"
                sorted_passage_indices,sorted_passage_scores = self.dense_passage_retrieval(question_embedding)
                final_passage_indices = sorted_passage_indices[:self.config.retrieval_top_k]
                final_passage_scores = sorted_passage_scores[:self.config.retrieval_top_k]
                final_passages = [self.passage_embedding_store.texts[idx] for idx in final_passage_indices]
            result = {
                "question": question,
                "sorted_passage": final_passages,
                "sorted_passage_scores": final_passage_scores,
                "retrieval_path": retrieval_path
            }
            retrieval_results.append(result)
        return retrieval_results


    def graph_search_with_seed_entities(self, question, question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores):
        entity_weights, actived_entities = self.calculate_entity_scores(
            question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores
        )
        passage_weights = self.calculate_passage_scores(question, question_embedding, actived_entities)
        node_weights = entity_weights + passage_weights
        ppr_sorted_passage_indices,ppr_sorted_passage_scores = self.run_ppr(node_weights)
        return ppr_sorted_passage_indices,ppr_sorted_passage_scores

    def run_ppr(self, node_weights):
        reset_prob = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=self.config.damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_indices])
        sorted_indices_in_doc_scores = np.argsort(doc_scores)[::-1]
        sorted_passage_scores = doc_scores[sorted_indices_in_doc_scores]

        sorted_passage_hash_ids = [
            self.vertex_idx_to_node_name[self.passage_node_indices[i]]
            for i in sorted_indices_in_doc_scores
        ]

        return sorted_passage_hash_ids, sorted_passage_scores.tolist()

    def calculate_entity_scores(self,question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores):
        """Propagate the strongest valid activation, independent of seed order.

        Paper Eq. (5) retains the previous activation with MAX. The pinned BFS
        instead accumulated repeated paths for PPR, overwrote passage bonuses
        with the last path, and let an ineffective source consume a shared
        sentence. Collect a whole round before consuming sentences, and use the
        same strongest-path score for entity resets and passage bonuses. This
        is a conservative Java adaptation of the monotonic activation rule,
        not an exact implementation of the paper's matrix aggregation. A stronger
        path keeps its actual tier; equally strong paths prefer the shorter one.
        This correction applies to BFS; the optional GPU implementation below
        still has the pinned upstream propagation semantics.
        """
        actived_entities = {}
        entity_weights = np.zeros(len(self.graph.vs["name"]))
        for seed_entity_idx,seed_entity,seed_entity_hash_id,seed_entity_score in zip(seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores):
            previous = actived_entities.get(seed_entity_hash_id)
            if not np.isfinite(seed_entity_score) or seed_entity_score <= 0:
                continue
            if previous is not None and previous[1] >= seed_entity_score:
                continue
            actived_entities[seed_entity_hash_id] = (seed_entity_idx, float(seed_entity_score), 1)
            seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score
        used_sentence_hash_ids = set()
        current_entities = actived_entities.copy()
        question_emb = np.asarray(question_embedding).reshape(-1, 1)
        for _ in range(1, self.config.max_iterations):
            if not current_entities or self.config.top_k_sentence <= 0:
                break
            sentence_activations = {}
            for entity_hash_id in sorted(current_entities):
                entity_id, entity_score, tier = current_entities[entity_hash_id]
                if entity_score < self.config.iteration_threshold:
                    continue
                sentence_hash_ids = [sid for sid in list(self.entity_hash_id_to_sentence_hash_ids[entity_hash_id]) if sid not in used_sentence_hash_ids]
                if not sentence_hash_ids:
                    continue
                sentence_indices = [self.sentence_embedding_store.hash_id_to_idx[sid] for sid in sentence_hash_ids]
                sentence_embeddings = self.sentence_embeddings[sentence_indices]
                sentence_similarities = np.dot(sentence_embeddings, question_emb).flatten()
                # Explicit tie-breaking also makes equal-similarity candidates
                # independent of cache/dictionary insertion order.
                top_sentence_indices = sorted(
                    range(len(sentence_hash_ids)),
                    key=lambda idx: (-float(sentence_similarities[idx]), sentence_hash_ids[idx]),
                )[:self.config.top_k_sentence]
                for top_sentence_index in top_sentence_indices:
                    top_sentence_hash_id = sentence_hash_ids[top_sentence_index]
                    next_entity_score = float(entity_score * sentence_similarities[top_sentence_index])
                    if not np.isfinite(next_entity_score) or next_entity_score < self.config.iteration_threshold:
                        continue
                    candidate = (next_entity_score, tier + 1)
                    previous = sentence_activations.get(top_sentence_hash_id)
                    if previous is None or (candidate[0], -candidate[1]) > (previous[0], -previous[1]):
                        sentence_activations[top_sentence_hash_id] = candidate

            # A below-threshold source cannot hide this sentence from another
            # source. Every source in the current round has now been considered.
            used_sentence_hash_ids.update(sentence_activations)
            new_entities = {}
            for sentence_hash_id in sorted(sentence_activations):
                next_entity_score, next_tier = sentence_activations[sentence_hash_id]
                for next_entity_hash_id in sorted(self.sentence_hash_id_to_entity_hash_ids[sentence_hash_id]):
                    previous = actived_entities.get(next_entity_hash_id)
                    if previous is not None and (previous[1], -previous[2]) >= (next_entity_score, -next_tier):
                        continue
                    next_entity_idx = self.entity_embedding_store.hash_id_to_idx[next_entity_hash_id]
                    activation = (next_entity_idx, next_entity_score, next_tier)
                    actived_entities[next_entity_hash_id] = activation
                    new_entities[next_entity_hash_id] = activation
                    next_entity_node_idx = self.node_name_to_vertex_idx[next_entity_hash_id]
                    entity_weights[next_entity_node_idx] = next_entity_score
            current_entities = new_entities
        return entity_weights, actived_entities


    def calculate_passage_scores(self, question, question_embedding, actived_entities):
        passage_weights = np.zeros(len(self.graph.vs["name"]))
        dpr_passage_indices, dpr_passage_scores = self.dense_passage_retrieval(question_embedding)
        dpr_passage_scores = min_max_normalize(dpr_passage_scores)
        for i, dpr_passage_index in enumerate(dpr_passage_indices):
            total_entity_bonus = 0
            passage_hash_id = self.passage_embedding_store.hash_ids[dpr_passage_index]
            dpr_passage_score = dpr_passage_scores[i]
            passage_text = self.passage_embedding_store.hash_id_to_text[passage_hash_id]
            for entity_hash_id, (entity_id, entity_score, tier) in actived_entities.items():
                entity_text = self.entity_embedding_store.hash_id_to_text[entity_hash_id]
                occurrences = entity_occurrences(passage_text, entity_text)
                if occurrences > 0:
                    denom = tier if tier >= 1 else 1
                    entity_bonus = entity_score * math.log(1 + occurrences) / denom
                    total_entity_bonus += entity_bonus

            passage_score = self.config.passage_ratio * dpr_passage_score + math.log(1 + total_entity_bonus)

            passage_node_idx = self.node_name_to_vertex_idx[passage_hash_id]
            passage_weights[passage_node_idx] = passage_score * self.config.passage_node_weight
        return passage_weights

    def dense_passage_retrieval(self, question_embedding):
        question_emb = question_embedding.reshape(1, -1)
        question_passage_similarities = np.dot(self.passage_embeddings, question_emb.T).flatten()
        sorted_passage_indices = np.argsort(question_passage_similarities)[::-1]
        sorted_passage_scores = question_passage_similarities[sorted_passage_indices].tolist()
        return sorted_passage_indices, sorted_passage_scores



    def get_seed_entities(self, question):
        if not self.entity_hash_ids:
            return [], [], [], []
        question_entities = list(dict.fromkeys(self.spacy_ner.question_ner(question)))
        if not question_entities:
            return [], [], [], []

        # Java symbols are case sensitive. Exact identities must not depend on
        # embedding similarity, ties, or a model's handling of unfamiliar names.
        entity_indices = {
            self.entity_embedding_store.hash_id_to_text[hash_id]: idx
            for idx, hash_id in enumerate(self.entity_hash_ids)
        }
        selected = {}
        unmatched_entities = []
        for entity in question_entities:
            if entity in entity_indices:
                selected[entity_indices[entity]] = 1.0
            else:
                unmatched_entities.append(entity)

        if unmatched_entities:
            question_entity_embeddings = self.config.embedding_model.encode(
                unmatched_entities, normalize_embeddings=True,
                show_progress_bar=False, batch_size=self.config.batch_size,
            )
            similarities = np.dot(self.entity_embeddings, question_entity_embeddings.T)
            min_similarity = getattr(self.config, "seed_min_similarity", 0.8)
            min_margin = getattr(self.config, "seed_min_margin", 0.1)
            for query_entity_idx in range(len(unmatched_entities)):
                entity_scores = similarities[:, query_entity_idx]
                if not np.isfinite(entity_scores).all():
                    continue
                best_entity_idx = int(np.argmax(entity_scores))
                best_entity_score = min(float(entity_scores[best_entity_idx]), 1.0)
                runner_up_score = (
                    float(np.partition(entity_scores, -2)[-2])
                    if len(entity_scores) > 1 else 0.0
                )
                # Reject weak/ambiguous alignments before graph activation.
                # iteration_threshold only controls subsequent propagation.
                if (
                    best_entity_score + 1e-7 < min_similarity
                    or best_entity_score - runner_up_score + 1e-7 < min_margin
                ):
                    continue
                selected[best_entity_idx] = max(
                    selected.get(best_entity_idx, 0.0), best_entity_score,
                )

        seed_entity_indices = list(selected)
        seed_entity_hash_ids = [self.entity_hash_ids[idx] for idx in seed_entity_indices]
        seed_entity_texts = [
            self.entity_embedding_store.hash_id_to_text[hash_id]
            for hash_id in seed_entity_hash_ids
        ]
        seed_entity_scores = list(selected.values())
        return seed_entity_indices, seed_entity_texts, seed_entity_hash_ids, seed_entity_scores

    def index(self, passages):
        self._retrieval_graph = None
        self.node_to_node_stats = defaultdict(dict)
        self.entity_to_sentence_stats = defaultdict(dict)
        self.passage_embedding_store.insert_text(passages)
        hash_id_to_passage = self.passage_embedding_store.get_hash_id_to_text()
        self._report_index("load_cache", "正在读取实体与句子提取缓存…")
        existing_passage_hash_id_to_entities,existing_sentence_to_entities, new_passage_hash_ids = self.load_existing_data(hash_id_to_passage.keys())
        self._report_index(
            "extract_entities",
            "正在从代码片段提取实体与句子…" if new_passage_hash_ids else "实体与句子提取结果已缓存，可直接复用。",
            0, len(new_passage_hash_ids),
        )
        if len(new_passage_hash_ids) > 0:
            new_hash_id_to_passage = {k : hash_id_to_passage[k] for k in new_passage_hash_ids}
            new_passage_hash_id_to_entities,new_sentence_to_entities = self.spacy_ner.batch_ner(new_hash_id_to_passage, self.config.max_workers)
            self.merge_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_id_to_entities, new_sentence_to_entities)
        self._report_index("save_cache", "正在保存实体与句子提取结果…")
        self.save_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities)
        self._report_index("extract_entities", "正在整理实体与句子关系…")
        entity_nodes, sentence_nodes,passage_hash_id_to_entities,self.entity_to_sentence,self.sentence_to_entity = self.extract_nodes_and_edges(existing_passage_hash_id_to_entities, existing_sentence_to_entities)
        self.sentence_embedding_store.insert_text(list(sentence_nodes))
        self.entity_embedding_store.insert_text(list(entity_nodes))
        self._report_index("graph", "正在构建实体、句子与代码片段关系图…")
        self.entity_hash_id_to_sentence_hash_ids = {}
        for entity, sentence in self.entity_to_sentence.items():
            entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
            self.entity_hash_id_to_sentence_hash_ids[entity_hash_id] = [self.sentence_embedding_store.text_to_hash_id[s] for s in sorted(sentence)]
        self.sentence_hash_id_to_entity_hash_ids = {}
        for sentence, entities in self.sentence_to_entity.items():
            sentence_hash_id = self.sentence_embedding_store.text_to_hash_id[sentence]
            self.sentence_hash_id_to_entity_hash_ids[sentence_hash_id] = [self.entity_embedding_store.text_to_hash_id[e] for e in sorted(entities)]
        self.add_entity_to_passage_edges(passage_hash_id_to_entities)
        self.add_adjacent_passage_edges()
        self.augment_graph()
        # GraphML/XML 1.0 cannot represent arbitrary source text (e.g. SOH/NUL
        # inside Java test fixtures). Retrieval resolves node hashes through the
        # embedding stores; the duplicate content attribute is never consulted.
        # Also remove it if this graph originated from a legacy in-memory graph.
        if "content" in self.graph.vs.attributes():
            del self.graph.vs["content"]
        output_graphml_path = os.path.join(self.config.working_dir,self.dataset_name, "LinearRAG.graphml")
        os.makedirs(os.path.dirname(output_graphml_path), exist_ok=True)
        self._report_index("save_graph", "正在保存 LinearRAG 关系图…")
        fd, temporary = tempfile.mkstemp(prefix=".graph-", suffix=".graphml", dir=os.path.dirname(output_graphml_path))
        os.close(fd)
        try:
            self.graph.write_graphml(temporary)
            os.replace(temporary, output_graphml_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def add_adjacent_passage_edges(self):
        passage_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        index_pattern = re.compile(r'^(\d+):')
        indexed_items = [
            (int(match.group(1)), node_key)
            for node_key, text in passage_id_to_text.items()
            if (match := index_pattern.match(text.strip()))
        ]
        indexed_items.sort(key=lambda x: x[0])
        for i in range(len(indexed_items) - 1):
            current_node = indexed_items[i][1]
            next_node = indexed_items[i + 1][1]
            self.node_to_node_stats[current_node][next_node] = 1.0

    def augment_graph(self):
        self.add_nodes()
        self.add_edges()

    def add_nodes(self):
        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}
        entity_hash_id_to_text = self.entity_embedding_store.get_hash_id_to_text()
        passage_hash_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        all_hash_id_to_text = {**entity_hash_id_to_text, **passage_hash_id_to_text}

        passage_hash_ids = set(passage_hash_id_to_text.keys())

        for hash_id in all_hash_id_to_text:
            if hash_id not in existing_nodes:
                self.graph.add_vertex(name=hash_id)

        self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()}
        self.passage_node_indices = [
            self.node_name_to_vertex_idx[passage_id]
            for passage_id in passage_hash_ids
            if passage_id in self.node_name_to_vertex_idx
        ]

    def add_edges(self):
        edges = []
        weights = []

        for node_hash_id, node_to_node_stats in self.node_to_node_stats.items():
            for neighbor_hash_id, weight in node_to_node_stats.items():
                if node_hash_id == neighbor_hash_id:
                    continue
                edges.append((node_hash_id, neighbor_hash_id))
                weights.append(weight)
        self.graph.add_edges(edges)
        self.graph.es['weight'] = weights

    def add_entity_to_passage_edges(self, passage_hash_id_to_entities):
        passage_to_entity_count ={}
        passage_to_all_score = defaultdict(int)
        for passage_hash_id, entities in passage_hash_id_to_entities.items():
            passage = self.passage_embedding_store.hash_id_to_text[passage_hash_id]
            for entity in entities:
                entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
                count = entity_occurrences(passage, entity)
                if not count:
                    continue
                passage_to_entity_count[(passage_hash_id, entity_hash_id)] = count
                passage_to_all_score[passage_hash_id] += count
        for (passage_hash_id, entity_hash_id), count in passage_to_entity_count.items():
            score = count / passage_to_all_score[passage_hash_id]
            self.node_to_node_stats[passage_hash_id][entity_hash_id] = score

    def extract_nodes_and_edges(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        entity_nodes = set()
        sentence_nodes = set()
        passage_hash_id_to_entities = defaultdict(set)
        entity_to_sentence= defaultdict(set)
        sentence_to_entity = defaultdict(set)
        for passage_hash_id, entities in existing_passage_hash_id_to_entities.items():
            for entity in entities:
                entity_nodes.add(entity)
                passage_hash_id_to_entities[passage_hash_id].add(entity)
        for sentence,entities in existing_sentence_to_entities.items():
            sentence_nodes.add(sentence)
            for entity in entities:
                entity_to_sentence[entity].add(sentence)
                sentence_to_entity[sentence].add(entity)
        return entity_nodes, sentence_nodes, passage_hash_id_to_entities, entity_to_sentence, sentence_to_entity

    def merge_ner_results(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_id_to_entities, new_sentence_to_entities):
        existing_passage_hash_id_to_entities.update(new_passage_hash_id_to_entities)
        existing_sentence_to_entities.update(new_sentence_to_entities)
        return existing_passage_hash_id_to_entities, existing_sentence_to_entities

    def save_ner_results(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        directory = os.path.dirname(self.ner_results_path)
        os.makedirs(directory, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".ner-", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as target:
                json.dump({"passage_hash_id_to_entities": existing_passage_hash_id_to_entities,
                           "sentence_to_entities": existing_sentence_to_entities}, target)
            os.replace(temporary, self.ner_results_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
