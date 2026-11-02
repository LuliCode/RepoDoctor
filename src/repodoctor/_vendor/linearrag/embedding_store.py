from copy import deepcopy
from .utils import compute_mdhash_id
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
import tempfile

class EmbeddingStore:
    # Keep Arrow's decoded buffers bounded even for old caches with huge row groups.
    CACHE_BATCH_SIZE = 256

    def __init__(self, embedding_model, db_filename, batch_size, namespace, progress=None, reuse_cache=None):
        self.embedding_model = embedding_model
        self.db_filename = db_filename
        self.batch_size = batch_size
        self.namespace = namespace
        self.progress = progress
        self.reuse_cache = reuse_cache

        self.hash_ids = []
        self.texts = []
        self.embeddings = []
        self.hash_id_to_text = {}
        self.hash_id_to_idx = {}
        self.text_to_hash_id = {}

        self._load_data()

    def _report(self, stage, message, completed=None, total=None):
        if self.progress is not None:
            self.progress({
                "stage": stage,
                "message": message,
                "completed": completed,
                "total": total,
                "unit": "条",
            })

    def _label(self):
        return {"passage": "代码片段", "sentence": "句子", "entity": "实体"}.get(
            self.namespace, self.namespace
        )

    def _load_data(self):
        self._report("load_cache", f"正在读取{self._label()}向量缓存…")
        if os.path.exists(self.db_filename):
            hash_ids, texts = [], []
            matrix = None
            parquet = pq.ParquetFile(self.db_filename, pre_buffer=False)
            try:
                total = parquet.metadata.num_rows
                self._report("load_cache", f"正在读取{self._label()}向量缓存：0/{total} 条。", 0, total)
                for batch in parquet.iter_batches(
                    batch_size=self.CACHE_BATCH_SIZE,
                    columns=["hash_id", "text", "embedding"],
                    use_threads=False,
                ):
                    vectors = batch.column("embedding")
                    if vectors.null_count:
                        raise ValueError("向量缓存包含空向量，请重新构建索引。")
                    if pa.types.is_fixed_size_list(vectors.type):
                        dimension = vectors.type.list_size
                    elif pa.types.is_list(vectors.type) or pa.types.is_large_list(vectors.type):
                        lengths = vectors.value_lengths().to_numpy(zero_copy_only=False)
                        dimension = int(lengths[0])
                        if not np.all(lengths == dimension):
                            raise ValueError("向量缓存中的向量维度不一致，请重新构建索引。")
                    else:
                        raise ValueError("向量缓存格式无效，请重新构建索引。")
                    if dimension <= 0 or (matrix is not None and matrix.shape[1] != dimension):
                        raise ValueError("向量缓存中的向量维度不一致，请重新构建索引。")
                    values = vectors.flatten()
                    if values.null_count:
                        raise ValueError("向量缓存包含空值，请重新构建索引。")
                    if matrix is None:
                        # One owned matrix replaces DataFrame/object arrays/row arrays.
                        # float64 caches written by previous versions remain readable.
                        matrix = np.empty((total, dimension), dtype=np.float32)
                    start = len(hash_ids)
                    end = start + batch.num_rows
                    matrix[start:end] = values.to_numpy(zero_copy_only=False).reshape(
                        batch.num_rows, dimension
                    )
                    hash_ids.extend(batch.column("hash_id").to_pylist())
                    texts.extend(batch.column("text").to_pylist())
                    self._report(
                        "load_cache", f"正在读取{self._label()}向量缓存：{end}/{total} 条。", end, total
                    )
                    # The generator's next Arrow batch must not overlap the previous
                    # batch through these references (in particular its value buffer).
                    del batch, vectors, values
            finally:
                parquet.close()
            self.hash_ids, self.texts = hash_ids, texts
            self.embeddings = matrix if matrix is not None else []
            self._rebuild_maps()
            print(f"[{self.namespace}] Loaded {len(self.hash_ids)} records from {self.db_filename}")
        self._report(
            "load_cache", f"已读取{self._label()}向量缓存：{len(self.hash_ids)} 条。",
            len(self.hash_ids), len(self.hash_ids),
        )

    def insert_text(self, text_list):
        nodes_dict = {}
        for text in text_list:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        all_hash_ids = list(nodes_dict.keys())

        existing = set(self.hash_ids)
        missing_ids = [h for h in all_hash_ids if h not in existing]
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]
        total = len(all_hash_ids)
        cached = total - len(missing_ids)
        stage = f"embedding_{self.namespace}"
        self._report(
            stage,
            f"正在生成{self._label()}向量；已复用 {cached} 条缓存，待生成 {len(missing_ids)} 条。",
            cached, total,
        )
        if not missing_ids and os.path.exists(self.db_filename):
            return
        if missing_ids and self.reuse_cache is not None:
            self._insert_with_shared_cache(missing_ids, texts_to_encode, cached, total)
            return
        all_embeddings = []
        if texts_to_encode and self.progress is None:
            # Preserve upstream's single-call behavior when no observer is attached.
            all_embeddings = self._validated_embeddings(self.embedding_model.encode(
                texts_to_encode, normalize_embeddings=True,
                show_progress_bar=False, batch_size=self.batch_size,
            ), len(texts_to_encode))
        elif texts_to_encode:
            # Report only completed encoder calls, never estimates of model work.
            # Accumulate all batches before updating memory/disk so a failed batch
            # does not leave a partially inserted cache.
            dimension = None
            for start in range(0, len(texts_to_encode), self.batch_size):
                batch = texts_to_encode[start:start + self.batch_size]
                embeddings = self._validated_embeddings(self.embedding_model.encode(
                    batch, normalize_embeddings=True,
                    show_progress_bar=False, batch_size=self.batch_size,
                ), len(batch))
                if dimension is not None and embeddings.shape[1] != dimension:
                    raise ValueError("向量模型返回的向量维度在批次之间发生变化。")
                if dimension is None:
                    dimension = embeddings.shape[1]
                    all_embeddings = np.empty((len(texts_to_encode), dimension), dtype=np.float32)
                end = start + len(batch)
                all_embeddings[start:end] = embeddings
                completed = cached + end
                self._report(
                    stage,
                    f"正在生成{self._label()}向量：{completed}/{total} 条（复用 {cached} 条缓存）。",
                    completed, total,
                )
                del embeddings

        self._upsert(missing_ids, texts_to_encode, all_embeddings)

    def _insert_with_shared_cache(self, ids, texts, cached, total):
        # The snapshot file remains atomic. Each successful encoder batch is
        # separately committed to the content cache before the next batch starts,
        # so a cancelled/failed build can resume without recomputing those vectors.
        cache = self.reuse_cache
        cache.prepare(self.namespace)
        all_embeddings = None
        reused = cached
        encoded_count = 0
        cache_batch_size = min(self.batch_size, cache.BATCH_SIZE)
        for start in range(0, len(ids), cache_batch_size):
            end = min(start + cache_batch_size, len(ids))
            batch_ids, batch_texts = ids[start:end], texts[start:end]
            found = cache.get(self.namespace, dict(zip(batch_ids, batch_texts)))
            missing = [i for i, h in enumerate(batch_ids) if h not in found]
            if missing:
                missing_texts = [batch_texts[i] for i in missing]
                vectors = self._validated_embeddings(self.embedding_model.encode(
                    missing_texts, normalize_embeddings=True,
                    show_progress_bar=False, batch_size=self.batch_size,
                ), len(missing))
                cache.put(self.namespace, [batch_ids[i] for i in missing], missing_texts, vectors)
                found.update((batch_ids[i], vector) for i, vector in zip(missing, vectors))
            batch = self._validated_embeddings([found[h] for h in batch_ids], len(batch_ids))
            if all_embeddings is None:
                all_embeddings = np.empty((len(ids), batch.shape[1]), dtype=np.float32)
            if batch.shape[1] != all_embeddings.shape[1]:
                raise ValueError("向量模型或共享缓存中的向量维度在批次之间发生变化。")
            all_embeddings[start:end] = batch
            reused += len(batch_ids) - len(missing)
            encoded_count += len(missing)
            self._report(f"embedding_{self.namespace}",
                         f"正在准备{self._label()}向量：{cached + end}/{total} 条（复用 {reused} 条，新生成 {encoded_count} 条）。",
                         cached + end, total)
        self._upsert(ids, texts, all_embeddings)
        self._report("reuse_embeddings", f"{self._label()}向量完成：复用 {reused} 条，新生成 {encoded_count} 条。", total, total)

    def _upsert(self, hash_ids, texts, embeddings):
        if not hash_ids:
            # Reopening a ready index must not rewrite the entire vector cache.
            if not os.path.exists(self.db_filename):
                self._save_data()
            return
        embeddings = self._validated_embeddings(embeddings, len(hash_ids))
        previous_embeddings = self.embeddings
        previous_count = len(self.hash_ids)
        if len(self.embeddings):
            if self.embeddings.shape[1] != embeddings.shape[1]:
                raise ValueError("向量模型返回的向量维度与现有缓存不一致。")
            self.embeddings = np.concatenate((self.embeddings, embeddings), axis=0)
        else:
            self.embeddings = embeddings
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)
        try:
            self._save_data()
        except Exception:
            # A failed atomic save must also leave this instance safe to retry.
            self.embeddings = previous_embeddings
            del self.hash_ids[previous_count:]
            del self.texts[previous_count:]
            raise
        self._rebuild_maps()

    @staticmethod
    def _validated_embeddings(embeddings, count):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if count == 1 and embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.ndim != 2 or embeddings.shape[0] != count or embeddings.shape[1] == 0:
            raise ValueError("向量模型返回的条数或形状与输入批次不一致。")
        return embeddings

    def _rebuild_maps(self):
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: t for h, t in zip(self.hash_ids, self.texts)}
        self.text_to_hash_id = {t: h for t, h in zip(self.texts, self.hash_ids)}

    def _save_data(self):
        self._report("save_cache", f"正在保存{self._label()}向量缓存…")
        directory = os.path.dirname(os.path.abspath(self.db_filename))
        os.makedirs(directory, exist_ok=True)
        schema = pa.schema([
            ("hash_id", pa.string()), ("text", pa.string()),
            ("embedding", pa.list_(pa.float32())),
        ])
        fd, temporary = tempfile.mkstemp(prefix=".embedding-", suffix=".parquet", dir=directory)
        os.close(fd)
        try:
            with pq.ParquetWriter(temporary, schema) as writer:
                total = len(self.hash_ids)
                for start in range(0, total, self.CACHE_BATCH_SIZE):
                    end = min(start + self.CACHE_BATCH_SIZE, total)
                    matrix = self.embeddings[start:end]
                    dimension = matrix.shape[1]
                    offsets = np.arange(end - start + 1, dtype=np.int32) * dimension
                    vectors = pa.ListArray.from_arrays(
                        pa.array(offsets), pa.array(matrix.reshape(-1), type=pa.float32())
                    )
                    batch = pa.RecordBatch.from_arrays([
                        pa.array(self.hash_ids[start:end], type=pa.string()),
                        pa.array(self.texts[start:end], type=pa.string()), vectors,
                    ], schema=schema)
                    writer.write_batch(batch)
                    self._report(
                        "save_cache", f"正在保存{self._label()}向量缓存：{end}/{total} 条。", end, total
                    )
                    del batch, vectors
            os.replace(temporary, self.db_filename)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def get_hash_id_to_text(self):
        return deepcopy(self.hash_id_to_text)

    def encode_texts(self, texts):
        return self.embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=self.batch_size)

    def get_embeddings(self, hash_ids):
        if not hash_ids:
            return np.array([])
        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        return self.embeddings[indices]
