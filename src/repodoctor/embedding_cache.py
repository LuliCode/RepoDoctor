"""Content-addressed vectors shared across snapshots, with durable batch checkpoints.

Only vectors are shared. Passage membership, Java extraction and graphs remain
snapshot-specific. Compatible legacy Parquet files are imported read-only in
bounded batches, including files from interrupted builds without ready markers.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing

from .index_progress import emit_index_progress
from .knowledge import model_key
from .models import digest

RECIPE = "windowed-v1-normalized-float32"
COMPATIBLE = {"linearrag": ("java-v3-windowed", "java-v4-statements-windowed"),
              "vector": ("passage-v1-windowed",)}


class EmbeddingCache:
    BATCH_SIZE = 256

    def __init__(self, index, model, progress=None, reuse=True, dimension=None):
        self.root = index.db_path.parent
        self.model = model_key(model)
        self.path = self.root / "embedding-cache" / f"{RECIPE}-{digest(self.model)}.sqlite3"
        self.progress = progress
        self.reuse = reuse
        self.dimension = dimension
        self.prepared = set()

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            connection.execute("CREATE TABLE IF NOT EXISTS metadata (identity TEXT PRIMARY KEY)")
            identity = json.dumps([RECIPE, self.model], ensure_ascii=False)
            existing = connection.execute("SELECT identity FROM metadata").fetchone()
            if existing is not None and existing[0] != identity:
                raise ValueError("共享向量缓存的模型或编码版本不一致。")
            connection.execute("INSERT OR IGNORE INTO metadata VALUES (?)", (identity,))
            connection.execute("""CREATE TABLE IF NOT EXISTS vectors (
                namespace TEXT, hash_id TEXT, text TEXT NOT NULL,
                dimension INTEGER NOT NULL, vector BLOB NOT NULL,
                PRIMARY KEY (namespace, hash_id))""")
            connection.execute("""CREATE TABLE IF NOT EXISTS sources (
                path TEXT PRIMARY KEY, size INTEGER NOT NULL, modified INTEGER NOT NULL)""")
            connection.commit()
        except BaseException:
            connection.close()
            raise
        return connection

    def _sources(self, namespace):
        suffix = digest(self.model)[:12]
        for backend, versions in COMPATIBLE.items():
            if namespace != "passage" and backend == "vector":
                continue
            parent = self.root / backend
            if not parent.is_dir():
                continue
            for directory in sorted(parent.iterdir()):
                if not directory.is_dir():
                    continue
                version = next((v for v in versions if directory.name.endswith(f"-{v}-{suffix}")), None)
                if version is None:
                    continue
                marker = directory / "ready.json"
                if marker.exists():
                    try:
                        data = json.loads(marker.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    if data != {"snapshot": directory.name.split("-", 1)[0], "model": self.model,
                                "adapter_version": version}:
                        continue
                path = directory / f"{namespace}_embedding.parquet"
                if path.is_file() and path.stat().st_size:
                    yield path

    def _valid_vector(self, vector):
        import numpy as np

        return (vector.ndim == 1 and vector.size > 0
                and (self.dimension is None or vector.size == self.dimension) and np.isfinite(vector).all()
                and np.isclose(np.linalg.norm(vector), 1, atol=1e-3))

    def prepare(self, namespace):
        """Import old vectors once per source revision, without loading old graphs."""
        if namespace in self.prepared:
            return
        if not self.reuse:
            return
        import numpy as np
        import pyarrow.parquet as pq
        from ._vendor.linearrag.utils import compute_mdhash_id

        with closing(self._connect()) as connection:
            for path in self._sources(namespace):
                stat = path.stat()
                revision = (stat.st_size, stat.st_mtime_ns)
                if connection.execute("SELECT size, modified FROM sources WHERE path=?", (str(path),)).fetchone() == revision:
                    continue
                emit_index_progress(self.progress, "import_vectors", f"正在复用已有{namespace}向量：{path.parent.name}…")
                try:
                    with pq.ParquetFile(path, pre_buffer=False) as parquet:
                        total = parquet.metadata.num_rows
                        completed = 0
                        for batch in parquet.iter_batches(
                            batch_size=self.BATCH_SIZE, columns=["hash_id", "text", "embedding"], use_threads=False,
                        ):
                            rows = []
                            for row in batch.to_pylist():
                                text, hash_id = row["text"], row["hash_id"]
                                if (not isinstance(text, str)
                                        or hash_id != compute_mdhash_id(text, prefix=namespace + "-")):
                                    continue
                                vector = np.asarray(row["embedding"], dtype="<f4")
                                if self._valid_vector(vector):
                                    rows.append((namespace, hash_id, text, vector.size, vector.tobytes()))
                            with connection:
                                connection.executemany("INSERT OR IGNORE INTO vectors VALUES (?,?,?,?,?)", rows)
                            completed += batch.num_rows
                            emit_index_progress(self.progress, "import_vectors", "正在读取可复用的历史向量…",
                                                completed=completed, total=total, unit="条")
                    with connection:
                        connection.execute("INSERT OR REPLACE INTO sources VALUES (?,?,?)", (str(path), *revision))
                except (OSError, ValueError, TypeError) as exc:
                    emit_index_progress(self.progress, "cache_warning", f"历史向量文件无法完整复用，将计算缺失内容：{path.name}（{type(exc).__name__}）")
        self.prepared.add(namespace)

    def get(self, namespace, ids_and_texts):
        import numpy as np

        found = {}
        if not self.reuse or not ids_and_texts:
            return found
        with closing(self._connect()) as connection:
            # Callers use bounded batches, never a corpus-sized SQL IN clause.
            placeholders = ",".join("?" for _ in ids_and_texts)
            records = connection.execute(
                f"SELECT hash_id,text,dimension,vector FROM vectors WHERE namespace=? AND hash_id IN ({placeholders})",
                [namespace, *ids_and_texts],
            )
            for hash_id, text, dimension, blob in records:
                if text != ids_and_texts[hash_id] or len(blob) != dimension * 4:
                    continue
                vector = np.frombuffer(blob, dtype="<f4")
                if self._valid_vector(vector):
                    found[hash_id] = vector
        return found

    def put(self, namespace, ids, texts, embeddings):
        import numpy as np

        rows = []
        for hash_id, text, vector in zip(ids, texts, embeddings):
            vector = np.asarray(vector, dtype="<f4")
            if not self._valid_vector(vector):
                raise ValueError("向量模型返回了非有限数值、零向量或未归一化向量，未保存共享缓存。")
            rows.append((namespace, hash_id, text, vector.size, vector.tobytes()))
        with closing(self._connect()) as connection, connection:
            connection.executemany("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?,?)", rows)
