from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import closing, contextmanager
from pathlib import Path

from .java import parse_java, resolve_calls
from .index_progress import emit_index_progress
from .models import Passage, digest
from .symbols import code_identifiers

EXCLUDED = {
    "target",
    "build",
    "out",
    "dist",
    ".git",
    ".idea",
    ".gradle",
    "node_modules",
    ".repodoctor",
    ".venv",
    ".runtime",
    "vendor",
    "cases",
    "answers",
    "patches",
}
TEST_DIRS = {"test", "tests", "testFixtures", "integrationTest"}
INDEX_VERSION = 4


def is_test_path(relative: Path) -> bool:
    parts = relative.parts
    for offset, name in enumerate(parts):
        # A package called "test" inside production sources is still production.
        if parts[offset : offset + 3] == ("src", "main", "java"):
            return False
        if name in TEST_DIRS:
            return True
    return False


class IndexErrorBase(ValueError):
    pass


class StaleIndexError(IndexErrorBase):
    pass


class ReadRangeError(IndexErrorBase):
    def __init__(self, file_path, total_lines):
        super().__init__(f"读取起始行超过文件总行数 {total_lines}，请检查文件路径或使用有效行号。")
        self.file_path = file_path
        self.total_lines = total_lines


def safe_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise IndexErrorBase("路径越过所选项目目录")
    return candidate


class Index:
    def __init__(self, root: str | Path, db_path: str | Path | None = None):
        self.root = Path(root).resolve()
        self.db_path = Path(db_path).resolve() if db_path else self.root / ".repodoctor/index.sqlite3"

    @contextmanager
    def connect(self):
        if not self.db_path.is_file():
            raise IndexErrorBase("尚未构建索引，请先运行 repodoctor index")
        conn = sqlite3.connect(self.db_path.as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def metadata(self) -> dict:
        with self.connect() as conn:
            data = {r["key"]: json.loads(r["value"]) for r in conn.execute("SELECT * FROM metadata")}
        if Path(data["root"]).resolve() != self.root:
            raise IndexErrorBase("该索引属于另一源码目录，请重新构建")
        if data.get("schema_version") != INDEX_VERSION:
            raise StaleIndexError("索引解析版本已更新，请刷新索引后重新定位（repodoctor index）。")
        return data

    def validate_project(self):
        if not self.root.is_dir():
            raise IndexErrorBase(f"项目目录不存在或不是文件夹：{self.root}")

    def source_files(self, include_tests=False):
        """Discover Java across the application, pruning outputs and directory links."""

        def on_error(error):
            raise error

        for directory, dirs, files in os.walk(self.root, followlinks=False, onerror=on_error):
            dirs[:] = sorted(
                name
                for name in dirs
                if name not in EXCLUDED
                and (include_tests or not is_test_path((Path(directory) / name).relative_to(self.root)))
                and not (Path(directory) / name).is_symlink()
                and not getattr(Path(directory) / name, "is_junction", lambda: False)()
            )
            for name in sorted(files):
                if name.endswith(".java"):
                    path = Path(directory) / name
                    relative = path.relative_to(self.root)
                    resolved = safe_path(self.root, relative.as_posix())
                    resolved_parts = resolved.relative_to(self.root).parts
                    if not EXCLUDED.intersection(resolved_parts) and (
                        include_tests or not is_test_path(resolved.relative_to(self.root).parent)
                    ):
                        yield path

    def source_module(self, path):
        relative = path.relative_to(self.root)
        if "src" in relative.parts:
            return Path(*relative.parts[: relative.parts.index("src")]).as_posix()
        # Nonstandard source layout: use the closest build descriptor if present.
        for parent in path.parents:
            if parent == self.root:
                break
            if any((parent / name).is_file() for name in ("pom.xml", "build.gradle", "build.gradle.kts")):
                return parent.relative_to(self.root).as_posix()
        return "."

    def document_files(self, modules, docs=None):
        default_docs = [
            str(Path(module) / name) for module in {".", *modules} for name in ("README.md", "docs")
        ]
        for doc in default_docs if docs is None else docs:
            path = safe_path(self.root, doc)
            if path.is_dir():
                candidates = (item for pattern in ("*.md", "*.txt") for item in path.rglob(pattern))
            elif path.is_file() and path.suffix.lower() in {".md", ".txt"}:
                candidates = [path]
            else:
                continue
            for candidate in candidates:
                relative = candidate.relative_to(self.root)
                resolved = safe_path(self.root, relative.as_posix())
                if not EXCLUDED.intersection(relative.parts) and not EXCLUDED.intersection(
                    resolved.relative_to(self.root).parts
                ):
                    yield candidate

    def build(self, include_tests=False, docs: list[str] | None = None, progress=None) -> dict:
        self.validate_project()
        emit_index_progress(progress, "scan", "正在扫描源码和文档…", completed=0, unit="个文件")
        paths = set()
        for path in self.source_files(include_tests):
            paths.add(path)
            emit_index_progress(
                progress, "scan", "正在扫描源码和文档…", completed=len(paths), unit="个文件",
                current_file=path.relative_to(self.root).as_posix(),
            )
        if not paths:
            raise IndexErrorBase("所选目录及子目录没有可索引的 Java 源码，请选择应用源码目录。")
        modules = sorted({self.source_module(path) for path in paths})
        for path in self.document_files(modules, docs):
            paths.add(path)
            emit_index_progress(
                progress, "scan", "正在扫描源码和文档…", completed=len(paths), unit="个文件",
                current_file=path.relative_to(self.root).as_posix(),
            )
        selected_paths = []
        for path in sorted(paths):
            relative = path.relative_to(self.root)
            resolved = safe_path(self.root, relative.as_posix())
            if not EXCLUDED.intersection(relative.parts) and not EXCLUDED.intersection(
                resolved.relative_to(self.root).parts
            ):
                selected_paths.append(path)
        snapshot = uuid.uuid4().hex
        files, passages, classes, methods, calls, warnings = {}, [], {}, [], [], []
        total_files = len(selected_paths)
        for completed, path in enumerate(selected_paths):
            relative = path.relative_to(self.root).as_posix()
            emit_index_progress(
                progress, "parse", "正在解析源码、分块和提取符号…", completed=completed,
                total=total_files, unit="个文件", current_file=relative,
            )
            resolved = safe_path(self.root, relative)
            if EXCLUDED.intersection(Path(relative).parts):
                continue
            # Resolve again to prevent symlink aliases from indexing an excluded answer directory.
            if EXCLUDED.intersection(resolved.relative_to(self.root).parts):
                continue
            raw = resolved.read_bytes()
            files[relative] = digest(raw)
            try:
                source = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                warnings.append(f"{relative}: 非 UTF-8 文件，已跳过；请转为 UTF-8 后刷新")
                continue
            if path.suffix == ".java":
                parsed = parse_java(self.root.name, relative, source)
                passages.extend(parsed.passages)
                for owner, info in parsed.classes.items():
                    if owner in classes:
                        info["ambiguous_definition"] = True
                        warnings.append(f"{owner}: 存在跨文件同名类，调用关系按不确定候选处理。")
                    classes[owner] = info
                methods.extend(parsed.methods)
                calls.extend(parsed.calls)
                warnings.extend(parsed.warnings)
            else:
                lines = source.splitlines()
                starts = sorted({1, *[i for i, line in enumerate(lines, 1) if line.startswith("#")]})
                for start, stop in zip(starts, starts[1:] + [len(lines) + 1]):
                    for a in range(start, stop, 120):
                        b = min(a + 119, stop - 1)
                        content = "\n".join(lines[a - 1 : b])
                        passages.append(
                            Passage(
                                digest(f"{relative}:{a}:{b}"),
                                self.root.name,
                                relative,
                                "",
                                "",
                                "document",
                                a,
                                b,
                                content,
                                digest(content),
                                entities=code_identifiers(content),
                            )
                        )
        emit_index_progress(
            progress, "parse", "源码和文档解析完成", completed=total_files,
            total=total_files, unit="个文件",
        )
        if not passages:
            raise IndexErrorBase("没有可索引内容")
        for passage in passages:
            passage.snapshot = snapshot
        meta = {
            "schema_version": INDEX_VERSION,
            "root": str(self.root),
            "snapshot": snapshot,
            "files": len(files),
            "passages": len(passages),
            "warnings": warnings,
            "include_tests": include_tests,
            "docs": docs,
            "modules": modules,
            "java_files": sum(path.suffix == ".java" for path in paths),
        }
        emit_index_progress(progress, "resolve_calls", "正在解析静态调用关系…")
        resolved_calls = resolve_calls(classes, methods, calls)
        emit_index_progress(progress, "save_source", "正在保存源码索引…")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # All tables and the snapshot change in one transaction; readers never see a partial index.
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            for table in ("metadata", "files", "passages", "calls"):
                conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("CREATE TABLE files (path TEXT PRIMARY KEY, hash TEXT NOT NULL)")
            conn.execute("CREATE TABLE passages (id TEXT PRIMARY KEY, data TEXT NOT NULL)")
            conn.execute("CREATE TABLE calls (data TEXT NOT NULL)")
            conn.executemany(
                "INSERT INTO metadata VALUES (?,?)", [(k, json.dumps(v)) for k, v in meta.items()]
            )
            conn.executemany("INSERT INTO files VALUES (?,?)", files.items())
            conn.executemany(
                "INSERT INTO passages VALUES (?,?)",
                [(p.id, json.dumps(p.to_dict(), ensure_ascii=False)) for p in passages],
            )
            conn.executemany(
                "INSERT INTO calls VALUES (?)",
                [(json.dumps(c),) for c in resolved_calls],
            )
        return meta

    def passages(self) -> list[Passage]:
        self.metadata()
        with self.connect() as conn:
            return [Passage(**json.loads(r["data"])) for r in conn.execute("SELECT data FROM passages")]

    def check_fresh(self, paths: list[str] | None = None):
        metadata = self.metadata()
        with self.connect() as conn:
            hashes = {r["path"]: r["hash"] for r in conn.execute("SELECT * FROM files")}
            passage_count = conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
            conn.execute("SELECT data FROM calls LIMIT 1").fetchone()
        if len(hashes) != metadata["files"] or passage_count != metadata["passages"]:
            raise IndexErrorBase("索引数据不完整，请重新构建索引。")
        for relative in hashes if paths is None else paths:
            if relative not in hashes:
                raise IndexErrorBase(f"文件不在当前索引内：{relative}")
            path = safe_path(self.root, relative)
            if not path.is_file() or digest(path.read_bytes()) != hashes[relative]:
                raise StaleIndexError(f"源码已变化，请刷新索引后重新定位：{relative}")
        if paths is None:
            for path in self.source_files(metadata["include_tests"]):
                relative = path.relative_to(self.root).as_posix()
                if relative not in hashes:
                    raise StaleIndexError(f"发现新增 Java 文件，请刷新索引：{relative}")
            for path in self.document_files(metadata.get("modules", []), metadata.get("docs")):
                relative = path.relative_to(self.root).as_posix()
                if relative not in hashes:
                    raise StaleIndexError(f"发现新增文档，请刷新索引：{relative}")

    def read(self, file_path: str, start_line: int, end_line: int, *, clamp=False) -> dict:
        if start_line < 1 or end_line < start_line or (not clamp and end_line - start_line + 1 > 160):
            raise IndexErrorBase("行号范围无效；每次最多读取 160 行")
        path = safe_path(self.root, file_path)
        relative = path.relative_to(self.root).as_posix()
        self.metadata()
        with self.connect() as conn:
            record = conn.execute("SELECT hash FROM files WHERE path = ?", (relative,)).fetchone()
        if record is None:
            raise IndexErrorBase(f"文件不在当前索引内：{relative}")
        if not path.is_file():
            raise StaleIndexError(f"源码已删除，请刷新索引：{relative}")
        raw = path.read_bytes()
        if digest(raw) != record["hash"]:
            raise StaleIndexError(f"源码已变化，请刷新索引后重新定位：{relative}")
        lines = raw.decode("utf-8-sig").splitlines()
        if start_line > len(lines):
            raise ReadRangeError(relative, len(lines))
        requested_end = end_line
        if clamp:
            end_line = min(end_line, len(lines), start_line + 159)
        if end_line > len(lines):
            raise IndexErrorBase(f"行号超过文件总行数 {len(lines)}")
        content = "\n".join(lines[start_line - 1 : end_line])
        result = {
            "file_path": relative,
            "start_line": start_line,
            "end_line": end_line,
            "content": content,
            "numbered_content": "\n".join(f"{i}: {lines[i - 1]}" for i in range(start_line, end_line + 1)),
            "content_hash": digest(content),
            "snapshot": self.metadata()["snapshot"],
            "total_lines": len(lines),
            "eof": end_line == len(lines),
            "truncated": end_line < min(requested_end, len(lines)),
        }
        if end_line != requested_end:
            result["requested_end_line"] = requested_end
            result["notice"] = (
                f"已按文件长度和单次 160 行限制返回 {start_line}–{end_line} 行；文件共 {len(lines)} 行。"
            )
        if result["truncated"]:
            result["next_start_line"] = end_line + 1
        return result

    def callers(self, symbol: str) -> dict:
        self.check_fresh()
        if symbol not in {p.symbol for p in self.passages() if p.symbol}:
            raise IndexErrorBase("方法符号不存在；请先搜索完整的 类#方法(参数类型) 符号键")
        with self.connect() as conn:
            calls = [json.loads(r["data"]) for r in conn.execute("SELECT data FROM calls")]
        confirmed = [c for c in calls if c["target"] == symbol]
        unknown = [c for c in calls if not c["target"] and symbol in c["candidates"]]
        return {
            "symbol": symbol,
            "callers": confirmed[:50],
            "unresolved_candidates": unknown[:50],
            "truncated": len(confirmed) > 50 or len(unknown) > 50,
            "snapshot": self.metadata()["snapshot"],
            "limitation": "仅有限静态解析；无结果不等于没有运行时调用方。",
        }
