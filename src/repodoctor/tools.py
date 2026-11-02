from .index import Index, ReadRangeError
from .retrieval import Search


class CodeTools:
    def __init__(self, index: Index, backend="lexical", embedding_model=None, *, exact_rerank=False):
        self.index = index
        self.searcher = Search(index, backend, embedding_model, exact_rerank=exact_rerank)

    def search_code(self, query: str, top_k: int = 5) -> dict:
        """搜索 Java 方法、类、字段、文档，按服务所选检索方式排序；长方法可再 read_code。"""
        return self.searcher.search(query, top_k)

    def read_code(self, file_path: str, start_line: int, end_line: int) -> dict:
        """按文件真实行号读取源码。结束行自动限于文件末尾和单次 160 行；返回 total_lines、eof 及分页提示。搜索已包含的源码不必重复读取。"""
        try:
            return self.index.read(file_path, start_line, end_line, clamp=True)
        except ReadRangeError as exc:
            return {
                "error": str(exc),
                "file_path": exc.file_path,
                "total_lines": exc.total_lines,
                "requested_start_line": start_line,
                "requested_end_line": end_line,
                "valid_range": {"start_line": 1, "end_line": exc.total_lines} if exc.total_lines else None,
                "hint": "不要把其他文件、日志输出行号或片段内相对行号用于本文件；先核对路径和真实行号。",
            }

    def find_callers(self, symbol: str) -> dict:
        """以完整 类#方法(参数类型) 符号查询静态调用方；候选和未知原因单独返回。"""
        return self.index.callers(symbol)
