# RepoDoctor

RepoDoctor 是面向 Java 应用的代码故障定位 Agent。围绕错误描述或异常堆栈，逐步检索相关代码、阅读关键实现、追踪静态调用关系，将分散的故障线索整理为带源码证据的原因分析和修改建议。

## 项目亮点

**沿着故障线索逐步追踪原因。** 根据已有证据动态决定下一步调查方向，在检索、源码阅读与调用方查询之间按需推进，并对候选结论进行复核，让分析逐步覆盖触发条件、相关实现与潜在原因。

**以 LinearRAG 关联分散的代码线索。** 利用 Java 实体与源码片段之间的关系补充检索上下文，发现直接报错位置之外的相关实现；同时提供 BM25 关键词检索与向量检索作为回退选择，兼顾精确符号匹配与自然语言描述。

**让定位结论有源码依据。** 原因分析关联实际读取的文件、行号和代码片段，引用经过逐行校验，便于回到源码核对判断。证据不足时明确说明缺少的信息，让修改建议建立在可检查的依据之上。

**面向多模块应用与持续变化的代码库。** 统一索引多个模块的源码，结合静态调用关系分析跨模块线索；源码更新后复用兼容向量，支持增量构建与断点续建，减少重复准备知识库的成本。

## 安装与启动

要求 Python 3.11+。使用 RepoDoctor 无需安装 Java 或 Maven；需要对被分析项目具有读取权限，并能在其根目录写入 `.repodoctor/` 索引。

在仓库根目录执行：

```powershell
# Windows PowerShell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

```bash
# Linux / macOS
python3 -m venv .venv
.venv/bin/python -m pip install -e .
cp .env.example .env
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

已有 `.env` 时保留自己的配置。打开终端显示的本地地址（默认 `http://127.0.0.1:8501`），按页面操作：

1. 填写本机 Java 应用的绝对路径，可直接选择多模块应用根目录。
2. 选择检索方式，点击对应的“构建 / 刷新索引”。
3. 输入错误描述或异常堆栈，点击“先查看相关代码”或“开始定位”。

关键词检索无需向量模型或 API 密钥；“开始定位”需要配置诊断模型。

## 模型配置

在 `.env` 中填写实际可用的模型与密钥，修改后重启应用：

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
REPODOCTOR_MODEL=your-model-name
REPODOCTOR_BACKEND=lexical
```

诊断模型需支持 Chat Completions 的 `tools`、`tool_calls` 和 `max_completion_tokens` 参数。分析时，问题和相关源码片段会发送到所配置的模型服务；密钥只配置在本地 `.env`。

### 启用语义检索

安装以下依赖前，先激活虚拟环境（PowerShell：`.\.venv\Scripts\Activate.ps1`；Linux / macOS：`source .venv/bin/activate`），或将 `python` 替换为虚拟环境中 Python 的完整路径。

```bash
python -m pip install -e '.[vector]'
# 或安装 LinearRAG 依赖，同时支持向量 RAG
python -m pip install -e '.[linearrag]'
```

| 后端 | 检索方式 | 适用输入 |
| --- | --- | --- |
| `lexical` | BM25 关键词匹配 | 堆栈、类名、方法名、错误码 |
| `vector` | 代码片段向量相似度 | 自然语言问题、语义描述 |
| `linearrag` | Java 实体激活、关系传播和 PageRank，无有效实体种子时使用向量检索 | 需要关联实体和代码片段的问题 |

两种语义检索默认使用 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`，首次使用需要下载权重，也可将 `REPODOCTOR_EMBEDDING_MODEL` 设置为完整本地模型目录。

常用设置见 `.env.example`：

| 配置项 | 作用 |
| --- | --- |
| `REPODOCTOR_EMBEDDING_DEVICE` | `auto`、`cpu`、`cuda` 或 `cuda:N`；`auto` 在 CUDA 可用时使用 GPU |
| `REPODOCTOR_EMBEDDING_BATCH_SIZE` | 编码批量；内存或显存不足时调小 |
| `HF_HOME` | 向量模型下载缓存目录 |
| `REPODOCTOR_EXACT_RERANK` | 开启后优先展示精确匹配堆栈或符号的结果，无需重建索引 |
| `REPODOCTOR_MAX_TOOL_CALLS` / `REPODOCTOR_MAX_SECONDS` | 单次分析的工具调用次数与耗时上限 |
| `REPODOCTOR_MAX_TOKENS` / `REPODOCTOR_MAX_CONTEXT_TOKENS` | 累计 Token 预算与单轮上下文上限 |

## 命令行

使用虚拟环境中的 Python，将 `/path/to/java-app` 替换为实际项目路径：

```bash
python -m repodoctor index /path/to/java-app --backend lexical
python -m repodoctor search /path/to/java-app "OrderService.createOrder 空指针" --backend lexical --top-k 5
python -m repodoctor read /path/to/java-app src/main/java/com/example/OrderService.java 1 80
python -m repodoctor callers /path/to/java-app "com.example.OrderService#createOrder(String)"
python -m repodoctor diagnose /path/to/java-app "创建订单时出现空指针" --backend lexical --mode agent
```

语义检索先用对应后端构建索引，例如 `python -m repodoctor index /path/to/java-app --backend linearrag`；搜索和诊断时使用相同的 `--backend`。`--embedding-model` 可指定向量模型，`--exact-rerank` / `--no-exact-rerank` 控制最终精确重排。

`index --include-tests` 将测试源码纳入阅读范围；`--docs` 可指定文档相对路径，不带路径时排除文档。源码变化后重新运行 `index` 刷新；需要完全重建时使用 `--force`。`diagnose --mode rag` 使用单次检索分析，`--output result.json` 可保存结果。

## MCP 接入

先构建索引，再由支持 stdio 的 MCP 客户端启动：

```bash
python -m repodoctor.mcp_server --root /path/to/java-app --backend lexical
```

客户端的启动命令应指向已安装依赖的 Python，参数使用上方 `-m` 之后的内容。服务器提供三个只读工具：`search_code`（检索代码）、`read_code`（按行阅读）、`find_callers`（查询静态调用方）。MCP Server 本身无需诊断模型密钥。

## 项目结构与范围

```text
app.py                    Streamlit 页面
src/repodoctor/            Agent、索引、检索、模型适配和 MCP 服务
  _vendor/linearrag/       使用中的上游组件及许可证
pyproject.toml            包配置、依赖和命令入口
.env.example              配置模板
.gitignore                本地环境与生成文件的忽略规则
LICENSE                   项目许可证
README.md                 使用说明
```

索引递归读取 Java 源码，默认纳入根目录及发现的源码模块中的 `README.md`、`docs/`，跳过构建产物、依赖、缓存和子目录链接。AST 解析以 Java 8 语法为基础，调用关系属于静态候选；运行时依赖、数据库状态和并发问题可能需要额外日志或环境信息。

RepoDoctor 读取源码并给出建议，不执行被分析项目、不自动修改代码。`.env`、索引、模型权重和运行缓存不属于发布内容。

## 许可证与来源

项目采用 GPL-3.0-only，完整条款见 [LICENSE](LICENSE)。LinearRAG 组件源自 [DEEP-PolyU/LinearRAG](https://github.com/DEEP-PolyU/LinearRAG)，固定上游提交为 `bcc94e66c221f798801255efba09311d6fbcd8d6`；来源与本地适配说明保留在组件包的 `__init__.py` 中，原许可证保留在 `LICENSE.txt`。
