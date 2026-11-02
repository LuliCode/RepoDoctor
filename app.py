"""Start with: python -m streamlit run app.py"""

import asyncio
import json
import os

import streamlit as st
from dotenv import load_dotenv

from repodoctor import ENGINE_REVISION
from repodoctor.errors import error_message
from repodoctor.index_build_process import build_knowledge_isolated
from repodoctor.index_progress import run_index_build
from repodoctor.knowledge import BACKENDS, SEMANTIC_BACKENDS, build_knowledge, knowledge_status
from repodoctor.models import Budget
from repodoctor.reporting import analysis_incomplete
from repodoctor.search_preview import search_preview
from repodoctor.service import diagnose
from repodoctor.retrieval_settings import default_exact_rerank

load_dotenv()
st.set_page_config(page_title="RepoDoctor · Java 故障定位", page_icon="🔎", layout="wide")
st.title("RepoDoctor")
st.caption("选择项目 → 构建代码知识库 → 输入问题并定位")

INDEX_LABELS = {"lexical": "离线关键词索引", "linearrag": "LinearRAG 语义索引", "vector": "向量 RAG 索引"}
SEARCH_LABELS = {"lexical": "离线关键词检索", "linearrag": "LinearRAG 语义检索", "vector": "向量 RAG 检索"}


def request_index_build(build_backend):
    if st.session_state.get("index_build_request"):
        return
    st.session_state["index_build_request"] = build_backend
    st.session_state.pop("index_build_result", None)
    st.session_state.pop("report", None)


index_building = bool(st.session_state.get("index_build_request"))

with st.sidebar:
    st.caption("诊断引擎版本：" + ENGINE_REVISION)
    st.subheader("1. 选择源码项目")
    entered_root = st.text_input(
        "本机 Java 应用目录",
        "",
        key="project_root",
        disabled=index_building,
        placeholder="填写要分析的 Java 应用绝对路径",
        help="可直接选择包含 web、rpc、service 等模块的应用根目录；递归发现 Java 源码，不要求根目录有 pom.xml 或 src/main/java。",
    )
    root = entered_root.strip().strip('"')
    backend = st.selectbox(
        "检索方式",
        BACKENDS,
        key="backend",
        disabled=index_building,
        index=BACKENDS.index(os.getenv("REPODOCTOR_BACKEND", "lexical"))
        if os.getenv("REPODOCTOR_BACKEND", "lexical") in BACKENDS else 0,
        format_func=lambda x: SEARCH_LABELS[x],
    )
    embedding = os.getenv("REPODOCTOR_EMBEDDING_MODEL")
    exact_rerank = st.checkbox(
        "启用最终精确重排", value=default_exact_rerank(), key="exact_rerank", disabled=index_building,
        help="开启后优先展示明确匹配堆栈、类名或方法名的代码；关闭时按所选后端原生分数排序。无需重建索引。",
    )
    if st.session_state.get("scope_project") != root:
        prior = knowledge_status(root, "lexical")
        st.session_state["include_tests"] = bool((prior["metadata"] or {}).get("include_tests", False))
        st.session_state["scope_project"] = root
    include_tests = st.checkbox(
        "同时索引测试源码（可选）",
        key="include_tests",
        disabled=index_building,
        help="额外读取各模块的 test、tests、testFixtures、integrationTest 目录，帮助理解断言和边界条件。不会执行测试；不勾选也能构建索引。",
    )
    st.caption("递归读取各模块 Java 源码与说明文档，跳过构建产物和依赖目录。测试源码是可选材料。")

    st.subheader("2. 准备代码知识库")
    st.caption("三种检索共用源码片段。向量 RAG 准备代码向量；LinearRAG 额外准备实体与关系图。兼容向量可复用。")
    st.button(
        "构建 / 刷新离线关键词索引", key="build_index", use_container_width=True,
        disabled=index_building, on_click=request_index_build, args=("lexical",),
    )
    st.button(
        "构建 / 刷新 LinearRAG 语义索引", key="build_linear_index", use_container_width=True,
        disabled=index_building, on_click=request_index_build, args=("linearrag",),
    )
    st.button(
        "构建 / 刷新向量 RAG 索引", key="build_vector_index", use_container_width=True,
        disabled=index_building, on_click=request_index_build, args=("vector",),
    )
    build_backend = st.session_state.get("index_build_request")
    if build_backend:
        build_label = INDEX_LABELS[build_backend]
        build_ui = {"last_stage": None, "stages": [], "last_event": None}
        try:
            with st.status(f"正在构建{build_label}…", expanded=True) as build_status:
                stage_text = st.empty()
                stage_bar = st.empty()
                current_file = st.empty()
                timing = st.empty()

                def show_build_progress(event):
                    build_ui["last_event"] = event
                    if event["stage"] != build_ui["last_stage"]:
                        build_status.write(event["message"])
                        build_ui["stages"].append(event["message"])
                        build_ui["last_stage"] = event["stage"]
                    completed, total = event.get("completed"), event.get("total")
                    text = event["message"]
                    if event["stage"] == "complete":
                        stage_bar.progress(1.0, text="索引构建完成")
                    elif total:
                        text += f" · {completed}/{total} {event.get('unit', '')}"
                        stage_bar.progress(completed / total, text="本阶段进度")
                    else:
                        stage_bar.empty()
                        if completed is not None:
                            action = "已发现" if event["stage"] == "scan" else "已完成"
                            text += f" · {action} {completed} {event.get('unit', '')}"
                    stage_text.write(text)
                    if event.get("current_file"):
                        current_file.caption("当前文件：" + event["current_file"])
                    else:
                        current_file.empty()
                    timing.caption(
                        f"已用时 {event['elapsed_seconds']:.1f} 秒 · 当前阶段 {event['phase_seconds']:.1f} 秒"
                    )

                try:
                    build = build_knowledge_isolated if build_backend in SEMANTIC_BACKENDS else build_knowledge
                    meta = run_index_build(
                        lambda callback: build(
                            root, build_backend, embedding, include_tests, progress=callback
                        ),
                        show_build_progress,
                    )
                except Exception:
                    build_status.update(label="索引构建失败", state="error", expanded=True)
                    raise
                build_status.update(label=f"{build_label}构建完成", state="complete", expanded=True)
            st.session_state["index_build_result"] = {
                "root": root, "include_tests": include_tests, "label": build_label,
                "metadata": meta, "stages": build_ui["stages"], "event": build_ui["last_event"],
            }
        except Exception as exc:
            st.session_state["index_build_result"] = {
                "root": root, "include_tests": include_tests, "label": build_label,
                "error": error_message(exc), "stages": build_ui["stages"], "event": build_ui["last_event"],
            }
        finally:
            st.session_state.pop("index_build_request", None)
        st.rerun()
    build_result = st.session_state.get("index_build_result")
    if build_result and build_result["root"] == root and build_result["include_tests"] == include_tests:
        failed = "error" in build_result
        suffix = "构建失败" if failed else "构建完成"
        with st.status(
            build_result["label"] + suffix, state="error" if failed else "complete", expanded=failed
        ):
            for message in build_result["stages"]:
                st.write(message)
            event = build_result["event"]
            if event:
                if failed:
                    st.write(event["message"])
                    if event.get("total") and event.get("completed", 0) < event["total"]:
                        st.progress(event["completed"] / event["total"], text="失败前的本阶段进度")
                    if event.get("current_file"):
                        st.caption("当前文件：" + event["current_file"])
                st.caption(f"已用时 {event['elapsed_seconds']:.1f} 秒")
            if failed:
                st.error("索引构建失败：" + build_result["error"])
            else:
                st.progress(1.0, text="索引构建完成")
                built = build_result["metadata"]
                st.success(f"{built['files']} 个文件 · {built['passages']} 个片段")
                for warning in built["warnings"]:
                    st.warning(warning)
    statuses = {
        mode: knowledge_status(root, mode, embedding, include_tests) for mode in INDEX_LABELS
    }
    for mode, status in statuses.items():
        if status["ready"]:
            st.success(INDEX_LABELS[mode] + "：已就绪")
        else:
            st.caption(INDEX_LABELS[mode] + "：" + status["message"])
    readiness = statuses[backend]
    if readiness["ready"]:
        st.success(readiness["message"])
    else:
        st.warning(readiness["message"])
    meta = readiness["metadata"]
    if meta:
        scope = "业务源码 + 测试源码 + 文档" if meta["include_tests"] else "业务源码 + 文档"
        st.caption(
            f"已建索引：{meta['files']} 个文件 · {meta['passages']} 个片段\n\n范围：{scope}\n\n快照：{meta['snapshot'][:12]}"
        )
        st.caption("已发现源码模块：" + "、".join(meta.get("modules", ["."])))
    with st.expander("分析设置"):
        max_calls = st.number_input("工具调用上限", 1, 30, int(os.getenv("REPODOCTOR_MAX_TOOL_CALLS", "8")))
        max_seconds = st.number_input(
            "耗时上限（秒）", 10, 600, int(os.getenv("REPODOCTOR_MAX_SECONDS", "120"))
        )
        max_tokens = st.number_input(
            "累计 Token 预算",
            key="max_tokens",
            min_value=1000,
            value=max(1000, int(os.getenv("REPODOCTOR_MAX_TOKENS", "24000"))),
            step=1000,
            help="所有模型轮次输入与输出的累计用量。长日志、跨模块调查可适当提高；增加预算可能增加 API 用量。",
        )
        max_context = st.number_input(
            "单轮上下文上限（估算 Token）",
            key="max_context_tokens",
            min_value=1000,
            value=max(1000, int(os.getenv("REPODOCTOR_MAX_CONTEXT_TOKENS", "12000"))),
            step=1000,
            help="发送前压缩重复元数据和较早工具结果；请按实际模型的上下文容量设置，并为输出预留空间。",
        )
        mode = st.radio(
            "定位方式",
            ["agent", "rag"],
            format_func=lambda x: "按需多轮分析" if x == "agent" else "单次检索分析",
        )
    st.caption("模型和密钥通过项目 .env 配置。检索和阅读代码无需模型密钥。")

st.subheader("3. 输入你的问题")
st.session_state.setdefault("problem_input", "")
problem = st.text_area(
    "错误描述 / Java 异常堆栈",
    key="problem_input",
    height=220,
    placeholder="输入问题、触发条件、预期行为，或粘贴 Java 异常堆栈。",
)
can_run = readiness["ready"] and bool(problem.strip())
if not readiness["ready"]:
    st.info("请先在左侧为当前项目完成索引构建，再检索或开始定位。")
elif not problem.strip():
    st.caption("请输入你的错误描述或异常堆栈，然后开始定位。")
left, right = st.columns(2)


def clear_previous_run():
    for key in ("report", "report_context"):
        st.session_state.pop(key, None)


search_clicked = left.button(
    "先查看相关代码",
    key="search",
    on_click=clear_previous_run,
    disabled=not can_run,
    use_container_width=True,
)
diagnose_clicked = right.button(
    "开始定位",
    key="diagnose",
    on_click=clear_previous_run,
    disabled=not can_run,
    type="primary",
    use_container_width=True,
)
if backend == "lexical":
    st.caption("当前使用关键词检索；加入类名、方法名或异常堆栈有助于定位。")
elif backend == "vector":
    st.caption("当前使用向量 RAG，按查询与代码片段的语义相似度排序。")
st.caption("最终精确重排：" + ("已开启" if exact_rerank else "已关闭，按检索后端原生排序"))

context = (
    root,
    backend,
    embedding,
    exact_rerank,
    include_tests,
    (meta or {}).get("snapshot"),
    problem,
    mode,
    max_tokens,
    max_context,
    max_calls,
    max_seconds,
)
if st.session_state.get("report_context") != context:
    st.session_state.pop("report", None)


def render_report(result):
    answer = result["diagnosis"]
    metrics = result["metrics"]
    stop = metrics.get("stop_reason", "")
    limits = {
        "tool_budget": "工具调用次数已达上限",
        "token_budget": "Token 预算或上下文容量不足",
        "token_budget_synthesis": "Token 预算接近上限",
        "model_budget": "模型决策次数或 Token 预算已达上限",
        "time_budget": "分析时间已达上限",
    }
    if stop in limits:
        if (
            metrics.get("final_summary_generated")
            and answer["status"] == "likely"
            and not result["validation_warnings"]
        ):
            st.info(limits[stop] + "；已根据已读源码生成结论并通过引用校验。详细用量见下方调用记录。")
        else:
            st.warning(
                limits[stop]
                + "，本轮尚未形成完整结论。请结合下方证据和缺失材料判断是否需要继续调查；可在左侧调整对应上限。"
            )
    elif stop == "no_new_information":
        st.info("连续两轮没有获得新的代码证据，已停止重复查询并总结现有发现。")
    elif stop in {"invalid_model_output", "model_error"}:
        st.error(
            {
                "invalid_model_output": "模型结果格式校验失败；这不表示代码证据不足。",
                "model_error": "模型调用失败，本轮分析未完成；具体类别与处理建议见下方。",
            }[stop]
        )
    incomplete = analysis_incomplete(result)
    if "max_tool_calls" in metrics:
        st.caption(
            f"工具执行 {metrics['tool_calls']}/{metrics['max_tool_calls']} 次 · 缓存复用 {metrics.get('cached_tool_results', 0)} 次 · 错误返回 {metrics.get('tool_errors', 0)} 次"
        )
    st.subheader(
        "分析未完成" if incomplete else "定位结果" if answer["status"] == "likely" else "当前信息不足"
    )
    if incomplete:
        st.caption(
            f"停止原因：{stop or '旧报告未记录'} · 模型轮次：{metrics.get('model_decisions', '未记录')}"
            f" · 累计 Token：{metrics.get('tokens', '未记录')}"
        )
        for detail in answer["missing_information"]:
            st.warning(detail)
    st.write(answer["cause"])
    for loc, code in zip(answer["locations"], result["evidence"]):
        st.markdown(f"**{loc['file_path']}:{loc['start_line']}–{loc['end_line']}**")
        st.write(loc["explanation"])
        st.code(code["numbered_content"], language="java")
    if answer["suggestions"]:
        st.subheader("修改建议")
        for suggestion in answer["suggestions"]:
            st.write("• " + suggestion)
    if answer["missing_information"]:
        st.subheader("未完成原因与待确认项" if incomplete else "需要补充的信息")
        for missing in answer["missing_information"]:
            st.write("• " + missing)
    for warning in result["validation_warnings"]:
        st.warning(warning)
    with st.expander("调用记录与统计"):
        st.caption("报告引擎版本：" + result.get("engine_revision", "旧版报告未记录"))
        st.json(result["metrics"])
        if result.get("model_calls"):
            st.write("模型轮次记录（结束原因及用量，不包含原始回答或内部推理）")
            st.json(result["model_calls"])
        if result.get("completion_checks"):
            st.write("结论复核记录")
            st.json(result["completion_checks"])
        st.json(result["trace"])
        if result.get("progress"):
            st.write("进度记录")
            st.json(result["progress"])
    st.download_button(
        "下载定位报告",
        json.dumps(result, ensure_ascii=False, indent=2),
        "repodoctor-report.json",
        "application/json",
    )


# Clear the previous result container before any blocking analysis work.
result_slot = st.empty()
result_slot.empty()
with result_slot.container():
    if search_clicked:
        try:
            result = asyncio.run(search_preview(root, problem, backend, embedding, exact_rerank=exact_rerank))
            if not result["hits"]:
                st.info("没有匹配到代码，请补充方法名或异常堆栈。")
            for hit in result["hits"]:
                with st.expander(
                    f"{hit['file_path']}:{hit['start_line']}–{hit['available_end_line']}", expanded=True
                ):
                    st.code(hit["content"], language="java")
            st.caption("检索路径：" + " → ".join(result["retrieval_path"]))
        except Exception as exc:
            st.error(error_message(exc))
    if diagnose_clicked:
        st.session_state.pop("report", None)
        if not os.getenv("OPENAI_API_KEY") or not os.getenv("REPODOCTOR_MODEL"):
            st.error("请先在 .env 配置 OPENAI_API_KEY 和 REPODOCTOR_MODEL，再开始模型定位。")
        else:
            try:
                with st.status("正在分析…", expanded=True) as status:
                    timing = st.empty()
                    activity = st.empty()
                    activities = []

                    def show_progress(event):
                        status.update(label=event["message"])
                        elapsed = event.get("elapsed_seconds", 0)
                        waiting = (
                            f" · 当前阶段等待 {event['phase_seconds']:g} 秒"
                            if event["event"] == "waiting"
                            else ""
                        )
                        timing.caption(f"本轮已用时 {elapsed:g} 秒{waiting}")
                        if event["event"] != "waiting":
                            activities.append(f"{elapsed:g}s · {event['message']}")
                            activity.text("\n".join(activities[-12:]))

                    def show_trace(trace):
                        with st.expander(f"{trace['tool']} · {trace['summary']}"):
                            st.json(
                                {
                                    "目的": trace["purpose"],
                                    "参数": trace["arguments"],
                                    "耗时毫秒": trace["duration_ms"],
                                    "结果": trace["result"],
                                }
                            )

                    result = asyncio.run(
                        diagnose(
                            root,
                            problem,
                            backend,
                            embedding,
                            mode,
                            Budget(
                                int(max_calls),
                                int(max_seconds),
                                int(max_tokens),
                                max_context_tokens=int(max_context),
                            ),
                            show_trace,
                            show_progress,
                            exact_rerank=exact_rerank,
                        )
                    )
                    incomplete = analysis_incomplete(result)
                    status.update(
                        label="分析未完成，请查看下方原因" if incomplete else "分析结束",
                        state="error" if incomplete else "complete",
                        expanded=incomplete,
                    )
                st.session_state["report"] = result
                st.session_state["report_context"] = context
            except Exception as exc:
                status.update(label="本轮分析未完成", state="error", expanded=True)
                st.error("定位失败：" + error_message(exc))
    if "report" in st.session_state:
        render_report(st.session_state["report"])
