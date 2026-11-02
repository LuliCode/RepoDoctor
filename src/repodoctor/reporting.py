"""Interpret current reports and legacy fallback reports consistently in the UI."""


def analysis_incomplete(report):
    metrics = report.get("metrics", {})
    if metrics.get("final_summary_generated") is False:
        return True
    if metrics.get("stop_reason") in {
        "invalid_model_output",
        "model_error",
    }:
        return True
    # Old reports lack the completion flag but contain this exact fallback prefix.
    return (
        report.get("diagnosis", {})
        .get("cause", "")
        .startswith(("本次分析提前结束，尚未完成故障原因判断。", "复核未完成，以下为未经复核的候选结论："))
    )
