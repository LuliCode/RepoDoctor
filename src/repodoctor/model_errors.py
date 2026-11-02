"""Safe, actionable model failures: never expose provider bodies or request headers."""

from openai import APIConnectionError, APIStatusError, APITimeoutError


def classify_model_error(exc):
    kind = type(exc).__name__
    status = exc.status_code if isinstance(exc, APIStatusError) else None
    code = getattr(exc, "code", None)
    category, retryable = "unknown", False
    summary = "模型调用失败"
    hint = "请检查模型服务和工具调用协议兼容性；可按异常类型排查服务端日志。"
    if isinstance(exc, APITimeoutError):
        category, retryable = "request_timeout", True
        summary = "模型单次请求超时"
        hint = "请检查模型响应速度与网络连接；单次请求超时不同于分析总耗时上限。"
    elif isinstance(exc, APIConnectionError):
        category, retryable = "connection", True
        summary = "无法连接模型服务或连接中断"
        hint = "请检查模型服务地址、网络、代理及 TLS 证书。"
    elif status == 401:
        category, summary = "authentication", "模型服务认证失败"
        hint = "请检查 API 密钥是否有效、是否属于当前服务地址，然后重启应用。"
    elif status == 403:
        category, summary = "permission", "模型服务拒绝访问"
        hint = "请检查账号、模型访问权限和服务商访问限制。"
    elif status == 404:
        category, summary = "not_found", "模型或 API 路径不存在"
        hint = "请核对模型名称和服务地址的 API 路径。"
    elif code == "insufficient_quota" and status in {400, 429}:
        category, summary = "quota", "模型服务账户额度不足"
        hint = "请检查服务商账户额度；增加页面中的分析预算不能增加账户额度。"
    elif status == 429:
        category, retryable, summary = "rate_limit", True, "模型服务请求被限流"
        hint = "请稍后重试，或检查服务商的请求速率和 Token 速率限制。"
    elif status == 408 or (status is not None and status >= 500):
        category, retryable, summary = "server", True, "模型服务暂时不可用"
        hint = "请检查服务商状态；本轮代码工具成功不代表模型 API 连接也成功。"
    elif status in {400, 413, 422}:
        if code == "context_length_exceeded" or status == 413:
            category, summary = "context_limit", "模型请求超过服务端上下文或请求体上限"
            hint = "请降低单轮上下文上限或精简日志，保留关键请求字段和业务堆栈；增加累计 Token 预算不会提高模型容量。"
        else:
            category, summary = "invalid_request", "模型服务拒绝了请求参数"
            hint = "请检查模型是否支持 tools、tool_calls、max_completion_tokens，及单轮上下文是否超限；同样请求不会自动重试。"
    suffix = f"{kind}" + (f"，HTTP {status}" if status is not None else "")
    result = {
        "error_type": kind,
        "error_category": category,
        "retryable": retryable,
        "summary": f"{summary}（{suffix}）",
        "hint": hint,
    }
    if status is not None:
        result["http_status"] = status
    return result
