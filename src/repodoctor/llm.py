from __future__ import annotations

import os

import httpx
from openai import AsyncOpenAI


class ChatModel:
    """OpenAI-compatible Chat Completions tool-calling adapter."""

    def __init__(self, model=None, api_key=None, base_url=None):
        self.model = model or os.getenv("REPODOCTOR_MODEL")
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.model or not key:
            raise ValueError("请在 .env 中配置 REPODOCTOR_MODEL 和 OPENAI_API_KEY；检索工具无需密钥。")
        self.client = AsyncOpenAI(
            api_key=key, base_url=base_url or os.getenv("OPENAI_BASE_URL") or None, timeout=60, max_retries=0
        )

    async def complete(self, messages, tools, max_tokens=2000, *, timeout=60):
        kwargs = {"model": self.model, "messages": messages, "max_completion_tokens": max_tokens}
        # The agent supplies its remaining wall-clock budget. A fixed 60s read
        # timeout would abort slow reasoning even when that budget still allows it.
        kwargs["timeout"] = httpx.Timeout(timeout, connect=min(10, timeout))
        if tools:
            kwargs.update(tools=tools, tool_choice="auto", parallel_tool_calls=False)
        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        usage = response.usage
        details = usage.completion_tokens_details if usage else None
        return {
            "message": choice.message.model_dump(exclude_none=True),
            "tokens": usage.total_tokens if usage else None,
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "reasoning_tokens": details.reasoning_tokens if details else None,
            }
            if usage
            else {},
        }

    async def close(self):
        await self.client.close()
