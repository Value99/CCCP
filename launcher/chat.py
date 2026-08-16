"""聊天代理:前端只与 WINUI-EXE 通信,由其转发到 CCCP 的 OpenAI 兼容 API。

支持 SSE 流式透传;CCCP 未启动时返回明确错误而非挂死。
"""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from .cccp_adapter import CCCPEngineAdapter, CCCPEngineError


class ChatProxy:
    def __init__(self, adapter: CCCPEngineAdapter):
        self.adapter = adapter

    def _base(self) -> str:
        if not self.adapter.instance:
            raise CCCPEngineError("模型未启动：请先选择模型和 profile 组合，然后点击“启动推理”")
        return self.adapter.instance.base_url

    def _headers(self) -> dict:
        if self.adapter.settings.cccp_api_key:
            return {"Authorization": f"Bearer {self.adapter.settings.cccp_api_key}"}
        return {}

    def _payload(self, payload: dict, *, stream: bool) -> dict:
        """把 WINUI 单实例代理请求规范化到 CCCP 实际注册的模型 ID。

        ``CCCPEngineInstance.model`` 是本地权重目录，只用于展示和重启；它不是
        OpenAI API 的 model ID。旧前端曾把这个 Windows 路径直接发给 CCCP，
        因而得到 model_not_found。代理层统一使用启动参数中的 served name，
        也能兼容旧客户端缓存下来的错误 model 字段。
        """
        if not self.adapter.instance:
            self._base()  # 复用明确的“模型未启动”错误
        served = self.adapter.instance.served_model_name
        return {**payload, "model": served, "stream": stream}

    async def models(self) -> dict:
        base = self._base()
        async with httpx.AsyncClient(timeout=10) as cli:
            r = await cli.get(f"{base}/v1/models", headers=self._headers())
            r.raise_for_status()
            return r.json()

    async def contract_get(self, endpoint: str) -> dict:
        """Forward one read-only CCCP interface-contract endpoint."""
        allowed = {"model-spec", "expert-bytes", "expert-stats"}
        if endpoint not in allowed:
            raise ValueError("不支持的 CCCP 契约端点")
        base = self._base()
        async with httpx.AsyncClient(timeout=30) as cli:
            response = await cli.get(
                f"{base}/v1/{endpoint}", headers=self._headers()
            )
            response.raise_for_status()
            return response.json()

    async def reset_expert_stats(self) -> dict:
        """Reset and enable the engine's live router counter while idle."""
        base = self._base()
        async with httpx.AsyncClient(timeout=30) as cli:
            response = await cli.post(
                f"{base}/v1/expert-stats/reset", headers=self._headers()
            )
            response.raise_for_status()
            return response.json()

    async def completions_stream(self, payload: dict) -> AsyncIterator[bytes]:
        """流式转发 /v1/chat/completions(SSE),逐 chunk 透传。"""
        base = self._base()
        payload = self._payload(payload, stream=True)
        async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10)) as cli:
            async with cli.stream(
                "POST", f"{base}/v1/chat/completions",
                json=payload, headers=self._headers(),
            ) as r:
                if r.status_code != 200:
                    body = await r.aread()
                    try:
                        error_payload = json.loads(body.decode("utf-8", errors="replace"))
                    except json.JSONDecodeError:
                        error_payload = {"error": {"message": body.decode(errors="replace")}}
                    yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                    return
                async for chunk in r.aiter_raw():
                    yield chunk

    async def completions_once(self, payload: dict) -> dict:
        base = self._base()
        payload = self._payload(payload, stream=False)
        async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10)) as cli:
            r = await cli.post(
                f"{base}/v1/chat/completions", json=payload, headers=self._headers()
            )
            r.raise_for_status()
            return r.json()
