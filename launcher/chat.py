"""聊天代理:前端只与 WINUI-EXE 通信,由其转发到 TPQ 的 OpenAI 兼容 API。

支持 SSE 流式透传;TPQ 未启动时返回明确错误而非挂死。
"""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from .tpq_adapter import TPQAdapter, TPQError


class ChatProxy:
    def __init__(self, adapter: TPQAdapter):
        self.adapter = adapter

    def _base(self) -> str:
        if not self.adapter.instance:
            raise TPQError("模型未启动:请先在「配置」页选择一个 profile 组合并发动")
        return self.adapter.instance.base_url

    def _headers(self) -> dict:
        if self.adapter.settings.tpq_api_key:
            return {"Authorization": f"Bearer {self.adapter.settings.tpq_api_key}"}
        return {}

    async def models(self) -> dict:
        base = self._base()
        async with httpx.AsyncClient(timeout=10) as cli:
            r = await cli.get(f"{base}/v1/models", headers=self._headers())
            r.raise_for_status()
            return r.json()

    async def completions_stream(self, payload: dict) -> AsyncIterator[bytes]:
        """流式转发 /v1/chat/completions(SSE),逐 chunk 透传。"""
        base = self._base()
        payload = {**payload, "stream": True}
        async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10)) as cli:
            async with cli.stream(
                "POST", f"{base}/v1/chat/completions",
                json=payload, headers=self._headers(),
            ) as r:
                if r.status_code != 200:
                    body = await r.aread()
                    yield f"data: {json.dumps({'error': body.decode(errors='replace')})}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                    return
                async for chunk in r.aiter_raw():
                    yield chunk

    async def completions_once(self, payload: dict) -> dict:
        base = self._base()
        payload = {**payload, "stream": False}
        async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10)) as cli:
            r = await cli.post(
                f"{base}/v1/chat/completions", json=payload, headers=self._headers()
            )
            r.raise_for_status()
            return r.json()
