"""聊天代理必须使用 CCCP 注册的模型 ID，而不是本地权重路径。"""
import asyncio
from types import SimpleNamespace

import launcher.chat as chat_module
from launcher.chat import ChatProxy
from launcher.cccp_adapter import CCCPEngineInstance


def test_chat_payload_normalizes_local_model_path_to_served_name():
    instance = CCCPEngineInstance(
        pid=1, port=8801,
        model=r"C:\models\dsv4-cccp-s-noblack-v2",
        served_model_name="winui-model",
        profiles=["roleplay-romance"], started_at=1.0,
        log_file="cccp.log", base_url="http://127.0.0.1:8801",
    )
    adapter = SimpleNamespace(
        instance=instance,
        settings=SimpleNamespace(cccp_api_key=""),
    )
    proxy = ChatProxy(adapter)
    payload = proxy._payload({"model": instance.model, "messages": []}, stream=True)
    assert payload["model"] == "winui-model"
    assert payload["stream"] is True


def test_chat_payload_preserves_generation_penalties():
    instance = CCCPEngineInstance(
        pid=1, port=8801, model=r"C:\models\x",
        served_model_name="winui-model", profiles=[], started_at=1.0,
        log_file="cccp.log", base_url="http://127.0.0.1:8801",
    )
    proxy = ChatProxy(SimpleNamespace(
        instance=instance, settings=SimpleNamespace(cccp_api_key=""),
    ))
    payload = proxy._payload({
        "messages": [], "repetition_penalty": 1.15, "presence_penalty": 0.4,
    }, stream=True)
    assert payload["repetition_penalty"] == 1.15
    assert payload["presence_penalty"] == 0.4


def test_non_stream_request_sends_served_name_to_downstream(monkeypatch):
    """覆盖实际 HTTP 转发边界，而不只测试字典辅助函数。"""
    instance = CCCPEngineInstance(
        pid=1, port=8801,
        model=r"C:\models\dsv4-cccp-s-noblack-v2",
        served_model_name="winui-model",
        profiles=["roleplay-romance"], started_at=1.0,
        log_file="cccp.log", base_url="http://127.0.0.1:8801",
    )
    adapter = SimpleNamespace(
        instance=instance,
        settings=SimpleNamespace(cccp_api_key=""),
    )
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, *, json, headers):
            captured.update(url=url, json=json, headers=headers)
            return FakeResponse()

    monkeypatch.setattr(chat_module.httpx, "AsyncClient", FakeAsyncClient)
    result = asyncio.run(ChatProxy(adapter).completions_once({
        "model": instance.model,
        "messages": [{"role": "user", "content": "你好"}],
        "stream": True,
    }))

    assert result == {"ok": True}
    assert captured["url"] == "http://127.0.0.1:8801/v1/chat/completions"
    assert captured["json"]["model"] == "winui-model"
    assert captured["json"]["stream"] is False


def test_interface_contract_proxy_uses_engine_v1_and_local_key(monkeypatch):
    instance = CCCPEngineInstance(
        pid=1, port=8801, model=r"C:\models\x",
        served_model_name="winui-model", profiles=[], started_at=1.0,
        log_file="cccp.log", base_url="http://127.0.0.1:8801",
    )
    adapter = SimpleNamespace(
        instance=instance,
        settings=SimpleNamespace(cccp_api_key="local-secret"),
    )
    captured = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"schema": "ok"}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, *, headers):
            captured.append(("GET", url, headers))
            return FakeResponse()

        async def post(self, url, *, headers):
            captured.append(("POST", url, headers))
            return FakeResponse()

    monkeypatch.setattr(chat_module.httpx, "AsyncClient", FakeAsyncClient)
    proxy = ChatProxy(adapter)
    assert asyncio.run(proxy.contract_get("model-spec")) == {"schema": "ok"}
    assert asyncio.run(proxy.reset_expert_stats()) == {"schema": "ok"}
    assert captured == [
        (
            "GET",
            "http://127.0.0.1:8801/v1/model-spec",
            {"Authorization": "Bearer local-secret"},
        ),
        (
            "POST",
            "http://127.0.0.1:8801/v1/expert-stats/reset",
            {"Authorization": "Bearer local-secret"},
        ),
    ]
