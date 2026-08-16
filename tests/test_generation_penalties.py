"""CCCP generation penalties: public schema mapping and exact logit transforms."""
from pathlib import Path
import sys

import pytest
import torch


CCCP_ROOT = Path(__file__).resolve().parents[1] / "engine" / "CCCP-Engine"
sys.path.insert(0, str(CCCP_ROOT))

from cccp.engine import Engine, _apply_token_penalties  # noqa: E402
from cccp.chat_adapters.base import ChatOptions  # noqa: E402
from cccp.chat_service import ChatService, _decode_token_rate  # noqa: E402
from cccp.openai_api import ChatCompletionRequest, _options_from_openai  # noqa: E402


def test_presence_and_repetition_penalties_share_unique_seen_tokens():
    logits = torch.tensor([8.0, -4.0, 2.0, 1.0])
    adjusted = _apply_token_penalties(
        logits, [0, 0, 1], repetition_penalty=2.0, presence_penalty=0.5,
    )
    assert adjusted.tolist() == pytest.approx([3.5, -8.5, 2.0, 1.0])
    assert logits.tolist() == [8.0, -4.0, 2.0, 1.0]


def test_openai_presence_penalty_range_is_supported():
    request = ChatCompletionRequest(
        model="winui-model",
        messages=[{"role": "user", "content": "hello"}],
        repetition_penalty=1.15,
        presence_penalty=0.4,
    )
    assert request.repetition_penalty == 1.15
    assert request.presence_penalty == 0.4
    options = _options_from_openai(type("Service", (), {"default_reasoning": False})(), request)
    assert options.repetition_penalty == 1.15
    assert options.presence_penalty == 0.4
    with pytest.raises(ValueError):
        ChatCompletionRequest(
            model="winui-model",
            messages=[{"role": "user", "content": "hello"}],
            presence_penalty=2.1,
        )


def test_penalty_request_bypasses_speculative_path():
    class FakeEngine:
        def __init__(self):
            self.generate_kwargs = None
            self.spec_called = False

        def generate(self, ids, **kwargs):
            self.generate_kwargs = kwargs
            return [9]

        def generate_speculative(self, ids, **kwargs):
            self.spec_called = True
            return [8]

    service = object.__new__(ChatService)
    service.engine = FakeEngine()
    service.spec = 3
    plan = type("Plan", (), {
        "input_ids": [1, 2], "kv_baseline_len": 0, "adapter_state": {},
    })()
    options = ChatOptions(
        thinking_mode="chat", reasoning_effort=None, temperature=0.7,
        top_p=1.0, max_new=10, repetition_penalty=1.1,
        presence_penalty=0.3,
    )
    assert service._generate(plan, options, None, None) == [9]
    assert service.engine.spec_called is False
    assert service.engine.generate_kwargs["presence_penalty"] == 0.3


def test_decode_rate_excludes_prefill_and_time_to_first_token():
    assert _decode_token_rate(
        5,
        generation_started=10.0,
        first_token_at=18.0,
        generation_finished=20.0,
    ) == pytest.approx(2.0)
    assert _decode_token_rate(
        0,
        generation_started=10.0,
        first_token_at=None,
        generation_finished=20.0,
    ) == 0.0


def _fake_glm_engine():
    class FakeModel:
        max_ctx = 64

        def __init__(self):
            self.forward_calls = []

        def forward(self, ids):
            self.forward_calls.append(list(ids))
            return torch.tensor([0.0, 9.0, -1.0])

    engine = object.__new__(Engine)
    engine.model = FakeModel()
    engine.arch = "glm"
    engine.eos = {2}
    engine._prepare_glm_prompt = lambda ids, **kwargs: torch.tensor([0.0, 9.0, -1.0])
    engine._glm_device_greedy_window = lambda **kwargs: 0
    engine.decode = lambda ids: "x"
    return engine


def test_glm_generation_does_not_decode_after_requested_output_limit():
    engine = _fake_glm_engine()

    assert engine.generate([7, 8], max_new=1, temp=0.0) == [1]
    assert engine.model.forward_calls == []
    # The sampled token has not entered KV, so it must not be advertised as
    # a reusable cached prefix.  A later request will replay it safely.
    assert engine._cache_ids == [7, 8]


def test_glm_generation_commits_only_tokens_needed_for_another_decode():
    engine = _fake_glm_engine()

    assert engine.generate([7, 8], max_new=2, temp=0.0) == [1, 1]
    assert engine.model.forward_calls == [[1]]
    assert engine._cache_ids == [7, 8, 1]


def test_glm_eos_role_token_is_hidden_from_output_and_callback():
    engine = _fake_glm_engine()
    engine._prepare_glm_prompt = (
        lambda ids, **kwargs: torch.tensor([0.0, -1.0, 9.0])
    )
    callback_ids = []

    output = engine.generate(
        [7, 8],
        max_new=8,
        temp=0.0,
        callback=lambda token_id, _text: callback_ids.append(token_id),
    )

    assert output == []
    assert callback_ids == []
    assert engine._cache_ids == [7, 8]
