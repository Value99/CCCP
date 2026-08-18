"""Qwen3.5 Dense VQ model, chat and launcher contracts."""

from __future__ import annotations

import json
import inspect
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine" / "CCCP-Engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from cccp.chat_adapters.base import ChatMessage, ChatOptions  # noqa: E402
from cccp.chat_adapters.qwen35 import Qwen35ChatAdapter  # noqa: E402
from cccp.qwen35_model import _qwen35_gpu_image_plan  # noqa: E402
from cccp.dense_vq import (  # noqa: E402
    DenseBF16SwiGLU,
    DenseVQArchive,
    DenseVQEmbedding,
    DenseVQLinear,
    DenseVQLinearGroup,
)
from cccp.store import PackedVQWeight  # noqa: E402
from cccp.presets import detect_architecture, resolve_preset  # noqa: E402
from cccp.launch import _spec_value  # noqa: E402
from cccp.chat_service import _decode_executor_name  # noqa: E402
from launcher.cccp_adapter import (  # noqa: E402
    estimate_gpu_vram_plan,
    full_model_combination,
    inspect_model,
)


def _pack(values: list[int], bits: int) -> torch.Tensor:
    result = bytearray((len(values) * bits + 7) // 8)
    for index, value in enumerate(values):
        offset = index * bits
        for shift in range(bits):
            if value & (1 << shift):
                bit = offset + shift
                result[bit // 8] |= 1 << (bit % 8)
    return torch.tensor(list(result), dtype=torch.uint8)


def _dense_model(root: Path) -> dict:
    root.mkdir(parents=True)
    codebook = torch.arange(32, dtype=torch.float32).reshape(16, 2)
    values = [0, 1, 2, 3, 4, 5]
    save_file(
        {"indices.000": _pack(values, 8), "codebook.000": codebook},
        root / "vq.global.safetensors",
    )
    save_file({"model.language_model.norm.weight": torch.ones(2)}, root / "dense.safetensors")
    for name in ("config.json", "tokenizer.json"):
        (root / name).write_text("{}", encoding="utf-8")
    manifest = {
        "format": "cccp-1",
        "architecture": "qwen3_5",
        "model_family": "qwen3.5-dense",
        "dense_file": "dense.safetensors",
        "tensor_files": ["vq.global.safetensors"],
        "tensor_vq": {
            "model.language_model.embed_tokens.weight": {
                "file": "vq.global.safetensors",
                "shape": [3, 4],
                "layout": "d2-k16",
                "index_packing": "packed-u8",
                "codebook_key": "codebook.000",
                "indices_key": "indices.000",
            }
        },
        "expert_files": {},
        "routed_experts": {"layer_files": {}},
        "quant": {"layouts": {"d2-k16": {"dim": 2, "size": 16}}},
        "tokenizer_files": ["tokenizer.json"],
        "config": {
            "n_layers": 2,
            "hidden": 4,
            "vocab": 3,
            "max_position_embeddings": 4096,
            "outer_model_type": "qwen3_5",
            "text_model_type": "qwen3_5_text",
        },
    }
    (root / "cccp.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return manifest


def test_dense_manifest_is_not_misclassified_as_glm_or_missing_experts(tmp_path):
    manifest = _dense_model(tmp_path / "dense")
    info = inspect_model(tmp_path / "dense")

    assert detect_architecture(manifest) == "qwen3_5_dense"
    assert info.complete
    assert info.execution_kind == "dense_vq"
    assert info.has_dynamic_experts is False
    assert info.supports_route_training is False
    assert info.expert_gb == 0
    assert info.total_bytes > 0
    assert full_model_combination(info.path).expert_count == 0


def test_dense_preset_and_vram_plan_have_no_expert_arena(tmp_path):
    _dense_model(tmp_path / "dense")
    preset = resolve_preset(tmp_path / "dense", profile="resident", tp=1)
    info = inspect_model(tmp_path / "dense")
    plan = estimate_gpu_vram_plan(info, max_ctx=4096, expert_cache_gb=8)

    assert preset.architecture == "qwen3_5_dense"
    assert preset.supports_parallel is False
    assert plan["architecture"] == "qwen3_5_dense"
    assert plan["minimum_expert_arena_gb"] == 0
    assert plan["preferred_expert_arena_gb"] == 0


def test_qwen_mtp_default_is_backend_specific(tmp_path, monkeypatch):
    manifest = _dense_model(tmp_path / "dense")
    preset = resolve_preset(tmp_path / "dense", profile="resident", tp=1)
    args = SimpleNamespace(spec=None, device="cuda")

    monkeypatch.delenv("CCCP_RUNTIME_BACKEND", raising=False)
    assert _spec_value(args, preset) == 0
    manifest["config"]["mtp_layers"] = 1
    (tmp_path / "dense" / "cccp.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    preset = resolve_preset(tmp_path / "dense", profile="resident", tp=1)
    assert _spec_value(args, preset) == 4
    args.device = "cpu"
    assert _spec_value(args, preset) == 0
    monkeypatch.setenv("CCCP_RUNTIME_BACKEND", "amd")
    args.device = "cuda"
    assert _spec_value(args, preset) == 0
    args.spec = 2
    assert _spec_value(args, preset) == 2


def test_qwen_mtp_execution_and_residency_are_config_driven():
    from cccp.speculative import provider_for_architecture

    provider_for_architecture.cache_clear()
    provider = provider_for_architecture("qwen3_5_dense")

    assert provider.policy.top_n == 3
    assert provider.policy.max_draft == 5
    assert provider.residency_value("cpu") == "disabled"
    assert provider.residency_value("nvidia_fp8") == "fp8"
    assert provider.residency_value("nvidia_legacy") == "int8"
    assert provider.residency_value("independent_of_lru") is True
    assert provider.execution_value("draft") == "fixed-block-cuda-graph"
    assert provider.execution_value("verify") == "fixed-block-cuda-graph"


def test_dense_embedding_expands_only_selected_rows(tmp_path):
    _dense_model(tmp_path / "dense")
    archive = DenseVQArchive(tmp_path / "dense")
    embedding = DenseVQEmbedding.from_archive(
        archive, "model.embed_tokens.weight", torch.device("cpu")
    )

    actual = embedding(torch.tensor([2, 0])).float()
    expected = torch.tensor([[8, 9, 10, 11], [0, 1, 2, 3]], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)


class _TextEngine:
    def __init__(self):
        self._cache_ids = None

    @staticmethod
    def encode(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(ids: list[int]) -> str:
        return bytes(ids).decode("utf-8")


def _options(thinking: str = "chat") -> ChatOptions:
    return ChatOptions(
        thinking_mode=thinking,
        reasoning_effort="high" if thinking == "thinking" else None,
        temperature=0.0,
        top_p=1.0,
        max_new=16,
    )


def test_qwen_chat_template_and_parser_hide_structural_tokens():
    engine = _TextEngine()
    adapter = Qwen35ChatAdapter()
    plan = adapter.prepare(
        engine,
        [ChatMessage(role="user", content="你好")],
        _options("chat"),
        None,
    )
    prompt = engine.decode(plan.input_ids)
    assert "<|im_start|>user\n你好<|im_end|>" in prompt
    assert prompt.endswith("<think>\n\n</think>\n\n")

    raw = "回答<|im_end|>"
    parsed = adapter.parse_complete(engine, engine.encode(raw), _options("chat"))
    assert parsed.content == "回答"
    assert parsed.reasoning_content is None


def test_qwen_thinking_is_folded_and_not_committed_for_hot_reuse():
    engine = _TextEngine()
    adapter = Qwen35ChatAdapter()
    options = _options("thinking")
    plan = adapter.prepare(
        engine, [ChatMessage(role="user", content="问题")], options, None
    )
    output = engine.encode("推理过程</think>\n\n正式回答<|im_end|>")
    engine._cache_ids = [*plan.input_ids, *output]
    parsed = adapter.parse_complete(engine, output, options)
    ledger = adapter.commit(engine, plan, output, parsed)

    assert parsed.reasoning_content == "推理过程"
    assert parsed.content == "正式回答"
    assert ledger.completed_ids is None
    assert ledger.committed_messages[-1].reasoning_content is None


def test_dense_training_rejected_by_api_before_job_creation(tmp_path, monkeypatch):
    import launcher.app as app_module
    import launcher.state as state_module
    from launcher.app import create_app
    from launcher.settings import Settings

    model = tmp_path / "models" / "dense"
    _dense_model(model)
    engine = tmp_path / "engine"
    (engine / "cccp").mkdir(parents=True)
    (engine / "cccp" / "__main__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(app_module, "USER_PROFILE_DIR", tmp_path / "profiles")
    monkeypatch.setattr(state_module, "STATE_FILE", tmp_path / "state.json")
    app = create_app(Settings(
        cccp_engine_path=str(engine),
        python_path=sys.executable,
        model_roots=[str(model.parent)],
    ))
    with TestClient(app) as client:
        response = client.post("/api/training/jobs", json={
            "model_path": str(model),
            "token_budget": 4096,
            "corpus_files": ["sample.jsonl"],
        })
    assert response.status_code == 400
    assert "Dense VQ" in response.json()["error"]["message"]


def test_dense_ui_has_no_expert_training_or_profile_language():
    script = (ROOT / "webui" / "app.js").read_text(encoding="utf-8")
    assert "Dense 模型不使用专家配置" in script
    assert "没有动态专家，无需且不支持语料路由扫描" in script
    assert "完整 Dense 模型" in script
    full_model_card = script.split("function fullModelCard(model)", 1)[1].split(
        "async function toggleFullModel", 1
    )[0]
    assert '${dense ? "Dense VQ"' not in full_model_card
    assert '${dense ? "完整权重"' not in full_model_card


def test_dense_gpu_plan_has_only_resident_and_compact_modes():
    gib = 1 << 30
    args = dict(
        resident_weight_bytes=27 * gib,
        compact_weight_bytes=12 * gib,
        fixed_bytes=2 * gib,
        runtime_bytes=4 * gib,
        resident_supported=True,
    )
    from types import SimpleNamespace

    config = SimpleNamespace(num_key_value_heads=8, head_dim=128, num_layers=48)
    small, small_plan = _qwen35_gpu_image_plan(
        free_bytes=16 * gib,
        linear_bf16_bytes=47 * gib,
        linear_fp8_bytes=23 * gib,
        linear_int4_bytes=12 * gib,
        packed_embedding_bytes=1 * gib,
        embedding_bf16_bytes=2 * gib,
        fixed_file_bytes=2 * gib,
        max_ctx=512,
        config=config,
        fp8_supported=True,
    )
    large, _ = _qwen35_gpu_image_plan(
        free_bytes=60 * gib,
        linear_bf16_bytes=47 * gib,
        linear_fp8_bytes=23 * gib,
        linear_int4_bytes=12 * gib,
        packed_embedding_bytes=1 * gib,
        embedding_bf16_bytes=2 * gib,
        fixed_file_bytes=2 * gib,
        max_ctx=512,
        config=config,
        fp8_supported=True,
    )
    # 真实空闲不足时按 int4 紧凑映像规划;宽裕时选更快的 fp8/bf16。
    assert small in {"int4", "fp8"}
    assert small_plan["free"] == 16 * gib
    assert large in {"bf16", "fp8"}


def test_dense_gpu_plan_never_selects_fp8_without_tensor_cores():
    from types import SimpleNamespace

    gib = 1 << 30
    config = SimpleNamespace(num_key_value_heads=8, head_dim=128, num_layers=48)
    image, _ = _qwen35_gpu_image_plan(
        free_bytes=60 * gib,
        linear_bf16_bytes=47 * gib,
        linear_fp8_bytes=23 * gib,
        linear_int4_bytes=12 * gib,
        packed_embedding_bytes=1 * gib,
        embedding_bf16_bytes=2 * gib,
        fixed_file_bytes=2 * gib,
        max_ctx=512,
        config=config,
        fp8_supported=False,
    )
    # FP8 Tensor Core 不可用时绝不选 fp8;宽裕显存走精确 bf16。
    assert image != "fp8"


def test_dense_gpu_linear_exposes_fp8_bf16_and_int4_images():
    source = inspect.getsource(DenseVQLinear)
    assert "compile_gpu_fp8" in source
    assert "compile_gpu_bf16" in source
    assert "compile_gpu_int4" in source
    # 旧的 compact VQ 现场解压路线已被三种常驻映像取代。
    assert "compile_gpu_compact" not in source
    assert "dense_vq_compact_mma" not in source
    assert "compact_codebook_lut_fp8" not in source


def test_dense_gpu_mode_has_no_legacy_third_route():
    source = (ENGINE / "cccp" / "qwen35_model.py").read_text(
        encoding="utf-8"
    )
    assert "CCCP_DENSE_VQ_GPU_IMAGE" in source
    assert "plan_dense_vq_gpu_execution" not in source
    assert "plan_dense_vq_gpu_execution" not in (
        ENGINE / "cccp" / "dense_vq.py"
    ).read_text(encoding="utf-8")


def test_qwen_gpu_fuses_both_decode_and_prefill_delta_entries():
    source = (ENGINE / "cccp" / "qwen35_model.py").read_text(
        encoding="utf-8"
    )
    assert '"torch_recurrent_gated_delta_rule"' in source
    assert '"torch_chunk_gated_delta_rule"' in source
    assert "qwen35_delta_recurrent_batch_fused" in source
    assert "torch.empty_like(value[0])" not in source
    assert "value[0].shape" in source


def test_qwen_gpu_projection_groups_are_architecture_owned_and_installed():
    model_source = (ENGINE / "cccp" / "qwen35_model.py").read_text(
        encoding="utf-8"
    )
    storage_source = (ENGINE / "cccp" / "dense_vq.py").read_text(
        encoding="utf-8"
    )

    assert "_install_qwen35_projection_groups(network)" in model_source
    assert '("gate_proj", "up_proj")' in model_source
    assert '("q_proj", "k_proj", "v_proj")' in model_source
    assert '("in_proj_qkv", "in_proj_z")' in model_source
    assert '("in_proj_b", "in_proj_a")' in model_source
    assert "class DenseVQLinearGroup" in storage_source
    assert "class DenseBF16LinearGroup" in storage_source
    assert "Qwen" not in storage_source


def test_dense_cpu_q4_projection_group_matches_individual_linears():
    torch.manual_seed(35)
    codebook = torch.randn(256, 2)
    linears = []
    for index, rows in enumerate((16, 24)):
        weight = PackedVQWeight(
            torch.randint(0, 256, (rows * 16,), dtype=torch.uint8),
            codebook,
            rows,
            32,
            8,
        )
        linear = DenseVQLinear(weight, name=f"test.{index}")
        assert linear.compile_cpu()
        linears.append(linear)
    value = torch.randn(1, 32, dtype=torch.bfloat16)
    expected = tuple(linear(value).clone() for linear in linears)
    group = DenseVQLinearGroup(tuple(linears))
    actual = tuple(group.view(index)(value) for index in range(2))

    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0.0, atol=0.0)

    batch = torch.randn(65, 32, dtype=torch.bfloat16)
    expected_batch = tuple(linear(batch).clone() for linear in linears)
    actual_batch = tuple(group.view(index)(batch) for index in range(2))
    for result, reference in zip(actual_batch, expected_batch):
        torch.testing.assert_close(result, reference, rtol=0.0, atol=0.0)


def test_dense_cpu_bf16_swiglu_matches_public_mlp():
    torch.manual_seed(350)

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(64, 128, bias=False).to(
                torch.bfloat16
            )
            self.up_proj = torch.nn.Linear(64, 128, bias=False).to(
                torch.bfloat16
            )
            self.down_proj = torch.nn.Linear(128, 64, bias=False).to(
                torch.bfloat16
            )

        def forward(self, value):
            return self.down_proj(
                torch.nn.functional.silu(self.gate_proj(value))
                * self.up_proj(value)
            )

    reference = MLP().eval()
    fused = DenseBF16SwiGLU(
        reference,
        reference.gate_proj,
        reference.up_proj,
        reference.down_proj,
    )
    value = torch.randn(1, 64, dtype=torch.bfloat16)
    expected = reference(value)
    actual = fused(value)
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)

def test_qwen_cpu_ordered_delta_batch_matches_reference():
    from cccp.cpuext import qwen35_delta_recurrent_batch_cpu

    torch.manual_seed(351)
    tokens, heads, key_dim, value_dim = 5, 3, 8, 6
    query = torch.randn(tokens, heads, key_dim)
    key = torch.randn_like(query)
    value = torch.randn(tokens, heads, value_dim)
    gate = -torch.rand(tokens, heads)
    beta = torch.sigmoid(torch.randn(tokens, heads))
    initial = torch.randn(heads, key_dim, value_dim)
    expected_state = initial.clone()
    expected_rows = []
    for token in range(tokens):
        q = torch.nn.functional.normalize(
            query[token], dim=-1, eps=1.0e-6
        )
        k = torch.nn.functional.normalize(
            key[token], dim=-1, eps=1.0e-6
        )
        expected_state.mul_(gate[token].exp()[:, None, None])
        prediction = torch.einsum("hkv,hk->hv", expected_state, k)
        delta = (value[token] - prediction) * beta[token, :, None]
        expected_state.add_(k[:, :, None] * delta[:, None, :])
        expected_rows.append(
            torch.einsum("hkv,hk->hv", expected_state, q)
            / math.sqrt(key_dim)
        )
    expected = torch.stack(expected_rows)

    actual = qwen35_delta_recurrent_batch_cpu(
        query, key, value, gate, beta, initial
    )
    assert actual is not None
    output, state = actual
    torch.testing.assert_close(output, expected, rtol=2.0e-5, atol=2.0e-5)
    torch.testing.assert_close(
        state, expected_state, rtol=2.0e-5, atol=2.0e-5
    )


def test_qwen_cuda_disables_high_overhead_cudnn_sdpa_only_in_adapter():
    model_source = (ENGINE / "cccp" / "qwen35_model.py").read_text(
        encoding="utf-8"
    )
    engine_source = (ENGINE / "cccp" / "engine.py").read_text(
        encoding="utf-8"
    )

    assert "enable_cudnn_sdp(False)" in model_source
    assert "flash-or-efficient" in model_source
    assert "enable_cudnn_sdp(False)" not in engine_source


def test_qwen_native_token_graph_is_static_cache_only_and_has_no_slow_fallback():
    model_source = (ENGINE / "cccp" / "qwen35_model.py").read_text(
        encoding="utf-8"
    )
    storage_source = (ENGINE / "cccp" / "dense_vq.py").read_text(
        encoding="utf-8"
    )

    assert "StaticCache" in model_source
    assert "Qwen3.5 native token graph capture failed" in model_source
    assert "eager-fallback=forbidden" in model_source
    assert "compile_gpu_fp8" in storage_source
    assert "dense_fp8_quantize_rows_fused" in storage_source
    assert "Qwen" not in storage_source


def test_decode_diagnostics_distinguish_hip_graph_and_dense(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.2")
    routed = SimpleNamespace(model=SimpleNamespace(
        device=torch.device("cuda"),
        packed_operator_name="cuda.packed_moe_topk_fused",
    ))
    assert _decode_executor_name(
        routed, {"decode_graph_submissions": 43}
    ) == "hip.tp1-token-graph"
    assert _decode_executor_name(
        routed, {"decode_graph_submissions": 0}
    ) == "hip.packed_moe_topk_fused"

    dense = SimpleNamespace(model=SimpleNamespace(
        device=torch.device("cuda"),
        packed_operator_name=(
            "dense_vq.compact.decode-direct-dot."
            "prefill-transient-fp8-gemm"
        ),
    ))
    assert _decode_executor_name(
        dense, {"decode_graph_submissions": 0}
    ) == (
        "dense_vq.compact.decode-direct-dot."
        "prefill-transient-fp8-gemm"
    )
