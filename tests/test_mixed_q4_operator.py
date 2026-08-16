from __future__ import annotations

import os
import sys
import gc
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine" / "CCCP-Engine"
TEST_DSV4_MODEL = Path(
    os.environ.get(
        "CCCP_TEST_DSV4_MODEL",
        str(ROOT / "models" / "dsv4-cccp-s-noblack-v2"),
    )
)
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

os.environ.setdefault("CCCP_CPU_AUTOBUILD", "0")

from cccp.cpuext import (  # noqa: E402
    configure_packed_resident_moe_cpu,
    make_packed_three_layer_cpu,
    make_packed_two_layer_cpu,
    moe_packed_rows_cpu,
    moe_packed_topk_cpu,
    vq_dequant_packed_cpu,
)
from cccp.kernels import BlockFP8Weight  # noqa: E402
from cccp.model import GLMModel, _create_glm_expert_pool  # noqa: E402
from cccp.ops import packed_moe_selected_rows  # noqa: E402
from cccp.store import Manifest, PackedCpuExpertPool, PackedVQWeight  # noqa: E402
from cccp.store import CCCPStore  # noqa: E402


def _vq(rows: int, cols: int, *, q4: bool) -> PackedVQWeight:
    torch.manual_seed(rows * 1000 + cols + int(q4))
    codebook = torch.randn(256, 4, dtype=torch.float32) * 0.02
    raw = torch.randint(0, 256, (rows * (cols // 4),), dtype=torch.uint8)
    weight = PackedVQWeight(raw, codebook, rows, cols, 8)
    if q4:
        assert weight.compile_cpu_q4_0()
    else:
        assert weight.optimize_cpu_row_tile(8)
    return weight


def _shared(rows: int, cols: int) -> BlockFP8Weight:
    # Zero is a valid E4M3 byte.  Q4 compilation is an execution-layout
    # transform only; it does not alter model files.
    compact = BlockFP8Weight(
        torch.zeros((rows, cols), dtype=torch.uint8),
        torch.ones(((rows + 127) // 128, (cols + 127) // 128), dtype=torch.float32),
        cols,
        rows=rows,
    )
    compiled = compact.compile_cpu_q4_0()
    assert compiled.layout == "q4_0"
    return compiled


def _shared_fp8(rows: int, cols: int) -> BlockFP8Weight:
    return BlockFP8Weight(
        torch.zeros((rows, cols), dtype=torch.uint8),
        torch.ones(((rows + 127) // 128, (cols + 127) // 128), dtype=torch.float32),
        cols,
        rows=rows,
    )


def _run_mixed_layer(hidden: int, intermediate: int) -> None:
    torch.set_num_threads(2)
    gates = [_vq(intermediate, hidden, q4=True), _vq(intermediate, hidden, q4=False)]
    ups = [_vq(intermediate, hidden, q4=True), _vq(intermediate, hidden, q4=False)]
    downs = [_vq(hidden, intermediate, q4=True), _vq(hidden, intermediate, q4=False)]
    executor = make_packed_three_layer_cpu(
        tuple(zip(gates, ups, downs)), force_mixed=True
    )
    assert executor is not None
    executor = configure_packed_resident_moe_cpu(
        executor,
        torch.zeros((2, hidden), dtype=torch.bfloat16),
        torch.zeros(2, dtype=torch.float32),
        torch.ones(2, dtype=torch.bool),
        (
            _shared(intermediate, hidden),
            _shared(intermediate, hidden),
            _shared(hidden, intermediate),
        ),
        top_k=2,
        normalize_route=True,
        routed_scaling=1.0,
    )
    assert executor is not None
    result = executor.forward_fused_moe(
        torch.randn((1, hidden), dtype=torch.float32),
        0.0,
        "swiglu",
        1.0,
        -1.0,
    )
    assert result.shape == (hidden,)
    assert torch.isfinite(result).all()


def test_fused_moe_accepts_hot_q4_and_cold_vq_experts() -> None:
    """A mixed hot/cold layer must not reinterpret either execution image."""
    _run_mixed_layer(32, 32)


def test_lru_operator_groups_q4_and_row_tile_payloads_separately() -> None:
    """Partial residency must not decode a cold VQ payload as hot Q4 bytes."""
    torch.manual_seed(5090)
    rows = cols = 32
    codebook = torch.randn(256, 4, dtype=torch.float32) * 0.02

    def weight(*, q4: bool) -> PackedVQWeight:
        raw = torch.randint(
            0, 256, (rows * (cols // 4),), dtype=torch.uint8
        )
        result = PackedVQWeight(raw, codebook, rows, cols, 8)
        if q4:
            assert result.compile_cpu_q4_0()
        else:
            assert result.optimize_cpu_row_tile(8)
        return result

    hot = (weight(q4=True), weight(q4=True), weight(q4=True))
    cold = (weight(q4=False), weight(q4=False), weight(q4=False))
    result = moe_packed_topk_cpu(
        torch.randn((1, cols), dtype=torch.float32),
        [hot, cold],
        torch.tensor([0.6, 0.4], dtype=torch.float32),
        0.0,
        activation="swiglu",
        activation_beta=1.0,
        activation_linear_beta=None,
    )
    assert result is not None
    assert result.shape == (cols,)
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("activation", ["swiglu", "situ"])
def test_packed_cpu_prefill_rows_match_exact_single_row_routes(
    activation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batching must preserve each token's original experts and weights."""
    monkeypatch.setenv("CCCP_CPU_GROUPED_DEQUANT_MIN_ROWS", "2")
    torch.manual_seed(5090)
    hidden = intermediate = 32
    codebooks = [
        torch.randn(256, 4, dtype=torch.float32) * 0.02
        for _ in range(3)
    ]

    def weight(rows: int, cols: int, codebook: torch.Tensor) -> PackedVQWeight:
        raw = torch.randint(0, 256, (rows * (cols // 4),), dtype=torch.uint8)
        return PackedVQWeight(raw, codebook, rows, cols, 8)

    bundles = [
        (
            weight(intermediate, hidden, codebooks[0]),
            weight(intermediate, hidden, codebooks[1]),
            weight(hidden, intermediate, codebooks[2]),
        )
        for _ in range(4)
    ]
    route_ids = torch.tensor([[0, 2], [3, 1], [2, 3]], dtype=torch.long)
    route_weights = torch.tensor(
        [[0.7, 0.3], [0.45, 0.55], [0.2, 0.8]], dtype=torch.float32
    )
    values = torch.randn(3, hidden, dtype=torch.float32)
    nested = [[bundles[int(index)] for index in row] for row in route_ids]
    batched = packed_moe_selected_rows(
        values,
        nested,
        route_weights,
        limit=0.75,
        activation=activation,
        activation_beta=4.0,
        activation_linear_beta=1.5,
    )
    def dense(weight: PackedVQWeight) -> torch.Tensor:
        return weight.cb[weight.unpack().long()].reshape(weight.rows, weight.cols)

    expected_rows = []
    for row in range(values.shape[0]):
        routed = torch.zeros(hidden, dtype=torch.float32)
        for slot, bundle in enumerate(nested[row]):
            gate = torch.mv(dense(bundle[0]), values[row]).clamp(max=0.75)
            up = torch.mv(dense(bundle[1]), values[row]).clamp(-0.75, 0.75)
            if activation == "situ":
                activated = (
                    4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
                    * (1.5 * torch.tanh(up / 1.5))
                )
            else:
                activated = torch.nn.functional.silu(gate) * up
            routed.add_(
                torch.mv(dense(bundle[2]), activated),
                alpha=float(route_weights[row, slot]),
            )
        expected_rows.append(routed)
    expected = torch.stack(expected_rows)
    assert batched is not None
    torch.testing.assert_close(batched, expected, rtol=3e-4, atol=3e-5)


def test_three_projection_cpu_prefill_requires_route_reuse_for_dense_grouping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = inspect.getsource(moe_packed_rows_cpu)

    assert "route_count >= 2 * unique_route_experts" in source
    assert "grouped_reuse_ok" in source


def test_combined_gu_cpu_prefill_rows_match_exact_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two-projection GU/Down archives use the same grouped rows operator."""
    monkeypatch.setenv("CCCP_CPU_GROUPED_DEQUANT_MIN_ROWS", "2")
    torch.manual_seed(334)
    hidden = intermediate = 32
    gu_codebook = torch.randn(256, 4, dtype=torch.float32) * 0.02
    down_codebook = torch.randn(256, 4, dtype=torch.float32) * 0.02

    def weight(rows: int, cols: int, codebook: torch.Tensor) -> PackedVQWeight:
        raw = torch.randint(0, 256, (rows * (cols // 4),), dtype=torch.uint8)
        return PackedVQWeight(raw, codebook, rows, cols, 8)

    bundles = [
        (
            weight(2 * intermediate, hidden, gu_codebook),
            weight(hidden, intermediate, down_codebook),
        )
        for _ in range(4)
    ]
    route_ids = torch.tensor([[0, 2], [3, 1], [2, 3]], dtype=torch.long)
    route_weights = torch.tensor(
        [[0.7, 0.3], [0.45, 0.55], [0.2, 0.8]], dtype=torch.float32
    )
    values = torch.randn(3, hidden, dtype=torch.float32)
    nested = [[bundles[int(index)] for index in row] for row in route_ids]
    batched = moe_packed_rows_cpu(
        values,
        nested,
        route_weights,
        0.0,
        activation="swiglu",
        activation_beta=1.0,
        activation_linear_beta=None,
    )

    def dense(weight: PackedVQWeight) -> torch.Tensor:
        return weight.cb[weight.unpack().long()].reshape(weight.rows, weight.cols)

    expected_rows = []
    for row in range(values.shape[0]):
        routed = torch.zeros(hidden, dtype=torch.float32)
        for slot, bundle in enumerate(nested[row]):
            gu = torch.mv(dense(bundle[0]), values[row])
            activated = torch.nn.functional.silu(gu[:intermediate]) * gu[intermediate:]
            routed.add_(
                torch.mv(dense(bundle[1]), activated),
                alpha=float(route_weights[row, slot]),
            )
        expected_rows.append(routed)
    assert batched is not None
    torch.testing.assert_close(
        batched,
        torch.stack(expected_rows),
        rtol=3e-4,
        atol=3e-5,
    )


def test_combined_gu_q4_decode_uses_layout_aware_executor() -> None:
    """Combined GU Q4 images must never enter the packed-index kernel."""
    torch.manual_seed(335)
    hidden = intermediate = 32
    bundles = []
    for _ in range(4):
        gu = _vq(2 * intermediate, hidden, q4=True)
        down = _vq(hidden, intermediate, q4=True)
        bundles.append((gu, down))

    executor = make_packed_two_layer_cpu(tuple(bundles))
    assert executor is not None
    result = moe_packed_topk_cpu(
        torch.randn((1, hidden), dtype=torch.float32),
        [bundles[0], bundles[2]],
        torch.tensor([0.7, 0.3], dtype=torch.float32),
        0.0,
        activation="swiglu",
        activation_beta=1.0,
        activation_linear_beta=None,
    )
    assert result is not None
    assert result.shape == (hidden,)
    assert torch.isfinite(result).all()


def test_combined_gu_native_pool_prepares_q4_layer() -> None:
    """The generic resident directory must include two-projection layers."""
    torch.manual_seed(336)
    bundles = tuple(
        (_vq(64, 32, q4=True), _vq(32, 32, q4=True))
        for _ in range(2)
    )
    pool = PackedCpuExpertPool.__new__(PackedCpuExpertPool)
    pool.store = SimpleNamespace(
        cfg={"n_experts": 2},
        route_allowlist=None,
        man=SimpleNamespace(expert_files={3: "experts.L03.safetensors"}),
    )
    pool.pinned = {(3, expert): bundle for expert, bundle in enumerate(bundles)}
    pool.compact_full_resident = True
    pool._native_layers = {}

    assert pool.prepare_native_layers() == 1
    assert pool.native_layer(3) is not None


def test_per_expert_projection_manifest_enables_generic_packed_pool(
    tmp_path: Path,
) -> None:
    """Capability comes from the manifest schema, never the model directory name."""
    manifest = {
        "format": "cccp-1",
        "config": {"n_experts": 2},
        "dense_file": "dense.safetensors",
        "expert_files": {"3": "experts.L03.safetensors"},
        "expert_audit_files": {"3": "experts.L03.audit.json"},
        "quant": {
            "method": "per-expert-per-projection-vq",
            "vq": {"gud4k256_dnd4k256": [4, 256]},
            "layer_kinds": {"3": "gud4k256_dnd4k256"},
            "projection_layouts": {
                "d4-k256": {"dim": 4, "codebook_size": 256, "bits": 2.0}
            },
            "layer_projection_layouts": {
                "3": {"gu": "d4-k256", "dn": "d4-k256"}
            },
            "index_packing": {"d4-k256": "u8"},
        },
    }
    (tmp_path / "cccp.json").write_text(json.dumps(manifest), encoding="utf-8")

    parsed = Manifest(str(tmp_path))

    assert parsed.combined_projection_vq is True
    assert parsed.projection_vq is False
    assert parsed.packed_expert_vq is True
    assert parsed.vq_dims == {"gud4k256_dnd4k256": (4, 256)}
    assert parsed.expert_audit_files == {3: "experts.L03.audit.json"}


def test_glm_prefill_uses_public_packed_rows_and_records_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GLM calibration batches routed rows and exposes heatmap counts."""
    monkeypatch.setenv("CCCP_ROUTE_COUNTS", "1")

    class Pool:
        prefill_rows_supported = True

        def __init__(self) -> None:
            self.prefetched = []
            self.calls = []

        def prefetch(self, keys) -> None:
            self.prefetched.append(tuple(keys))

        def run_rows(self, layer, value, ids, weights, **options):
            self.calls.append((layer, value.clone(), ids.clone(), options))
            return torch.full_like(value, 2.0)

    model = GLMModel.__new__(GLMModel)
    model.cfg = {
        "top_k": 2,
        "routed_scaling": 1.0,
        "situ_beta": 4.0,
        "swiglu_limit": 0.0,
    }
    model.device = torch.device("cpu")
    model.expert_parallel = None
    model.pool = Pool()
    model.store = SimpleNamespace(man=SimpleNamespace(projection_vq=True))
    model.operator_config = SimpleNamespace(expert_activation="swiglu")
    model._mask = lambda _layer: torch.ones(4, dtype=torch.bool)
    model._shared_expert_eager = lambda value, _layer: torch.zeros_like(value)
    route_weight = torch.zeros((4, 8), dtype=torch.float32)
    route_bias = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float32)
    model.w = lambda name: (
        route_weight if name.endswith("gate.weight") else route_bias
    )

    values = torch.randn(3, 8)
    output = model._moe(values, 7)

    torch.testing.assert_close(output, torch.full_like(values, 2.0))
    assert len(model.pool.calls) == 1
    assert model.pool.calls[0][0] == 7
    assert model.pool.calls[0][1].shape == (3, 8)
    assert model.pool.calls[0][2].shape == (3, 2)
    assert model.pool.calls[0][3]["activation"] == "swiglu"
    assert sum(model.pool.route_counts.values()) == 6
    assert {layer for layer, _expert in model.pool.route_counts} == {7}
    assert model.pool.prefetched


def test_glm_projection_manifest_selects_common_packed_cpu_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CCCP_CPU_PACKED", "1")
    store = SimpleNamespace(man=SimpleNamespace(projection_vq=True))
    pool = _create_glm_expert_pool(
        store,
        device="cpu",
        cache_gb=3.5,
        vram_cache_gb=0.0,
        pin_gb=0.0,
    )
    assert isinstance(pool, PackedCpuExpertPool)
    assert pool.budget == int(3.5 * 2**30)


def test_glm_single_token_uses_public_packed_decode() -> None:
    class Pool:
        def __init__(self) -> None:
            self.calls = []

        def run_native(self, layer, value, ids, weights, **options):
            self.calls.append((layer, value.clone(), ids.clone(), options))
            return torch.full_like(value, 3.0)

    model = GLMModel.__new__(GLMModel)
    model.cfg = {"top_k": 2, "routed_scaling": 1.0}
    model.device = torch.device("cpu")
    model.expert_parallel = None
    model.pool = Pool()
    model.store = SimpleNamespace(man=SimpleNamespace(projection_vq=True))
    model.operator_config = SimpleNamespace(expert_activation="swiglu")
    model._mask = lambda _layer: torch.ones(4, dtype=torch.bool)
    model._shared_expert_eager = lambda value, _layer: torch.zeros_like(value)
    model._prev_ids = {}
    route_weight = torch.zeros((4, 8), dtype=torch.float32)
    route_bias = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float32)
    model.w = lambda name: (
        route_weight if name.endswith("gate.weight") else route_bias
    )

    output = model._moe(torch.randn(1, 8), 5)

    torch.testing.assert_close(output, torch.full_like(output, 3.0))
    assert len(model.pool.calls) == 1
    assert model.pool.calls[0][0] == 5
    assert model.pool.calls[0][2].shape == (2,)
    assert len(model._prev_ids[5]) == 2


def test_glm_layer_first_prefill_chunks_and_releases_kv() -> None:
    """The calibration entry retains only the current layer's KV state."""
    model = GLMModel.__new__(GLMModel)
    model.cfg = {"n_layers": 3, "rms_eps": 1.0e-6}
    model.pos = 0
    model.max_ctx = 32
    model.kv = [None, None, None]
    model.device = torch.device("cpu")
    model.latent_kv = False
    model._flashinfer_mla_state = None
    model._ensure_rope_capacity = lambda _required: None
    model._ensure_latent_capacity = lambda _required: None
    model.embed = lambda ids: torch.as_tensor(ids).float().reshape(-1, 1).repeat(1, 4)
    calls = []

    def forward_layer(value, layer, start):
        calls.append((layer, start, int(value.shape[0])))
        model.kv[layer] = torch.ones(1)
        return value + float(layer + 1)

    model._forward_layer = forward_layer
    model.w = lambda name: torch.ones(4) if name == "model.norm.weight" else None
    layers = []
    completed = []
    output = model.prefill_chunked(
        torch.arange(10).reshape(1, 10),
        chunk_size=4,
        progress_callback=completed.append,
        layer_progress_callback=lambda start, stop, layer, count: layers.append(
            (start, stop, layer, count)
        ),
    )

    assert calls == [
        (0, 0, 4), (0, 4, 4), (0, 8, 2),
        (1, 0, 4), (1, 4, 4), (1, 8, 2),
        (2, 0, 4), (2, 4, 4), (2, 8, 2),
    ]
    assert completed == [10]
    assert layers[-1] == (8, 10, 3, 3)
    assert model.kv == [None, None, None]
    assert model.pos == 10
    assert output.shape == (10, 4)


def test_grouped_prefill_accepts_q4_and_row_tile_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal chat prefill can mix compiled hot and compact cold experts."""
    monkeypatch.setenv("CCCP_CPU_GROUPED_DEQUANT_MIN_ROWS", "2")
    hidden = intermediate = 32
    hot = tuple(_vq(rows, cols, q4=True) for rows, cols in (
        (intermediate, hidden),
        (intermediate, hidden),
        (hidden, intermediate),
    ))
    cold = tuple(_vq(rows, cols, q4=False) for rows, cols in (
        (intermediate, hidden),
        (intermediate, hidden),
        (hidden, intermediate),
    ))
    values = torch.randn(3, hidden)
    nested = [[hot, cold], [cold, hot], [hot, cold]]
    weights = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]])
    result = moe_packed_rows_cpu(
        values,
        nested,
        weights,
        0.0,
        activation="swiglu",
        activation_beta=4.0,
        activation_linear_beta=None,
    )
    assert result is not None
    assert result.shape == values.shape
    assert torch.isfinite(result).all()


def test_native_dequant_preserves_row_tile_vq() -> None:
    weight = _vq(32, 32, q4=False)
    got = vq_dequant_packed_cpu(
        weight.raw,
        weight.cb,
        weight.rows,
        weight.blocks,
        weight.bits,
        weight.layout,
    )
    expected = weight.cb[weight.unpack().long()].reshape(
        weight.rows, weight.cols
    )
    assert got is not None
    torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)


def test_q4_dense_prefill_uses_multirow_gemm() -> None:
    compact = _shared_fp8(32, 32)
    compiled = compact.compile_cpu_q4_0()
    values = torch.randn(5, 32)
    result = compiled.matmul_T(values)
    assert result.shape == (5, 32)
    torch.testing.assert_close(result, torch.zeros_like(result))


def test_fused_moe_mixed_layout_at_dsv4_dimensions() -> None:
    """Exercise the tile counts used by the target model (4096 x 2048)."""
    _run_mixed_layer(4096, 2048)


def test_target_layer_zero_hot_images_in_mixed_directory() -> None:
    """Local integration probe for the target archive's heterogeneous widths."""
    model = TEST_DSV4_MODEL
    if not (model / "cccp.json").is_file():
        pytest.skip("target CCCP archive is not present")
    store = CCCPStore(str(model))
    hot_ids = (246, 73, 76, 153, 215, 5)
    bundles = []
    for expert_id in hot_ids:
        bundle = store.load_expert_packed(0, expert_id)
        assert all(weight.compile_cpu_q4_0() for weight in bundle)
        bundles.append(bundle)
    cold = store.load_expert_packed(0, 0)
    assert all(weight.optimize_cpu_row_tile(8) for weight in cold)
    bundles.append(cold)
    executor = make_packed_three_layer_cpu(tuple(bundles), force_mixed=True)
    hidden = 4096
    intermediate = 2048
    executor = configure_packed_resident_moe_cpu(
        executor,
        torch.zeros((len(bundles), hidden), dtype=torch.bfloat16),
        torch.zeros(len(bundles), dtype=torch.float32),
        torch.ones(len(bundles), dtype=torch.bool),
        (
            _shared(intermediate, hidden),
            _shared(intermediate, hidden),
            _shared(hidden, intermediate),
        ),
        top_k=6,
        normalize_route=True,
        routed_scaling=1.5,
    )
    result = executor.forward_fused_moe(
        torch.randn((1, hidden), dtype=torch.float32),
        10.0,
        "swiglu",
        1.0,
        -1.0,
    )
    assert result.shape == (hidden,)
    assert torch.isfinite(result).all()


def test_all_target_layers_hot_images_in_mixed_directory() -> None:
    """Probe every target layer because its expert VQ widths are heterogeneous."""
    model = TEST_DSV4_MODEL
    profile_path = model / "profiles" / "roleplay-romance.json"
    if not (model / "cccp.json").is_file() or not profile_path.is_file():
        pytest.skip("target CCCP archive/profile is not present")
    profile_config = json.loads(profile_path.read_text(encoding="utf-8"))
    counts: dict[str, dict[str, int]] = {}
    for expert in profile_config["experts"]:
        layer, expert_id = str(expert["key"]).split(":", 1)
        counts.setdefault(layer, {})[expert_id] = max(
            1, int(expert.get("route_count", 1))
        )
    store = CCCPStore(str(model))
    hidden = 4096
    intermediate = 2048
    torch.set_num_threads(2)
    for layer in range(43):
        layer_counts = counts[str(layer)]
        ranked = sorted(
            layer_counts,
            key=lambda key: (-int(layer_counts[key]), int(key)),
        )
        hot_ids = tuple(int(value) for value in ranked[:6])
        cold_id = next(int(value) for value in ranked if int(value) not in hot_ids)
        print(f"probe-layer={layer} hot={hot_ids} cold={cold_id}", flush=True)
        bundles = []
        for expert_id in hot_ids:
            bundle = store.load_expert_packed(layer, expert_id)
            assert all(weight.compile_cpu_q4_0() for weight in bundle)
            bundles.append(bundle)
        cold = store.load_expert_packed(layer, cold_id)
        assert all(weight.optimize_cpu_row_tile(8) for weight in cold)
        bundles.append(cold)
        executor = make_packed_three_layer_cpu(tuple(bundles), force_mixed=True)
        executor = configure_packed_resident_moe_cpu(
            executor,
            torch.zeros((len(bundles), hidden), dtype=torch.bfloat16),
            torch.zeros(len(bundles), dtype=torch.float32),
            torch.ones(len(bundles), dtype=torch.bool),
            (
                _shared_fp8(intermediate, hidden),
                _shared_fp8(intermediate, hidden),
                _shared_fp8(hidden, intermediate),
            ),
            top_k=6,
            normalize_route=True,
            routed_scaling=1.5,
        )
        result = executor.forward_fused_moe(
            torch.randn((1, hidden), dtype=torch.float32),
            10.0,
            "swiglu",
            1.0,
            -1.0,
        )
        assert result.shape == (hidden,)
        assert torch.isfinite(result).all()
        del result, executor, bundles, cold
        store._cb_cache.clear()
        gc.collect()


def test_target_prefill_route_with_one_hot_q4_expert() -> None:
    """Reproduce the first real prefill row that mixes one hot and five cold experts."""
    model = TEST_DSV4_MODEL
    if not (model / "cccp.json").is_file():
        pytest.skip("target CCCP archive is not present")
    store = CCCPStore(str(model))
    route_ids = (143, 245, 171, 73, 50, 229)
    bundles = []
    for expert_id in route_ids:
        bundle = store.load_expert_packed(0, expert_id)
        if expert_id == 73:
            assert all(weight.compile_cpu_q4_0() for weight in bundle)
        else:
            assert all(weight.optimize_cpu_row_tile(8) for weight in bundle)
        bundles.append(bundle)
    executor = make_packed_three_layer_cpu(tuple(bundles), force_mixed=True)
    result = executor.forward(
        torch.randn((1, 4096), dtype=torch.float32),
        torch.arange(6, dtype=torch.int64),
        torch.full((6,), 1.0 / 6.0, dtype=torch.float32),
        10.0,
        "swiglu",
        1.0,
        -1.0,
    )
    assert result.shape == (4096,)
    assert torch.isfinite(result).all()
