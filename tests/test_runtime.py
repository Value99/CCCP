"""CPU 离线运行策略：设置门禁、模型检查、命令与预检。"""
from __future__ import annotations

import asyncio
import json
import inspect
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import launcher.cccp_adapter as cccp_module

ENGINE_ROOT = Path(__file__).resolve().parents[1] / "engine" / "CCCP-Engine"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from cccp.presets import resolve_preset  # noqa: E402
from cccp import cpuext  # noqa: E402
from cccp.packed_hybrid import (  # noqa: E402
    HostPackedWeight,
    PackedHybridPool,
    PackedExpertSignature,
    automatic_host_pin_budget,
)
from cccp.dsv4model import (  # noqa: E402
    DSV4CCCPModel,
    _automatic_prefetch_policy,
    _compressor_decode_cccp,
    _compressor_prefill_cccp,
    _indexer_candidate_capacity,
    _prefill_sliding_window,
    _requires_flashmla_splitkv,
    _tp1_token_graph_bucket,
)
from cccp.dsv4cache import PagedKV  # noqa: E402
from cccp.kimi_experts import (  # noqa: E402
    PackedExpertPool,
    _packed_grouped_prefill_supported,
    _select_packed_grouped_prefill,
)
from cccp.ops import packed_moe_topk, packed_moe_topk_grouped  # noqa: E402
from cccp.engine import Engine  # noqa: E402
from cccp.model import GLMModel, _latent_attention_context_batched  # noqa: E402
from cccp.kernels import VQWeight  # noqa: E402
from cccp.engine import (  # noqa: E402
    _dsv4_prefill_workspace_reserve_gb,
    _initial_expert_vram_request_gb,
    _profile_dsv4_stage_call,
    _safe_expert_budget,
    _use_hip_short_reset_decode,
)
from cccp.store import ExpertPool  # noqa: E402
from cccp.store import _prefer_direct_pinned_h2d  # noqa: E402
from cccp.chat_service import _expert_cache_snapshot  # noqa: E402
from cccp.expert_slots import SlotBook  # noqa: E402
from cccp.expert_parallel import GpuResidentExpertParallel  # noqa: E402
from cccp.launch import _apply_environment  # noqa: E402
from cccp.prefill import prefill_block_size, prefill_ranges  # noqa: E402
from launcher.profiles import Combination, ExpertRef
from launcher.settings import Settings
from launcher.cccp_adapter import (
    FULL_MODEL_PROFILE_ID, LaunchConfig, CCCPEngineAdapter, CCCPEngineInstance,
    discover_models, estimate_gpu_vram_plan, full_model_combination,
    inspect_model,
)
from launcher.training import load_expert_sizes


def test_cpu_thread_auto_keeps_large_smt_bandwidth_headroom(monkeypatch):
    selected: dict[str, int] = {}
    import psutil

    monkeypatch.setenv("CCCP_CPU_THREADS", "auto")
    monkeypatch.setattr(cpuext, "configure_windows_performance", lambda: False)
    monkeypatch.setattr(cpuext.os, "cpu_count", lambda: 192)
    monkeypatch.setattr(
        psutil,
        "cpu_count",
        lambda logical=True: 192 if logical else 96,
    )
    monkeypatch.setattr(
        cpuext.torch,
        "set_num_threads",
        lambda value: selected.__setitem__("threads", int(value)),
    )
    monkeypatch.setattr(
        cpuext.torch,
        "get_num_threads",
        lambda: selected.get("threads", 1),
    )

    assert cpuext.configure_cpu_threads() == 72
    assert selected["threads"] == 72


def test_shared_prefill_scheduler_defaults_to_4096(monkeypatch):
    monkeypatch.delenv("CCCP_PREFILL_BLOCK_TOKENS", raising=False)

    assert prefill_block_size() == 4096
    assert list(prefill_ranges(9000)) == [
        (0, 4096),
        (4096, 8192),
        (8192, 9000),
    ]


def test_fixed_packed_arena_allows_shrink_but_forbids_fake_growth():
    source = inspect.getsource(PackedHybridPool.__init__)

    assert "self.supports_vram_watch = True" in source
    assert "self.supports_vram_growth = False" in source


def test_decode_arena_one_layer_short_causes_complete_cyclic_lru_thrash():
    route = [
        (layer, expert)
        for layer in range(43)
        for expert in range(6)
    ]

    def second_round_hits(capacity: int) -> int:
        slots = SlotBook(capacity)
        for key in route:
            slots.acquire(key)
        hits = 0
        for key in route:
            hits += int(key in slots._by_key)
            slots.acquire(key)
        return hits

    assert len(route) == 258
    assert second_round_hits(244) == 0
    assert second_round_hits(258) == 258


def test_packed_arena_uses_one_unprotected_strict_lru():
    source = inspect.getsource(PackedHybridPool.build_gpu_arenas)

    assert "policy=strict-lru" in source
    assert "permanent_protection=0" in source
    assert "route_history_minimum" not in source
    assert "_protect_previous" not in source
    assert "[cccp-cache-plan]" in source


def test_automatic_gpu_launch_discards_stale_parent_vram_limits(monkeypatch):
    monkeypatch.setenv("CCCP_VRAM_RESERVE_GB", "3")
    monkeypatch.setenv("CCCP_VRAM_LIMIT_GB", "16")
    monkeypatch.setenv("CCCP_EXTREME_VRAM_CAP_GB", "16")
    args = SimpleNamespace(
        gpus=None,
        extreme=False,
        vram_reserve_gb=None,
        vram_limit_gb=None,
        dense_bf16=None,
        ram_reserve_gb=None,
        prefill_block_tokens=None,
        prefill_moe_batch=None,
        pin_gb=None,
        single_gpu_layer_graph=None,
        h2d_batch=None,
        cpu_compile=None,
    )
    preset = SimpleNamespace(
        tp=1,
        ep_layout=None,
        environment={"CCCP_VRAM_RESERVE_GB": "1"},
    )

    _apply_environment(args, preset)

    assert os.environ["CCCP_VRAM_RESERVE_GB"] == "1"
    assert "CCCP_VRAM_LIMIT_GB" not in os.environ
    assert "CCCP_EXTREME_VRAM_CAP_GB" not in os.environ


def test_cache_diagnostics_split_host_staging_from_device_dma():
    stage = SimpleNamespace(
        host_staging_seconds=1.25,
        direct_upload_bytes=3 * 2**20,
        staged_upload_bytes=5 * 2**20,
        upload_submissions=7,
        upload_copies=11,
        batch_submissions=0,
        batch_copies=0,
        batch_fallbacks=0,
    )
    pool = SimpleNamespace(
        hits=2,
        miss=3,
        prefetch_hits=4,
        uploaded_bytes=8 * 2**20,
        transfer_seconds=0.5,
        gpu_arena_bytes=9 * 2**30,
        _host_pinned_bytes=10 * 2**30,
        _stage=stage,
    )
    engine = SimpleNamespace(
        model=SimpleNamespace(pool=pool),
        _vwatch=SimpleNamespace(trims=0, grows=0),
    )

    snapshot = _expert_cache_snapshot(engine)

    assert snapshot["transfer_seconds"] == pytest.approx(0.5)
    assert snapshot["host_staging_seconds"] == pytest.approx(1.25)
    assert snapshot["direct_upload_bytes"] == 3 * 2**20
    assert snapshot["staged_upload_bytes"] == 5 * 2**20
    assert snapshot["upload_submissions"] == 7
    assert snapshot["upload_copies"] == 11
    assert snapshot["watcher_trims"] == 0


def test_cache_log_names_make_transfer_bottleneck_unambiguous():
    source = inspect.getsource(
        sys.modules["cccp.chat_service"].ChatService.complete
    )

    for field in (
        "dma=", "host_stage=", "direct=", "staged=", "submissions=",
        "copies=", "compiled_batches=", "arena_slots=", "initial_free_slots=",
        "decode_executor=", "decode_fused=",
        "decode_graph=", "decode_reference=",
        "policy=strict-lru", "warmup_hot_slots=", "permanent_protection=0",
        "prefetch=off", "cuda_alloc=",
        "driver_free=", "process_limit=", "headroom=",
        "watcher_trims=", "watcher_grows=",
    ):
        assert field in source


def test_glm_reuses_lcp_when_chat_template_boundary_retokens():
    class FakeModel:
        def __init__(self):
            self.truncated_to = None

        def truncate_kv(self, keep):
            self.truncated_to = keep

    engine = object.__new__(Engine)
    engine.arch = "glm"
    engine.model = FakeModel()
    engine.quiet = True
    engine._cache_ids = [10, 11, 12, 13]
    engine._cache_media_digest = None
    engine._cache_media_slots = ()
    engine._active_media_slots = ()
    engine._prefill_glm_suffix = (
        lambda ids, skip, **kwargs: torch.tensor([float(skip)])
    )

    logits = engine._prepare_glm_prompt([10, 11, 90, 91, 92])

    assert logits.item() == 2.0
    assert engine.model.truncated_to == 2
    assert engine.last_kv_stats.mode == "lcp-replay"
    assert engine.last_kv_stats.reason == "live-prefix-diverged"
    assert engine.last_kv_stats.baseline_tokens == 2
    assert engine.last_kv_stats.lcp_tokens == 2
    assert engine.last_kv_stats.suffix_tokens == 3


def test_glm_sequential_prefill_is_cuda_only():
    source = inspect.getsource(Engine._prefill_glm_suffix)

    assert 'getattr(self.model, "device", torch.device("cpu")).type' in source
    assert '== "cuda"' in source


def test_dsv4_long_incremental_range_uses_layer_first_batch(monkeypatch):
    class FakePool:
        def __init__(self):
            self.phases = []

        def activate_prefill_arena(self):
            self.phases.append("prefill")

        def activate_decode_arena(self):
            self.phases.append("decode")

        def release_host_rows_workspace(self):
            self.phases.append("release")

    class FakeModel:
        def __init__(self):
            self.pos = 4
            self.pool = FakePool()
            self.batches = []

        def forward_incremental_batch(self, values):
            self.batches.append(list(values))
            self.pos += len(values)
            return torch.tensor([float(self.pos)])

        def forward(self, _values):
            raise AssertionError("long canonical suffix replayed tokenwise")

    engine = object.__new__(Engine)
    engine.model = FakeModel()
    engine.quiet = True
    engine._with_kv_capacity_retry = lambda fn, *args, **kwargs: fn(*args)
    logits = engine._dsv4_prefill_range(list(range(25)), 4, 25)

    assert logits.item() == 25.0
    assert engine.model.batches == [list(range(4, 25))]
    assert engine.model.pool.phases == ["prefill", "release", "decode"]


def test_dsv4_short_incremental_range_uses_fused_decode_without_repartition(
    monkeypatch,
):
    monkeypatch.setenv("CCCP_DSV4_WINDOWS_SHORT_PREFILL", "1")
    class FakePool:
        def __init__(self):
            self.phases = []

        def activate_prefill_arena(self):
            self.phases.append("prefill")

        def activate_decode_arena(self):
            self.phases.append("decode")

        def release_host_rows_workspace(self):
            self.phases.append("release")

    class FakeModel:
        def __init__(self):
            self.pos = 3
            self.pool = FakePool()
            self.batches = []

        def forward_incremental_batch(self, values):
            raise AssertionError("short continuation entered grouped Prefill")

        def forward(self, values):
            assert self._canonical_short_decode is True
            self.batches.append(list(values))
            self.pos += len(values)
            return torch.tensor([float(self.pos)])

    engine = object.__new__(Engine)
    engine.model = FakeModel()
    engine.quiet = True
    engine._with_kv_capacity_retry = lambda fn, *args, **kwargs: fn(*args)
    logits = engine._dsv4_prefill_range(list(range(8)), 3, 8)

    assert logits.item() == 8.0
    assert engine.model.batches == [[3, 4, 5, 6, 7]]
    assert engine.model.pool.phases == []
    assert engine.model._canonical_short_decode is False


def test_dsv4_single_token_suffix_uses_fused_decode(monkeypatch):
    monkeypatch.setenv("CCCP_DSV4_WINDOWS_SHORT_PREFILL", "1")
    class FakePool:
        def __init__(self):
            self.phases = []

        def activate_prefill_arena(self):
            self.phases.append("prefill")

        def activate_decode_arena(self):
            self.phases.append("decode")

        def release_host_rows_workspace(self):
            self.phases.append("release")

    class FakeModel:
        def __init__(self):
            self.pos = 4
            self.pool = FakePool()
            self.batches = []

        def forward_incremental_batch(self, values):
            raise AssertionError("one-token continuation entered grouped Prefill")

        def forward(self, values):
            assert self._canonical_short_decode is True
            self.batches.append(list(values))
            self.pos += len(values)
            return torch.tensor([float(self.pos)])

    engine = object.__new__(Engine)
    engine.model = FakeModel()
    engine.quiet = True
    engine._with_kv_capacity_retry = lambda fn, *args, **kwargs: fn(*args)

    logits = engine._dsv4_prefill_range([10, 11, 12, 13, 14], 4, 5)

    assert logits.item() == 5.0
    assert engine.model.batches == [[14]]
    assert engine.model.pool.phases == []
    assert engine.model._canonical_short_decode is False


@pytest.mark.skipif(os.name != "nt", reason="Windows execution policy")
def test_dsv4_windows_short_suffix_keeps_verified_fused_decode(monkeypatch):
    monkeypatch.delenv("CCCP_DSV4_WINDOWS_SHORT_PREFILL", raising=False)

    class FakePool:
        def __init__(self):
            self._arena_phase = "decode"
            self.phases = []

        def activate_prefill_arena(self):
            self._arena_phase = "prefill"
            self.phases.append("prefill")

        def activate_decode_arena(self):
            self._arena_phase = "decode"
            self.phases.append("decode")

        def release_host_rows_workspace(self):
            self.phases.append("release")

    class FakeModel:
        def __init__(self):
            self.pos = 3
            self.pool = FakePool()
            self.batches = []

        def forward(self, values):
            assert self._canonical_short_decode is True
            self.batches.append(list(values))
            self.pos += len(values)
            return torch.tensor([float(self.pos)])

        def forward_incremental_batch(self, _values):
            raise AssertionError("short suffix entered unverified WDDM batch path")

    engine = object.__new__(Engine)
    engine.model = FakeModel()
    engine.quiet = True
    engine._with_kv_capacity_retry = lambda fn, *args, **kwargs: fn(*args)

    logits = engine._dsv4_prefill_range(list(range(8)), 3, 8)

    assert logits.item() == 8.0
    assert engine.model.batches == [[3, 4, 5, 6, 7]]
    assert engine.model.pool.phases == []


def test_dsv4_nonfinite_incremental_logits_fail_without_hidden_rebuild():
    class FakePool:
        def __init__(self):
            self.phases = []

        def activate_prefill_arena(self):
            self.phases.append("prefill")

        def activate_decode_arena(self):
            self.phases.append("decode")

        def release_host_rows_workspace(self):
            self.phases.append("release")

    class FakeModel:
        def __init__(self):
            self.device = torch.device("cpu")
            self.pos = 2
            self.pool = FakePool()
            self._last_prefill_scheduler = "incremental-layer-first"

    engine = object.__new__(Engine)
    engine.model = FakeModel()
    engine.quiet = True
    engine._cache_ids = [10, 11]
    engine._kv_baseline = None
    engine._kv_prefill_events = None
    engine._save_dsv4_baseline = lambda *_args: 0

    incremental_calls = []
    rebuild_calls = []

    def invalid_incremental(ids, start, stop, *, manage_arena=True):
        incremental_calls.append((list(ids), start, stop, manage_arena))
        engine.model.pos = stop
        return torch.full((8,), float("nan"))

    def forbidden_rebuild(ids, boundary, *, manage_arena=True):
        rebuild_calls.append((list(ids), boundary, manage_arena))
        raise AssertionError("non-finite incremental output was rerun")

    engine._dsv4_prefill_range = invalid_incremental
    engine._prefill_from_reset_to_boundary = forbidden_rebuild

    with pytest.raises(RuntimeError, match="non-finite logits before sampling"):
        engine._prepare_dsv4_prompt([10, 11, 12], 3)

    assert incremental_calls == [([10, 11, 12], 2, 3, False)]
    assert rebuild_calls == []
    assert engine.model.pool.phases == []


def test_dsv4_canonical_short_decode_bypasses_static_graph_packets():
    block_source = inspect.getsource(DSV4CCCPModel._block)
    attention_source = inspect.getsource(DSV4CCCPModel._attn_batch)

    assert "and not canonical_short_decode" in block_source
    assert '_canonical_short_decode", False' in attention_source


def test_cuda_route_kernels_keep_masked_experts_out_on_nonfinite_scores():
    source = (
        ENGINE_ROOT / "cccp" / "csrc" / "vq_gemv.cu"
    ).read_text(encoding="utf-8")

    dsv4 = source.split(
        "__global__ void dsv4_route_post_kernel", 1
    )[1].split("std::vector<torch::Tensor> dsv4_route_post", 1)[0]
    glm = source.split(
        "__global__ void sigmoid_route_select_kernel", 1
    )[1].split("__device__ __forceinline__ uint32_t route_ordered_float", 1)[0]
    radix = source.split(
        "__global__ void sigmoid_route_radix_kernel", 1
    )[1].split("void launch_sigmoid_route", 1)[0]

    assert "? (isfinite(corrected) ? corrected : -FLT_MAX)" in dsv4
    assert "if (!isfinite(value))" in dsv4
    assert "? (isfinite(corrected) ? corrected : -FLT_MAX)" in glm
    assert "choice = isfinite(corrected) ? corrected : -FLT_MAX" in radix


def test_decode_arena_repairs_missing_runtime_buffers():
    pool = object.__new__(PackedHybridPool)
    pool._decode_arena_target_budget = 100
    pool._extreme_specs = None
    pool._arena_phase = "prefill"
    pool._arenas = SimpleNamespace(nbytes=100)
    pool._workspaces = None
    pool._metadata = None
    pool._route_ids = None
    pool._ordered_weights = None
    pool.pinned = {(0, 0): ()}
    calls = []

    def rebuild(budget, *, force_rebuild=False, **_kwargs):
        calls.append((budget, force_rebuild))
        pool._workspaces = (object(),)
        pool._metadata = object()
        pool._route_ids = object()
        pool._ordered_weights = object()
        return 100, 100

    pool.resize_gpu_arenas = rebuild

    assert pool.activate_decode_arena() == (100, 100)
    assert calls == [(100, True)]
    assert pool._arena_phase == "decode"


def test_dsv4_canonical_commit_removes_private_replay_from_next_turn():
    class FakeSnapshot:
        def __init__(self, pos):
            self.pos = pos
            self.nbytes = pos * 8

    class FakeModel:
        def __init__(self):
            # The live branch contains private reasoning and is deliberately
            # different from the public conversation that the adapter keeps.
            self.pos = 6
            self.device = torch.device("cpu")
            self.restored = []

        def restore_kv(self, snapshot):
            self.restored.append(snapshot.pos)
            self.pos = snapshot.pos

        def snapshot_kv(self):
            return FakeSnapshot(self.pos)

    engine = object.__new__(Engine)
    engine.arch = "dsv4"
    engine.model = FakeModel()
    engine.quiet = True
    engine._cache_ids = [10, 11, 90, 91, 92, 93]
    engine._kv_baseline = SimpleNamespace(
        ids=[10, 11],
        snapshot=FakeSnapshot(2),
    )
    engine._kv_prefill_events = None
    ranges = []

    def advance(ids, start, stop, **_kwargs):
        ranges.append((start, stop, list(ids[start:stop])))
        engine.model.pos = stop
        return torch.tensor([float(stop)])

    engine._dsv4_prefill_range = advance
    canonical = [10, 11, 20, 21, 22]

    engine.commit_canonical_history(canonical)

    assert engine.model.restored == [2]
    assert ranges == [(2, 5, [20, 21, 22])]
    assert engine._cache_ids == canonical
    assert engine._kv_baseline.ids == canonical
    assert engine._kv_baseline.snapshot.pos == len(canonical)

    logits = engine._prepare_dsv4_prompt(
        canonical + [30, 31],
        baseline_len=len(canonical) + 2,
    )

    assert logits.item() == 7.0
    assert ranges[-1] == (5, 7, [30, 31])
    assert engine.last_kv_stats.reason == "live-prefix"
    assert engine.last_kv_stats.replay_tokens == 0
    assert engine.last_kv_stats.suffix_tokens == 2
    assert engine.last_kv_stats.processed_tokens == 2


def test_dsv4_full_batch_workspace_estimate_scales_to_4096():
    assert _dsv4_prefill_workspace_reserve_gb(512) == pytest.approx(1.09375)
    assert _dsv4_prefill_workspace_reserve_gb(4096) == pytest.approx(8.75)
    assert _dsv4_prefill_workspace_reserve_gb(32768) == pytest.approx(8.75)


def test_reported_20g_plan_deducts_the_single_total_reserve_once():
    # The GUI contract exposes one total 1-GiB physical safety line. Dynamic
    # Prefill scratch is live-chunked and must not be pre-deducted a second
    # time from the fixed expert arena.
    selected = _safe_expert_budget(
        limit_bytes=int(18.77 * 2**30),
        allocated_bytes=int(17.21 * 2**30),
        expert_bytes=int(8.49 * 2**30),
        requested_bytes=int(8.50 * 2**30),
        reserve_bytes=int(1.0 * 2**30),
    )

    assert selected / 2**30 == pytest.approx(8.50, abs=0.01)
    assert selected / 2**30 > 2.58


def test_fixed_arenas_use_one_total_reserve_and_live_chunked_workspace():
    engine_source = inspect.getsource(Engine.__init__)
    arena_source = inspect.getsource(PackedHybridPool._safe_budget)

    assert 'os.environ["CCCP_VRAM_HEADROOM_GB"]' in engine_source
    assert 'live_chunked_workspace = arch_hint in {' in engine_source
    assert '"dsv4", "qwen3_5_dense"' in engine_source
    assert "if live_chunked_workspace" in engine_source
    assert "'live-chunked' if live_chunked_workspace" in engine_source
    assert '"CCCP_VRAM_HEADROOM_GB"' in arena_source
    assert "phase=runtime-headroom" in engine_source


def test_prefill_expert_chunk_keeps_configured_vram_headroom():
    source = inspect.getsource(
        PackedHybridPool._prefill_dequant_chunk_capacity
    )

    assert '"CCCP_VRAM_RESERVE_GB"' in source
    assert '"CCCP_VRAM_HEADROOM_GB"' in source
    assert "configured_headroom * 2**30" in source


def test_prefill_cleanup_runs_before_decode_arena_restore():
    reset_source = inspect.getsource(Engine._prefill_from_reset_to_boundary)
    range_source = inspect.getsource(Engine._dsv4_prefill_range)

    for source in (reset_source, range_source):
        assert "release_host_rows_workspace" in source
        assert source.index("release_workspace()") < source.index(
            "activate_decode()"
        )


def test_windows_defaults_to_registered_source_direct_dma(monkeypatch):
    store_module = sys.modules["cccp.store"]
    monkeypatch.setattr(store_module, "_WINDOWS", True)
    monkeypatch.setattr(store_module, "_ROCM", False)
    monkeypatch.delenv("CCCP_WDDM_DIRECT_PIN", raising=False)

    assert _prefer_direct_pinned_h2d() is True

    monkeypatch.setenv("CCCP_WDDM_DIRECT_PIN", "0")
    assert _prefer_direct_pinned_h2d() is False


def test_windows_rocm_never_uses_externally_registered_direct_dma(monkeypatch):
    store_module = sys.modules["cccp.store"]
    monkeypatch.setattr(store_module, "_WINDOWS", True)
    monkeypatch.setattr(store_module, "_ROCM", True)
    monkeypatch.setenv("CCCP_WDDM_DIRECT_PIN", "1")

    assert store_module._prefer_direct_pinned_h2d() is False


def test_windows_rocm_hybrid_arena_has_fixed_safe_single_allocation_cap():
    source = inspect.getsource(PackedHybridPool._safe_budget)

    assert "hip_single_arena_cap = int(1.5 * 2**30)" in source
    assert "selected = min(selected, hip_single_arena_cap)" in source


def test_profile_hot_plan_uses_global_counts_after_topk_reserve():
    weight = HostPackedWeight(
        raw=torch.empty(4, dtype=torch.uint8),
        cb=torch.empty(2, 2, dtype=torch.bfloat16),
        rows=2,
        cols=2,
        blocks=1,
        dim=2,
        bits=8,
    )
    expert = (weight, weight)
    signature = PackedExpertSignature.of(expert)
    pool = object.__new__(PackedHybridPool)
    pool._adaptive_decode_arena = False
    pool._extreme_specs = None
    pool.store = SimpleNamespace(
        heat_ranks={0: [0, 1, 2, 3], 1: [0, 1, 2, 3]},
        heat_counts={
            0: {0: 1000, 1: 900, 2: 800, 3: 700},
            1: {0: 10, 1: 9, 2: 8, 3: 7},
        },
    )
    pool.pinned = {
        (layer, expert_id): expert
        for layer in range(2)
        for expert_id in range(4)
    }

    selected = pool._plan_profile_hot_keys(
        {signature: 8},
        {signature: 6},
    )

    assert len(selected) == 2
    assert selected == ((0, 0), (0, 1))
    assert len(set(pool.pinned) - set(selected)) == 6


def test_profile_hot_is_only_lru_start_order_and_keeps_ram_complete():
    activate_source = inspect.getsource(
        PackedHybridPool.activate_prefill_arena
    )
    resize_source = inspect.getsource(PackedHybridPool.resize_gpu_arenas)
    profile_warm_source = inspect.getsource(
        PackedHybridPool._warm_profile_hot_locked
    )

    assert not hasattr(PackedHybridPool, "_release_profile_hot_hosts_locked")
    assert not hasattr(PackedHybridPool, "_rehydrate_gpu_only_hot_hosts")
    assert "_rehydrate_gpu_only_hot_hosts" not in activate_source
    assert "_rehydrate_gpu_only_hot_hosts" not in resize_source
    assert "reversed(self.profile_hot_keys)" in profile_warm_source
    assert "self.pinned.pop" not in profile_warm_source
    assert "permanent_protection=0" in profile_warm_source


def test_residency_plan_accounts_dense_topk_prefill_and_hot_remainder():
    source = inspect.getsource(PackedHybridPool.build_gpu_arenas)

    assert "dense_and_fixed=" in source
    assert "prefill_layer_max=" in source
    assert "largest_expert=" in source
    assert "topk_largest_floor=" in source
    assert "hot_start_bytes=" in source
    assert "top_k * self.max_expert_slot_bytes" in source


def test_glm_expert_arena_uses_live_post_dense_capacity_clamp():
    assert _initial_expert_vram_request_gb(
        architecture="glm",
        planning_free_gb=24.0,
        dense_estimate_gb=16.9,
        runtime_margin_gb=3.0,
        extreme=False,
    ) == pytest.approx(24.0)
    assert _initial_expert_vram_request_gb(
        architecture="dsv4",
        planning_free_gb=24.0,
        dense_estimate_gb=9.9,
        runtime_margin_gb=9.75,
        extreme=False,
    ) == pytest.approx(24.0)


def test_glm_cuda_prefill_cannot_fall_back_to_per_expert_projection():
    moe_source = inspect.getsource(GLMModel._moe)

    assert "per-token/per-expert projection implementation was deleted" in moe_source
    assert "for e in eids:" not in moe_source
    assert "gu.matmul_T(x[toks])" not in moe_source
    assert "dn.matmul_T(inter)" not in moe_source


def test_legacy_glm_cpu_prefill_is_grouped_by_expert_not_token():
    source = inspect.getsource(ExpertPool._run_rows_cpu)
    dispatch = inspect.getsource(GLMModel._moe)

    assert "for expert in unique_ids:" in source
    assert "vq_gemv_list_cpu(" in source
    assert "gate_up.matmul_T(expert_source)" in source
    assert "down.matmul_T(activated)" in source
    assert "for token" not in source
    row_branch = dispatch.split("if (\n            x.shape[0] > 1", 1)[1].split(
        "if (\n            x.shape[0] == 1", 1
    )[0]
    assert "packed_expert_vq" not in row_branch
    assert "projection_vq" not in row_branch


def test_legacy_glm_cpu_prefill_grouped_rows_matches_dense_reference():
    def weight(indices, codebook, cols):
        return VQWeight(
            torch.tensor(indices, dtype=torch.uint8),
            torch.tensor(codebook, dtype=torch.float32).reshape(-1, 1),
            cols,
        )

    experts = {
        (3, 0): (
            weight([[0, 1], [1, 0], [2, 0], [0, 2]], [0.0, 1.0, -1.0], 2),
            weight([[1, 0], [0, 1]], [0.0, 1.0, -1.0], 2),
        ),
        (3, 1): (
            weight([[1, 1], [2, 2], [1, 0], [0, 1]], [0.0, 0.5, -0.5], 2),
            weight([[2, 0], [0, 2]], [0.0, 0.5, -0.5], 2),
        ),
    }
    pool = object.__new__(ExpertPool)
    pool._prefill_executor_announced = True
    pool.prefill_batch_submissions = 0
    pool.prefill_batch_rows = 0
    pool.prefill_batch_max = 0
    pool.prefill_layer_unique_max = 0
    pool.get_many = lambda keys: {key: experts[key] for key in keys}
    values = torch.tensor([[1.0, 2.0], [-1.0, 0.5], [0.25, -2.0]])
    route_ids = torch.tensor([[0, 1], [1, 0], [0, 0]])
    route_weights = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.25, 0.75]])

    actual = pool._run_rows_cpu(
        3,
        values,
        route_ids,
        route_weights,
        activation="silu",
        activation_beta=4.0,
        activation_linear_beta=25.0,
    )
    expected = torch.zeros_like(actual)
    for token in range(values.shape[0]):
        for slot in range(route_ids.shape[1]):
            gate_up, down = experts[(3, int(route_ids[token, slot]))]
            hidden = values[token] @ gate_up.dequant().t()
            gate, up = hidden.chunk(2)
            projected = (torch.nn.functional.silu(gate) * up) @ down.dequant().t()
            expected[token] += route_weights[token, slot] * projected

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert pool.prefill_batch_submissions == 1
    assert pool.prefill_batch_rows == 3


def test_glm_decode_cannot_restore_single_expert_projection_fallback():
    moe_source = inspect.getsource(GLMModel._moe)

    assert "single-token expert projection implementation was deleted" in moe_source
    assert 'os.environ.get("CCCP_GROUPED"' not in moe_source
    assert "moe_mlp_grouped_mixed(" in moe_source


def test_profile_gpu_preload_interleaves_layers_before_next_heat_rank():
    source = inspect.getsource(ExpertPool.preload_profile_gpu)

    rank_loop = source.index("for rank in range(max_rank):")
    layer_loop = source.index("for layer, experts in per_layer.items():")
    admission = source.index("for key in ordered:")
    assert rank_loop < layer_loop < admission


def test_glm_mtp_has_no_per_expert_projection_loop():
    from cccp.mtp import MTPHead

    moe_source = inspect.getsource(MTPHead._moe)
    assert "for e in idx.unique().tolist()" not in moe_source
    assert "gu.matmul_T(x[toks])" not in moe_source
    assert "dn.matmul_T(inter)" not in moe_source
    assert "run_rows(" in moe_source
    assert "moe_mlp_grouped_mixed(" in moe_source


def test_dsv_cuda_cannot_enter_generic_expert_projection_fallback():
    source = inspect.getsource(DSV4CCCPModel._moe)
    guard = source.index("if x_rows.is_cuda:")
    generic_loop = source.index("for e in present:")

    assert guard < generic_loop
    assert "single-token/per-expert projection fallback was deleted" in source


def test_legacy_glm_prefill_metadata_describes_combined_gu_and_down():
    gu = VQWeight(
        torch.zeros(8, 4, dtype=torch.uint8),
        torch.randn(256, 2),
        8,
    )
    down = VQWeight(
        torch.zeros(4, 4, dtype=torch.uint16),
        torch.randn(512, 2),
        8,
    )

    rows = ExpertPool._legacy_prefill_metadata_rows([(gu, down)])

    assert len(rows) == 10
    assert all(len(row) == 1 for row in rows)
    assert rows[0][0] == gu.idx.data_ptr()
    assert rows[5][0] == down.idx.data_ptr()
    assert rows[2][0] == rows[7][0] == 4
    assert rows[3][0] == rows[8][0] == 2
    assert rows[4][0] == 0
    assert rows[9][0] == 1


def test_hybrid_cuda_prefill_has_no_decode_gemv_fallback():
    source = inspect.getsource(PackedHybridPool.run_rows)
    default_rows = inspect.signature(
        PackedHybridPool.run_rows
    ).parameters["prefill_default"].default

    assert "packed_moe_topk(" not in source
    assert source.count("torch._grouped_mm(") == 2
    assert "hip_packed_prefill" in source
    assert "packed_moe_topk_grouped(" in source
    assert "decode GEMV fallback=forbidden" in source
    assert "count = max(1, count // 2)" not in source
    assert "expert_chunks" in source
    assert default_rows == 4096


def test_hybrid_cuda_prefill_waits_for_current_h2d_before_dequant():
    source = inspect.getsource(PackedHybridPool.run_rows)
    upload = source.index("selected = self._ensure_locked(")
    wait = source.index("self._stage.wait()", upload)
    dequant = source.index("projection_dequant(", upload)

    assert upload < wait < dequant


def test_windows_auto_h2d_uses_compiled_layer_batch(monkeypatch):
    from cccp import store

    monkeypatch.delenv("CCCP_H2D_BATCH", raising=False)
    monkeypatch.setattr(store, "_WINDOWS", True)
    monkeypatch.setattr(store, "_ROCM", False)
    assert store._should_batch_h2d(1) is False
    assert store._should_batch_h2d(2) is True
    assert store._should_batch_h2d(128) is True


def test_windows_compiled_h2d_batch_avoids_native_cuda_batch_api():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "engine"
        / "CCCP-Engine"
        / "cccp"
        / "csrc"
        / "vq_gemv.cu"
    )
    source = source_path.read_text(encoding="utf-8")
    windows_start = source.index("#if defined(_WIN32)")
    linux_start = source.index("#elif CUDART_VERSION >= 12080", windows_start)
    windows_branch = source[windows_start:linux_start]
    linux_branch = source[linux_start:]

    assert "cudaMemcpyAsync(" in windows_branch
    assert "cudaMemcpyBatchAsync(" not in windows_branch
    assert "cudaMemcpyBatchAsync(" in linux_branch


def test_linux_auto_h2d_keeps_native_batch_submission(monkeypatch):
    from cccp import store

    monkeypatch.delenv("CCCP_H2D_BATCH", raising=False)
    monkeypatch.setattr(store, "_WINDOWS", False)
    monkeypatch.setattr(store, "_ROCM", False)
    assert store._should_batch_h2d(1) is False
    assert store._should_batch_h2d(2) is True


def test_windows_rocm_h2d_uses_safe_async_copy_stream(monkeypatch):
    from cccp import store

    monkeypatch.setenv("CCCP_H2D_BATCH", "1")
    monkeypatch.setattr(store, "_WINDOWS", True)
    monkeypatch.setattr(store, "_ROCM", True)

    assert store._should_batch_h2d(128) is False


def test_cuda_audit_explains_windows_shared_memory_without_claiming_offload():
    source = inspect.getsource(Engine.__init__)
    assert "Windows 的“共享 GPU 内存”可能包含" in source
    assert "这是异步 DMA 缓存，不代表专用显存溢出，也不是磁盘卸载" in source
    assert "未占满部分保留给 Attention、KV、Prefill 与 WDDM" in source


def test_hybrid_cuda_prefill_bridges_complete_host_rows_once():
    source = inspect.getsource(PackedHybridPool.run_rows)
    prepare = inspect.getsource(
        PackedHybridPool._prepare_host_rows_bridge
    )
    finish = inspect.getsource(
        PackedHybridPool._finish_host_rows_bridge
    )

    assert "if not value.is_cuda:" in source
    assert "self._prepare_host_rows_bridge(" in source
    assert "self._finish_host_rows_bridge(device_result)" in source
    assert "input_buffer.copy_(value.to(torch.bfloat16))" in prepare
    assert "ids_buffer.copy_(route_ids.to(torch.long))" in prepare
    assert "host_result.copy_(device_result, non_blocking=True)" in finish
    assert "self._retain_prefill_workspace = True" in source
    release = inspect.getsource(
        PackedHybridPool.release_host_rows_workspace
    )
    assert "self._prefill_dequant_workspace = None" in release


def test_kimi_ram_dense_cuda_prefill_uses_layer_major_rows():
    from cccp.kimi_model import KimiK3CCCPModel

    block_source = inspect.getsource(
        KimiK3CCCPModel.forward_hidden_block_cpu
    )
    hidden_source = inspect.getsource(KimiK3CCCPModel.forward_hidden)
    logits_source = inspect.getsource(KimiK3CCCPModel.forward)

    assert "self._ram_dense_cuda" not in block_source.split(
        'raise RuntimeError("Kimi single-device prefill', 1
    )[0]
    assert "and not self._ram_dense_cuda" not in hidden_source
    assert "and not self._ram_dense_cuda" not in logits_source

    kda_source = inspect.getsource(
        KimiK3CCCPModel._kda_attention_block_cpu
    )
    mla_source = inspect.getsource(
        KimiK3CCCPModel._mla_attention_block_cpu
    )
    assert "self._kda_dynamic_hybrid_rows(" in kda_source
    assert "self._mla_dynamic_hybrid_rows(" in mla_source
    assert "release_host_rows_workspace" in block_source


def test_kimi_full_resident_payload_precedes_runtime_graph_capture():
    """Large TP expert writes must finish before KDA/Dense graph capture."""
    from cccp.kimi_model import KimiK3CCCPModel

    source = inspect.getsource(KimiK3CCCPModel.preload)
    payload = source.index("self.pool.preload(capture_graphs=False)")
    kda = source.index("self._prepare_tp_kda()")
    runtime_graphs = source.index("self.pool.preload()", kda)
    assert payload < kda < runtime_graphs
    final_kda = source.rindex("self._tp_kda.capture()")
    parent_plans = source.index("self._prepare_no_owner_moe_plans()")
    assert runtime_graphs < parent_plans < final_kda

    from cccp.ops.tensor_parallel import TensorParallelKDA

    capture_source = inspect.getsource(TensorParallelKDA.capture)
    assert "if layers is None" in capture_source
    assert "state.output_replicas is None" in capture_source
    assert "state.output_events is None" in capture_source

    config = json.loads(
        (
            ENGINE_ROOT / "cccp" / "configs" / "kimi_k3.json"
        ).read_text(encoding="utf-8")
    )
    for profile in ("parallel", "parallel_tp4", "parallel_tp8"):
        assert (
            config["profiles"][profile]["environment"]
            ["CCCP_TP_DECODE_LAYER_PLAN"]
            == "0"
        )


def test_small_vram_uses_fixed_slab_route_repartition_not_token_split():
    build_source = inspect.getsource(PackedHybridPool.build_gpu_arenas)
    route_source = inspect.getsource(
        PackedHybridPool._ensure_decode_route_capacity_locked
    )

    assert "topk_minimum_bytes > arena_budget" in build_source
    assert "signature: min(int(count), 1)" in build_source
    assert "signature: min(int(count), 1)" in route_source
    assert "min(4, top_k)" not in build_source + route_source
    assert "self._arenas.repartition(specs)" in route_source
    assert "self._default_arena_specs = dict(specs)" in route_source
    assert "run_rows" not in route_source

    prefill_partition = inspect.getsource(
        PackedHybridPool._partition_prefill_layer_locked
    )
    assert "if required > self._arenas.nbytes" in prefill_partition
    assert "return" in prefill_partition
    assert "one complete packed expert layer exceeds" not in prefill_partition

    host_decode = inspect.getsource(PackedHybridPool.prepare_host_run)
    assert host_decode.index("_restore_decode_arena_locked") < host_decode.index(
        "_prepare_selected_run_locked"
    )


def test_dsv4_prefill_compressor_projects_the_complete_batch_once():
    source = inspect.getsource(_compressor_prefill_cccp)

    assert "_compressor_decode_cccp(" not in source
    assert "kv = _cccp_lin(x, w[\"wkv\"])" in source
    assert "score = _cccp_lin(x, w[\"wgate\"])" in source


def test_dsv4_sliding_attention_query_tiles_equal_full_outer_block():
    torch.manual_seed(19)
    ring = torch.randn(1, 4, 3)
    ring_positions = torch.tensor([[-4, -3, -2, -1]])
    current = torch.randn(1, 11, 3)
    full_values, full_valid = _prefill_sliding_window(
        ring, ring_positions, current, 0
    )
    value_tiles = []
    valid_tiles = []
    for start, count in ((0, 3), (3, 5), (8, 3)):
        values, valid = _prefill_sliding_window(
            ring,
            ring_positions,
            current,
            0,
            query_start=start,
            query_count=count,
        )
        value_tiles.append(values)
        valid_tiles.append(valid)

    assert torch.equal(torch.cat(value_tiles, dim=1), full_values)
    assert torch.equal(torch.cat(valid_tiles, dim=1), full_valid)


def test_glm_latent_attention_query_batches_equal_full_matrix():
    torch.manual_seed(23)
    heads, tokens, prefix, latent, rope = 3, 11, 5, 7, 4
    sequence = prefix + tokens
    qa = torch.randn(heads, tokens, latent)
    qrot = torch.randn(heads, tokens, rope)
    ckv = torch.randn(sequence, latent)
    krot = torch.randn(sequence, rope)

    full = _latent_attention_context_batched(
        qa,
        qrot,
        ckv,
        krot,
        scale=3.25,
        pos0=prefix,
        query_batch=tokens,
    )
    tiled = _latent_attention_context_batched(
        qa,
        qrot,
        ckv,
        krot,
        scale=3.25,
        pos0=prefix,
        query_batch=3,
    )

    torch.testing.assert_close(tiled, full, rtol=1e-5, atol=1e-6)


def test_dsv4_decode_compressor_rejects_multi_token_projection():
    with pytest.raises(RuntimeError, match="decode-only"):
        _compressor_decode_cccp(
            torch.empty(1, 2, 4),
            {},
            4,
            2,
            0,
            torch.empty(1, 1, 0),
            torch.empty(1, 1, 0),
            1e-6,
            {},
            0,
        )


def test_hip_short_reset_prompt_uses_fused_decode_without_changing_cuda_cpu(
    monkeypatch,
):
    monkeypatch.setattr(torch.version, "hip", "6.4")
    assert _use_hip_short_reset_decode(1) is True
    assert _use_hip_short_reset_decode(16) is True
    assert _use_hip_short_reset_decode(17) is False

    monkeypatch.setattr(torch.version, "hip", None)
    assert _use_hip_short_reset_decode(4) is False


def test_dsv4_tp1_graph_buckets_and_hip_capture_policy_are_bounded():
    assert _tp1_token_graph_bucket(0) == "direct32"
    assert _tp1_token_graph_bucket(130) == "direct32"
    assert _tp1_token_graph_bucket(131) == "direct128"
    assert _tp1_token_graph_bucket(514) == "direct128"
    assert _tp1_token_graph_bucket(515) == "direct512"
    assert _tp1_token_graph_bucket(2050) == "direct512"
    assert _tp1_token_graph_bucket(2051) == "topk512"
    assert _tp1_token_graph_bucket(0, hip_runtime=True) == "direct128"
    assert _tp1_token_graph_bucket(514, hip_runtime=True) == "direct128"
    assert _tp1_token_graph_bucket(515, hip_runtime=True) == "direct512"

    source = inspect.getsource(DSV4CCCPModel._prepare_tp1_token_graphs)
    assert "hip_runtime = torch.version.hip is not None" in source
    assert "spec[0] == requested_bucket" in source
    assert "AMD/HIP 按需单 bucket" in source
    assert "else '全 bucket 预捕获'" in source


def test_dsv4_long_cuda_requires_flashmla_and_indexer_alignment():
    assert _requires_flashmla_splitkv(
        device_type="cuda", hip_runtime=False, max_ctx=2051
    ) is False
    assert _requires_flashmla_splitkv(
        device_type="cuda", hip_runtime=False, max_ctx=2052
    ) is True
    assert _requires_flashmla_splitkv(
        device_type="cuda", hip_runtime=True, max_ctx=4096
    ) is False
    assert _requires_flashmla_splitkv(
        device_type="cpu", hip_runtime=False, max_ctx=4096
    ) is False
    assert _indexer_candidate_capacity(1) == 16
    assert _indexer_candidate_capacity(1024) == 1024
    assert _indexer_candidate_capacity(1092) == 1104
    with pytest.raises(ValueError, match="positive"):
        _indexer_candidate_capacity(0)

    attention_source = inspect.getsource(
        DSV4CCCPModel._tp_controlled_attention
    )
    assert "禁止退回 BF16 Indexer" in attention_source
    assert "sparse_selected" not in attention_source


def test_hip_full_resident_selects_tp1_before_pool_construction():
    source = inspect.getsource(DSV4CCCPModel.__init__)
    forced = source.index(
        'os.environ["CCCP_SINGLE_GPU_LAYER_GRAPH"] = "1"'
    )
    constructed = source.index("self.pool = PackedExpertPool(")

    assert forced < constructed
    assert "AMD/HIP full-resident pool must be initialized" in source


def test_dsv4_paged_kv_reserved_during_inference_remains_mutable():
    cache = PagedKV(
        batch=1,
        page_items=8,
        dim=4,
        device="cpu",
    )
    with torch.inference_mode():
        cache.reserve(7)

    assert cache.pages[0].is_inference() is False
    cache.write_many(0, torch.ones(1, 3, 4))
    torch.testing.assert_close(
        cache.pages[0][:, :3],
        torch.ones(1, 3, 4, dtype=torch.bfloat16),
    )


def test_hip_full_resident_never_leaves_token_graph_for_stage_probe(
    monkeypatch,
):
    class FakeModel:
        device = torch.device("cuda")
        tp_size = 1
        _tp_attention_contexts = object()
        _profile_enabled = False
        _packed_full_gpu = True

        def __init__(self):
            self.starts = 0

        def start_profile(self):
            self.starts += 1

        def finish_profile(self):
            return {
                "tensor_parallel": {
                    "critical_path_ms": 12.5,
                    "totals": {
                        "attention_ms": 2.0,
                        "moe_ms": 9.0,
                        "ffn_post_ms": 1.5,
                    },
                },
            }

    monkeypatch.setattr(torch.version, "hip", "6.4")
    model = FakeModel()
    function = lambda value: value + 1

    assert _profile_dsv4_stage_call(
        model, "decode-token", 1, function, 4
    ) == 5
    assert model.starts == 0
    assert not hasattr(model, "_cccp_hip_stage_probe_done")


def test_hip_full_resident_decode_cannot_profile_around_token_graph():
    source = inspect.getsource(DSV4CCCPModel._decode_tp)

    assert "hip_full_resident = bool(" in source
    assert "not self._profile_enabled or hip_full_resident" in source
    graph_call = source.index("self._decode_tp1_token_graph(ids, pos)")
    hard_failure = source.index(
        "AMD/HIP full-resident decode requires the TP1 TokenGraph"
    )
    assert graph_call < hard_failure


def test_hip_hybrid_stage_probe_runs_once(monkeypatch, capsys):
    class FakeModel:
        device = torch.device("cuda")
        tp_size = 1
        _tp_attention_contexts = object()
        _profile_enabled = False
        _packed_full_gpu = False

        def __init__(self):
            self.starts = 0

        def start_profile(self):
            self.starts += 1

        def finish_profile(self):
            return {
                "tensor_parallel": {
                    "critical_path_ms": 12.5,
                    "totals": {
                        "attention_ms": 2.0,
                        "moe_ms": 9.0,
                        "ffn_post_ms": 1.5,
                    },
                },
            }

    monkeypatch.setattr(torch.version, "hip", "6.4")
    model = FakeModel()
    function = lambda value: value + 1

    assert _profile_dsv4_stage_call(
        model, "decode-token", 1, function, 4
    ) == 5
    assert model.starts == 1
    assert model._cccp_hip_stage_probe_done is True
    assert "[cccp-stage-tp]" in capsys.readouterr().out

    assert _profile_dsv4_stage_call(
        model, "decode-token", 1, function, 5
    ) == 6
    assert model.starts == 1


def test_dsv4_vectorized_compressor_matches_projected_state_machine():
    torch.manual_seed(7)
    ratio = 4
    width = 4
    rope_width = 2
    tokens = 11
    hidden = 8
    x = torch.randn(1, tokens, hidden)
    weights = {
        "wkv": torch.randn(2 * width, hidden),
        "wgate": torch.randn(2 * width, hidden),
        "ape": torch.randn(ratio, 2 * width),
        "norm": torch.randn(width),
    }
    cache = SimpleNamespace(
        cos=torch.ones(32, rope_width // 2),
        sin=torch.zeros(32, rope_width // 2),
    )

    def state():
        return {
            "ckv": torch.zeros(1, 2 * ratio, 2 * width),
            "cscore": torch.full(
                (1, 2 * ratio, 2 * width), float("-inf")
            ),
        }

    vector_state = state()
    reference_state = state()
    vector, vector_start, _ = _compressor_prefill_cccp(
        x,
        weights,
        ratio,
        width,
        rope_width,
        cache,
        1e-6,
        vector_state,
        0,
    )
    reference, reference_start, snapshots = _compressor_prefill_cccp(
        x,
        weights,
        ratio,
        width,
        rope_width,
        cache,
        1e-6,
        reference_state,
        0,
        capture_steps=True,
    )

    torch.testing.assert_close(vector, reference)
    torch.testing.assert_close(vector_state["ckv"], reference_state["ckv"])
    torch.testing.assert_close(
        vector_state["cscore"], reference_state["cscore"]
    )
    assert vector_start == reference_start == 0
    assert len(snapshots) == tokens


def test_replicated_cuda_prefill_has_no_decode_gemv_fallback():
    source = inspect.getsource(PackedExpertPool.run_rows_replicated)

    assert "packed_moe_topk(" not in source
    assert "_run_rows_dequant_rank(" in source
    assert "decode GEMV fallback is forbidden" in source


def test_rocm_resident_prefill_accepts_heterogeneous_three_projection_layouts(
    monkeypatch,
):
    monkeypatch.setattr(torch.version, "hip", "6.4")
    manifest = SimpleNamespace(
        projection_vq=True,
        projection_operator_capabilities=lambda _layer: (
            {
                "packed_formats": ("p13", "p13", "p13"),
                "code_dims": (4, 4, 4),
                "codebook_sizes": (8192, 8192, 8192),
            },
            {
                "packed_formats": ("p14", "p14", "p15"),
                "code_dims": (4, 4, 4),
                "codebook_sizes": (16384, 16384, 32768),
            },
        ),
    )

    assert _packed_grouped_prefill_supported(manifest, 0) is True


def test_rocm_resident_prefill_rejects_incomplete_projection_metadata(
    monkeypatch,
):
    monkeypatch.setattr(torch.version, "hip", "6.4")
    manifest = SimpleNamespace(
        projection_vq=True,
        projection_operator_capabilities=lambda _layer: ({
            "packed_formats": ("p13", "p13"),
            "code_dims": (4, 4),
            "codebook_sizes": (8192, 8192),
        },),
    )

    assert _packed_grouped_prefill_supported(manifest, 0) is False


def test_cuda_resident_prefill_uses_packed_grouped_when_dequant_scratch_does_not_fit():
    """Kimi TP must not reject valid [15,E] metadata after a scratch miss."""
    manifest = SimpleNamespace(
        projection_vq=True,
        projection_operator_capabilities=lambda _layer: ({
            "packed_formats": ("p16", "p16", "p14"),
            "code_dims": (16, 16, 16),
            "codebook_sizes": (65536, 65536, 16384),
        },),
    )

    assert _select_packed_grouped_prefill(
        manifest,
        1,
        metadata_rows=15,
        dequant_prefill=False,
        hip_runtime=False,
    ) is True
    assert _select_packed_grouped_prefill(
        manifest,
        1,
        metadata_rows=15,
        dequant_prefill=True,
        hip_runtime=False,
    ) is False
    assert _select_packed_grouped_prefill(
        manifest,
        1,
        metadata_rows=15,
        dequant_prefill=True,
        hip_runtime=True,
    ) is True


def test_resident_prefill_accepts_original_two_projection_directory():
    pool = object.__new__(PackedExpertPool)
    metadata = torch.zeros((10, 3), dtype=torch.int64)
    metadata[0] = 1
    metadata[1] = 2
    metadata[2] = 3
    metadata[5] = 4
    metadata[6] = 5
    metadata[7] = 6
    pool._metadata = [[metadata]]
    pool._grouped_local_masks = {}

    mask = pool._grouped_local_mask(0, 0)

    assert mask.tolist() == [True, True, True]


def test_grouped_packed_prefill_accepts_two_projection_metadata():
    api_source = inspect.getsource(packed_moe_topk_grouped)
    fused_source = (
        ENGINE_ROOT / "cccp" / "fusedext.py"
    ).read_text(encoding="utf-8")
    cuda_source = (
        ENGINE_ROOT / "cccp" / "csrc" / "vq_gemv.cu"
    ).read_text(encoding="utf-8")

    assert "metadata.shape[0] not in (10, 15)" in api_source
    assert "metadata.shape[0] not in (10, 15)" in fused_source
    assert "metadata_rows == 10 ? 0 : 5" in cuda_source
    assert "metadata_rows == 10 ? 5 : 10" in cuda_source


def test_grouped_packed_prefill_decodes_heterogeneous_gate_up_layouts():
    cuda_source = (
        ENGINE_ROOT / "cccp" / "csrc" / "vq_gemv.cu"
    ).read_text(encoding="utf-8")
    kernel = cuda_source.split(
        "__global__ void vq_projection_gate_up_grouped_kernel", 1
    )[1].split(
        "__global__ void vq_projection_down_grouped_kernel", 1
    )[0]

    assert "const bool same_layout" in kernel
    assert "if (same_layout)" in kernel
    assert "block < gate_meta.blocks" in kernel
    assert "block < up_meta.blocks" in kernel
    assert "gate_meta.blocks != up_meta.blocks" not in kernel


def test_glm_tp2_final_reduce_uses_contribution_list_contract():
    source = inspect.getsource(
        GpuResidentExpertParallel._compute_decode_device_routed
    )

    assert "[*partials, shared]" in source
    assert "partials[0],\n                partials[1]" not in source


def test_kimi_decode_and_prefill_use_disjoint_packed_entries():
    from cccp.kimi_model import KimiK3CCCPModel

    source = inspect.getsource(KimiK3CCCPModel._moe_block_cpu)
    assert 'native is not None and value.shape[0] == 1' in source
    assert "value.shape[0] == 1" in source
    assert "run_decode(" in source
    assert source.index("run_decode(") < source.index("run_rows(")


def test_full_profile_prefill_keeps_the_global_packed_directory():
    source = inspect.getsource(
        PackedHybridPool._partition_prefill_layer_locked
    )

    assert "_default_arena_covers_all_pinned" in source
    assert "if covers_all:" in source
    assert source.index("if covers_all:") < source.index(
        "specs: Counter[PackedExpertSignature]"
    )


def test_prefill_chunk_capacity_respects_other_process_vram(monkeypatch):
    pool = PackedHybridPool.__new__(PackedHybridPool)
    pool.store = SimpleNamespace(cfg={
        "hidden": 8192,
        "routed_hidden": 8192,
        "moe_inter": 384,
        "n_experts": 896,
    })
    pool.device = torch.device("cuda", 0)
    pool._prefill_dequant_workspace = None
    gib = 2**30
    monkeypatch.setenv("CCCP_VRAM_LIMIT_GB", "136")
    monkeypatch.setattr(
        torch.cuda, "memory_allocated", lambda _device: 120 * gib
    )
    monkeypatch.setattr(
        torch.cuda, "memory_reserved", lambda _device: 120 * gib
    )
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda _device: (1 * gib, 140 * gib)
    )

    capacity = pool._prefill_dequant_chunk_capacity(896)

    # Only the configured 1-GiB physical safety line remains, so the planner
    # permits just the mandatory one-expert chunk instead of consuming it.
    assert capacity == 1


def test_prefill_chunk_releases_only_stale_allocator_cache(monkeypatch):
    pool = PackedHybridPool.__new__(PackedHybridPool)
    pool.store = SimpleNamespace(cfg={
        "hidden": 8192,
        "routed_hidden": 8192,
        "moe_inter": 384,
        "n_experts": 896,
    })
    pool.device = torch.device("cuda", 0)
    pool._prefill_dequant_workspace = None
    gib = 2**30
    released = []
    monkeypatch.setenv("CCCP_VRAM_LIMIT_GB", "16")
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 10 * gib)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 12 * gib)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: released.append("sync"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: released.append(True))
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda _device: (100 * gib, 140 * gib)
    )

    capacity = pool._prefill_dequant_chunk_capacity(896)

    assert released == ["sync", True]
    assert capacity > 1


def test_public_packed_gemv_rejects_multi_token_input():
    with pytest.raises(RuntimeError, match="decode-only"):
        packed_moe_topk(
            torch.empty(2, 4),
            torch.empty(2, 1, dtype=torch.long),
            torch.empty(2, 1),
            torch.empty(15, 1, dtype=torch.long),
            activation="swiglu",
            activation_beta=1.0,
            activation_linear_beta=0.0,
            hidden_workspace=torch.empty(2, 4),
            output_workspace=torch.empty(2, 4),
            result=torch.empty(2, 4),
            grouped_prefix=-1,
        )


def test_cuda_packed_gemv_is_registered_and_guarded_as_decode_only():
    backend_source = (
        ENGINE_ROOT / "cccp" / "ops" / "cuda_backend.py"
    ).read_text(encoding="utf-8")
    wrapper_text = (
        ENGINE_ROOT / "cccp" / "fusedext.py"
    ).read_text(encoding="utf-8")
    wrapper_source = wrapper_text.split(
        "    def packed_moe_topk_fused(", 1
    )[1].split("    def packed_stage_topk_three_projection_fused(", 1)[0]
    native_source = (
        ENGINE_ROOT / "cccp" / "csrc" / "vq_gemv.cu"
    ).read_text(encoding="utf-8")

    registration = backend_source.split(
        '"cuda.packed_moe_topk.three_projection.mixed.gated"', 1
    )[1].split("registry.register(", 1)[0]
    assert "batch_sizes=(1,)" in registration
    assert "value.shape[0] != 1" in wrapper_source
    assert "route_ids.ndim != 1" in wrapper_source
    assert "input.size(0) == 1" in native_source
    assert "route_ids.is_contiguous() && route_ids.dim() == 1" in native_source


def test_cuda_kda_wrapper_does_not_forward_optional_normalization_keys():
    backend_source = (
        ENGINE_ROOT / "cccp" / "ops" / "cuda_backend.py"
    ).read_text(encoding="utf-8")
    recurrent_wrapper = backend_source.split(
        "def _kda_recurrent(**kwargs):", 1
    )[1].split("def _kda_recurrent_batch(**kwargs):", 1)[0]

    assert 'query=kwargs["query"]' in recurrent_wrapper
    assert "kda_recurrent_fused(**kwargs)" not in recurrent_wrapper
    assert "output_gate" not in recurrent_wrapper
    assert "norm_weight" not in recurrent_wrapper


def test_hybrid_route_probe_never_speculates_with_unverified_slots():
    prepare_source = inspect.getsource(PackedHybridPool.prepare_run)
    finish_source = inspect.getsource(PackedHybridPool.finish_run)
    config_root = ENGINE_ROOT / "cccp" / "configs"

    assert "_begin_device_route_metadata" in prepare_source
    assert "_launch_packed_run" not in prepare_source
    assert "speculative_result" not in prepare_source + finish_source
    assert "speculative_done" not in prepare_source + finish_source
    for path in config_root.glob("*.json"):
        assert "CCCP_SPECULATIVE_PACKED_HIT" not in path.read_text(
            encoding="utf-8"
        )


def test_hybrid_dequant_chunk_respects_explicit_vram_cap(monkeypatch):
    gib = 2**30
    pool = PackedHybridPool.__new__(PackedHybridPool)
    pool.store = SimpleNamespace(cfg={
        "hidden": 4096,
        "moe_inter": 2048,
    })
    pool.device = torch.device("cuda:0")
    monkeypatch.setenv("CCCP_VRAM_LIMIT_GB", "16")
    monkeypatch.delenv("CCCP_PREFILL_DEQUANT_EXPERTS", raising=False)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 15 * gib)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 15 * gib)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda _device: (8 * gib, 24 * gib)
    )

    # Exactly the configured 1-GiB safety line remains. Keep it intact and
    # admit only the mandatory one-expert chunk.
    assert pool._prefill_dequant_chunk_capacity(256) == 1


def test_dsv4_reset_prefill_batches_the_entire_baseline():
    phases = []

    class Pool:
        def activate_prefill_arena(self):
            phases.append("prefill")

        def activate_decode_arena(self):
            phases.append("decode")

    class Model:
        def __init__(self):
            self.pos = 9
            self.calls = []
            self.last_prefill_block_size = 4096
            self.pool = Pool()

        def forward(self, values):
            self.calls.append(list(values))
            self.pos = len(values)
            return torch.tensor([float(values[-1])])

    model = Model()
    engine = Engine.__new__(Engine)
    engine.model = model
    engine.quiet = True
    engine.reset = lambda: setattr(model, "pos", 0)
    engine._with_kv_capacity_retry = (
        lambda function, *args, **kwargs: function(*args, **kwargs)
    )

    logits = engine._prefill_from_reset_to_boundary(
        [11, 12, 13, 14, 15],
        4,
    )

    assert model.calls == [[11, 12, 13, 14]]
    assert phases == ["prefill", "decode"]
    assert model.pos == 4
    assert logits.item() == 14.0


def test_terminal_log_decodes_utf8_and_windows_compiler_code_page():
    payload = (
        "[cccp] 专家映射完成\n".encode("utf-8")
        + "正在创建库和对象\n".encode("gb18030")
    )

    assert cccp_module._decode_mixed_process_log(payload).splitlines() == [
        "[cccp] 专家映射完成",
        "正在创建库和对象",
    ]


def test_terminal_session_files_are_reset_on_launcher_start(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    log_file = runtime / "cccp-serve.log"
    metrics_file = runtime / "chat-metrics.jsonl"
    log_file.write_text("old engine output", encoding="utf-8")
    metrics_file.write_text('{"old": true}', encoding="utf-8")
    monkeypatch.setattr(cccp_module, "RUNTIME_DIR", runtime)
    monkeypatch.setattr(cccp_module, "CHAT_METRICS_FILE", metrics_file)

    CCCPEngineAdapter(Settings()).reset_terminal_session()

    assert log_file.read_bytes() == b""
    assert metrics_file.read_bytes() == b""


def _model(tmp_path):
    root = tmp_path / "model" / "tiny"
    root.mkdir(parents=True)
    (root / "dense.safetensors").write_bytes(b"dense")
    (root / "experts.L00.safetensors").write_bytes(b"experts")
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "cccp.json").write_text(json.dumps({
        "architecture": "deepseek_v4",
        "config": {"n_layers": 1, "n_experts": 2,
                   "max_position_embeddings": 4096},
        "dense_file": "dense.safetensors",
        "expert_files": {"0": "experts.L00.safetensors"},
    }), encoding="utf-8")
    return root


def _kimi_model(tmp_path):
    root = tmp_path / "model" / "kimi"
    (root / "dense").mkdir(parents=True)
    (root / "dense" / "model-00001.safetensors").write_bytes(b"dense-a")
    (root / "dense" / "model-00002.safetensors").write_bytes(b"dense-b")
    (root / "experts.L001.safetensors").write_bytes(b"0123456789")
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "tiktoken.model").write_bytes(b"tokens")
    (root / "tokenization_kimi.py").write_text("", encoding="utf-8")
    (root / "dense.audit.json").write_text(json.dumps({
        "fixed_bytes": 30 * 2**20,
        "entries": {
            "language_model.shared_experts.weight": {"stored_bytes": 5 * 2**20},
            "language_model.other.weight": {"stored_bytes": 25 * 2**20},
        },
    }), encoding="utf-8")
    (root / "experts.L001.audit.json").write_text(json.dumps({
        "layer": 1,
        "file_bytes": 10,
        "experts": {
            "0": {"gate": {"packed_bytes": 1}, "up": {"packed_bytes": 2}},
            "1": {"gate": {"packed_bytes": 3}, "down": {"packed_bytes": 4}},
        },
    }), encoding="utf-8")
    (root / "cccp.json").write_text(json.dumps({
        "format": "cccp-1",
        "model_family": "kimi_k3",
        "config": {
            "n_layers": 2,
            "n_experts": 2,
            "top_k": 1,
            "max_position_embeddings": 4096,
            "kda_layers": [0],
            "routed_hidden": 8,
        },
        "dense_files": [
            "dense/model-00001.safetensors",
            "dense/model-00002.safetensors",
        ],
        "dense_audit_file": "dense.audit.json",
        "routed_experts": {
            "layers": 1,
            "experts_per_layer": 2,
            "layer_files": {
                "1": {
                    "path": "experts.L001.safetensors",
                    "audit_path": "experts.L001.audit.json",
                    "bytes": 10,
                },
            },
        },
    }), encoding="utf-8")
    return root


def _adapter(tmp_path):
    cccp = tmp_path / "engine"
    (cccp / "cccp").mkdir(parents=True)
    (cccp / "cccp" / "__init__.py").write_text("", encoding="utf-8")
    (cccp / "cccp" / "__main__.py").write_text("", encoding="utf-8")
    (cccp / "cccp" / "engine.py").write_text("", encoding="utf-8")
    settings = Settings(cccp_engine_path=str(cccp), python_path=sys.executable,
                        model_roots=[str(tmp_path / "model")])
    return CCCPEngineAdapter(settings), settings


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_settings_memory_guards():
    Settings().validate()
    with pytest.raises(ValueError):
        Settings(memory_limit_gb=2049).validate()
    with pytest.raises(ValueError):
        Settings(expert_cache_gb=2049).validate()
    with pytest.raises(ValueError):
        Settings(default_context=32).validate()
    with pytest.raises(ValueError):
        Settings(cpu_compile_mode="invalid").validate()
    with pytest.raises(ValueError):
        Settings(theme_mode="invalid").validate()


@pytest.mark.parametrize(
    (
        "resident_all", "packed_device_pool", "packed_full_gpu",
        "extreme_staging", "route_history_resident", "expected",
    ),
    [
        # Sequential layer-first Prefill/decode stages RAM-resident experts on
        # demand.  Cross-layer speculative prefetch thrashes the small global
        # arena and measured 2.82 tok/s versus 5.43 tok/s with it disabled.
        (True, True, False, False, False, False),
        # A real all-VRAM pool has nothing to prefetch.
        (True, True, True, False, False, False),
        # Non-resident pools retain their existing disk/RAM prefetch policy.
        (False, True, False, False, False, True),
        # Extreme staging prefetches only when a whole route history cannot fit.
        (True, True, False, True, False, True),
        (True, True, False, True, True, False),
    ],
)
def test_dsv4_automatic_prefetch_uses_device_residency(
    resident_all, packed_device_pool, packed_full_gpu,
    extreme_staging, route_history_resident, expected,
):
    assert _automatic_prefetch_policy(
        resident_all=resident_all,
        packed_device_pool=packed_device_pool,
        packed_full_gpu=packed_full_gpu,
        extreme_staging=extreme_staging,
        route_history_resident=route_history_resident,
    ) is expected


def test_model_inspection_and_discovery(tmp_path):
    root = _model(tmp_path)
    info = inspect_model(root)
    assert info.complete and info.layers == 1 and info.experts_per_layer == 2
    assert info.expert_layers == [0] and info.expert_layer_count == 1
    found = discover_models([str(tmp_path / "model")])
    assert [m.name for m in found] == ["tiny"]
    (root / "tokenizer.json").unlink()
    assert not inspect_model(root).complete


def test_model_discovery_hides_transient_copy_directories(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    copying = _model(tmp_path).rename(models / "large-model.copying")
    final = _model(tmp_path / "other").rename(models / "ready-model")

    found = discover_models([str(models)])

    assert [Path(item.path).name for item in found] == [final.name]
    assert copying.is_dir()


def test_model_inspection_reads_top_level_glm_model_family(tmp_path):
    root = _model(tmp_path)
    manifest_path = root / "cccp.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("architecture")
    manifest["model_family"] = "glm_moe_dsa"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert inspect_model(root).architecture == "glm_moe_dsa"


def test_kimi_model_inspection_and_full_model_profile(tmp_path):
    root = _kimi_model(tmp_path)

    info = inspect_model(root)
    combo = full_model_combination(root)

    assert info.complete
    assert info.architecture == "kimi_k3"
    assert info.layers == 2
    assert info.expert_layers == [1]
    assert info.expert_layer_count == 1
    assert info.dense_gb == pytest.approx(30 / 1024, abs=0.001)
    assert info.shared_expert_gb == pytest.approx(5 / 1024, abs=0.001)
    assert set(combo.union) == {"1:0", "1:1"}
    assert sum(item.size_mb for item in combo.union.values()) == pytest.approx(
        10 / 2**20, abs=1e-6
    )


def test_mapped_capacity_policy_falls_back_to_published_ram_capability(tmp_path):
    """没有专用 mapped 算子的架构仍可使用公共 RAM/系统分页路径。"""
    model = tmp_path / "glm"
    model.mkdir()
    (model / "cccp.json").write_text(json.dumps({
        "format": "cccp-1",
        "model_family": "glm_moe_dsa",
        "config": {},
    }), encoding="utf-8")

    preset = resolve_preset(model, profile="mapped", tp=1)

    assert preset.architecture == "glm"
    assert preset.profile == "ram"
    assert preset.config_profile == "ram"


def test_dsv4_mapped_profile_uses_bounded_hybrid_pool(tmp_path):
    model = _model(tmp_path)
    manifest_path = model / "cccp.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format"] = "cccp-1"
    manifest["config"]["hc_mult"] = 1
    manifest["quant"] = {"method": "projection-vq"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    preset = resolve_preset(model, profile="mapped", tp=1)

    assert preset.environment["CCCP_PACKED_FULL_GPU"] == "0"
    assert preset.environment["CCCP_PACKED_HOST_MAPPED"] == "0"
    assert preset.environment["CCCP_FULL_RESIDENT"] == "1"
    assert preset.environment["CCCP_ALLOW_PAGEFILE_RESIDENT"] == "1"
    assert preset.environment["CCCP_RAM_RESERVE_GB"] == "2"
    assert preset.environment["CCCP_RESIDENT_RESERVE_GB"] == "2"
    assert preset.environment["CCCP_VRAM_RESERVE_GB"] == "1"
    assert "CCCP_VRAM_RUNTIME_GB" not in preset.environment


def test_hybrid_resident_pool_loads_every_configured_expert_only(
    tmp_path, monkeypatch, capsys,
):
    store = SimpleNamespace(
        root=str(tmp_path),
        cfg={"n_experts": 4},
        man=SimpleNamespace(
            expert_files={0: "L0", 1: "L1"},
            projection_vq=False,
            no_expert_drop=False,
        ),
        route_allowlist={0: {1, 3}, 1: {0, 2}},
        expert_kind=lambda _layer, _expert: "projection-vq",
        _cb_cache={},
    )
    pool = object.__new__(PackedHybridPool)
    pool.store = store
    pool.pinned = {}
    pool._host_codebooks = {}
    pool._load_one = lambda layer, expert: ()
    monkeypatch.setenv("CCCP_FULL_RESIDENT", "1")
    monkeypatch.setenv("CCCP_LOAD_WORKERS", "1")

    assert pool.preload_all(reserve_gb=0)
    assert set(pool.pinned) == {(0, 1), (0, 3), (1, 0), (1, 2)}
    progress = capsys.readouterr().out
    assert "phase=experts current=1 total=4" in progress
    assert "phase=experts current=4 total=4" in progress


def test_gpu_profile_ram_preload_uses_exact_allowlist(
    monkeypatch, capsys,
):
    loaded = []

    def load_expert(layer, expert):
        loaded.append((layer, expert))
        item = SimpleNamespace(nbytes=1024)
        return item, item

    store = SimpleNamespace(
        cfg={"hidden": 8, "moe_inter": 4, "top_k": 2},
        heat_ranks={0: [3], 1: [2]},
        route_allowlist={0: {1, 3}, 1: {0, 2}},
        load_expert=load_expert,
    )
    pool = ExpertPool(
        store,
        budget_gb=1.0,
        device="cpu",
        ram_gb=1.0,
        pin_gb=0.0,
    )
    monkeypatch.setenv("CCCP_PROFILE_FULL_LOAD", "1")

    pool.preload_pinned()

    expected = {(0, 1), (0, 3), (1, 0), (1, 2)}
    assert set(loaded) == expected
    assert set(pool.pinned) == expected
    progress = capsys.readouterr().out
    assert "phase=experts current=4 total=4" in progress
    assert "配置专家 RAM 预载完成" in progress


def test_strict_profile_skips_unfiltered_full_model_preload(monkeypatch):
    pool = object.__new__(ExpertPool)
    pool.store = SimpleNamespace(route_allowlist={0: {1, 3}})
    monkeypatch.setenv("CCCP_PROFILE_FULL_LOAD", "1")
    monkeypatch.setenv("CCCP_FULL_RESIDENT", "1")

    assert pool.preload_all() is False


def test_host_pin_budget_keeps_reasonable_ram_floor_without_disabling_dma():
    gib = 2**30
    budget, floor = automatic_host_pin_budget(
        payload_bytes=36 * gib,
        available_ram_bytes=int(39.5 * gib),
        device_bytes=20 * gib,
        driver_multiplier=1.25,
    )

    assert floor == 2 * gib
    assert budget == 25 * gib


def test_host_pin_auto_budget_uses_ram_not_vram_when_uncapped():
    gib = 2**30
    budget, floor = automatic_host_pin_budget(
        payload_bytes=36 * gib,
        available_ram_bytes=int(39.5 * gib),
        device_bytes=20 * gib,
        driver_multiplier=0.0,
    )

    assert floor == 2 * gib
    assert budget == 36 * gib


def test_compact_pool_auto_mode_registers_existing_ram_in_place(monkeypatch):
    class FakeCudaRuntime:
        def __init__(self):
            self.registered = []

        def cudaHostRegister(self, pointer, size, flags):
            self.registered.append((int(pointer), int(size), int(flags)))
            return 0

    runtime = FakeCudaRuntime()
    monkeypatch.delenv("CCCP_WDDM_DIRECT_PIN", raising=False)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: runtime)

    first_raw = torch.empty(64, dtype=torch.uint8)
    second_raw = torch.empty(96, dtype=torch.uint8)
    codebook = torch.empty(2, 2, dtype=torch.bfloat16)
    first = HostPackedWeight(first_raw, codebook, 2, 2, 1, 2, 8)
    second = HostPackedWeight(second_raw, codebook, 2, 2, 1, 2, 8)
    pool = object.__new__(PackedHybridPool)
    pool.pinned = {(0, 0): (first, second)}
    pool._host_registrations = {}
    pool._host_pinned_bytes = 0

    pinned_gib = pool.pin_host_resident(budget_gb=1.0)

    assert pinned_gib > 0
    assert {pointer for pointer, _size, _flags in runtime.registered} == {
        first_raw.data_ptr(),
        second_raw.data_ptr(),
    }
    assert pool._host_pinned_bytes == first_raw.nbytes + second_raw.nbytes


def test_rocm_compact_pool_skips_external_host_registration(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "6.4")
    monkeypatch.setattr(
        torch.cuda,
        "cudart",
        lambda: (_ for _ in ()).throw(AssertionError("must not register")),
    )
    raw = torch.empty(64, dtype=torch.uint8)
    codebook = torch.empty(2, 2, dtype=torch.bfloat16)
    weight = HostPackedWeight(raw, codebook, 2, 2, 1, 2, 8)
    pool = object.__new__(PackedHybridPool)
    pool.pinned = {(0, 0): (weight,)}
    pool._host_registrations = {}
    pool._host_pinned_bytes = 0

    pinned_gib = pool.pin_host_resident(budget_gb=1.0)

    assert pinned_gib == 0.0
    assert pool.host_dma_mode == "hip-pinned-stage"
    assert pool._host_registrations == {}


def test_full_model_combination_contains_every_expert(tmp_path):
    root = _model(tmp_path)
    combo = full_model_combination(root)
    assert combo.profile_ids == [FULL_MODEL_PROFILE_ID]
    assert set(combo.union) == {"0:0", "0:1"}
    assert combo.model_name == "tiny"


def test_full_model_combination_uses_manifest_expert_layers_only(tmp_path):
    root = _model(tmp_path)
    manifest_path = root / "cccp.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["n_layers"] = 4
    manifest["expert_files"] = {"3": "experts.L00.safetensors"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    info = inspect_model(root)
    combo = full_model_combination(root)

    assert info.layers == 4
    assert info.expert_layers == [3]
    assert info.expert_layer_count == 1
    assert set(combo.union) == {"3:0", "3:1"}


def test_expert_sizes_support_projection_map_audit_schema(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "experts.L03.safetensors.audit.json").write_text(json.dumps({
        "layer": 3,
        "bytes": 1000,
        "experts": {
            "0": {"projections": {"gu": {"packed_bytes": 100}, "dn": {"packed_bytes": 100}}},
            "1": {"projections": {"gu": {"packed_bytes": 300}, "dn": {"packed_bytes": 500}}},
        },
    }), encoding="utf-8")

    sizes = load_expert_sizes(model)

    assert sizes["3:0"] == pytest.approx(200 / 2**20, abs=1e-6)
    assert sizes["3:1"] == pytest.approx(800 / 2**20, abs=1e-6)
    assert sum(sizes.values()) == pytest.approx(1000 / 2**20, abs=2e-6)


def test_full_model_env_disables_route_profile(tmp_path):
    model = _model(tmp_path)
    adapter, _ = _adapter(tmp_path)
    combo = full_model_combination(model)
    cfg = LaunchConfig(
        model_path=str(model), profiles=[], combination=combo,
        port=_free_port(), device="cpu", profile_mode="auto",
        cache_gb=1, cpu_compile="off",
    )
    env = adapter._env(cfg)
    assert env["CCCP_ROUTE_PROFILE"] == "0"
    assert env["CCCP_PROFILE_FULL_LOAD"] == "0"
    assert env["CCCP_FULL_RESIDENT"] == "1"
    assert "CCCP_PROFILE_JSON" not in env


def test_full_model_mapped_cache_does_not_reduce_or_reject_experts(tmp_path):
    model = _model(tmp_path)
    adapter, _ = _adapter(tmp_path)
    combo = full_model_combination(model)
    cfg = LaunchConfig(
        model_path=str(model), profiles=[], combination=combo,
        port=_free_port(), device="cpu", profile_mode="mapped",
        cache_gb=0.25, cpu_compile="off",
    )
    result = adapter.preflight(cfg)
    assert result["ok"]
    assert len(combo.union) == 2
    assert not any("严格路由" in error for error in result["errors"])


def test_cpu_command_and_preflight(tmp_path):
    model = _model(tmp_path)
    adapter, settings = _adapter(tmp_path)
    combo = Combination(profile_ids=["x"], union={}, overlap_mb=0)
    cfg = LaunchConfig(model_path=str(model), profiles=["x"], combination=combo,
                       port=_free_port(), profile_mode="mapped", device="cpu",
                       cache_gb=2, max_ctx=512, memory_limit_gb=24,
                       cpu_compile="off", extreme=False)
    command = adapter.build_command(cfg, dry_run=True)
    assert "--device" in command and command[command.index("--device") + 1] == "cpu"
    assert "--cache-gb" in command and "--no-extreme" in command
    assert command[-1] == "--dry-run"
    preflight = adapter.preflight(cfg)
    assert preflight["ok"] and preflight["memory"]["total_estimate_gb"] < 24
    env = adapter._env(cfg)
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["CCCP_FULL_RESIDENT"] == "0"
    assert env["CCCP_CPU_AUTOBUILD"] == "1"
    assert preflight["native_cpu_operator"]["auto_build_fallback"] is True


def test_preflight_rejects_cpu_extreme(tmp_path):
    model = _model(tmp_path)
    adapter, _ = _adapter(tmp_path)
    cfg = LaunchConfig(model_path=str(model), profiles=["x"],
                       combination=Combination(["x"], {}, 0),
                       port=_free_port(), device="cpu", extreme=True)
    result = adapter.preflight(cfg)
    assert not result["ok"]
    assert any("extreme" in item for item in result["errors"])


def test_preflight_distinguishes_busy_memory_from_device_capacity(tmp_path, monkeypatch):
    model = _model(tmp_path)
    adapter, _ = _adapter(tmp_path)
    cfg = LaunchConfig(
        model_path=str(model), profiles=["x"],
        combination=Combination(["x"], {}, 0), port=_free_port(),
        device="cpu", cache_gb=2, memory_limit_gb=24, cpu_compile="off",
    )
    monkeypatch.setattr(cccp_module, "_memory_status", lambda: (32.0, 1.0))
    busy = adapter.preflight(cfg)
    assert busy["ok"] and busy["status"] == "warning"
    assert busy["memory"]["capacity_kind"] == "ram"
    assert busy["memory"]["capacity_label"] == "内存"
    codes = {item["code"] for item in busy["memory"]["risk_reasons"]}
    assert codes == {"system_memory_busy"}
    assert any("关闭" in warning for warning in busy["warnings"])

    monkeypatch.setattr(cccp_module, "_memory_status", lambda: (32.0, 32.0))
    cfg.memory_limit_gb = 1.0
    capacity = adapter.preflight(cfg)
    assert capacity["ok"] and capacity["status"] == "danger"
    codes = {item["code"] for item in capacity["memory"]["risk_reasons"]}
    assert codes == {"device_capacity_exceeded"}


def test_gpu_preflight_degrades_vram_to_ram_before_disk(tmp_path, monkeypatch):
    model = _model(tmp_path)
    adapter, _ = _adapter(tmp_path)
    cfg = LaunchConfig(
        model_path=str(model), profiles=["x"],
        combination=Combination(["x"], {}, 0), port=_free_port(),
        device="cuda", cache_gb=20, memory_limit_gb=0, cpu_compile="off",
    )
    monkeypatch.setattr(cccp_module, "probe_backend", lambda *args: {
        "ready": True,
        "label": "NVIDIA CUDA",
        "device_memory_gb": 8.0,
        "device_available_memory_gb": 7.0,
    })

    monkeypatch.setattr(cccp_module, "_memory_status", lambda: (32.0, 32.0))
    ram = adapter.preflight(cfg)
    assert ram["ok"] and ram["status"] == "warning"
    assert ram["memory"]["ram_offload_likely"] is True
    assert ram["memory"]["disk_offload_likely"] is False
    assert ram["memory"]["offload_target"] == "ram"
    assert ram["memory"]["automatic_offload_mode"] == "gpu_to_host_ram"
    assert ram["memory"]["gpu_execution_tier"] == "reduced_expert_arena"
    assert ram["memory"]["minimum_vram_gb"] < 7.0
    assert ram["memory"]["recommended_vram_gb"] > 7.0
    assert any("主机内存" in item for item in ram["warnings"])
    assert not any("磁盘映射" in item for item in ram["warnings"])

    monkeypatch.setattr(cccp_module, "_memory_status", lambda: (32.0, 4.0))
    disk = adapter.preflight(cfg)
    assert disk["ok"] and disk["status"] == "danger"
    assert disk["memory"]["ram_offload_likely"] is True
    assert disk["memory"]["disk_offload_likely"] is True
    assert disk["memory"]["offload_target"] == "disk"
    codes = {item["code"] for item in disk["memory"]["risk_reasons"]}
    assert "host_memory_insufficient" in codes


def test_gpu_vram_plan_counts_only_bounded_expert_arena(tmp_path):
    model = inspect_model(_model(tmp_path))

    short = estimate_gpu_vram_plan(
        model, max_ctx=512, expert_cache_gb=80.0,
    )
    long = estimate_gpu_vram_plan(
        model, max_ctx=4096, expert_cache_gb=80.0,
    )

    assert short["preferred_expert_arena_gb"] == pytest.approx(4.0)
    assert (
        short["recommended_vram_gb"] - short["minimum_vram_gb"]
    ) == pytest.approx(4.0)
    assert long["minimum_vram_gb"] > short["minimum_vram_gb"]


def test_kimi_consumer_gpu_uses_ram_dense_tier_without_dropping_experts(
    tmp_path, monkeypatch,
):
    model_root = _kimi_model(tmp_path)
    audit_path = model_root / "dense.audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["fixed_bytes"] = 56 * 2**30
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    model = inspect_model(model_root)
    plan = estimate_gpu_vram_plan(
        model, max_ctx=512, expert_cache_gb=55.0,
    )
    assert plan["architecture"] == "kimi_k3"
    assert plan["hybrid_dense_ram"] is True
    assert plan["minimum_vram_gb"] < 16.0
    assert plan["recommended_vram_gb"] > 60.0

    adapter, _ = _adapter(tmp_path)
    cfg = LaunchConfig(
        model_path=str(model_root), profiles=["x"],
        combination=Combination(
            ["x"], {"1:0": ExpertRef(key="1:0", size_mb=1)}, 0
        ),
        port=_free_port(),
        device="cuda", cache_gb=55.0, max_ctx=512,
    )
    monkeypatch.setattr(cccp_module, "probe_backend", lambda *args: {
        "ready": True,
        "label": "NVIDIA CUDA",
        "device_memory_gb": 16.0,
        "device_available_memory_gb": 16.0,
    })
    monkeypatch.setattr(
        cccp_module, "_memory_status", lambda: (128.0, 128.0)
    )
    result = adapter.preflight(cfg)

    assert result["ok"] is True
    assert result["memory"]["gpu_execution_tier"] == "reduced_expert_arena"
    assert result["memory"]["hybrid_dense_ram"] is True
    assert result["memory"]["offload_target"] == "ram"
    assert result["memory"]["disk_offload_likely"] is False

    monkeypatch.setattr(cccp_module, "probe_backend", lambda *args: {
        "ready": True,
        "label": "NVIDIA CUDA",
        "device_memory_gb": 8.0,
        "device_available_memory_gb": 8.0,
    })
    blocked = adapter.preflight(cfg)
    assert blocked["ok"] is False
    assert blocked["memory"]["gpu_execution_tier"] == "below_minimum"
    assert "RAM Dense 混合模式" in blocked["errors"][0]
    assert "无法释放 Dense" not in blocked["errors"][0]


def test_kimi_engine_auto_residency_uses_hybrid_before_cpu_fallback():
    source = (ENGINE_ROOT / "cccp" / "engine.py").read_text(encoding="utf-8")
    hybrid = source.index('arch_hint == "kimi_k3"')
    selection = source.index('dense_residency = "ram"', hybrid)
    cpu_fallback = source.index('dev = "cpu"', selection)
    assert selection < cpu_fallback
    assert "margin = 10.0" in source[hybrid:cpu_fallback]


def test_prefill_restores_global_arena_after_wider_layer():
    source = inspect.getsource(PackedHybridPool._partition_prefill_layer_locked)
    wider = source.index("if required > self._arenas.nbytes:")
    restore = source.index(
        "self._arenas.repartition(self._default_arena_specs)", wider,
    )
    early_return = source.index("return", restore)
    assert wider < restore < early_return
    assert "self._prefill_partition_layer = None" in source[restore:early_return]


def test_dsv4_switches_long_prefill_and_decode_arena_phases():
    build = inspect.getsource(PackedHybridPool.build_gpu_arenas)
    prefill = inspect.getsource(PackedHybridPool.activate_prefill_arena)
    decode = inspect.getsource(PackedHybridPool.activate_decode_arena)
    engine_prefill = inspect.getsource(Engine._prefill_from_reset_to_boundary)

    assert "layer_envelope" in build
    assert "phase=prefill-switch" in prefill
    assert "self.resize_gpu_arenas" in prefill
    assert "phase=decode-switch" in decode
    assert "activate_prefill()" in engine_prefill
    assert "activate_decode()" in engine_prefill


def test_dsv4_prefetch_startup_message_matches_effective_policy():
    source = inspect.getsource(DSV4CCCPModel.preload)
    enabled_message = source.index("RAM+VRAM 跨层专家预取已启用")
    policy_guard = source.rfind("elif self._prefetch_enabled():", 0, enabled_message)
    demand_message = source.index("RAM+VRAM 按需专家 staging 已启用")
    assert policy_guard >= 0
    assert enabled_message < demand_message


def test_gpu_preflight_blocks_only_below_fixed_cuda_working_set(
    tmp_path, monkeypatch,
):
    model = _model(tmp_path)
    adapter, _ = _adapter(tmp_path)
    cfg = LaunchConfig(
        model_path=str(model), profiles=["x"],
        combination=Combination(["x"], {}, 0), port=_free_port(),
        device="cuda", cache_gb=20, memory_limit_gb=0, cpu_compile="off",
    )
    monkeypatch.setattr(cccp_module, "probe_backend", lambda *args: {
        "ready": True,
        "label": "NVIDIA CUDA",
        "device_memory_gb": 2.0,
        "device_available_memory_gb": 2.0,
    })
    monkeypatch.setattr(cccp_module, "_memory_status", lambda: (64.0, 64.0))

    result = adapter.preflight(cfg)

    assert result["ok"] is False
    assert result["status"] == "blocked"
    assert result["memory"]["gpu_execution_tier"] == "below_minimum"
    assert result["memory"]["offload_target"] == "cpu"
    assert result["memory"]["automatic_offload_mode"] == "gpu_to_cpu_fallback"
    codes = {item["code"] for item in result["memory"]["risk_reasons"]}
    assert codes == {"cuda_working_set_insufficient"}
    assert "缩小专家显存块也无法" in result["errors"][0]


def test_loading_progress_parses_expert_counter(tmp_path, monkeypatch):
    adapter, _ = _adapter(tmp_path)
    adapter.instance = CCCPEngineInstance(
        pid=123, port=8801, model="D:/model", served_model_name="winui-model",
        profiles=["x"], started_at=1.0, log_file="x.log",
        base_url="http://127.0.0.1:8801",
    )
    monkeypatch.setattr(adapter, "tail_log", lambda _lines=80: (
        "===== launch now =====\n"
        "[cccp-winui-progress] phase=experts current=50 total=100 loaded=49"
    ))
    progress = adapter.loading_progress({"running": True, "ready": False})
    assert progress["state"] == "loading"
    assert progress["phase"] == "experts"
    assert progress["percent"] == 50
    assert "50/100" in progress["detail"]


def test_engine_readiness_is_sticky_while_fused_generation_is_busy(
    tmp_path,
    monkeypatch,
):
    adapter, _ = _adapter(tmp_path)
    adapter.instance = CCCPEngineInstance(
        pid=123,
        port=8801,
        model="D:/model",
        served_model_name="winui-model",
        profiles=["x"],
        started_at=1.0,
        log_file="x.log",
        base_url="http://127.0.0.1:8801",
    )
    adapter._proc = SimpleNamespace(poll=lambda: None)

    class ReadyResponse:
        headers = {"content-type": "application/json"}

        @staticmethod
        def json():
            return {"status": "ok", "ready": True}

    class ReadyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, _url):
            return ReadyResponse()

    monkeypatch.setattr(
        cccp_module.httpx,
        "AsyncClient",
        lambda **_kwargs: ReadyClient(),
    )
    first = asyncio.run(adapter.health())

    class BusyClient(ReadyClient):
        async def get(self, _url):
            raise cccp_module.httpx.ReadTimeout("fused kernel busy")

    monkeypatch.setattr(
        cccp_module.httpx,
        "AsyncClient",
        lambda **_kwargs: BusyClient(),
    )
    second = asyncio.run(adapter.health())

    assert first["ready"] is True
    assert second["ready"] is True
    assert second["running"] is True
    assert second["busy"] is True
    assert adapter.loading_progress(second)["state"] == "ready"


def test_loading_progress_parses_host_pin_counter(tmp_path, monkeypatch):
    adapter, _ = _adapter(tmp_path)
    adapter.instance = CCCPEngineInstance(
        pid=123, port=8801, model="D:/model", served_model_name="winui-model",
        profiles=["x"], started_at=1.0, log_file="x.log",
        base_url="http://127.0.0.1:8801",
    )
    monkeypatch.setattr(adapter, "tail_log", lambda _lines=80: (
        "===== launch now =====\n"
        "[cccp-winui-progress] phase=experts current=3599 total=3599\n"
        "[cccp-winui-progress] phase=expert-pin current=12800 total=25600"
    ))

    progress = adapter.loading_progress({"running": True, "ready": False})

    assert progress["phase"] == "expert-pin"
    assert progress["percent"] == 92
    assert "12.5/25.0 GiB" in progress["detail"]


def test_loading_progress_parses_profile_gpu_upload(tmp_path, monkeypatch):
    adapter, _ = _adapter(tmp_path)
    adapter.instance = CCCPEngineInstance(
        pid=123, port=8801, model="D:/model", served_model_name="winui-model",
        profiles=["x"], started_at=1.0, log_file="x.log",
        base_url="http://127.0.0.1:8801",
    )
    monkeypatch.setattr(adapter, "tail_log", lambda _lines=80: (
        "===== launch now =====\n"
        "[cccp-winui-progress] phase=experts current=811 total=811\n"
        "[cccp-winui-progress] phase=expert-upload current=512 total=811"
    ))

    progress = adapter.loading_progress({"running": True, "ready": False})

    assert progress["phase"] == "expert-upload"
    assert progress["percent"] == 95
    assert "512/811" in progress["detail"]


def test_loading_progress_marks_disk_offload(tmp_path, monkeypatch):
    adapter, _ = _adapter(tmp_path)
    adapter.instance = CCCPEngineInstance(
        pid=123, port=8801, model="D:/model", served_model_name="winui-model",
        profiles=["x"], started_at=1.0, log_file="x.log",
        base_url="http://127.0.0.1:8801",
    )
    monkeypatch.setattr(adapter, "tail_log", lambda _lines=80: (
        "===== launch now =====\n"
        "[cccp-winui-offload] target=disk 内存不足，使用磁盘\n"
        "[cccp-winui-progress] phase=experts current=10 total=100 loaded=9"
    ))
    progress = adapter.loading_progress({"running": True, "ready": False})
    assert progress["disk_offload"] is True
    assert "磁盘卸载" in progress["detail"]


def test_loading_progress_distinguishes_ram_offload(tmp_path, monkeypatch):
    adapter, _ = _adapter(tmp_path)
    adapter.instance = CCCPEngineInstance(
        pid=123, port=8801, model="D:/model", served_model_name="winui-model",
        profiles=["x"], started_at=1.0, log_file="x.log",
        base_url="http://127.0.0.1:8801",
    )
    monkeypatch.setattr(adapter, "tail_log", lambda _lines=80: (
        "===== launch now =====\n"
        "[cccp-winui-offload] target=ram 显存不足，卸载到内存\n"
        "[cccp-winui-progress] phase=experts current=10 total=100 loaded=9"
    ))
    progress = adapter.loading_progress({"running": True, "ready": False})
    assert progress["ram_offload"] is True
    assert progress["disk_offload"] is False
    assert "主机内存" in progress["detail"]
    assert "未使用磁盘" in progress["detail"]


def test_loading_progress_keeps_quiet_operator_build_visibly_alive(tmp_path, monkeypatch):
    adapter, _ = _adapter(tmp_path)
    adapter.instance = CCCPEngineInstance(
        pid=123, port=8801, model="D:/model", served_model_name="winui-model",
        profiles=["x"], started_at=1.0, log_file="x.log",
        base_url="http://127.0.0.1:8801",
    )
    monkeypatch.setattr(adapter, "tail_log", lambda _lines=80: (
        "===== launch now =====\n"
        "[cccp-winui-progress] phase=operator-build "
        "event=running backend=NVIDIA-CUDA elapsed=125"
    ))
    progress = adapter.loading_progress({"running": True, "ready": False})
    assert progress["phase"] == "operator-build"
    assert progress["indeterminate"] is True
    assert "NVIDIA CUDA" in progress["label"]
    assert "125 秒" in progress["detail"]
    assert "编译器仍在运行" in progress["detail"]


def test_launcher_enables_operator_build_heartbeat(tmp_path):
    model = _model(tmp_path)
    adapter, _ = _adapter(tmp_path)
    cfg = LaunchConfig(
        model_path=str(model), profiles=["x"], combination=Combination(["x"], {}, 0),
        port=_free_port(), device="cuda",
    )
    env = adapter._env(cfg)
    assert env["CCCP_OPERATOR_BUILD_PROGRESS"] == "1"
    assert env["CCCP_OPERATOR_BUILD_HEARTBEAT_S"] == "5"


def test_launcher_cuda_ignores_inherited_arch_and_uses_active_gpu(tmp_path, monkeypatch):
    model = _model(tmp_path)
    adapter, _ = _adapter(tmp_path)
    cfg = LaunchConfig(
        model_path=str(model), profiles=["x"], combination=Combination(["x"], {}, 0),
        port=_free_port(), device="cuda",
    )
    monkeypatch.setenv("CCCP_CUDA_ARCH", "8.6")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.6")

    env = adapter._env(cfg)

    assert "CCCP_CUDA_ARCH" not in env
    assert "TORCH_CUDA_ARCH_LIST" not in env


def test_launcher_windows_cuda_keeps_h2d_batch_automatic(tmp_path, monkeypatch):
    import launcher.cccp_adapter as adapter_module

    model = _model(tmp_path)
    adapter, _ = _adapter(tmp_path)
    cfg = LaunchConfig(
        model_path=str(model), profiles=["x"], combination=Combination(["x"], {}, 0),
        port=_free_port(), device="cuda",
    )
    monkeypatch.setattr(adapter_module, "_WINDOWS", True)
    monkeypatch.setenv("CCCP_H2D_BATCH", "1")
    monkeypatch.setenv("CCCP_WDDM_DIRECT_PIN", "0")

    env = adapter._env(cfg)

    assert "CCCP_H2D_BATCH" not in env
    assert "CCCP_WDDM_DIRECT_PIN" not in env


def test_runtime_profile_files_are_content_addressed(tmp_path):
    adapter, _ = _adapter(tmp_path)
    first = Combination(profile_ids=["plain-a"], union={}, overlap_mb=0)
    second = Combination(profile_ids=["plain-b"], union={}, overlap_mb=0)
    first_path = adapter._profile_counts_file(first, job="test")
    second_path = adapter._profile_counts_file(second, job="test")
    assert first_path != second_path


def test_preflight_never_suggests_reducing_experts(tmp_path):
    model = _model(tmp_path)
    adapter, _ = _adapter(tmp_path)
    combo = Combination(
        profile_ids=["x"],
        union={"0:0": ExpertRef(key="0:0", size_mb=4096)},
        overlap_mb=0,
    )
    cfg = LaunchConfig(
        model_path=str(model), profiles=["x"], combination=combo,
        port=_free_port(), device="cpu", profile_mode="auto", cache_gb=1,
    )
    result = adapter.preflight(cfg)
    text = "\n".join(result["errors"])
    assert "减少配置" not in text
    assert "完整保留所选专家" in text


def test_latest_chat_metrics_reads_last_complete_record(tmp_path, monkeypatch):
    adapter, _ = _adapter(tmp_path)
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        '{"kv_mode":"rebuild","tokens_per_second":1.0}\n'
        '{"kv_mode":"extend","tokens_per_second":3.25}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(cccp_module, "CHAT_METRICS_FILE", metrics)
    latest = adapter.latest_chat_metrics()
    assert latest["kv_mode"] == "extend"
    assert latest["tokens_per_second"] == 3.25


def test_latest_chat_metrics_can_match_request_id(tmp_path, monkeypatch):
    adapter, _ = _adapter(tmp_path)
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        '{"request_id":"chatcmpl-first","tokens_per_second":1.0}\n'
        '{"request_id":"chatcmpl-second","tokens_per_second":3.25}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(cccp_module, "CHAT_METRICS_FILE", metrics)

    matched = adapter.latest_chat_metrics("chatcmpl-first")
    assert matched["request_id"] == "chatcmpl-first"
    assert adapter.latest_chat_metrics("missing") == {}


def test_block_major_weight_uses_logical_row_bounds():
    """块主序 q 的第一维是 tile 数；批量预填充必须按逻辑行数检查。"""
    engine_root = Path(__file__).resolve().parents[1] / "engine" / "CCCP-Engine"
    sys.path.insert(0, str(engine_root))
    import torch
    from cccp.kernels import BlockFP8Weight

    weight = BlockFP8Weight(
        torch.zeros((1, 4, 1, 32, 128), dtype=torch.uint8),
        torch.ones((1, 1), dtype=torch.float32),
        cols=128,
        block=128,
        rows=128,
        layout="block-major32",
    )
    assert weight.dequant_rows(0, 128).shape == (128, 128)


def test_tensor_fp8_weight_preserves_scalar_scale_and_slices():
    """公共 Tensor-FP8 执行映像保持一字节权重和标量反量化语义。"""
    engine_root = Path(__file__).resolve().parents[1] / "engine" / "CCCP-Engine"
    sys.path.insert(0, str(engine_root))
    import torch
    from cccp.kernels import BlockFP8Weight

    values = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0], [5.0, -6.0, 7.0, -8.0]],
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    weight = BlockFP8Weight(
        values.view(torch.uint8),
        torch.tensor([0.25], dtype=torch.float32),
        cols=4,
        rows=2,
        layout="tensor-fp8",
    )
    expected = values.float() * 0.25
    assert torch.equal(weight.dequant_rows(0, 2, torch.float32), expected)
    assert torch.equal(
        weight.row_view(1, 2).dequant_rows(0, 1, torch.float32),
        expected[1:2],
    )
    assert torch.equal(
        weight.column_slice(1, 3).dequant_rows(0, 2, torch.float32),
        expected[:, 1:3],
    )


def test_compressed_kv_decode_registry_uses_token_axis(monkeypatch):
    """TP MLA 的 leading axis 是本地 heads，不能误当请求 batch。"""
    from cccp.ops import api as ops_api

    key = ("compressed_kv_decode", "cuda")
    ops_api._ATTENTION_IMPLEMENTATIONS.pop(key, None)
    captured = {}

    def resolve(request):
        captured["request"] = request
        return SimpleNamespace(implementation=lambda **kwargs: "resolved")

    monkeypatch.setattr(ops_api.REGISTRY, "resolve", resolve)
    try:
        result = ops_api.attention_step(
            "compressed_kv_decode",
            "cuda",
            query_nope=torch.empty((24, 1, 128)),
        )
    finally:
        ops_api._ATTENTION_IMPLEMENTATIONS.pop(key, None)

    assert result == "resolved"
    assert captured["request"].batch_size == 1
