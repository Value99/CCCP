"""Packed GPU capacity checks must account for the actual startup phase."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ENGINE_ROOT = Path(__file__).resolve().parents[1] / "engine" / "CCCP-Engine"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from cccp.extreme import inspect_compact_projection_archive  # noqa: E402
from cccp.kimi_experts import (  # noqa: E402
    _packed_startup_required_bytes,
    build_kimi_layer_plan,
)
from cccp.packed_hybrid import (  # noqa: E402
    PackedExpertSignature,
    PackedWeightSignature,
    _PackedArenas,
)


GIB = 2**30


def _plan():
    return SimpleNamespace(
        dense_bytes_by_rank=(8 * GIB,),
        bytes_by_rank=(18 * GIB,),
        expert_aux_by_layer=(256 * 2**20,),
    )


def test_mapped_capacity_does_not_charge_already_resident_dense_twice():
    before_dense = _packed_startup_required_bytes(
        _plan(), 0, 10 * GIB,
        host_mapped=True, parallelism="tensor", dense_resident=False,
    )
    after_dense = _packed_startup_required_bytes(
        _plan(), 0, 10 * GIB,
        host_mapped=True, parallelism="tensor", dense_resident=True,
    )

    assert before_dense == 8 * GIB + 768 * 2**20
    assert after_dense == 768 * 2**20


def test_full_gpu_capacity_keeps_experts_but_not_resident_dense():
    before_dense = _packed_startup_required_bytes(
        _plan(), 0, 10 * GIB,
        host_mapped=False, parallelism="pipeline", dense_resident=False,
    )
    after_dense = _packed_startup_required_bytes(
        _plan(), 0, 10 * GIB,
        host_mapped=False, parallelism="pipeline", dense_resident=True,
    )

    assert before_dense == 18 * GIB + 512 * 2**20
    assert after_dense == 10 * GIB + 512 * 2**20


def test_layer_partition_reuses_one_fixed_byte_slab():
    def signature(raw_bytes: int) -> PackedExpertSignature:
        weight = PackedWeightSignature(
            raw_bytes=raw_bytes,
            cb_shape=(8, 4),
            rows=8,
            cols=8,
            blocks=2,
            dim=4,
            bits=8,
        )
        return PackedExpertSignature((weight, weight, weight))

    small = signature(16)
    large = signature(32)
    arenas = _PackedArenas(
        {small: 4, large: 2},
        torch.device("cpu"),
        resident_codebooks=True,
    )
    pointer = arenas._raw_storage.data_ptr()
    allocated = arenas.nbytes

    arenas.repartition({small: 2, large: 3})

    assert arenas._raw_storage.data_ptr() == pointer
    assert arenas.nbytes == allocated
    assert arenas.arenas[small].book.count == 2
    assert arenas.arenas[large].book.count == 3


def test_packed_arena_inflight_slots_are_atomic_within_route_batch():
    weight_signature = PackedWeightSignature(
        raw_bytes=16,
        cb_shape=(8, 4),
        rows=8,
        cols=8,
        blocks=2,
        dim=4,
        bits=8,
    )
    signature = PackedExpertSignature((weight_signature, weight_signature))
    arenas = _PackedArenas(
        {signature: 2},
        torch.device("cpu"),
        resident_codebooks=True,
    )
    host_weight = SimpleNamespace(
        raw=torch.zeros(16, dtype=torch.uint8),
        cb=torch.zeros(8, 4, dtype=torch.bfloat16),
        rows=8,
        cols=8,
        blocks=2,
        dim=4,
        bits=8,
    )
    expert = (host_weight, host_weight)
    device_codebooks = {host_weight.cb.data_ptr(): host_weight.cb}

    arenas.lease((0, 0), expert, device_codebooks)
    arenas.mark_inflight((0, 0))
    arenas.lease((0, 1), expert, device_codebooks)
    arenas.mark_inflight((0, 1))

    try:
        import pytest

        with pytest.raises(RuntimeError, match="in-flight"):
            arenas.lease((0, 2), expert, device_codebooks)
    finally:
        arenas.clear_inflight((0, 0))
        arenas.clear_inflight((0, 1))


def test_compact_capacity_uses_strict_profile_calibrated_expert_bytes(
    tmp_path, monkeypatch,
):
    from cccp import store as store_module

    root = tmp_path / "model"
    root.mkdir()
    (root / "experts.L00.safetensors").write_bytes(b"x" * 1200)
    (root / "experts.L00.audit.json").write_text(json.dumps({
        "layer": 0,
        "experts": {
            "0": {"projections": {
                "gu": {"packed_bytes": 100},
                "dn": {"packed_bytes": 100},
            }},
            "1": {"projections": {
                "gu": {"packed_bytes": 300},
                "dn": {"packed_bytes": 500},
            }},
        },
    }), encoding="utf-8")
    profile = root / "selected.json"
    profile.write_text(json.dumps({
        "strict_route": True,
        "allowed_experts": {"0": [0]},
    }), encoding="utf-8")

    class FakeManifest:
        packed_expert_vq = True
        projection_vq = True
        expert_files = {0: "experts.L00.safetensors"}
        expert_audit_files = {0: "experts.L00.audit.json"}

        def __init__(self, _root):
            pass

        @staticmethod
        def projection_operator_capability(_layer):
            return {
                "packed_formats": ("p13",),
                "code_dims": (4,),
                "codebook_sizes": (8192,),
            }

    monkeypatch.setattr(store_module, "Manifest", FakeManifest)
    monkeypatch.setenv("CCCP_ROUTE_PROFILE", "1")
    monkeypatch.setenv("CCCP_PROFILE_JSON", str(profile))

    selected = inspect_compact_projection_archive(root)
    monkeypatch.setenv("CCCP_ROUTE_PROFILE", "0")
    complete = inspect_compact_projection_archive(root)

    assert selected.expert_bytes == 240
    assert complete.expert_bytes == 1200


def test_resident_plan_allocates_only_strict_profile_experts(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "experts.L00.safetensors").write_bytes(b"x" * 1200)
    (root / "experts.L00.audit.json").write_text(json.dumps({
        "layer": 0,
        "experts": {
            "0": {"projections": {
                "gu": {"packed_bytes": 100},
                "dn": {"packed_bytes": 100},
            }},
            "1": {"projections": {
                "gu": {"packed_bytes": 300},
                "dn": {"packed_bytes": 500},
            }},
        },
    }), encoding="utf-8")
    store = SimpleNamespace(
        root=str(root),
        cfg={"n_layers": 1, "n_experts": 2},
        man=SimpleNamespace(
            expert_files={0: "experts.L00.safetensors"},
            expert_audit_files={0: "experts.L00.audit.json"},
        ),
        route_allowlist={0: {0}},
        dense_names=lambda: (),
        dense_nbytes=lambda _name: 0,
    )

    plan = build_kimi_layer_plan(store, 1)

    assert plan.expert_payload_by_expert == ((200, 0),)
    assert plan.expert_payload_by_layer == (200,)
    assert plan.expert_aux_by_layer == (40,)
    assert plan.expert_bytes_by_rank == (240,)
