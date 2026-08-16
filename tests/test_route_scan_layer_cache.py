from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine" / "CCCP-Engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from cccp.model import GLMModel  # noqa: E402
from cccp.store import PackedCpuExpertPool  # noqa: E402


class _Future:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def result(self) -> None:
        return None


class _Store:
    def __init__(self) -> None:
        self.codebook_clears = 0
        self.released_layers: list[int] = []

    def clear_codebook_cache(self) -> int:
        self.codebook_clears += 1
        return 1

    def release_expert_layer(self, layer: int) -> bool:
        self.released_layers.append(int(layer))
        return True


def test_route_scan_cache_releases_previous_layer(monkeypatch) -> None:
    monkeypatch.setenv("CCCP_ROUTE_SCAN_LAYER_LOCAL", "1")
    store = _Store()
    pool = PackedCpuExpertPool(store, budget_gb=0.01)
    first_future = _Future()
    pool.cache[(0, 3)] = (SimpleNamespace(nbytes=64),)
    pool.bytes = 64
    pool._pending[(0, 4)] = first_future

    pool._enter_layer(0)
    assert not pool.cache
    assert pool.bytes == 0
    assert first_future.cancelled is True
    assert store.codebook_clears == 1
    assert pool.layer_cache_resets == 0

    pool.cache[(0, 7)] = (SimpleNamespace(nbytes=128),)
    pool.bytes = 128
    pool._enter_layer(0)
    assert pool.cache
    assert store.codebook_clears == 1

    pool._enter_layer(1)
    assert not pool.cache
    assert pool.bytes == 0
    assert store.codebook_clears == 2
    assert pool.layer_cache_resets == 1


def test_normal_runtime_keeps_cross_layer_lru(monkeypatch) -> None:
    monkeypatch.delenv("CCCP_ROUTE_SCAN_LAYER_LOCAL", raising=False)
    store = _Store()
    pool = PackedCpuExpertPool(store, budget_gb=0.01)
    pool.cache[(0, 3)] = (SimpleNamespace(nbytes=64),)
    pool.bytes = 64

    pool._enter_layer(1)
    assert (0, 3) in pool.cache
    assert pool.bytes == 64
    assert store.codebook_clears == 0


def test_route_scan_explicitly_releases_completed_layer(monkeypatch) -> None:
    monkeypatch.setenv("CCCP_ROUTE_SCAN_LAYER_LOCAL", "1")
    store = _Store()
    pool = PackedCpuExpertPool(store, budget_gb=0.01)
    pool._enter_layer(3)
    pool.cache[(3, 7)] = (SimpleNamespace(nbytes=128),)
    pool.bytes = 128

    assert pool.release_scan_layer(3) is True
    assert not pool.cache
    assert pool.bytes == 0
    assert pool._active_layer is None
    assert store.released_layers == [3]
    assert pool.layer_cache_resets == 1


def test_normal_runtime_does_not_explicitly_release_layer(monkeypatch) -> None:
    monkeypatch.delenv("CCCP_ROUTE_SCAN_LAYER_LOCAL", raising=False)
    store = _Store()
    pool = PackedCpuExpertPool(store, budget_gb=0.01)

    assert pool.release_scan_layer(3) is False
    assert store.released_layers == []


class _ReleasePool:
    def __init__(self) -> None:
        self.layers: list[int] = []

    def release_scan_layer(self, layer: int) -> bool:
        self.layers.append(int(layer))
        return True


def test_glm_route_scan_releases_only_completed_layer_weights(monkeypatch) -> None:
    monkeypatch.setenv("CCCP_ROUTE_SCAN_LAYER_LOCAL", "1")
    model = GLMModel.__new__(GLMModel)
    model._wcache = {
        "model.layers.3.input_layernorm.weight": object(),
        "model.layers.3.mlp.gate.weight": object(),
        "model.layers.4.input_layernorm.weight": object(),
        "model.embed_tokens.weight": object(),
    }
    model._wuk = {3: object(), 4: object()}
    model._wuv = {3: object(), 4: object()}
    model._masks = {3: object(), 4: object()}
    model._prev_ids = {3: [1], 4: [2]}
    model.pool = _ReleasePool()

    released = model._release_completed_scan_layer(3)

    assert released == {"dense_objects": 2, "expert_shards": 1}
    assert all(not name.startswith("model.layers.3.") for name in model._wcache)
    assert "model.layers.4.input_layernorm.weight" in model._wcache
    assert "model.embed_tokens.weight" in model._wcache
    assert 3 not in model._wuk and 4 in model._wuk
    assert 3 not in model._masks and 4 in model._masks
    assert model.pool.layers == [3]


def test_glm_normal_runtime_keeps_layer_weights(monkeypatch) -> None:
    monkeypatch.delenv("CCCP_ROUTE_SCAN_LAYER_LOCAL", raising=False)
    model = GLMModel.__new__(GLMModel)
    marker = object()
    model._wcache = {"model.layers.3.mlp.gate.weight": marker}

    assert model._release_completed_scan_layer(3) == {
        "dense_objects": 0,
        "expert_shards": 0,
    }
    assert model._wcache["model.layers.3.mlp.gate.weight"] is marker
