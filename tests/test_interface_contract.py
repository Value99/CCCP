from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient


ENGINE_ROOT = Path(__file__).resolve().parents[1] / "engine" / "CCCP-Engine"
sys.path.insert(0, str(ENGINE_ROOT))

from cccp.interface_contract import expert_bytes_payload, model_spec_payload  # noqa: E402
from cccp.openai_api import create_app  # noqa: E402


def _model(tmp_path: Path) -> Path:
    manifest = {
        "format": "cccp-1",
        "version": "fixture-v1",
        "architecture": "deepseek_v4",
        "config": {
            "n_layers": 1,
            "n_experts": 2,
            "top_k": 1,
            "max_position_embeddings": 4096,
        },
        "expert_files": {"0": "experts.L00.safetensors"},
        "tiers_per_layer": {"0": "vv"},
    }
    (tmp_path / "cccp.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "experts.L00.safetensors").write_bytes(b"0123456789")
    audit = {
        "layer": 0,
        "file_bytes": 10,
        "experts": {
            "0": {"gate": {"packed_bytes": 1}, "up": {"packed_bytes": 2}},
            "1": {"gate": {"packed_bytes": 3}, "down": {"packed_bytes": 4}},
        },
    }
    (tmp_path / "experts.L00.audit.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )
    return tmp_path


def _service(model_root: Path):
    manifest = SimpleNamespace(
        expert_files={0: "experts.L00.safetensors"},
        tiers_per_layer={0: "vv"},
    )
    store = SimpleNamespace(
        root=str(model_root),
        cfg={"n_experts": 2, "top_k": 1},
        man=manifest,
        profile_path=str(model_root / "profile.json"),
        profile_loaded=True,
    )
    pool = SimpleNamespace(
        pinned={(0, 0): object()},
        cache={(0, 1): object()},
        full_resident=False,
        host_expert_bytes=3,
        gpu_storage_bytes=7,
        route_counts=Counter({(0, 0): 4, (0, 1): 2}),
    )
    model = SimpleNamespace(
        store=store,
        pool=pool,
        device=SimpleNamespace(type="cpu"),
    )
    engine = SimpleNamespace(model=model, arch="dsv4")
    return SimpleNamespace(engine=engine, busy=False)


def test_exact_bytes_and_model_spec(tmp_path):
    root = _model(tmp_path)
    byte_table = expert_bytes_payload(root)
    assert byte_table["schema"] == "cccp-expert-bytes-v1"
    assert byte_table["calibrated"] is True
    assert byte_table["layer_expert_counts"] == {"0": 2}
    assert byte_table["bytes"] == {"0:0": 3, "0:1": 7}
    assert byte_table["total_bytes"] == 10

    spec = model_spec_payload(root)
    assert spec["schema"] == "cccp-model-spec-v1"
    assert spec["model_version"] == "fixture-v1"
    assert spec["routed_layers"] == [0]
    assert spec["layer_expert_counts"] == {"0": 2}
    assert spec["expert_bytes_calibrated"] is True


def test_kimi_routed_layer_contract(tmp_path):
    manifest = {
        "format": "cccp-1",
        "model_family": "kimi_k3",
        "config": {
            "n_layers": 2,
            "n_experts": 2,
            "top_k": 1,
            "max_position_embeddings": 4096,
        },
        "routed_experts": {
            "layer_files": {
                "1": {
                    "path": "experts.L001.safetensors",
                    "audit_path": "experts.L001.audit.json",
                },
            },
        },
    }
    (tmp_path / "cccp.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "experts.L001.safetensors").write_bytes(b"0123456789")
    (tmp_path / "experts.L001.audit.json").write_text(json.dumps({
        "layer": 1,
        "file_bytes": 10,
        "experts": {
            "0": {"gate": {"packed_bytes": 1}, "up": {"packed_bytes": 2}},
            "1": {"gate": {"packed_bytes": 3}, "down": {"packed_bytes": 4}},
        },
    }), encoding="utf-8")

    byte_table = expert_bytes_payload(tmp_path)
    spec = model_spec_payload(tmp_path)

    assert byte_table["calibrated"] is True
    assert byte_table["bytes"] == {"1:0": 3, "1:1": 7}
    assert spec["architecture"] == "kimi_k3"
    assert spec["routed_layers"] == [1]


def test_http_i1_i2_i3_i5_contracts(tmp_path, monkeypatch):
    root = _model(tmp_path)
    (root / "profile.json").write_text('{"counts":{}}', encoding="utf-8")
    monkeypatch.setenv("CCCP_PROFILE_JSON", str(root / "profile.json"))
    service = _service(root)
    app = create_app(
        service,
        served_model_name="fixture",
        api_key="secret",
        context_length=4096,
    )

    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["plan_file_ok"] is True
        assert health["experts_resident_ram"] == 2
        assert health["experts_resident_vram"] == 0
        assert health["expert_resident_ram_bytes"] == 3

        assert client.get("/v1/model-spec").status_code == 401
        headers = {"Authorization": "Bearer secret"}
        assert client.get("/v1/model-spec", headers=headers).json()["top_k"] == 1
        assert client.get("/v1/expert-bytes", headers=headers).json()["total_bytes"] == 10

        stats = client.get("/v1/expert-stats", headers=headers).json()
        assert stats["counts"] == {"0:0": 4, "0:1": 2}
        assert stats["tokens_seen"] == 6
        reset = client.post("/v1/expert-stats/reset", headers=headers).json()
        assert reset["enabled"] is True
        assert reset["counts"] == {}
