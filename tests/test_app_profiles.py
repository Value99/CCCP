"""配置分享 API：导出文件必须完整、可重新导入。"""
import json
import hashlib
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient

from launcher.app import create_app
from launcher.profiles import load_profile_dict
from launcher.settings import Settings


def test_profile_export_is_downloadable_and_reimportable(tmp_path, monkeypatch):
    app, _, _ = _isolated_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        response = client.get("/api/profiles/editable/export")
    assert response.status_code == 200, response.text
    assert "attachment" in response.headers["content-disposition"]
    payload = json.loads(response.content.decode("utf-8"))
    loaded = load_profile_dict(payload)
    assert loaded.id == "editable"
    assert loaded.expert_count == 1
    assert loaded.meta["model_version"] == "v2"
    assert loaded.meta["model_total_bytes"] == 2**30
    assert loaded.meta["model_manifest_sha256"]
    assert loaded.meta["model_top_k"] == 1


def _isolated_app(tmp_path, monkeypatch):
    import launcher.app as app_module
    import launcher.state as state_module
    user = tmp_path / "profiles" / "user"
    user.mkdir(parents=True)
    models = tmp_path / "models"
    model = models / "model-v2"
    profiles = model / "profiles"
    profiles.mkdir(parents=True)
    manifest = b"{}"
    (model / "cccp.json").write_bytes(manifest)
    fingerprint = hashlib.sha256(manifest).hexdigest()
    (profiles / "editable.json").write_text(json.dumps({
        "schema": "winui-expert-profile-v1", "id": "editable",
        "name": "原名", "description": "原说明", "tags": [],
        "experts": [{"key": "0:0", "size_mb": 1.0}],
        "drop": {"enabled": True, "hint_tags": []},
        "meta": {
            "model_name": "model-v2", "model_version": "v2", "model_format": "cccp-1",
            "model_manifest_sha256": fingerprint, "model_total_bytes": 2**30,
            "model_total_gib": 1.0, "model_layers": 1, "model_experts_per_layer": 1,
            "model_top_k": 1, "fixed_model_gib": 0.5, "dense_without_shared_gib": 0.4,
            "shared_expert_gib": 0.1, "configuration_budget_gib": 1.0,
            "configuration_resident_gib": 0.5 + 1 / 1024,
        },
    }), encoding="utf-8")
    monkeypatch.setattr(app_module, "USER_PROFILE_DIR", user)
    monkeypatch.setattr(state_module, "STATE_FILE", tmp_path / "state.json")
    engine = tmp_path / "engine"
    (engine / "cccp").mkdir(parents=True)
    (engine / "cccp" / "__main__.py").write_text("", encoding="utf-8")
    settings = Settings(cccp_engine_path=str(engine), python_path=sys.executable,
                        model_roots=[str(models)])
    return create_app(settings), user, profiles


def test_json_api_explicitly_declares_utf8_for_chinese_model_paths(
    tmp_path, monkeypatch,
):
    app, _, _ = _isolated_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        response = client.get("/api/models")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json; charset=utf-8"


def test_profile_name_and_description_can_be_edited_without_changing_experts(
    tmp_path, monkeypatch,
):
    app, user, profiles = _isolated_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        response = client.patch("/api/profiles/editable", json={
            "name": "自定义名称", "description": "自定义说明",
        })
        detail = client.get("/api/profiles/editable").json()
    assert response.status_code == 200, response.text
    assert detail["name"] == "自定义名称"
    assert detail["description"] == "自定义说明"
    assert [expert["key"] for expert in detail["experts"]] == ["0:0"]
    assert not (user / "editable.yaml").exists()
    assert json.loads((profiles / "editable.json").read_text(encoding="utf-8"))["name"] == "自定义名称"


def test_model_profile_can_be_deleted(tmp_path, monkeypatch):
    app, _, profiles = _isolated_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        assert client.delete("/api/profiles/editable").status_code == 200
        ids = {item["id"] for item in client.get("/api/profiles").json()["profiles"]}
        assert "editable" not in ids
    assert not (profiles / "editable.json").exists()


def test_model_delete_only_accepts_discovered_cccp_directory(tmp_path, monkeypatch):
    app, _, _ = _isolated_app(tmp_path, monkeypatch)
    model = tmp_path / "models" / "delete-me"
    model.mkdir(parents=True)
    (model / "cccp.json").write_text("{}", encoding="utf-8")
    (model / "payload.bin").write_bytes(b"model")
    outside = tmp_path / "outside"
    outside.mkdir()
    with TestClient(app) as client:
        refused = client.request("DELETE", "/api/models", json={"path": str(outside)})
        deleted = client.request("DELETE", "/api/models", json={"path": str(model)})
    assert refused.status_code == 400
    assert deleted.status_code == 200, deleted.text
    assert not model.exists()
    assert outside.is_dir()


def test_training_route_derives_dynamic_model_limit(tmp_path, monkeypatch):
    app, _, _ = _isolated_app(tmp_path, monkeypatch)
    import launcher.app as app_module
    monkeypatch.setattr(app_module, "inspect_model", lambda _path: SimpleNamespace(
        complete=True, errors=[], name="model-v2", model_version="v2",
        model_format="cccp-1", manifest_sha256="a" * 64,
        total_bytes=80_000_000_000, dense_gb=8.0,
        dense_without_shared_gb=7.0, shared_expert_gb=1.0,
        expert_gb=69.0, layers=43, experts_per_layer=256, top_k=6,
        expert_layers=list(range(43)), expert_layer_count=43,
        max_context=1_048_576,
    ))
    captured = {}

    def fake_submit(spec):
        captured.update(spec)
        return SimpleNamespace(to_dict=lambda: {"id": "job", "status": "pending"})

    monkeypatch.setattr(app.state.training, "submit", fake_submit)
    with TestClient(app) as client:
        response = client.post("/api/training/jobs", json={
            "model_path": "D:/model", "token_budget": 500_000,
            "corpus_files": ["sample.jsonl"],
        })
    assert response.status_code == 200
    assert captured["fixed_model_gib"] == 8.0
    assert captured["model_max_configuration_gib"] == 77.0
    assert captured["model_version"] == "v2"
    assert captured["token_budget"] == 500_000


def test_api_key_can_be_generated_saved_and_enforced(tmp_path, monkeypatch):
    app, _, _ = _isolated_app(tmp_path, monkeypatch)
    app.state.settings.save = lambda: None
    app.state.adapter.instance = SimpleNamespace()
    app.state.adapter.instance = None
    with TestClient(app) as client:
        generated = client.post("/api/settings/api-key/generate").json()["api_key"]
        assert generated.startswith("cccp_") and len(generated) > 32
        saved = client.post("/api/settings", json={"cccp_api_key": generated})
        assert saved.status_code == 200
        rejected = client.get("/v1/models")
        accepted = client.get(
            "/v1/models", headers={"Authorization": f"Bearer {generated}"}
        )
    assert rejected.status_code == 401
    assert accepted.status_code != 401


def test_launch_remembers_last_selected_device(tmp_path, monkeypatch):
    app, _, _ = _isolated_app(tmp_path, monkeypatch)
    saved = []
    app.state.settings.save = lambda: saved.append(
        app.state.settings.default_device
    )
    instance = SimpleNamespace(
        pid=123,
        port=8801,
        model=str(tmp_path / "models" / "model-v2"),
        served_model_name="winui-model",
        profiles=["editable"],
        started_at=1.0,
        log_file="test.log",
        base_url="http://127.0.0.1:8801",
        full_model=False,
    )
    monkeypatch.setattr(app.state.adapter, "launch", lambda _cfg: instance)

    with TestClient(app) as client:
        response = client.post("/api/launch", json={
            "profile_ids": ["editable"],
            "model_path": str(tmp_path / "models" / "model-v2"),
            "device": "cuda",
            "port": 8801,
        })

    assert response.status_code == 200, response.text
    assert app.state.settings.default_device == "cuda"
    assert saved == ["cuda"]


def test_theme_selection_is_saved_by_settings_api(tmp_path, monkeypatch):
    app, _, _ = _isolated_app(tmp_path, monkeypatch)
    saved = []
    app.state.settings.save = lambda: saved.append(app.state.settings.theme_mode)

    with TestClient(app) as client:
        response = client.post("/api/settings", json={"theme_mode": "dark"})
        restored = client.get("/api/settings")

    assert response.status_code == 200, response.text
    assert restored.json()["theme_mode"] == "dark"
    assert saved == ["dark"]


def test_invalid_theme_is_rejected(tmp_path, monkeypatch):
    app, _, _ = _isolated_app(tmp_path, monkeypatch)
    app.state.settings.save = lambda: None

    with TestClient(app) as client:
        response = client.post("/api/settings", json={"theme_mode": "neon"})

    assert response.status_code == 400
    assert app.state.settings.theme_mode == "system"
