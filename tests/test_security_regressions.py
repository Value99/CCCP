"""Regression coverage for security and GPU-only launcher paths."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import launcher.app as app_module
import launcher.io_utils as io_utils
import launcher.state as state_module
import launcher.training as training_module
from launcher.settings import Settings


def _isolated_app(tmp_path, monkeypatch):
    user_profiles = tmp_path / "profiles" / "user"
    user_profiles.mkdir(parents=True)
    models = tmp_path / "models"
    models.mkdir()
    engine = tmp_path / "engine"
    (engine / "cccp").mkdir(parents=True)
    (engine / "cccp" / "__main__.py").write_text("", encoding="utf-8")
    corpus = tmp_path / "data" / "corpus"
    training = tmp_path / "data" / "training"
    corpus.mkdir(parents=True)
    training.mkdir(parents=True)
    monkeypatch.setattr(app_module, "USER_PROFILE_DIR", user_profiles)
    monkeypatch.setattr(state_module, "STATE_FILE", tmp_path / "data" / "state.json")
    monkeypatch.setattr(training_module, "CORPUS_DIR", corpus)
    monkeypatch.setattr(training_module, "TRAIN_DIR", training)
    settings = Settings(
        cccp_engine_path=str(engine), python_path=sys.executable,
        model_roots=[str(models)],
    )
    return app_module.create_app(settings)


@pytest.mark.parametrize("device", ["cuda", "amd"])
def test_gpu_automatic_context_calls_imported_backend_probe(monkeypatch, device):
    calls = []
    monkeypatch.setattr(app_module, "_memory_status", lambda: (32.0, 24.0))
    monkeypatch.setattr(
        app_module,
        "probe_backend",
        lambda settings, backend, root: calls.append((backend, root))
        or {"device_memory_gb": 12.0},
    )
    settings = Settings(cccp_engine_path="")
    assert app_module._automatic_context("", 1.0, device, settings) == 4096
    assert calls == [(device, settings.cccp_engine_path)]


@pytest.mark.parametrize("device", ["cuda", "amd"])
def test_gpu_context_does_not_subtract_host_expert_cache(
    monkeypatch, device, tmp_path
):
    settings = Settings()
    monkeypatch.setattr(
        app_module,
        "probe_backend",
        lambda *_args, **_kwargs: {"device_memory_gb": 16.0},
    )
    monkeypatch.setattr(app_module, "_memory_status", lambda: (64.0, 64.0))

    assert app_module._automatic_context("", 80.0, device, settings) == 4096


def test_gpu_context_steps_down_until_fixed_cuda_working_set_fits(
    monkeypatch,
):
    settings = Settings()
    model = SimpleNamespace(max_context=32768, dense_gb=8.0)
    monkeypatch.setattr(app_module, "inspect_model", lambda _path: model)
    monkeypatch.setattr(
        app_module,
        "probe_backend",
        lambda *_args, **_kwargs: {
            "device_memory_gb": 16.0,
            "device_available_memory_gb": 14.0,
        },
    )
    minimums = {4096: 20.0, 2048: 16.0, 1024: 13.0, 512: 12.0}
    monkeypatch.setattr(
        app_module,
        "estimate_gpu_vram_plan",
        lambda _model, *, max_ctx, expert_cache_gb: {
            "minimum_vram_gb": minimums[max_ctx],
        },
    )

    assert app_module._automatic_context(
        "D:/model", 80.0, "cuda", settings,
    ) == 1024


def test_kimi_ram_dense_gpu_uses_safe_automatic_context(monkeypatch):
    settings = Settings()
    model = SimpleNamespace(max_context=131072, dense_gb=56.0)
    monkeypatch.setattr(app_module, "inspect_model", lambda _path: model)
    monkeypatch.setattr(
        app_module,
        "probe_backend",
        lambda *_args, **_kwargs: {
            "device_memory_gb": 16.0,
            "device_available_memory_gb": 16.0,
        },
    )
    monkeypatch.setattr(app_module, "_memory_status", lambda: (128.0, 96.0))
    monkeypatch.setattr(
        app_module,
        "estimate_gpu_vram_plan",
        lambda _model, *, max_ctx, expert_cache_gb: {
            "architecture": "kimi_k3",
            "minimum_vram_gb": 10.5,
            "recommended_vram_gb": 66.5,
        },
    )

    assert app_module._automatic_context(
        "D:/kimi", 55.0, "cuda", settings,
    ) == 512


def test_cross_site_write_is_rejected_before_state_change(tmp_path, monkeypatch):
    app = _isolated_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        blocked = client.post(
            "/api/profiles/select",
            json={"ids": ["attacker"]},
            headers={"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"},
        )
        assert blocked.status_code == 403
        assert app.state.app_state.data["selected_profiles"] == []
        allowed = client.post(
            "/api/profiles/select",
            json={"ids": []},
            headers={"Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"},
        )
        assert allowed.status_code == 200


@pytest.mark.parametrize(
    ("path", "filename", "limit_name"),
    [
        ("/api/profiles/import", "large.yaml", "MAX_PROFILE_UPLOAD_BYTES"),
        ("/api/training/corpus", "large.txt", "MAX_CORPUS_UPLOAD_BYTES"),
    ],
)
def test_upload_size_is_bounded(tmp_path, monkeypatch, path, filename, limit_name):
    monkeypatch.setattr(app_module, limit_name, 8)
    monkeypatch.setitem(app_module._UPLOAD_LIMITS, path, 8)
    app = _isolated_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        response = client.post(path, files={"file": (filename, b"123456789")})
    assert response.status_code == 413


def test_upload_request_rate_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "UPLOAD_REQUESTS_PER_MINUTE", 2)
    app = _isolated_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        for _ in range(2):
            client.post(
                "/api/profiles/import", files={"file": ("invalid.yaml", b"invalid")}
            )
        response = client.post(
            "/api/profiles/import", files={"file": ("invalid.yaml", b"invalid")}
        )
    assert response.status_code == 429
    assert response.headers["retry-after"] == "60"


def test_atomic_write_keeps_previous_file_when_replace_fails(tmp_path, monkeypatch):
    target = tmp_path / "settings.json"
    target.write_text("previous", encoding="utf-8")

    def fail_replace(_source, _target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(io_utils.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        io_utils.atomic_write_text(target, "new value")
    assert target.read_text(encoding="utf-8") == "previous"
    assert list(tmp_path.glob(".settings.json.*.tmp")) == []


def test_corpus_save_is_atomic(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    target = corpus / "sample.txt"
    target.write_text("previous", encoding="utf-8")
    monkeypatch.setattr(training_module, "CORPUS_DIR", corpus)

    def fail_replace(_source, _target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(io_utils.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        training_module.save_corpus_file("sample.txt", b"new corpus")
    assert target.read_text(encoding="utf-8") == "previous"
