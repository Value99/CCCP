"""Update checks are asynchronous, source-aware, and failure tolerant."""
from __future__ import annotations

import sys
import time

from fastapi.testclient import TestClient

import launcher.app as app_module
import launcher.state as state_module
import launcher.updates as updates_module
from launcher.settings import Settings
from launcher.updates import UPDATE_SCHEMA, UpdateChecker, version_key


def manifest(version: str = "0.9.1") -> dict:
    return {
        "schema": UPDATE_SCHEMA,
        "version": version,
        "title": f"CCCP 启动器 {version}",
        "summary": "更新说明",
        "release_notes": ["第一项", "第二项"],
    }


def test_version_comparison_is_numeric():
    assert version_key("v0.10.0") > version_key("0.9.9")
    assert version_key("0.9") == version_key("0.9.0")


def test_checker_prefers_visionsic_and_uses_its_download_page():
    calls = []

    def fetch(source):
        calls.append(source.id)
        return manifest()

    checker = UpdateChecker("0.9.0", lambda: "", fetcher=fetch)
    started = time.perf_counter()
    checker.start()
    assert time.perf_counter() - started < 0.1
    result = checker.wait()
    assert calls == ["visionsic"]
    assert result["status"] == "available"
    assert result["source"] == "visionsic"
    assert result["download_url"] == "https://www.visionsic.com/cccp/"


def test_checker_falls_back_to_github_without_blocking_caller():
    calls = []

    def fetch(source):
        calls.append(source.id)
        if source.id == "visionsic":
            raise TimeoutError("offline")
        return manifest()

    checker = UpdateChecker("0.9.0", lambda: "", fetcher=fetch)
    checker.start()
    result = checker.wait()
    assert calls == ["visionsic", "github"]
    assert result["source"] == "github"
    assert result["download_url"] == "https://github.com/Value99/CCCP"
    assert "visionsic" in result["errors"]


def test_both_sources_unavailable_is_a_silent_status():
    checker = UpdateChecker(
        "0.9.0", lambda: "", fetcher=lambda source: (_ for _ in ()).throw(OSError(source.id))
    )
    checker.start()
    result = checker.wait()
    assert result["status"] == "unavailable"
    assert set(result["errors"]) == {"visionsic", "github"}


def test_same_or_skipped_version_does_not_prompt():
    current = UpdateChecker("0.9.0", lambda: "", fetcher=lambda source: manifest("0.9.0"))
    current.start()
    assert current.wait()["status"] == "current"
    skipped = UpdateChecker("0.9.0", lambda: "0.9.1", fetcher=lambda source: manifest())
    skipped.start()
    assert skipped.wait()["status"] == "ignored"


def _app(tmp_path, monkeypatch):
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    engine = tmp_path / "engine"
    (engine / "cccp").mkdir(parents=True)
    (engine / "cccp" / "__main__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(app_module, "USER_PROFILE_DIR", profiles)
    monkeypatch.setattr(state_module, "STATE_FILE", tmp_path / "state.json")
    settings = Settings(
        cccp_engine_path=str(engine), python_path=sys.executable, model_roots=[str(models)]
    )
    return app_module.create_app(settings)


def test_update_api_persists_ignore_and_opens_only_detected_source(tmp_path, monkeypatch):
    monkeypatch.setattr(
        updates_module, "UPDATE_SOURCES", updates_module.UPDATE_SOURCES,
    )
    app = _app(tmp_path, monkeypatch)
    app.state.settings.save = lambda: None
    app.state.updates = UpdateChecker(
        "0.9.0", lambda: app.state.settings.skipped_update_version,
        fetcher=lambda source: manifest(),
    )
    app.state.updates.start()
    assert app.state.updates.wait()["status"] == "available"
    opened = []
    monkeypatch.setattr(app_module.webbrowser, "open", opened.append)
    with TestClient(app) as client:
        ignored = client.post("/api/update/ignore", json={"version": "0.9.1"})
        assert ignored.status_code == 200
        assert ignored.json()["status"] == "ignored"
        rejected = client.post("/api/update/open", json={"source": "github"})
        assert rejected.status_code == 400
        opened_response = client.post("/api/update/open", json={"source": "visionsic"})
        assert opened_response.status_code == 200
    for _ in range(100):
        if opened:
            break
        time.sleep(0.01)
    assert opened == ["https://www.visionsic.com/cccp/"]


def test_update_ui_contains_three_explicit_choices():
    html = (app_module.WEBUI_STATIC / "index.html").read_text(encoding="utf-8")
    script = (app_module.WEBUI_STATIC / "app.js").read_text(encoding="utf-8")
    assert "稍后提醒" in html
    assert "本版本不升级" in html
    assert "updateOpenBtn" in html
    assert "checkForUpdates(false)" in script
    assert 'source: state.update.source' in script
