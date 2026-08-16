"""downloads:注入 fake backend 的离线任务生命周期 + 社区索引解析 + 默认落盘目录。"""
import asyncio
import time

import pytest

from launcher.downloads import DownloadEngine, default_target, fetch_index
from launcher.settings import Settings


def _settled(job):
    for _ in range(200):
        if job.finished_at:
            return
        time.sleep(0.01)
    raise AssertionError("job 未结束")


@pytest.fixture(autouse=True)
def _isolated_models_dir(monkeypatch, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr("launcher.downloads.default_models_dir", lambda: models)
    return models


def test_download_done(tmp_path):
    eng = DownloadEngine(Settings(), backend=lambda job, s: str(job.target_dir))
    job = eng.submit({"repo": "a/b", "target_dir": str(tmp_path / "models" / "b")})
    _settled(job)
    assert job.status == "done" and job.result_path == str(tmp_path / "models" / "b")
    assert eng.list()[0]["id"] == job.id
    assert eng.get(job.id) is job
    assert eng.delete(job.id) is True


def test_download_defaults_to_modelscope(tmp_path):
    eng = DownloadEngine(Settings(), backend=lambda job, s: str(job.target_dir))
    job = eng.submit({"repo": "a/b", "target_dir": str(tmp_path / "models" / "b")})
    _settled(job)
    assert job.source == "modelscope"


def test_download_failed(tmp_path):
    def bad(job, s):
        raise RuntimeError("缺少 huggingface_hub")
    eng = DownloadEngine(Settings(), backend=bad)
    job = eng.submit({"repo": "a/b", "source": "modelscope", "target_dir": str(tmp_path / "models" / "b")})
    _settled(job)
    assert job.status == "failed" and "huggingface_hub" in job.error
    assert eng.delete(job.id) is True


def test_download_validate(tmp_path):
    eng = DownloadEngine(Settings(), backend=lambda j, s: "")
    with pytest.raises(ValueError):
        eng.submit({"repo": ""})
    with pytest.raises(ValueError):
        eng.submit({"repo": "a/b", "source": "xx"})


def test_download_rejects_target_outside_bundled_models(tmp_path):
    eng = DownloadEngine(Settings(), backend=lambda j, s: "")
    with pytest.raises(ValueError, match="models"):
        eng.submit({"repo": "a/b", "target_dir": str(tmp_path / "outside" / "b")})
    with pytest.raises(ValueError, match="models"):
        eng.submit({"repo": "a/.."})


def test_default_target_is_bundled_models(monkeypatch, tmp_path):
    models = tmp_path / "models"
    monkeypatch.setattr("launcher.downloads.default_models_dir", lambda: models)
    settings = Settings(model_download_dir=str(tmp_path / "ignored"),
                        model_roots=[str(tmp_path / "ignored-roots")])
    assert default_target(settings, "org/My-Model") == models / "My-Model"


def test_fetch_index_parse(monkeypatch):
    class R:
        def raise_for_status(self): pass
        def json(self): return {"profiles": [{"name": "x", "url": "u"}, {"name": "no-url"}]}

    class C:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url): return R()

    monkeypatch.setattr("launcher.downloads.httpx.AsyncClient", lambda **kw: C())
    entries = asyncio.run(fetch_index("http://example/x.json"))
    assert entries == [{"name": "x", "url": "u"}]
