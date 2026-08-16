"""CCCP 品牌完整性与自动运行参数门禁。"""
from pathlib import Path

from launcher.settings import Settings


ROOT = Path(__file__).resolve().parents[1]


def test_settings_api_cannot_override_internal_paths_or_tuning() -> None:
    settings = Settings(
        cccp_engine_path="built-in-engine",
        python_path="built-in-python",
        cpu_python_path="built-in-cpu",
        cuda_python_path="built-in-cuda",
        amd_python_path="built-in-amd",
    )
    settings.update({
        "cccp_engine_path": "user-engine",
        "python_path": "user-python",
        "cpu_python_path": "user-cpu",
        "cuda_python_path": "user-cuda",
        "amd_python_path": "user-amd",
        "expert_cache_gb": 999,
        "default_context": 32768,
        "cpu_threads": 512,
        "cpu_compile_mode": "off",
        "memory_limit_gb": 2048,
        "discord_url": "https://invalid.example",
        "modelscope_profile_url": "https://invalid.example",
        "community_index_url": "https://invalid.example/index.json",
        "default_download_source": "hf",
        "hf_endpoint": "https://invalid.example",
        "model_download_dir": "D:/invalid",
    })
    assert settings.cccp_engine_path == str(Path("built-in-engine").resolve())
    assert settings.python_path == str(Path("built-in-python").resolve())
    assert settings.cpu_python_path == str(Path("built-in-cpu").resolve())
    assert settings.cuda_python_path == str(Path("built-in-cuda").resolve())
    assert settings.amd_python_path == str(Path("built-in-amd").resolve())
    assert settings.expert_cache_gb == 8.0
    assert settings.default_context == 512
    assert settings.cpu_threads == 0
    assert settings.cpu_compile_mode == "q4"
    assert settings.memory_limit_gb == 32.0
    assert settings.discord_url == "https://discord.gg/eNnwmAUY4M"
    assert settings.modelscope_profile_url == "https://www.modelscope.cn/profile/ValueFX"
    assert settings.community_index_url == ""
    assert settings.default_download_source == "modelscope"
    assert settings.hf_endpoint == "https://hf-mirror.com"
    assert settings.model_download_dir == ""


def test_webui_has_no_manual_runtime_path_or_tuning_controls() -> None:
    html = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "webui" / "app.js").read_text(encoding="utf-8")
    forbidden_ids = (
        "setCccpPath", "setPythonPath", "setCudaPythonPath", "setAmdPythonPath",
        "setExpertCache", "setDefaultContext", "setCpuThreads", "setCpuCompile",
        "setMemoryLimit", "cacheGbInput", "maxCtxInput", "cpuThreadsInput",
        "cpuCompileInput", "memoryLimitInput",
        "setDiscord", "setIndexUrl", "setModelScopeProfile",
        "setDownloadSource", "setHfEndpoint", "setDlDir",
    )
    for control_id in forbidden_ids:
        assert control_id not in html
        assert control_id not in script


def test_theme_is_restored_from_backend_and_saved_on_change() -> None:
    script = (ROOT / "webui" / "app.js").read_text(encoding="utf-8")
    assert "applyTheme(theme);" in script
    assert 'body: JSON.stringify({ theme_mode: theme })' in script
    assert '$("#settingsHint").textContent = "主题已保存";' in script


def test_training_profile_controls_wait_for_completed_heatmap_plan() -> None:
    html = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "webui" / "app.js").read_text(encoding="utf-8")
    assert 'id="profileSaveActions" hidden' in html
    assert 'class="field-block final-profile-fields" hidden' in html
    assert (
        "const canSaveProfile = done && hasHeatmap && "
        "(j.plan_keys || []).length > 0;"
    ) in script
    assert '$("#profileSaveActions").hidden = !canSaveProfile;' in script
    assert '$(".final-profile-fields").hidden = !done;' in script
    assert "let coveragePlanning = false;" in script
    assert "let coveragePendingValue = null;" in script
    assert "async function flushCoveragePlan()" in script


def test_terminal_has_indeterminate_first_compile_progress() -> None:
    script = (ROOT / "webui" / "app.js").read_text(encoding="utf-8")
    style = (ROOT / "webui" / "style.css").read_text(encoding="utf-8")
    assert 'classList.toggle("indeterminate", indeterminate)' in script
    assert '$("#termPercent").textContent = indeterminate ? "编译中"' in script
    assert ".load-progress.indeterminate > div" in style
    assert "@keyframes terminal-compile-progress" in style


def test_public_docs_only_contain_end_user_basics() -> None:
    assert {path.name for path in (ROOT / "docs").iterdir() if path.is_file()} == {
        "中文使用手册.md",
        "依赖与离线环境说明.md",
        "AMD核显兼容性说明.md",
    }
    packaging = (ROOT / "scripts" / "package_offline_release.ps1").read_text(
        encoding="utf-8"
    )
    assert "测试报告.md" not in packaging
    assert "INTERFACE.md" not in packaging
    assert 'Copy-Tree (Join-Path $engineSource "cccp")' in packaging
    assert 'Copy-Tree (Join-Path $engineSource "_vendor")' in packaging
    assert 'Copy-Tree (Join-Path $root "engine\\CCCP-Engine")' not in packaging
    assert "audit_offline_release.py" in packaging


def test_default_models_directory_is_bundled(monkeypatch, tmp_path) -> None:
    from launcher import resources

    monkeypatch.setattr(resources, "runtime_root", lambda: tmp_path)
    assert resources.default_models_dir() == tmp_path / "models"
    assert (tmp_path / "models").is_dir()


def test_active_source_contains_no_legacy_brand() -> None:
    needle = bytes((116, 112, 113))
    text_suffixes = {
        ".py", ".js", ".css", ".html", ".md", ".txt", ".json", ".toml",
        ".ini", ".cfg", ".yaml", ".yml", ".ps1", ".bat", ".sh", ".spec",
        ".cpp", ".cu", ".cuh", ".h", ".hpp",
    }
    roots = [
        ROOT / name for name in (
            "launcher", "webui", "scripts", "tests", "packaging", "docs",
            "profiles", "engine", "data",
        )
    ]
    paths = [ROOT / name for name in ("README.md", "列表.md", "CHANGELOG.md", "VERSION")]
    for directory in roots:
        if directory.is_dir():
            paths.extend(directory.rglob("*"))
    for path in paths:
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        assert needle not in path.name.lower().encode("utf-8")
        if path.suffix.lower() in text_suffixes or path.name in {"LICENSE", "VERSION"}:
            assert needle not in path.read_bytes().lower(), path


def test_public_channels_contain_only_cccp_brand() -> None:
    legacy_tokens = (
        bytes((116, 112, 113)),
        bytes((116, 121, 108, 111, 113, 117, 97, 110, 116)),
    )
    public_repo = ROOT / "发布" / "CCCP-github"
    paths = [
        ROOT / "官网" / "index.html",
        ROOT / "CCCP框架介绍.html",
        public_repo / "README.md",
        public_repo / "README_EN.md",
        public_repo / "README_RU.md",
        public_repo / "assets" / "cccp-quality-chart-v2.svg",
    ]
    for path in paths:
        content = path.read_bytes().lower()
        for token in legacy_tokens:
            assert token not in content, path
