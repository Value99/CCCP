"""Static guards for model-aware chat and unambiguous profile capacity UI."""
from __future__ import annotations

from pathlib import Path

from launcher.app import _training_terminal_progress


ROOT = Path(__file__).resolve().parents[1]


def test_profile_cards_and_home_chips_show_total_residency() -> None:
    script = (ROOT / "webui" / "app.js").read_text(encoding="utf-8")

    assert "profileResidentGiB(p)" in script
    assert "GiB 配置总驻留" in script
    assert "G 总驻留" in script


def test_real_engine_kv_modes_have_user_labels() -> None:
    script = (ROOT / "webui" / "app.js").read_text(encoding="utf-8")

    assert '"full-prefill": "KV 冷启动"' in script
    assert '"lcp-replay": "KV 前缀复用"' in script
    assert '"exact-prefix": "KV 续接"' in script


def test_thinking_levels_follow_the_running_model_capabilities() -> None:
    script = (ROOT / "webui" / "app.js").read_text(encoding="utf-8")
    html = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")

    assert 'api("/api/chat/models")' in script
    assert "think_efforts?.valid_efforts" in script
    assert "state.thinkingDefaultEffort" in script
    assert "reasoning_effort = state.thinkingDefaultEffort" in script
    assert 'id="thinkingScale"' in html


def test_training_start_switches_to_terminal_and_uses_shared_progress_contract() -> None:
    script = (ROOT / "webui" / "app.js").read_text(encoding="utf-8")
    assert "primeTrainingTerminal(r.job)" in script
    assert 'activateTab("terminal")' in script

    progress = _training_terminal_progress({
        "status": "running",
        "progress": 0.435,
        "message": "块 2/3 · 层 18/43",
        "created_at": 10.0,
        "finished_at": 20.0,
    })
    assert progress["phase"] == "token-route-scan"
    assert progress["label"] == "正在扫描专家路由"
    assert progress["percent"] == 44
    assert progress["detail"] == "块 2/3 · 层 18/43"


def test_model_delete_uses_the_shared_icon_library() -> None:
    script = (ROOT / "webui" / "app.js").read_text(encoding="utf-8")
    assert 'data-model-delete="${esc(m.path)}"' in script
    assert '<use href="#i-trash"/></svg>删除模型' in script


def test_profile_metadata_editor_is_an_in_app_dialog() -> None:
    script = (ROOT / "webui" / "app.js").read_text(encoding="utf-8")
    html = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")

    assert "prompt(" not in script
    assert "openProfileEdit(p)" in script
    assert 'id="profileEditOverlay"' in html
    assert 'id="profileEditForm"' in html


def test_model_discovery_immediately_rescans_adjacent_profiles() -> None:
    script = (ROOT / "webui" / "app.js").read_text(encoding="utf-8")
    load_models = script.split("async function loadModels()", 1)[1].split(
        "function updateTrainingModelLimit()", 1
    )[0]
    init = script.split("(async function init()", 1)[1].split(
        "/* ---------- 无边框自绘标题栏", 1
    )[0]

    assert "await loadProfiles();" in load_models
    assert init.index("await loadModels();") < init.index("refreshSessions();")
    assert '$("#modelSelect").dispatchEvent(new Event("change"));' in script


def test_selecting_a_model_rescans_its_profiles_without_restart() -> None:
    script = (ROOT / "webui" / "app.js").read_text(encoding="utf-8")
    home_handler = script.split(
        '$("#homeModelSelect").addEventListener("change", async () => {', 1
    )[1].split('$("#modelSelect").addEventListener', 1)[0]
    profile_handler = script.split(
        '$("#modelSelect").addEventListener("change", async () => {', 1
    )[1].split("async function refreshHome()", 1)[0]

    assert "await loadProfiles();" in home_handler
    assert "await loadProfiles();" in profile_handler


def test_completed_model_download_refreshes_models_and_adjacent_profiles() -> None:
    script = (ROOT / "webui" / "app.js").read_text(encoding="utf-8")
    refresh_jobs = script.split("async function refreshDlJobs()", 1)[1].split(
        "function refreshModelsPage()", 1
    )[0]

    assert "completedDownloadScans: new Set()" in script
    assert 'job.status === "done"' in refresh_jobs
    assert "await loadModels();" in refresh_jobs


def test_modelscope_download_has_a_launchable_default_repository() -> None:
    html = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")

    assert 'value="modelscope" selected' in html
    assert (
        'id="dlRepo" value="ValueFX/DeepSeek-V4-Flash-0731-CCCP-L"'
        in html
    )
    assert "下载完成后会自动扫描模型与相邻配置" in html


def test_home_model_summary_does_not_show_racy_profile_count() -> None:
    script = (ROOT / "webui" / "app.js").read_text(encoding="utf-8")
    summary = script.split("function updateHomeSelectionChip()", 1)[1].split(
        '$("#homeModelSelect").addEventListener', 1
    )[0]

    assert "state.selected.size" not in summary
    assert "个配置" not in summary
    assert 'ms?.selectedOptions[0]?.textContent || "未选择模型"' in summary


def test_low_vram_ui_distinguishes_reduced_arena_from_hard_minimum() -> None:
    script = (ROOT / "webui" / "app.js").read_text(encoding="utf-8")

    assert 'gpu_execution_tier === "reduced_expert_arena"' in script
    assert 'gpu_execution_tier === "below_minimum"' in script
    assert "CUDA 最低工作集" in script
    assert "RAM Dense 混合模式的 CUDA 基础工作区与最小专家块" in script
    assert "自动切换为 CPU 推理继续预检" in script
    assert 'body.device = "cpu"' in script
