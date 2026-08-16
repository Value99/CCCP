"""多后端目录、命令映射与环境隔离。"""
from pathlib import Path
from types import SimpleNamespace

import launcher.resources as resources
from launcher.profiles import Combination
from launcher.settings import Settings
from launcher.cccp_adapter import LaunchConfig, CCCPEngineAdapter


def test_runtime_candidates_are_vendor_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "runtime_root", lambda: tmp_path)
    cpu = resources.runtime_python_candidates("cpu")
    cuda = resources.runtime_python_candidates("cuda")
    amd = resources.runtime_python_candidates("amd")
    assert cpu[0] == tmp_path / "runtime" / "cpu" / "env" / "python.exe"
    assert cuda == (tmp_path / "runtime" / "cuda" / "env" / "python.exe",)
    assert amd == (tmp_path / "runtime" / "amd" / "env" / "python.exe",)


def test_amd_maps_to_torch_cuda_api_but_keeps_backend_identity(tmp_path, monkeypatch):
    engine = tmp_path / "engine"
    (engine / "cccp").mkdir(parents=True)
    (engine / "cccp" / "__main__.py").write_text("", encoding="utf-8")
    python = tmp_path / "runtime" / "amd" / "env" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    settings = Settings(cccp_engine_path=str(engine), amd_python_path=str(python))
    adapter = CCCPEngineAdapter(settings)
    cfg = LaunchConfig(
        model_path=str(tmp_path / "model"), profiles=["roleplay"],
        combination=Combination(["roleplay"], {}, 0), device="amd", port=8801,
        cache_gb=18,
    )
    monkeypatch.setattr(adapter, "_profile_counts_file", lambda *a, **k: tmp_path / "profile.json")
    command = adapter.build_command(cfg)
    assert command[command.index("--device") + 1] == "cuda"
    assert command[0] == str(python.resolve())
    assert "--cache-gb" not in command
    assert "--no-extreme" not in command
    env = adapter._env(cfg)
    assert env["CCCP_RUNTIME_BACKEND"] == "amd"
    assert env["CCCP_REQUIRE_FUSED"] == "1"
    assert env["CCCP_FLASHINFER_MLA"] == "0"


def test_amd_launch_drops_stale_parent_graph_overrides(tmp_path, monkeypatch):
    engine = tmp_path / "engine"
    (engine / "cccp").mkdir(parents=True)
    (engine / "cccp" / "__main__.py").write_text("", encoding="utf-8")
    python = tmp_path / "runtime" / "amd" / "env" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    adapter = CCCPEngineAdapter(
        Settings(cccp_engine_path=str(engine), amd_python_path=str(python))
    )
    cfg = LaunchConfig(
        model_path=str(tmp_path / "model"),
        profiles=["roleplay"],
        combination=Combination(["roleplay"], {}, 0),
        device="amd",
        port=8801,
    )
    monkeypatch.setattr(
        adapter,
        "_profile_counts_file",
        lambda *a, **k: tmp_path / "profile.json",
    )
    for key in (
        "CCCP_PACKED_FULL_GPU",
        "CCCP_SINGLE_GPU_LAYER_GRAPH",
        "CCCP_DSV4_TOKEN_GRAPH",
        "CCCP_TP_LAYER_GRAPH",
        "CCCP_TP_HIDDEN",
        "CCCP_TP_NO_OWNER",
        "CCCP_STATIC_DECODE_GRAPHS",
        "CCCP_STATIC_FFN_GRAPH",
    ):
        monkeypatch.setenv(key, "0")

    env = adapter._env(cfg)

    for key in (
        "CCCP_PACKED_FULL_GPU",
        "CCCP_SINGLE_GPU_LAYER_GRAPH",
        "CCCP_DSV4_TOKEN_GRAPH",
        "CCCP_TP_LAYER_GRAPH",
        "CCCP_TP_HIDDEN",
        "CCCP_TP_NO_OWNER",
        "CCCP_STATIC_DECODE_GRAPHS",
        "CCCP_STATIC_FFN_GRAPH",
    ):
        assert key not in env


def test_amd_forced_mapped_mode_keeps_explicit_hybrid_budget(tmp_path, monkeypatch):
    engine = tmp_path / "engine"
    (engine / "cccp").mkdir(parents=True)
    (engine / "cccp" / "__main__.py").write_text("", encoding="utf-8")
    python = tmp_path / "runtime" / "amd" / "env" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    adapter = CCCPEngineAdapter(
        Settings(cccp_engine_path=str(engine), amd_python_path=str(python))
    )
    cfg = LaunchConfig(
        model_path=str(tmp_path / "model"),
        profiles=["roleplay"],
        combination=Combination(["roleplay"], {}, 0),
        device="amd",
        profile_mode="mapped",
        cache_gb=18,
        port=8801,
    )
    monkeypatch.setattr(
        adapter,
        "_profile_counts_file",
        lambda *a, **k: tmp_path / "profile.json",
    )

    command = adapter.build_command(cfg)

    assert command[command.index("--cache-gb") + 1] == "18"
    assert "--no-extreme" in command
