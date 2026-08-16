"""Token 路由扫描、覆盖率规划与配置导出。"""
from __future__ import annotations

import json

import pytest

import launcher.training as training_module
from launcher.training import (
    TrainingEngine,
    TrainingJob,
    delete_corpus,
    export_counts,
    export_profile,
    export_scores,
    inspect_corpus,
    iter_corpus,
    iter_corpus_records,
    list_corpus,
    measured_activation,
    plan_route_coverage,
    prepare_scan_input,
    save_corpus_file,
)


def test_iter_corpus_preserves_roles_and_long_system(tmp_path):
    system = "角色设定" * 20_000
    path = tmp_path / "roleplay.jsonl"
    path.write_text(
        json.dumps({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "继续故事"},
                {"role": "assistant", "content": "她轻轻握住你的手。"},
            ]
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    records = list(iter_corpus_records([path]))
    assert [message["role"] for message in records[0]["messages"]] == [
        "system", "user", "assistant"
    ]
    assert records[0]["messages"][0]["content"] == system
    assert list(iter_corpus([path]))[0].endswith("她轻轻握住你的手。")


def test_prepare_scan_input_is_token_budget_driven_and_deterministic(tmp_path):
    source = tmp_path / "many.jsonl"
    source.write_text(
        "".join(
            json.dumps({"messages": [{"role": "user", "content": f"line-{i}-" + "长" * 200}]}, ensure_ascii=False) + "\n"
            for i in range(200)
        ),
        encoding="utf-8",
    )
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    count_a, chars_a = prepare_scan_input([source], first, 500_000, 5090)
    count_b, chars_b = prepare_scan_input([source], second, 500_000, 5090)
    assert count_a == count_b == 200
    assert chars_a == chars_b
    assert first.read_bytes() == second.read_bytes()
    raw = json.loads(first.read_text(encoding="utf-8").splitlines()[0])
    assert raw["messages"][0]["role"] == "user"


def test_corpus_metadata_contains_message_and_context_information(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    monkeypatch.setattr(training_module, "CORPUS_DIR", corpus_dir)
    info = save_corpus_file(
        "roleplay.jsonl",
        json.dumps({
            "messages": [
                {"role": "system", "content": "设定"},
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好呀"},
            ]
        }, ensure_ascii=False).encode("utf-8") + b"\ninvalid\n",
    )
    assert info["samples"] == 1
    assert info["messages"] == 3
    assert info["roles"] == ["assistant", "system", "user"]
    assert info["characters"] > 0
    assert info["invalid_lines"] == 1
    assert inspect_corpus(corpus_dir / "roleplay.jsonl")["meta_schema"] == 2
    assert list_corpus()[0]["stored_path"] == "data/corpus/roleplay.jsonl"
    assert delete_corpus("roleplay.jsonl") is True


def test_measured_activation_uses_route_scan_and_progress(tmp_path, monkeypatch):
    engine = tmp_path / "engine"
    (engine / "cccp").mkdir(parents=True)
    (engine / "cccp" / "__main__.py").write_text("", encoding="utf-8")
    python = tmp_path / "python.exe"
    python.write_bytes(b"")
    scan_input = tmp_path / "scan.jsonl"
    scan_input.write_text('{"messages":[{"role":"user","content":"hello"}]}\n', encoding="utf-8")
    job = TrainingJob(
        id="measured", corpus_files=["x.jsonl"], model_path=str(tmp_path / "model")
    )
    captured = {}

    class FakeProcess:
        def __init__(self, command, **kwargs):
            captured["command"] = command
            route = command[command.index("--output") + 1]
            report = command[command.index("--report") + 1]
            with open(route, "w", encoding="utf-8") as handle:
                json.dump({
                    "format": "cccp-expert-residency-scores-v1",
                    "scores": {"0:1": 7, "0:2": 0, "1:3": 2},
                    "observations": 9,
                }, handle)
            with open(report, "w", encoding="utf-8") as handle:
                json.dump({
                    "processed_tokens": 500_000,
                    "documents": 9,
                    "max_context_tokens": 65_536,
                    "truncated_documents": 0,
                }, handle)
            self.stdout = iter([
                'CCCP_ROUTE_SCAN_PROGRESS {"processed_tokens":4096,"token_budget":500000,"stage":"prefill"}\n'
            ])

        def wait(self):
            return 0

    monkeypatch.setattr(training_module.subprocess, "Popen", FakeProcess)
    events = []
    counts, report = measured_activation(
        job,
        scan_input,
        engine_root=engine,
        python_path=python,
        train_dir=tmp_path,
        progress_callback=lambda *event: events.append(event),
    )
    assert counts == {"0:1": 7, "1:3": 2}
    assert report["processed_tokens"] == 500_000
    assert job.route_observations == 9
    assert events == [(4096, 500_000, "prefill")]
    assert captured["command"][1:4] == ["-m", "cccp", "route-scan"]
    assert captured["command"][captured["command"].index("--prefill-block-tokens") + 1] == "4096"


def test_plan_route_coverage_is_per_layer_and_capacity_exact():
    counts = {
        "0:0": 60, "0:1": 30, "0:2": 10,
        "1:0": 50, "1:1": 25, "1:2": 25,
    }
    sizes = {key: 8.0 for key in counts}
    keys, used, actual, layers = plan_route_coverage(
        counts, 0.8, sizes, top_k=1, layers=2
    )
    assert set(keys) == {"0:0", "0:1", "1:0", "1:1", "1:2"}
    assert used == 40.0
    assert actual == 0.95
    assert layers == {"0": 0.9, "1": 1.0}


def test_uniform_layer_keeps_expert_count_needed_for_each_layer_coverage():
    counts = {
        **{f"0:{expert}": 1 for expert in range(256)},
        **{f"1:{expert}": 100 for expert in range(6)},
    }
    keys, _used, actual, layer_coverages = plan_route_coverage(
        counts, 0.95, top_k=6, layers=2
    )
    picked = {
        layer: sum(key.startswith(f"{layer}:") for key in keys)
        for layer in range(2)
    }
    assert picked == {0: 244, 1: 6}
    assert layer_coverages == {"0": 0.953125, "1": 1.0}
    assert actual == 0.985981


def test_low_coverage_keeps_native_top_k_per_expert_layer():
    counts = {
        **{f"3:{expert}": 100 - expert for expert in range(20)},
        **{f"7:{expert}": 50 - expert for expert in range(20)},
    }

    keys, _used, actual, layers = plan_route_coverage(
        counts,
        0.01,
        top_k=8,
        layers=8,
        expert_layers=[3, 7],
    )

    assert len([key for key in keys if key.startswith("3:")]) == 8
    assert len([key for key in keys if key.startswith("7:")]) == 8
    assert actual > 0.01
    assert set(layers) == {"3", "7"}


def test_coverage_uses_manifest_expert_layer_ids():
    counts = {
        **{f"3:{expert}": 10 - expert for expert in range(8)},
        **{f"7:{expert}": 10 - expert for expert in range(8)},
    }
    keys, _used, _actual, layer_coverages = plan_route_coverage(
        counts,
        0.5,
        top_k=2,
        layers=8,
        expert_layers=[3, 7],
    )
    assert keys
    assert {int(key.split(":", 1)[0]) for key in keys} == {3, 7}
    assert set(layer_coverages) == {"3", "7"}


def _job() -> TrainingJob:
    job = TrainingJob(
        id="t12345678901",
        corpus_files=["c.jsonl"],
        model_path="D:/model",
        token_budget=500_000,
        layers=2,
        expert_layers=[0, 1],
        experts_per_layer=4,
        model_top_k=1,
    )
    job.counts = {"0:1": 5, "1:3": 10}
    job.plan_keys = ["0:1", "1:3"]
    job.plan_sizes_mb = {"0:1": 6.4, "1:3": 6.4}
    job.plan_bytes_mb = 12.8
    job.processed_tokens = 500_000
    job.actual_coverage = job.coverage_target = 1.0
    job.status = "done"
    return job


def test_exports_record_tokens_coverage_capacity_and_model_identity():
    job = _job()
    job.fixed_model_gib = 8.239
    job.dense_without_shared_gib = 7.231
    job.shared_expert_gib = 1.008
    job.model_name = "dsv4"
    job.model_version = "v2"
    job.model_format = "cccp-1"
    job.model_manifest_sha256 = "a" * 64
    job.model_total_bytes = 80_000_000_000
    job.model_total_gib = job.model_total_bytes / 2**30
    scores = export_scores(job)
    counts = export_counts(job)
    profile = export_profile(job, name="爱情角色扮演", description="侧重关系推进。")
    assert len(scores["scores"]) == 8
    assert scores["meta"]["processed_tokens"] == 500_000
    assert counts["meta"]["processed_tokens"] == 500_000
    assert profile["name"] == "爱情角色扮演"
    assert profile["description"] == "侧重关系推进。"
    assert profile["meta"]["model_manifest_sha256"] == "a" * 64
    assert profile["meta"]["model_expert_layers"] == [0, 1]
    assert profile["meta"]["route_token_budget"] == 500_000
    assert profile["meta"]["prefill_block_tokens"] == 4096
    assert profile["meta"]["configuration_budget_gib"] == profile["meta"]["configuration_resident_gib"]


def test_one_heatmap_can_save_multiple_independent_capacity_profiles():
    job = _job()
    job.fixed_model_gib = 8.239
    first = export_profile(job, name="角色扮演 32G", description="较低覆盖率版本")
    job.coverage_target = job.actual_coverage = 1.0
    job.plan_keys = ["0:1", "0:2", "1:3"]
    job.counts["0:2"] = 1
    job.plan_sizes_mb["0:2"] = 6.4
    job.plan_bytes_mb = 19.2
    second = export_profile(job, name="角色扮演 64G", description="较高覆盖率版本")
    repeated = export_profile(job, name="角色扮演 64G", description="说明可以更新")
    assert first["id"] != second["id"]
    assert repeated["id"] == second["id"]
    assert first["meta"]["selected_experts"] == 2
    assert second["meta"]["selected_experts"] == 3


def test_training_job_records_every_saved_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(training_module, "TRAIN_DIR", tmp_path / "training")
    engine = TrainingEngine(type("Registry", (), {})())
    job = _job()
    engine._jobs[job.id] = job
    engine.mark_registered(job.id, "profile-32g", "32G 版", "精简容量")
    job.plan_keys.append("0:2")
    job.plan_bytes_mb = 19.2
    engine.mark_registered(job.id, "profile-64g", "64G 版", "更高覆盖")
    assert [item["id"] for item in job.registered_profiles] == [
        "profile-32g", "profile-64g"
    ]
    assert job.registered_profiles[0]["selected_experts"] == 2
    assert job.registered_profiles[1]["selected_experts"] == 3
    assert job.registered_profile_id == "profile-64g"


def test_submit_rejects_unknown_training_fields(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "x.txt").write_text("hello", encoding="utf-8")
    monkeypatch.setattr(training_module, "CORPUS_DIR", corpus)
    monkeypatch.setattr(training_module, "TRAIN_DIR", tmp_path / "training")
    registry = type("Registry", (), {})()
    engine = TrainingEngine(registry)
    with pytest.raises(ValueError, match="不支持的训练字段"):
        engine.submit({
            "model_path": "D:/model",
            "corpus_files": ["x.txt"],
            "unexpected_option": 1,
        })
    with pytest.raises(ValueError, match="4,096"):
        engine.submit({
            "model_path": "D:/model",
            "corpus_files": ["x.txt"],
            "token_budget": 4095,
            "model_top_k": 1,
        })


def test_running_training_job_can_be_cancelled_from_ui_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(training_module, "TRAIN_DIR", tmp_path / "training")
    engine = TrainingEngine(type("Registry", (), {})())
    job = TrainingJob(
        id="cancel-me",
        corpus_files=["x.txt"],
        model_path="D:/model",
        model_top_k=1,
        status="running",
    )
    engine._jobs[job.id] = job
    engine._cancel_events[job.id] = training_module.threading.Event()
    assert engine.cancel(job.id) is True
    assert engine._cancel_events[job.id].is_set()
    assert job.message == "正在停止 token 路由扫描…"


def test_layer_progress_moves_bar_before_first_block_completes(tmp_path, monkeypatch):
    monkeypatch.setattr(training_module, "TRAIN_DIR", tmp_path / "training")
    engine = TrainingEngine(type("Registry", (), {})())
    job = TrainingJob(
        id="layer-progress",
        corpus_files=["x.txt"],
        model_path="D:/model",
        token_budget=12_288,
        model_top_k=1,
        status="running",
    )
    engine._jobs[job.id] = job
    engine._update_scan_progress(
        job, 0, 12_288, "prefill 1/2 · 块 1/3 · 层 4/43"
    )
    assert 0.08 < job.progress < 0.12
    assert job.processed_tokens == 0


def test_layer_first_glm_progress_counts_layers_and_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(training_module, "TRAIN_DIR", tmp_path / "training")
    engine = TrainingEngine(type("Registry", (), {})())
    job = TrainingJob(
        id="glm-layer-progress",
        corpus_files=["x.txt"],
        model_path="D:/model",
        token_budget=12_288,
        model_top_k=1,
        status="running",
    )
    engine._jobs[job.id] = job
    engine._update_scan_progress(
        job,
        0,
        12_288,
        "分层 prefill 1/1 · 层 2/78 · 块 2/3 · token 12288",
    )
    expected_work = 12_288 * ((2 - 1) + 2 / 3) / 78
    expected = 0.08 + 0.82 * expected_work / 12_288
    assert job.progress == pytest.approx(expected)
    assert job.processed_tokens == 0
    assert "层 2/78 · 块 2/3" in job.message


def test_submit_rejects_second_running_training_job(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "x.txt").write_text("hello", encoding="utf-8")
    monkeypatch.setattr(training_module, "CORPUS_DIR", corpus)
    monkeypatch.setattr(training_module, "TRAIN_DIR", tmp_path / "training")
    engine = TrainingEngine(type("Registry", (), {})())
    engine._jobs["active"] = TrainingJob(
        id="active",
        corpus_files=["x.txt"],
        model_path="D:/model",
        model_top_k=1,
        status="running",
    )
    with pytest.raises(ValueError, match="已有 token 扫描"):
        engine.submit({
            "model_path": "D:/model",
            "corpus_files": ["x.txt"],
            "token_budget": 4096,
            "model_top_k": 1,
        })
