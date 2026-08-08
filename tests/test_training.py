"""training 模块:语料解析、目标体积预算、导出 schema 全覆盖。"""
import json

from launcher.training import (
    TrainingJob, export_counts, export_profile, export_scores,
    iter_corpus, plan_target_size,
)


def test_iter_corpus_jsonl(tmp_path):
    f = tmp_path / "c.jsonl"
    f.write_text(
        '{"prompt": "写代码"}\n{"text": "第二行"}\n'
        '{"messages": [{"content": "多轮"}]}\n# 注释\n\n',
        encoding="utf-8",
    )
    out = list(iter_corpus([f]))
    assert out == ["写代码", "第二行", "多轮"]


def test_iter_corpus_txt(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("第一条\n\n# 注释\n第二条\n", encoding="utf-8")
    assert list(iter_corpus([f])) == ["第一条", "第二条"]


def test_plan_target_size_never_exceeds_budget():
    counts = {f"{i}:{j}": (i * j) % 13 + 1 for i in range(5) for j in range(40)}
    sizes = {k: 24.0 for k in counts}
    keys, used_mb = plan_target_size(counts, 10.0, sizes)  # 10 GiB 预算
    assert used_mb <= 10.0 * 1024
    assert keys and all(k in counts for k in keys)
    # 高优先级(score/bytes)优先:命中率最高的专家必须在选中集(若放得下)
    top = max(counts, key=lambda k: counts[k])
    assert top in keys


def test_plan_zero_target_returns_all():
    counts = {"1:1": 3, "1:2": 9}
    keys, used = plan_target_size(counts, 0.0)
    assert set(keys) == set(counts)
    assert keys[0] == "1:2"  # 按命中率降序


def _job(counts, plan_keys, layers=3, epl=8):
    job = TrainingJob(
        id="t12345678901", corpus_files=["c.jsonl"], layers=layers,
        experts_per_layer=epl, target_gb=1.0,
    )
    job.counts = counts
    job.plan_keys = plan_keys
    job.status = "done"
    return job


def test_export_scores_full_coverage():
    counts = {"0:1": 5, "2:3": 10}
    job = _job(counts, plan_keys=["0:1", "2:3"])
    exp = export_scores(job)
    assert exp["schema"] == "tpq-expert-residency-scores-v1"
    assert len(exp["scores"]) == job.layers * job.experts_per_layer  # 全覆盖补零
    assert exp["scores"]["2:3"] > exp["scores"]["0:1"] > 0
    # 未选中的专家必须为 0(发动时不会被 TPQ 驻留)
    assert exp["scores"]["1:0"] == 0.0
    assert exp["meta"]["calibrated"] is False


def test_export_counts_schema():
    job = _job({"0:1": 5, "2:3": 10}, plan_keys=[])
    exp = export_counts(job)
    assert exp["counts"] == {"0": {"1": 5}, "2": {"3": 10}}


def test_export_profile_schema():
    job = _job({"0:1": 5, "2:3": 10}, plan_keys=["2:3"])
    prof = export_profile(job, name="测试产物")
    assert prof["schema"] == "winui-expert-profile-v1"
    assert prof["name"] == "测试产物"
    assert [e["key"] for e in prof["experts"]] == ["2:3"]
    assert prof["meta"]["source"] == "trained"
    # 导出产物可被 registry 直接导入
    from launcher.profiles import load_profile_dict
    p = load_profile_dict(prof, source="trained")
    assert p.expert_count == 1
