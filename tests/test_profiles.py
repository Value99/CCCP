"""profiles 模块：模型配置发现、组合重叠算数、drop 路由与导入。"""
import pytest

from launcher.profiles import (
    ProfileError, combine, load_profile_dict, resolve_drop,
)


def test_model_profiles_loaded(registry):
    ids = {p.id for p in registry.list()}
    expected = {"roleplay-romance", "roleplay-immersive", "roleplay-worldbuilding"}
    assert expected <= ids
    for p in registry.list():
        if p.id in expected:
            assert p.source == "model"
            assert p.calibrated and p.expert_count > 2000
            assert 23.9 <= p.meta["configuration_resident_gib"] <= 24.0
            assert p.meta["model_version"]
            assert p.meta["model_total_bytes"] > 80_000_000_000
            assert p.meta["model_manifest_sha256"]


def test_recipe_deterministic(registry):
    raw = {
        "schema": "winui-expert-profile-v1", "id": "recipe-test", "name": "recipe",
        "tags": ["test"],
        "recipe": {"seed": "stable", "layers": 4, "experts_per_layer": 64,
                   "density": 0.25, "mean_size_mb": 6.4, "size_jitter": 0.1},
    }
    a = load_profile_dict(raw)
    b = load_profile_dict(raw)
    a.materialize(); b.materialize()
    assert [e.key for e in a.experts][:50] == [e.key for e in b.experts][:50]
    assert a.memory_mb == b.memory_mb


def test_combine_overlap_only_once(registry):
    """两个语料方向按专家编号求并集，重复专家与固定权重都只计一次。"""
    romance = registry.require("roleplay-romance")
    immersive = registry.require("roleplay-immersive")
    combo = registry.combine(["roleplay-romance", "roleplay-immersive"])
    per_sum_gb = romance.memory_gb + immersive.memory_gb
    assert combo.memory_gb < per_sum_gb
    assert combo.overlap_mb > 0
    assert abs((per_sum_gb * 1024 - combo.overlap_mb) - combo.memory_mb) < 1024
    assert combo.expert_count == len(combo.union)
    assert combo.fixed_deduplicated_gib == pytest.approx(8.239, abs=0.001)
    assert combo.total_deduplicated_gib > combo.overlap_mb / 1024
    assert combo.model_version == romance.meta["model_version"]
    assert combo.model_total_bytes == romance.meta["model_total_bytes"]


def test_model_specific_combination_deduplicates_fixed_and_expert_memory():
    common_meta = {
            "model_name": "m",
            "model_version": "m-v1",
            "model_format": "cccp-1",
            "model_manifest_sha256": "a" * 64,
            "model_total_bytes": 80 * 2**30,
            "model_total_gib": 80.0,
            "model_layers": 43,
            "model_experts_per_layer": 256,
            "model_top_k": 6,
            "fixed_model_gib": 8.0,
            "dense_without_shared_gib": 7.0,
            "shared_expert_gib": 1.0,
            "configuration_budget_gib": 24.0,
    }
    common = {
        "schema": "winui-expert-profile-v1",
        "name": "模型配置",
    }
    first = load_profile_dict({
        **common, "id": "model-a",
        "experts": [{"key": "0:1", "size_mb": 10},
                    {"key": "0:2", "size_mb": 20}],
        "meta": {**common_meta, "configuration_resident_gib": 8 + 30 / 1024},
    })
    second = load_profile_dict({
        **common, "id": "model-b",
        "experts": [{"key": "0:2", "size_mb": 20},
                    {"key": "0:3", "size_mb": 30}],
        "meta": {**common_meta, "configuration_resident_gib": 8 + 50 / 1024},
    })
    combo = combine([first, second])
    assert combo.expert_count == 3
    assert combo.memory_mb == 60
    assert combo.overlap_mb == 20
    assert combo.configuration_resident_gib == pytest.approx(8 + 60 / 1024, abs=0.001)


def test_strict_route_validates_only_manifest_expert_layers():
    """前置 Dense 层不能被误判为缺少 top-k 专家。"""
    data = {
        "schema": "winui-expert-profile-v1",
        "id": "sparse-expert-layers",
        "name": "稀疏专家层模型",
        "experts": [
            {"key": f"{layer}:{expert}", "size_mb": 1.0, "route_count": 1}
            for layer in (3, 4)
            for expert in (0, 1)
        ],
        "meta": {
            "model_name": "sparse-model",
            "model_version": "v1",
            "model_format": "cccp-1",
            "model_manifest_sha256": "c" * 64,
            "model_total_bytes": 10 * 2**30,
            "model_total_gib": 10.0,
            "model_layers": 5,
            "model_expert_layers": [3, 4],
            "model_experts_per_layer": 256,
            "model_top_k": 2,
            "fixed_model_gib": 1.0,
            "dense_without_shared_gib": 0.8,
            "shared_expert_gib": 0.2,
            "configuration_budget_gib": 1.0 + 4 / 1024,
            "configuration_resident_gib": 1.0 + 4 / 1024,
            "strict_route": True,
        },
    }
    profile = load_profile_dict(data)
    assert profile.expert_count == 4

    missing_real_layer = {
        **data,
        "experts": data["experts"][:2],
        "meta": {
            **data["meta"],
            "configuration_budget_gib": 1.0 + 2 / 1024,
            "configuration_resident_gib": 1.0 + 2 / 1024,
        },
    }
    with pytest.raises(ProfileError, match=r"\[4\]"):
        load_profile_dict(missing_real_layer)

    expert_on_dense_layer = {
        **data,
        "experts": [
            *data["experts"],
            {"key": "0:0", "size_mb": 1.0, "route_count": 1},
        ],
        "meta": {
            **data["meta"],
            "configuration_budget_gib": 1.0 + 5 / 1024,
            "configuration_resident_gib": 1.0 + 5 / 1024,
        },
    }
    with pytest.raises(ProfileError, match="位于非专家层"):
        load_profile_dict(expert_on_dense_layer)


def test_combination_rejects_different_model_versions():
    def profile(pid, digest):
        return load_profile_dict({
            "schema": "winui-expert-profile-v1", "id": pid, "name": pid,
            "experts": [{"key": "0:1", "size_mb": 1}],
            "meta": {
                "model_name": "m", "model_version": pid, "model_format": "cccp-1",
                "model_manifest_sha256": digest,
                "model_total_bytes": 80 * 2**30, "model_total_gib": 80.0,
                "model_layers": 43, "model_experts_per_layer": 256, "model_top_k": 6,
                "fixed_model_gib": 8.0, "dense_without_shared_gib": 7.0,
                "shared_expert_gib": 1.0, "configuration_budget_gib": 24.0,
                "configuration_resident_gib": 8 + 1 / 1024,
            },
        })
    with pytest.raises(ProfileError, match="不同模型版本"):
        combine([profile("version-a", "a" * 64), profile("version-b", "b" * 64)])


def test_drop_resolution(registry):
    combo = registry.combine(["roleplay-romance", "roleplay-immersive"])
    assert set(combo.drop_resolution) == {"roleplay-romance", "roleplay-immersive"}
    for key in combo.drop_resolution.values():
        assert key in combo.union


def test_drop_prefers_hint_tags(registry):
    profile = registry.require("roleplay-romance")
    profile.materialize()
    combo = combine([profile])
    key = resolve_drop(profile, combo.union)
    assert key in combo.union
    assert set(profile.tags) & set(combo.union[key].tags)


def test_schema_validation():
    with pytest.raises(ProfileError):
        load_profile_dict({"schema": "wrong", "id": "x", "experts": [{"key": "1:1", "size_mb": 1}]})
    with pytest.raises(ProfileError):
        load_profile_dict({"schema": "winui-expert-profile-v1", "id": "x;rm -rf",
                           "experts": [{"key": "1:1", "size_mb": 1}]})
    with pytest.raises(ProfileError):
        load_profile_dict({"schema": "winui-expert-profile-v1", "id": "x",
                           "experts": [{"key": "bad", "size_mb": 1}]})
    with pytest.raises(ProfileError):
        load_profile_dict({"schema": "winui-expert-profile-v1", "id": "x",
                           "experts": [{"key": "1:1", "size_mb": 0}]})
    with pytest.raises(ProfileError):
        load_profile_dict({"schema": "winui-expert-profile-v1", "id": "x",
                           "experts": [{"key": "1:1", "size_mb": 1},
                                       {"key": "1:1", "size_mb": 2}]})
    with pytest.raises(ProfileError):
        load_profile_dict({"schema": "winui-expert-profile-v1", "id": "x",
                           "recipe": {"seed": "x", "layers": 1,
                                      "experts_per_layer": 2, "density": 2.0}})


def test_route_statistics_and_meta_round_trip():
    raw = {
        "schema": "winui-expert-profile-v1",
        "id": "route-measured",
        "name": "实测路由",
        "experts": [{"key": "0:7", "size_mb": 6.25,
                     "route_count": 9, "route_score": 0.75}],
        "meta": {"calibrated": True, "strict_route": True,
                 "prompt_sha256": "0" * 64},
    }
    profile = load_profile_dict(raw)
    exported = profile.to_dict(with_experts=True)
    assert profile.calibrated is True
    assert exported["experts"][0]["route_count"] == 9
    assert exported["experts"][0]["route_score"] == 0.75
    assert exported["meta"]["strict_route"] is True


def test_model_romance_profile_is_real_route_calibrated(registry):
    profile = registry.require("roleplay-romance")
    assert profile.calibrated is True
    assert profile.meta["configuration_only"] is True
    assert profile.meta["strict_route"] is True
    assert 0.45 <= profile.meta["route_coverage"] <= 1.0
    assert profile.memory_gb <= 24.0
    by_layer = {}
    for expert in profile.experts:
        by_layer[expert.layer] = by_layer.get(expert.layer, 0) + 1
        assert expert.route_count > 0
    assert len(by_layer) == 43
    assert profile.expert_count > 2000
    assert min(by_layer.values()) >= profile.meta["model_top_k"] == 6
    assert profile.meta["model_version"] == "dsv4-cccp-s-noblack-v2"
    assert profile.meta["model_total_gib"] == pytest.approx(
        profile.meta["model_total_bytes"] / 2**30, abs=0.000001
    )
    assert profile.meta["configuration_resident_gib"] == pytest.approx(
        profile.meta["fixed_model_gib"] + profile.memory_mb / 1024, abs=0.001
    )


def test_import_requires_model_identity(registry):
    import yaml
    data = {
        "schema": "winui-expert-profile-v1", "id": "my-domain", "name": "自定义",
        "experts": [{"key": "3:7", "size_mb": 5.0}],
    }
    with pytest.raises(ProfileError, match="模型配置必须记录"):
        registry.import_text(yaml.safe_dump(data, allow_unicode=True), "my.yaml")


def test_import_model_profile_and_delete_only_import(registry):
    import json
    original = registry.require("roleplay-romance")
    data = original.to_dict(with_experts=True)
    data["id"] = "shared-romance"
    p = registry.import_text(json.dumps(data, ensure_ascii=False), "shared.json")
    assert p.source == "imported"
    assert registry.combine([p.id]).drop_resolution[p.id] in registry.combine([p.id]).union
    registry.delete(p.id)
    assert registry.get(p.id) is None
    assert registry.require("roleplay-romance").source == "model"
