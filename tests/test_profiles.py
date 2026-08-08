"""profiles 模块:schema 校验、组合重叠算数、drop 路由、导入覆盖/还原。"""
import pytest

from launcher.profiles import (
    ProfileError, combine, load_profile_dict, resolve_drop,
)


def test_builtins_loaded(registry):
    ids = {p.id for p in registry.list()}
    assert {"roleplay", "python-code", "contract"} <= ids
    for p in registry.list():
        if p.id in ("roleplay", "python-code", "contract"):
            assert p.builtin and p.source == "builtin"
            assert p.expert_count > 0 and p.memory_gb > 0


def test_recipe_deterministic(registry):
    a = registry.require("python-code")
    b = registry.require("python-code")
    a.materialize(); b.materialize()
    assert [e.key for e in a.experts][:50] == [e.key for e in b.experts][:50]
    assert a.memory_mb == b.memory_mb


def test_combine_overlap_only_once(registry):
    """合同+代码并集 < 逐项之和;重叠 > 0(100G+200G->~250G 型)。"""
    contract = registry.require("contract")
    code = registry.require("python-code")
    combo = registry.combine(["contract", "python-code"])
    per_sum_gb = contract.memory_gb + code.memory_gb
    assert combo.memory_gb < per_sum_gb
    assert combo.overlap_mb > 0
    assert abs((per_sum_gb * 1024 - combo.overlap_mb) - combo.memory_mb) < 1024
    assert combo.expert_count == len(combo.union)


def test_drop_resolution(registry):
    combo = registry.combine(["contract", "python-code"])
    assert set(combo.drop_resolution) == {"contract", "python-code"}
    for key in combo.drop_resolution.values():
        assert key in combo.union


def test_drop_prefers_hint_tags(registry):
    code = registry.require("python-code")
    code.materialize()
    combo = combine([code])
    key = resolve_drop(code, combo.union)
    assert key in combo.union
    # hint_tags=[code,python] 命中 => 解析专家的 tags 应为 profile tags
    assert set(code.tags) & set(combo.union[key].tags)


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


def test_import_override_and_restore(registry, tmp_path):
    """导入同 id 覆盖内置 -> 删除覆盖 -> 还原内置定义。"""
    original = registry.require("contract").memory_mb
    override = {
        "schema": "winui-expert-profile-v1",
        "id": "contract",
        "name": "合同·覆盖版",
        "experts": [{"key": "1:1", "size_mb": 10.0}],
    }
    import yaml
    p = registry.import_text(yaml.safe_dump(override, allow_unicode=True), "contract.yaml")
    assert p.source == "imported"
    assert registry.require("contract").memory_mb == 10.0
    registry.delete("contract")
    assert registry.require("contract").builtin
    assert registry.require("contract").memory_mb == original


def test_import_new_profile_and_delete(registry):
    data = {
        "schema": "winui-expert-profile-v1",
        "id": "my-domain",
        "name": "自定义",
        "experts": [{"key": "3:7", "size_mb": 5.0}],
        "drop": {"enabled": True, "hint_tags": ["x"]},
    }
    import yaml
    p = registry.import_text(yaml.safe_dump(data, allow_unicode=True), "my.yaml")
    assert p.id == "my-domain"
    combo = registry.combine(["my-domain"])
    assert combo.drop_resolution["my-domain"] == "3:7"
    registry.delete("my-domain")
    assert registry.get("my-domain") is None


def test_delete_builtin_forbidden(registry):
    with pytest.raises(ProfileError):
        registry.delete("roleplay")
