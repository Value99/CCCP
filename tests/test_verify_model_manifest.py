from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_model_manifest.py"
SPEC = importlib.util.spec_from_file_location("verify_model_manifest", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _model(tmp_path: Path) -> Path:
    root = tmp_path / "model"
    root.mkdir()
    artifacts = {"dense.bin": b"dense", "experts/L00.bin": b"expert"}
    for name, data in artifacts.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = {"artifact_sha256": {name: _digest(data) for name, data in artifacts.items()}}
    (root / "cccp.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "verify.json").write_text(
        json.dumps({"errors": [], "full_hash": True, "hashes_checked": 2}),
        encoding="utf-8",
    )
    return root


def test_manifest_verifier_accepts_exact_model(tmp_path):
    root = _model(tmp_path)
    result = MODULE.verify_model_directory(root, workers=1, progress=False)
    assert result["ok"]
    assert result["artifacts_expected"] == result["artifacts_hashed"] == 2
    assert result["missing"] == result["extras"] == result["hash_mismatches"] == []


def test_manifest_verifier_reports_missing_extra_and_corruption(tmp_path):
    root = _model(tmp_path)
    (root / "dense.bin").write_bytes(b"corrupt")
    (root / "experts" / "L00.bin").unlink()
    (root / "unexpected.txt").write_text("extra", encoding="utf-8")
    result = MODULE.verify_model_directory(root, workers=1, progress=False)
    assert not result["ok"]
    assert result["missing"] == ["experts/L00.bin"]
    assert result["extras"] == ["unexpected.txt"]
    assert result["hash_mismatches"] == ["dense.bin"]


def test_manifest_verifier_rejects_path_escape(tmp_path):
    root = _model(tmp_path)
    manifest = json.loads((root / "cccp.json").read_text(encoding="utf-8"))
    manifest["artifact_sha256"]["../outside.bin"] = "0" * 64
    (root / "cccp.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="不安全"):
        MODULE.verify_model_directory(root, workers=1, progress=False)


def test_manifest_verifier_checks_upstream_full_hash_status(tmp_path):
    root = _model(tmp_path)
    (root / "verify.json").write_text(
        json.dumps({"errors": ["failed"], "full_hash": False, "hashes_checked": 1}),
        encoding="utf-8",
    )
    result = MODULE.verify_model_directory(root, workers=1, progress=False)
    assert not result["ok"]
    assert len(result["verify_errors"]) == 3
