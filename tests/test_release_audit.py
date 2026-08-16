from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDITOR = REPO_ROOT / "scripts" / "audit_offline_release.py"
SMOKE = REPO_ROOT / "scripts" / "smoke_offline_release.py"
MANIFEST_GENERATOR = REPO_ROOT / "scripts" / "generate_release_manifest.py"
MANIFEST_VERIFIER = REPO_ROOT / "scripts" / "verify_release_manifest.py"
PUBLIC_DOCS = (
    "AMD核显兼容性说明.md",
    "中文使用手册.md",
    "依赖与离线环境说明.md",
)


def _minimal_release(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    docs = root / "docs"
    docs.mkdir(parents=True)
    for name in PUBLIC_DOCS:
        (docs / name).write_text("用户文档\n", encoding="utf-8")
    (root / "使用手册.md").write_text("用户手册\n", encoding="utf-8")
    engine = root / "engine" / "CCCP-Engine" / "cccp"
    engine.mkdir(parents=True)
    (engine / "__init__.py").write_text("__version__ = '0.9.0'\n", encoding="utf-8")
    (root / "models").mkdir()
    (root / "profiles" / "user").mkdir(parents=True)
    (root / "data").mkdir()
    return root


def _audit(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(AUDITOR), str(root)],
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )


def test_release_audit_allows_inference_dequant_reader(tmp_path: Path) -> None:
    root = _minimal_release(tmp_path)
    fp4io = root / "engine" / "CCCP-Engine" / "cccp" / "fp4io.py"
    fp4io.write_text("def dequant_fp4(value): return value\n", encoding="utf-8")

    result = _audit(root)

    assert result.returncode == 0, result.stderr
    assert "inference runtime only" in result.stdout


def test_release_audit_rejects_confidential_quantization_source(tmp_path: Path) -> None:
    root = _minimal_release(tmp_path)
    leaked = root / "engine" / "CCCP-Engine" / "cccp" / "quantize.py"
    leaked.write_text("# confidential quantizer\n", encoding="utf-8")

    result = _audit(root)

    assert result.returncode != 0
    assert "confidential quantization framework leaked" in result.stderr
    assert "quantize.py" in result.stderr


def test_release_audit_rejects_developer_only_engine_modules(tmp_path: Path) -> None:
    root = _minimal_release(tmp_path)
    leaked = root / "engine" / "CCCP-Engine" / "cccp" / "benchmark.py"
    leaked.write_text("# developer benchmark\n", encoding="utf-8")

    result = _audit(root)

    assert result.returncode != 0
    assert "developer-only engine modules leaked" in result.stderr


def test_release_audit_rejects_machine_local_settings(tmp_path: Path) -> None:
    root = _minimal_release(tmp_path)
    (root / "data" / "settings.json").write_text(
        '{"model_roots":["C:/build-machine/models"]}', encoding="utf-8"
    )

    result = _audit(root)

    assert result.returncode != 0
    assert "data directory contains machine-local state" in result.stderr


def test_packaged_smoke_cannot_dirty_release_with_python_bytecode() -> None:
    source = SMOKE.read_text(encoding="utf-8")

    assert 'child_env["PYTHONDONTWRITEBYTECODE"] = "1"' in source
    assert 'child_env["PYTHONNOUSERSITE"] = "1"' in source
    assert "env=child_env" in source


def _generate_and_verify_manifest(root: Path) -> subprocess.CompletedProcess[str]:
    subprocess.run(
        [sys.executable, str(MANIFEST_GENERATOR), str(root), "--version", "0.9.2"],
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    return subprocess.run(
        [sys.executable, str(MANIFEST_VERIFIER), str(root)],
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )


def test_release_manifest_verifier_accepts_exact_directory(tmp_path: Path) -> None:
    root = tmp_path / "release"
    root.mkdir()
    (root / "CCCP-Launcher.exe").write_bytes(b"launcher")
    (root / "payload.bin").write_bytes(b"payload")
    (root / "封装信息.json").write_text("{}", encoding="utf-8")

    result = _generate_and_verify_manifest(root)

    assert result.returncode == 0, result.stderr
    assert "no missing/extra/mismatched files" in result.stdout


def test_release_manifest_verifier_rejects_changed_or_extra_file(tmp_path: Path) -> None:
    root = tmp_path / "release"
    root.mkdir()
    (root / "CCCP-Launcher.exe").write_bytes(b"launcher")
    (root / "封装信息.json").write_text("{}", encoding="utf-8")
    subprocess.run(
        [sys.executable, str(MANIFEST_GENERATOR), str(root), "--version", "0.9.2"],
        check=True,
    )
    (root / "CCCP-Launcher.exe").write_bytes(b"tampered")
    (root / "unexpected.bin").write_bytes(b"extra")

    result = subprocess.run(
        [sys.executable, str(MANIFEST_VERIFIER), str(root)],
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "release file set mismatch" in result.stderr
    assert "unexpected.bin" in result.stderr
