"""Guards for the native first-run offline installer."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_windows_argument_quote_preserves_trailing_directory_separator() -> None:
    source = (ROOT / "packaging" / "offline_bootstrap.cs").read_text(encoding="utf-8")

    assert "trailingBackslashes" in source
    assert "new string('\\\\', trailingBackslashes)" in source
    assert "AppDomain.CurrentDomain.BaseDirectory" in source


def test_installer_validates_parts_archive_and_root_before_extracting() -> None:
    source = (ROOT / "packaging" / "offline_bootstrap.cs").read_text(encoding="utf-8")

    assert "partHash.Hash" in source
    assert "archiveHash.Hash" in source
    assert "name.Contains(\"../\")" in source
    assert 'Stage = "正在解压完整离线环境"' in source
