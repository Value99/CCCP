"""Static release-publishing safety guards for Windows PowerShell 5.1."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_release_notes_are_serialized_as_a_plain_json_string() -> None:
    script = (ROOT / "scripts" / "publish_github_release.ps1").read_text(encoding="utf-8")

    assert "[System.IO.File]::ReadAllText" in script
    assert "body = $notesText" in script
    assert "body = Get-Content" not in script
    assert '-Method Patch' in script


def test_publisher_rejects_models_and_verifies_remote_asset_sizes() -> None:
    script = (ROOT / "scripts" / "publish_github_release.ps1").read_text(encoding="utf-8")

    assert "Refusing to upload model weight files" in script
    assert "Remote asset verification failed" in script
    assert "Final remote audit failed" in script
    assert "Verified existing" in script
    assert "$attempt -le 3 -and -not $final" in script
