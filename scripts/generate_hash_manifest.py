"""Generate an auditable SHA-256 manifest for the offline distribution."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SHA256SUMS.txt"
SKIP_DIR_NAMES = {
    ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache", "__pycache__",
}
SKIP_PREFIXES = {
    "archive", "data/logs", "data/runtime", "data/sessions",
}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    if path == OUTPUT or path.name == OUTPUT.name + ".tmp":
        return False
    if any(part in SKIP_DIR_NAMES for part in path.relative_to(ROOT).parts):
        return False
    if relative.endswith((".pyc", ".pyo")):
        return False
    return not any(relative == prefix or relative.startswith(prefix + "/")
                   for prefix in SKIP_PREFIXES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    previous: dict[str, str] = {}
    previous_mtime_ns = 0
    if OUTPUT.is_file():
        previous_mtime_ns = OUTPUT.stat().st_mtime_ns
        for line in OUTPUT.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#") or " *" not in line:
                continue
            digest, relative = line.split(" *", 1)
            if len(digest) == 64:
                previous[relative] = digest
    files = sorted(
        (path for path in ROOT.rglob("*") if path.is_file() and included(path)),
        key=lambda path: path.relative_to(ROOT).as_posix().lower(),
    )
    total_bytes = sum(path.stat().st_size for path in files)
    lines = [
        "# CCCP-Launcher offline distribution SHA-256",
        f"# files={len(files)} bytes={total_bytes}",
    ]
    hardlink_cache: dict[tuple[int, int, int, int], str] = {}
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        digest = hardlink_cache.get(identity)
        if digest is None and stat.st_mtime_ns <= previous_mtime_ns:
            digest = previous.get(relative)
        if digest is None:
            digest = sha256(path)
        hardlink_cache[identity] = digest
        lines.append(f"{digest} *{relative}")
    temp = OUTPUT.with_name(OUTPUT.name + ".tmp")
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    temp.replace(OUTPUT)
    print(f"{OUTPUT}: {len(files)} files, {total_bytes} bytes")


if __name__ == "__main__":
    main()
