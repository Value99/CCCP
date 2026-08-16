"""Generate a compact SHA-256 manifest for one versioned release directory."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path


def _long_path(path: str | os.PathLike[str]) -> str:
    """Return a Windows extended-length path for deeply nested dependencies."""
    value = os.path.abspath(os.fspath(path))
    if os.name != "nt" or value.startswith("\\\\?\\"):
        return value
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value[2:]
    return "\\\\?\\" + value


def sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(_long_path(path), "rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def iter_files(root: Path, output: Path):
    root_text = str(root)
    output_text = os.path.normcase(os.path.abspath(output))
    for parent, directories, names in os.walk(_long_path(root_text)):
        directories.sort(key=str.lower)
        names.sort(key=str.lower)
        for name in names:
            long_name = os.path.join(parent, name)
            normal_name = long_name[4:] if long_name.startswith("\\\\?\\") else long_name
            if normal_name.startswith("UNC\\"):
                normal_name = "\\\\" + normal_name[4:]
            if os.path.normcase(os.path.abspath(normal_name)) == output_text:
                continue
            relative = os.path.relpath(normal_name, root_text).replace(os.sep, "/")
            yield normal_name, relative


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    root = args.release_dir.resolve()
    output = root / "SHA256SUMS.txt"
    info = root / "封装信息.json"
    payload = json.loads(info.read_text(encoding="utf-8")) if info.is_file() else {}
    payload.update({
        "version": args.version,
        "platform": "win-x64",
        "offline": True,
        "launcher_sha256": sha256(root / "CCCP-Launcher.exe"),
    })
    info.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    files = list(iter_files(root, output))
    total_bytes = sum(os.stat(_long_path(path)).st_size for path, _ in files)
    lines = [
        f"# CCCP CCCP Launcher v{args.version} win-x64 offline",
        f"# files={len(files)} bytes={total_bytes}",
    ]
    workers = max(1, min(args.workers, 16))
    paths = [path for path, _ in files]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="release-sha256") as pool:
        for index, (digest, (_, relative)) in enumerate(zip(pool.map(sha256, paths), files), 1):
            lines.append(f"{digest} *{relative}")
            if index % 10_000 == 0:
                print(f"hashed {index}/{len(files)} files", flush=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{output}: {len(files)} files")


if __name__ == "__main__":
    main()
