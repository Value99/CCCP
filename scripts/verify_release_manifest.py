"""Verify every file in a versioned offline release against SHA256SUMS.txt."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re


HEADER_RE = re.compile(r"^# files=(\d+) bytes=(\d+)$")
ENTRY_RE = re.compile(r"^([0-9a-fA-F]{64}) \*(.+)$")


def _long_path(path: str | os.PathLike[str]) -> str:
    value = os.path.abspath(os.fspath(path))
    if os.name != "nt" or value.startswith("\\\\?\\"):
        return value
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value[2:]
    return "\\\\?\\" + value


def _normal_path(value: str) -> str:
    if not value.startswith("\\\\?\\"):
        return value
    value = value[4:]
    return "\\\\" + value[4:] if value.startswith("UNC\\") else value


def sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(_long_path(path), "rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def iter_files(root: Path, manifest: Path):
    manifest_text = os.path.normcase(os.path.abspath(manifest))
    for parent, directories, names in os.walk(_long_path(root)):
        directories.sort(key=str.lower)
        names.sort(key=str.lower)
        for name in names:
            absolute = _normal_path(os.path.join(parent, name))
            if os.path.normcase(os.path.abspath(absolute)) == manifest_text:
                continue
            relative = os.path.relpath(absolute, root).replace(os.sep, "/")
            yield absolute, relative


def parse_manifest(path: Path) -> tuple[int, int, dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2 or not lines[0].startswith("# CCCP CCCP Launcher v"):
        raise ValueError("invalid release manifest title")
    header = HEADER_RE.fullmatch(lines[1])
    if not header:
        raise ValueError("invalid release manifest summary")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(lines[2:], 3):
        match = ENTRY_RE.fullmatch(line)
        if not match:
            raise ValueError(f"invalid manifest entry at line {line_number}")
        relative = match.group(2)
        pure = PurePosixPath(relative)
        if pure.is_absolute() or not pure.parts or ".." in pure.parts or "\\" in relative:
            raise ValueError(f"unsafe manifest path at line {line_number}: {relative!r}")
        key = relative.casefold()
        if key in entries:
            raise ValueError(f"duplicate manifest path at line {line_number}: {relative!r}")
        entries[key] = match.group(1).lower()
    return int(header.group(1)), int(header.group(2)), entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    root = args.release_dir.resolve()
    manifest = root / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise SystemExit(f"release manifest is missing: {manifest}")
    try:
        declared_files, declared_bytes, expected = parse_manifest(manifest)
    except ValueError as exc:
        raise SystemExit(f"release manifest is invalid: {exc}") from exc

    actual_pairs = list(iter_files(root, manifest))
    actual = {relative.casefold(): (path, relative) for path, relative in actual_pairs}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing[:20]))
        if extra:
            details.append("extra=" + ", ".join(extra[:20]))
        raise SystemExit("release file set mismatch: " + "; ".join(details))

    total_bytes = sum(os.stat(_long_path(path)).st_size for path, _ in actual_pairs)
    if declared_files != len(actual_pairs) or declared_bytes != total_bytes:
        raise SystemExit(
            "release manifest summary mismatch: "
            f"declared files={declared_files} bytes={declared_bytes}; "
            f"actual files={len(actual_pairs)} bytes={total_bytes}"
        )

    ordered = sorted(actual_pairs, key=lambda item: item[1].casefold())
    workers = max(1, min(args.workers, 16))
    mismatches: list[str] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="release-verify") as pool:
        for index, (digest, (_, relative)) in enumerate(
            zip(pool.map(sha256, [path for path, _ in ordered]), ordered), 1
        ):
            if digest.lower() != expected[relative.casefold()]:
                mismatches.append(relative)
            if index % 10_000 == 0:
                print(f"verified {index}/{len(ordered)} files", flush=True)
    if mismatches:
        raise SystemExit("release SHA-256 mismatch: " + ", ".join(mismatches[:20]))

    info_path = root / "封装信息.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        recorded = str(info.get("launcher_sha256") or "").lower()
        actual_launcher = expected.get("cccp-launcher.exe")
        if not recorded or recorded != actual_launcher:
            raise SystemExit("封装信息.json launcher_sha256 does not match the manifest")

    print(
        f"release manifest verified: {len(ordered)} files, {total_bytes} bytes, "
        "no missing/extra/mismatched files"
    )


if __name__ == "__main__":
    main()
