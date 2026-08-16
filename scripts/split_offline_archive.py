"""Split one verified offline ZIP into GitHub Release-sized assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--root-directory", required=True)
    parser.add_argument("--part-mib", type=int, default=1900)
    args = parser.parse_args()
    archive = args.archive.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        raise SystemExit(f"archive not found: {archive}")
    if not 128 <= args.part_mib <= 1950:
        raise SystemExit("--part-mib must stay between 128 and 1950")

    with zipfile.ZipFile(archive, "r") as package:
        bad = package.testzip()
        if bad:
            raise SystemExit(f"ZIP CRC validation failed: {bad}")
        names = [info.filename.replace("\\", "/") for info in package.infolist()]
        prefix = args.root_directory.rstrip("/") + "/"
        if not names or any(name != args.root_directory and not name.startswith(prefix) for name in names):
            raise SystemExit("ZIP contains entries outside the expected root directory")
        entry_count = len(names)

    part_bytes = args.part_mib * 1024 * 1024
    archive_digest = hashlib.sha256()
    parts: list[dict[str, object]] = []
    with archive.open("rb") as source:
        number = 1
        while True:
            part_name = f"{archive.name}.{number:03d}"
            part_path = output / part_name
            part_digest = hashlib.sha256()
            written = 0
            with part_path.open("wb") as target:
                while written < part_bytes:
                    block = source.read(min(8 * 1024 * 1024, part_bytes - written))
                    if not block:
                        break
                    target.write(block)
                    part_digest.update(block)
                    archive_digest.update(block)
                    written += len(block)
            if written == 0:
                part_path.unlink()
                break
            parts.append({
                "name": part_name,
                "size_bytes": written,
                "sha256": part_digest.hexdigest(),
            })
            print(f"part {number}: {written} bytes", flush=True)
            number += 1

    manifest_name = f"CCCP-Launcher-v{args.version}-offline.parts.json"
    payload = {
        "schema": "cccp-offline-parts-v1",
        "version": args.version,
        "archive_name": archive.name,
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": archive_digest.hexdigest(),
        "root_directory": args.root_directory,
        "entry_count": entry_count,
        "parts": parts,
    }
    (output / manifest_name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    checksums = [f"{item['sha256']}  {item['name']}" for item in parts]
    checksums.append(f"{digest_file(output / manifest_name)}  {manifest_name}")
    (output / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
