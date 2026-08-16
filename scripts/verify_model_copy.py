"""Compare two model trees byte-for-byte with parallel SHA-256 hashing."""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def files_by_relative(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    source_files = files_by_relative(source)
    destination_files = files_by_relative(destination)
    missing = sorted(set(source_files) - set(destination_files))
    extras = sorted(set(destination_files) - set(source_files))
    common = sorted(set(source_files) & set(destination_files))
    size_mismatches = [
        relative
        for relative in common
        if source_files[relative].stat().st_size
        != destination_files[relative].stat().st_size
    ]
    candidates = [relative for relative in common if relative not in size_mismatches]

    started = time.perf_counter()
    futures: dict[Future[str], tuple[str, str]] = {}
    hashes: dict[tuple[str, str], str] = {}
    workers = max(1, min(int(args.workers), 16))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="model-sha256") as pool:
        for relative in candidates:
            futures[pool.submit(sha256, source_files[relative])] = (relative, "source")
            futures[pool.submit(sha256, destination_files[relative])] = (
                relative,
                "destination",
            )
        for completed, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            hashes[key] = future.result()
            if completed % 10 == 0 or completed == len(futures):
                print(
                    f"hashed {completed}/{len(futures)} files; "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )

    hash_mismatches = [
        relative
        for relative in candidates
        if hashes[(relative, "source")] != hashes[(relative, "destination")]
    ]
    source_bytes = sum(path.stat().st_size for path in source_files.values())
    destination_bytes = sum(path.stat().st_size for path in destination_files.values())
    result = {
        "ok": not (missing or extras or size_mismatches or hash_mismatches),
        "source_files": len(source_files),
        "destination_files": len(destination_files),
        "source_bytes": source_bytes,
        "destination_bytes": destination_bytes,
        "sha256_files_checked": len(candidates),
        "missing": missing,
        "extras": extras,
        "size_mismatches": size_mismatches,
        "hash_mismatches": hash_mismatches,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
