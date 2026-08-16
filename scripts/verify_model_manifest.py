"""Verify a CCCP model directory against its own signed artifact inventory."""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
from typing import Any


_METADATA_FILES = {"cccp.json", "verify.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(name: str) -> str:
    normalized = name.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if (
        not normalized
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"不安全的制品路径: {name!r}")
    return relative.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return value


def verify_model_directory(
    model_dir: Path,
    *,
    workers: int = 4,
    expected_manifest_sha256: str = "",
    progress: bool = True,
) -> dict[str, Any]:
    root = model_dir.resolve()
    manifest_path = root / "cccp.json"
    if not manifest_path.is_file():
        raise ValueError(f"缺少 cccp.json: {root}")

    manifest_hash = sha256(manifest_path)
    if expected_manifest_sha256 and manifest_hash.lower() != expected_manifest_sha256.lower():
        raise ValueError(
            "cccp.json SHA-256 不匹配: "
            f"expected={expected_manifest_sha256.lower()} actual={manifest_hash}"
        )

    manifest = _read_json(manifest_path)
    raw_inventory = manifest.get("artifact_sha256")
    if not isinstance(raw_inventory, dict) or not raw_inventory:
        raise ValueError("cccp.json 缺少非空 artifact_sha256")

    inventory: dict[str, str] = {}
    for raw_name, raw_hash in raw_inventory.items():
        name = _safe_relative(str(raw_name))
        digest = str(raw_hash).strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"无效 SHA-256: {raw_name!r}")
        if name in inventory:
            raise ValueError(f"重复制品路径: {name}")
        inventory[name] = digest

    present = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
    }
    expected_names = set(inventory)
    missing = sorted(expected_names - set(present))
    extras = sorted(set(present) - expected_names - _METADATA_FILES)
    candidates = sorted(expected_names & set(present))

    started = time.perf_counter()
    hashes: dict[str, str] = {}
    worker_count = max(1, min(int(workers), 16))
    futures: dict[Future[str], str] = {}
    with ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="manifest-sha256"
    ) as pool:
        for name in candidates:
            futures[pool.submit(sha256, present[name])] = name
        for completed, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            hashes[name] = future.result()
            if progress and (completed % 5 == 0 or completed == len(futures)):
                print(
                    f"hashed {completed}/{len(futures)} artifacts; "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )

    hash_mismatches = sorted(
        name for name in candidates if hashes[name].lower() != inventory[name]
    )

    upstream_verify: dict[str, Any] | None = None
    verify_path = root / "verify.json"
    verify_errors: list[str] = []
    if verify_path.is_file():
        upstream_verify = _read_json(verify_path)
        if upstream_verify.get("errors") not in (None, []):
            verify_errors.append("verify.json errors 非空")
        if upstream_verify.get("full_hash") is not True:
            verify_errors.append("verify.json 未声明 full_hash=true")
        if int(upstream_verify.get("hashes_checked", -1)) != len(inventory):
            verify_errors.append("verify.json hashes_checked 与清单数量不一致")

    elapsed = time.perf_counter() - started
    result = {
        "ok": not (missing or extras or hash_mismatches or verify_errors),
        "model_directory": str(root),
        "manifest_sha256": manifest_hash,
        "artifacts_expected": len(inventory),
        "artifacts_hashed": len(candidates),
        "artifact_bytes": sum(present[name].stat().st_size for name in candidates),
        "missing": missing,
        "extras": extras,
        "hash_mismatches": hash_mismatches,
        "verify_errors": verify_errors,
        "upstream_verify": upstream_verify,
        "elapsed_seconds": round(elapsed, 3),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按 cccp.json 的 artifact_sha256 验收本地 CCCP 模型"
    )
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--manifest-sha256", default="")
    args = parser.parse_args()

    try:
        result = verify_model_directory(
            args.model_directory,
            workers=args.workers,
            expected_manifest_sha256=args.manifest_sha256,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {"ok": False, "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
