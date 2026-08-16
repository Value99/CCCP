"""Reject launcher/developer documentation leakage from an offline release."""
from __future__ import annotations

import argparse
from pathlib import Path


PUBLIC_DOCS = {
    "AMD核显兼容性说明.md",
    "中文使用手册.md",
    "依赖与离线环境说明.md",
}
ENGINE_DEVELOPER_DOC_NAMES = {
    "readme.md",
    "changelog.md",
    "parameters.md",
}
# These are distinctive implementation files/directories from the separate,
# confidential CCCP quantization framework.  None is required by CCCP-Engine
# inference.  Runtime dequantization readers such as ``fp4io.py`` are
# deliberately not forbidden: they decode existing weights but cannot create
# or convert a model.
QUANTIZATION_FRAMEWORK_SENTINELS = {
    "activation_aware.py",
    "cccp_release.py",
    "densepilot.py",
    "dsv4_awq.py",
    "dsv4_projection_sensitivity.py",
    "dsv4_release.py",
    "exploration_sop.py",
    "kimi_k3_quantize.py",
    "quantize.py",
    "release_workflow.py",
    "stage_search.py",
    "validation_gate.py",
}
BROAD_QUANTIZATION_SENTINELS = QUANTIZATION_FRAMEWORK_SENTINELS - {"quantize.py"}
QUANTIZATION_FRAMEWORK_DIRS = {
    "model_configs",
    "release_configs",
    "results",
}
NON_RUNTIME_ENGINE_MODULES = {
    "api_cli_chat.py",
    "benchmark.py",
}
ALLOWED_RELEASE_ROOT_ENTRIES = {
    "CCCP-Launcher.exe",
    "SHA256SUMS.txt",
    "VERSION",
    "data",
    "docs",
    "engine",
    "models",
    "profiles",
    "runtime",
    "toolchain",
    "使用手册.md",
    "封装信息.json",
}
FORBIDDEN_PUBLIC_PHRASES = {
    "CLI 对话只需给模型和要使用的物理显卡",
    "待 CCCP 开发",
    "详细测试报告",
    "开发审计",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_directory", type=Path)
    args = parser.parse_args()
    root = args.release_directory.resolve()
    errors: list[str] = []

    unexpected_root_entries = sorted(
        item.name for item in root.iterdir() if item.name not in ALLOWED_RELEASE_ROOT_ENTRIES
    )
    if unexpected_root_entries:
        errors.append(f"unexpected release root entries: {unexpected_root_entries}")

    docs = root / "docs"
    actual_docs = {item.name for item in docs.iterdir() if item.is_file()}
    if actual_docs != PUBLIC_DOCS:
        errors.append(f"public docs mismatch: {sorted(actual_docs)}")
    root_markdown = {item.name for item in root.glob("*.md")}
    if root_markdown != {"使用手册.md"}:
        errors.append(f"root markdown mismatch: {sorted(root_markdown)}")

    engine = root / "engine" / "CCCP-Engine"
    engine_package = engine / "cccp"
    if not (engine_package / "__init__.py").is_file():
        errors.append("CCCP inference runtime is missing")
    leaked_quantization_files = sorted(
        path.relative_to(root).as_posix()
        for path in engine.rglob("*")
        if path.is_file()
        and (
            path.name.lower() in BROAD_QUANTIZATION_SENTINELS
            or (
                path.name.lower() == "quantize.py"
                and engine in path.parents
            )
        )
    )
    leaked_quantization_dirs = sorted(
        path.relative_to(root).as_posix()
        for path in engine.rglob("*")
        if path.is_dir() and path.name.lower() in QUANTIZATION_FRAMEWORK_DIRS
    )
    if leaked_quantization_files or leaked_quantization_dirs:
        leaked = leaked_quantization_files + leaked_quantization_dirs
        errors.append("confidential quantization framework leaked: " + ", ".join(leaked))
    python_site_packages = [
        root / "runtime" / backend / "env" / "Lib" / "site-packages"
        for backend in ("cpu", "cuda", "amd")
    ] + [root / "runtime" / "miniconda" / "Lib" / "site-packages"]
    leaked_quantization_packages = sorted(
        (site_packages / package_name).relative_to(root).as_posix()
        for site_packages in python_site_packages
        for package_name in ("CCCP", "cccp_quantization", "cccp_quantizer")
        if (site_packages / package_name).exists()
    )
    if leaked_quantization_packages:
        errors.append(
            "confidential CCCP package leaked into Python environment: "
            + ", ".join(leaked_quantization_packages)
        )
    leaked_non_runtime = sorted(
        path.relative_to(root).as_posix()
        for path in engine_package.rglob("*")
        if path.is_file() and path.name.lower() in NON_RUNTIME_ENGINE_MODULES
    )
    if leaked_non_runtime:
        errors.append("developer-only engine modules leaked: " + ", ".join(leaked_non_runtime))
    leaked_engine_docs = [
        path.relative_to(root).as_posix()
        for path in engine.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() in {".md", ".rst"}
            or path.name.lower() in ENGINE_DEVELOPER_DOC_NAMES
        )
    ]
    if leaked_engine_docs:
        errors.append("engine developer docs leaked: " + ", ".join(leaked_engine_docs))

    user_texts = [root / "使用手册.md", *sorted(docs.glob("*.md"))]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in user_texts)
    for phrase in FORBIDDEN_PUBLIC_PHRASES:
        if phrase in combined:
            errors.append(f"forbidden public phrase: {phrase}")

    forbidden_roots = {
        "launcher", "webui", "tests", "scripts", "packaging", "build", "dist", "archive"
    }
    present = sorted(name for name in forbidden_roots if (root / name).exists())
    if present:
        errors.append(f"engineering roots leaked: {present}")
    if any(item.is_file() for item in (root / "models").rglob("*")):
        errors.append("models directory is not empty")
    if any(item.is_file() for item in (root / "profiles" / "user").rglob("*")):
        errors.append("profiles/user directory is not empty")
    if any(item.is_file() for item in (root / "data").rglob("*")):
        errors.append("data directory contains machine-local state")

    if errors:
        raise SystemExit("offline release audit failed:\n- " + "\n- ".join(errors))
    print(
        "offline release content audit ok: "
        "inference runtime only; confidential quantization framework absent; "
        "end-user docs only; models/profiles empty"
    )


if __name__ == "__main__":
    main()
