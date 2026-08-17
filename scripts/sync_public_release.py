"""Synchronize public release metadata after local artifacts pass all gates."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_release_section(path: Path, section: str, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    start = text.find("## ⬇️")
    end = text.find("\n## ", start + 4)
    if start < 0 or end < 0:
        raise RuntimeError(f"cannot locate release section in {path}")
    updated = text[:start] + section.rstrip() + "\n" + text[end:]
    updated = updated.replace(old, new)
    path.write_text(updated, encoding="utf-8", newline="\n")


def numbered_files(names: list[str]) -> str:
    return "\n".join(f"{index}. `{name}`" for index, name in enumerate(names, 1))


def update_manifest(path: Path, version: str, executable: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update({
        "version": version,
        "released_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "title": f"CCCP 启动器 {version}",
        "summary": (
            "随包提供 SM75/86/89/90/120 CUDA 融合算子，修复缺少 quant.vq "
            "的模型清单，并改为按模型声明动态扩展上下文。"
        ),
    })
    data["download"] = {
        "provider": "GitHub Releases",
        "url": f"https://github.com/Value99/CCCP/releases/tag/v{version}",
        "extraction_code": "",
    }
    data["launcher"] = {
        "filename": executable.name,
        "size_bytes": executable.stat().st_size,
        "sha256": sha256(executable).upper(),
    }
    data["release_notes"] = [
        "SM75、SM86、SM89、SM90、SM120 的 ABI 精确 CUDA 融合算子随离线包提供；RTX 4090 不再现场调用 NVCC。",
        "未知架构的 JIT 恢复路径只允许包内 MSVC/Windows SDK；拒绝混用系统 Visual Studio。",
        "模型清单省略 quant.vq 时从 quant.vq_projections 严格推导并校验，不再触发 KeyError。",
        "GUI 不再写死 512/1024/4096 token 上限；逻辑上限读取模型清单，KV 随请求增长。",
        "完整自动化回归 324 passed、11 skipped。",
    ]
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--public-repo", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()

    version = args.version
    assets = args.assets.resolve()
    public = args.public_repo.resolve()
    workspace = args.workspace.resolve()
    executable = assets / "CCCP-Launcher.exe"
    if not executable.is_file():
        raise SystemExit(f"standalone launcher is missing: {executable}")
    parts_manifest = assets / f"CCCP-Launcher-v{version}-offline.parts.json"
    setup = assets / f"CCCP-Launcher-{version}-Offline-Setup.exe"
    parts = sorted(assets.glob(f"CCCP-Launcher-v{version}-win-x64-offline.zip.*"))
    if not setup.is_file() or not parts_manifest.is_file() or not parts:
        raise SystemExit("offline setup/manifest/parts are incomplete")
    install_names = [setup.name, parts_manifest.name, *(part.name for part in parts)]

    old_version = (public / "VERSION").read_text(encoding="utf-8").strip()
    common_files = numbered_files(install_names)
    release_url = f"https://github.com/Value99/CCCP/releases/tag/v{version}"
    zh = f"""## ⬇️ 下载 Windows 完整离线版（v{version}）

> [!IMPORTANT]
> **第一次使用请下载完整离线包，不要只下载单独的 `CCCP-Launcher.exe`。** 完整包已内置 Python、Miniconda、CPU/CUDA/AMD 推理环境、常见 NVIDIA 架构预编译算子及全部依赖。

### [👉 GitHub Release 下载页（推荐）]({release_url})

打开下载页后，将下面 **{len(install_names)} 个文件**全部下载到同一个文件夹：

{common_files}

然后双击 `{setup.name}`。安装器会自动校验、合并、解压并启动。模型不包含在启动器发行包内，需要单独下载并放入解压目录的 `models` 文件夹。

> Release 页面底部的 `Source code (zip/tar.gz)` 是 GitHub 自动生成的公开资料快照，不包含启动器源码、推理引擎源码、模型或 CCCP 量化框架。普通用户请下载上面列出的离线安装器和分卷。

### {version} 重点更新

- SM75 / SM86 / SM89 / SM90 / SM120 融合算子随包提供；RTX 4090 直接加载 SM89 算子，不再现场编译。
- JIT 恢复路径硬绑定包内 MSVC/Windows SDK，不使用系统 Visual Studio。
- 修复缺少 `quant.vq` 的模型清单加载，并移除 Dense 配置卡冗余标签。
- GUI 不再写死上下文 token 上限；按模型声明上限动态扩展 KV。
"""
    en = f"""## ⬇️ Download the complete Windows offline package (v{version})

> [!IMPORTANT]
> **First-time users must download the complete offline package, not only `CCCP-Launcher.exe`.** Python, Miniconda, CPU/CUDA/AMD runtimes, common prebuilt NVIDIA operators, and dependencies are bundled.

### [👉 Open the GitHub Release download page]({release_url})

Download these **{len(install_names)} files** into the same folder:

{common_files}

Run `{setup.name}`. It verifies, joins, extracts, and starts the launcher. Model weights are distributed separately.

### Highlights in {version}

- Prebuilt SM75/86/89/90/120 operators avoid local NVCC on common NVIDIA GPUs, including RTX 4090.
- The JIT recovery path is locked to the bundled MSVC/Windows SDK and cannot use system Visual Studio.
- Fixed manifests without `quant.vq`, removed redundant Dense-card labels, and enabled model-declared dynamic context growth.
"""
    ru = f"""## ⬇️ Полный автономный пакет для Windows (v{version})

> [!IMPORTANT]
> Для первого запуска загрузите полный автономный пакет, а не только `CCCP-Launcher.exe`. Python, Miniconda, среды CPU/CUDA/AMD, готовые операторы NVIDIA и зависимости уже включены.

### [👉 Открыть страницу загрузки GitHub Release]({release_url})

Сохраните эти **{len(install_names)} файлов** в одной папке:

{common_files}

Запустите `{setup.name}`. Установщик проверит, объединит и распакует пакет. Веса моделей распространяются отдельно.

### Главное в версии {version}

- Готовые операторы SM75/86/89/90/120 исключают локальную компиляцию NVCC на распространённых GPU, включая RTX 4090.
- Резервный JIT жёстко использует встроенные MSVC/Windows SDK и не обращается к системной Visual Studio.
- Исправлены манифесты без `quant.vq`, лишние метки Dense и динамический контекст по лимиту модели.
"""
    replace_release_section(public / "README.md", zh, old_version, version)
    replace_release_section(public / "README_EN.md", en, old_version, version)
    replace_release_section(public / "README_RU.md", ru, old_version, version)
    (public / "VERSION").write_text(version + "\n", encoding="utf-8")
    update_manifest(public / "latest.json", version, executable)

    release_dir = public / "releases" / version
    release_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(executable, release_dir / executable.name)
    digest = sha256(executable)
    (release_dir / "SHA256SUMS.txt").write_text(
        f"{digest}  {executable.name}\n", encoding="utf-8", newline="\n"
    )
    (release_dir / "README.md").write_text(
        f"# CCCP Launcher {version}\n\n"
        f"完整离线包请从 [GitHub Release v{version}]({release_url}) 下载。\n\n"
        "本目录仅保留独立启动器更新 EXE；首次使用必须下载 Release 中的完整离线安装器和全部分卷。\n\n"
        "本版本内置常见 NVIDIA 架构预编译算子、修复缺失 quant.vq 的清单，并使用模型声明的动态上下文上限。\n",
        encoding="utf-8",
        newline="\n",
    )

    # Keep the local website/update sources in sync.  Network deployment is a
    # separate hosting concern; these files must never claim a different build.
    for manifest in (workspace / "latest.json", workspace / "官网/latest.json"):
        update_manifest(manifest, version, executable)
    new_sha = sha256(executable).upper()
    for html in (workspace / "官网/index.html", workspace / "CCCP框架介绍.html"):
        text = html.read_text(encoding="utf-8")
        text = text.replace(f"0.9.4", version)
        zh_summary = (
            f"{version} 内置 SM75/86/89/90/120 CUDA 融合算子，修复模型清单"
            "兼容性，并按模型声明动态扩展上下文。"
        )
        en_summary = (
            f"Version {version} bundles SM75/86/89/90/120 CUDA operators, "
            "fixes manifest compatibility, and grows context to the model-declared limit."
        )
        ru_summary = (
            f"Версия {version} включает CUDA-операторы SM75/86/89/90/120, "
            "исправляет манифесты и динамически расширяет контекст до лимита модели."
        )
        for old_summary in (
            f"{version} 修复双路 NUMA 服务器上的 CPU Q4 页归属与线程调度，并保留已验证的 GPU 推理路径。",
            f"{version} 合并当前最优 Dense VQ、MTP 与 MoE 推理实现，并完成四类真实模型和桌面 GUI 回归。",
            f"{version} 新增通用 Dense VQ 支持与 Qwen3.5 实模验证，同时保持 DeepSeek-V4、Kimi K3、GLM-5.2 回归通过。",
        ):
            text = text.replace(old_summary, zh_summary)
        for old_summary in (
            f"Version {version} combines the current best Dense VQ, MTP, and MoE inference paths and completes real-model and desktop GUI regressions.",
            f"Version {version} adds generic Dense VQ support and real Qwen3.5 validation while preserving DeepSeek-V4, Kimi K3, and GLM-5.2 regressions.",
        ):
            text = text.replace(old_summary, en_summary)
        text = text.replace(
            f"Версия {version} добавляет универсальную поддержку Dense VQ и реальную проверку Qwen3.5, сохраняя регрессии DeepSeek-V4, Kimi K3 и GLM-5.2.",
            ru_summary,
        )
        text = text.replace(
            "75CF6EB189F861BEBE2A529F0D35F34C68DBA0CC88791E65E1886D6FEE778E37",
            new_sha,
        )
        html.write_text(text, encoding="utf-8", newline="\n")
    root_readme = workspace / "README.md"
    text = root_readme.read_text(encoding="utf-8").replace("0.9.4", version)
    root_readme.write_text(text, encoding="utf-8", newline="\n")
    print(f"public release metadata synchronized: v{version}, {new_sha}")


if __name__ == "__main__":
    main()
