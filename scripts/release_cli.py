"""CCCP Launcher reproducible release CLI (Windows, offline)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "runtime" / "cpu" / "env" / "python.exe"
CURRENT_VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def run(command: list[str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def powershell(script: str, *arguments: str) -> None:
    run([
        "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", str(ROOT / "scripts" / script), *arguments,
    ])


def doctor() -> None:
    required = [
        PYTHON,
        ROOT / "runtime" / "cuda" / "env" / "python.exe",
        ROOT / "runtime" / "amd" / "env" / "python.exe",
        ROOT / "engine" / "CCCP-Engine" / "cccp" / "__main__.py",
        ROOT / "toolchain",
        ROOT / "webui" / "images" / "icon.ico",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("缺少发行依赖:\n" + "\n".join(missing))
    run([str(PYTHON), "-m", "pip", "check"])
    run([str(ROOT / "runtime" / "cuda" / "env" / "python.exe"), "-m", "pip", "check"])
    run([str(ROOT / "runtime" / "amd" / "env" / "python.exe"), "-m", "pip", "check"])
    run([str(PYTHON), "-c", "import PyInstaller,fastapi,webview,modelscope,torch;print('doctor ok',torch.__version__)"])


def main() -> None:
    parser = argparse.ArgumentParser(description="CCCP Launcher 离线构建 CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="检查三套环境、引擎、工具链和构建依赖")
    sub.add_parser("test", help="运行完整回归测试")
    sub.add_parser("build-exe", help="重建根目录 CCCP-Launcher.exe")
    package = sub.add_parser("package", help="生成按版本命名的离线封装目录")
    package.add_argument("--version", default="", help="必须与源码版本一致；留空自动读取")
    package.add_argument("--force", action="store_true", help="覆盖同版本封装目录")
    smoke = sub.add_parser("smoke", help="从封装目录启动 EXE 并检查核心 API")
    smoke.add_argument("--version", default=CURRENT_VERSION)
    verify = sub.add_parser("verify", help="逐文件验证离线目录 SHA-256 清单")
    verify.add_argument("--version", default=CURRENT_VERSION)
    assets = sub.add_parser("assets", help="生成首次解压安装器、ZIP64 与 GitHub 分卷")
    assets.add_argument("--version", default=CURRENT_VERSION)
    assets.add_argument("--force", action="store_true")
    publish = sub.add_parser("publish", help="创建/更新 GitHub Release 并上传完整离线分卷")
    publish.add_argument("--version", default=CURRENT_VERSION)
    publish.add_argument("--repo", default="Value99/CCCP")
    publish.add_argument("--git-repo", default=str(ROOT / "发布" / "CCCP-github"))
    sub.add_parser("all", help="doctor + test + build-exe + package + assets（不上传）")
    full = sub.add_parser("full-release", help="从测试一直执行到 GitHub Release 远端校验")
    full.add_argument("--version", default=CURRENT_VERSION)
    full.add_argument("--repo", default="Value99/CCCP")
    full.add_argument("--git-repo", default=str(ROOT / "发布" / "CCCP-github"))
    args = parser.parse_args()

    os.environ["PYTHONNOUSERSITE"] = "1"
    if args.command == "doctor":
        doctor()
    elif args.command == "test":
        run([str(PYTHON), "-m", "pytest", "-q"])
    elif args.command == "build-exe":
        powershell("build_app.ps1")
    elif args.command == "package":
        options = []
        if args.version:
            options += ["-Version", args.version]
        if args.force:
            options.append("-Force")
        powershell("package_offline_release.ps1", *options)
    elif args.command == "smoke":
        run([
            str(PYTHON), str(ROOT / "scripts" / "smoke_offline_release.py"),
            str(ROOT / "封装" / f"CCCP-Launcher-v{args.version}-win-x64-offline"),
            "--version", args.version,
        ])
    elif args.command == "verify":
        run([
            str(PYTHON), str(ROOT / "scripts" / "verify_release_manifest.py"),
            str(ROOT / "封装" / f"CCCP-Launcher-v{args.version}-win-x64-offline"),
        ])
    elif args.command == "assets":
        options = [
            "-ReleaseDirectory", str(ROOT / "封装" / f"CCCP-Launcher-v{args.version}-win-x64-offline"),
            "-Version", args.version,
        ]
        if args.force:
            options.append("-Force")
        powershell("create_offline_release_assets.ps1", *options)
    elif args.command == "publish":
        powershell(
            "publish_github_release.ps1",
            "-AssetsDirectory", str(ROOT / "release-assets" / f"v{args.version}"),
            "-Version", args.version,
            "-Repository", args.repo,
            "-GitRepository", args.git_repo,
        )
    else:
        version = getattr(args, "version", CURRENT_VERSION)
        doctor()
        run([str(PYTHON), "-m", "pytest", "-q"])
        powershell("build_app.ps1")
        powershell("package_offline_release.ps1", "-Version", version, "-Force")
        run([
            str(PYTHON), str(ROOT / "scripts" / "smoke_offline_release.py"),
            str(ROOT / "封装" / f"CCCP-Launcher-v{version}-win-x64-offline"),
            "--version", version,
        ])
        powershell(
            "create_offline_release_assets.ps1",
            "-ReleaseDirectory", str(ROOT / "封装" / f"CCCP-Launcher-v{version}-win-x64-offline"),
            "-Version", version, "-Force",
        )
        if args.command == "full-release":
            assets_dir = ROOT / "release-assets" / f"v{version}"
            public_repo = Path(args.git_repo).resolve()
            run([
                str(PYTHON), str(ROOT / "scripts" / "sync_public_release.py"),
                "--version", version,
                "--assets", str(assets_dir),
                "--public-repo", str(public_repo),
                "--workspace", str(ROOT),
            ])
            run(["git", "-C", str(public_repo), "add", "--all"])
            staged = subprocess.run(
                ["git", "-C", str(public_repo), "diff", "--cached", "--quiet"],
                cwd=ROOT,
            )
            if staged.returncode == 1:
                run([
                    "git", "-C", str(public_repo), "commit", "-m",
                    f"release: publish CCCP Launcher {version}",
                ])
            elif staged.returncode != 0:
                raise subprocess.CalledProcessError(staged.returncode, staged.args)
            run(["git", "-C", str(public_repo), "push", "origin", "main"])
            powershell(
                "publish_github_release.ps1",
                "-AssetsDirectory", str(assets_dir),
                "-Version", version,
                "-Repository", args.repo,
                "-GitRepository", args.git_repo,
            )


if __name__ == "__main__":
    main()
