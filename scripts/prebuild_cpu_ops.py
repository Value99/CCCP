"""预编译 CCCP CPU 算子并封装为离线可加载的 Python 扩展。"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def choose_runtime_dlls(toolchain: Path) -> dict[str, Path]:
    wanted = {"vcomp140.dll", "msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"}
    found: dict[str, Path] = {}
    for path in toolchain.rglob("*.dll"):
        lower = path.name.lower()
        if lower not in wanted:
            continue
        parts = str(path).lower()
        if not any(token in parts for token in ("x64", "amd64")):
            continue
        current = found.get(lower)
        if current is None or ("redist" in parts and "redist" not in str(current).lower()):
            found[lower] = path
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--toolchain", type=Path)
    args = parser.parse_args()
    engine = args.engine.resolve()
    build_dir = args.build_dir.resolve()
    native_dir = engine / "cccp" / "native"
    native_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(engine))
    vendor = engine / "_vendor"
    if vendor.is_dir():
        sys.path.insert(0, str(vendor))
    os.environ["TORCH_EXTENSIONS_DIR"] = str(build_dir)
    os.environ["CCCP_CPU_FUSED"] = "1"
    os.environ["CCCP_CPU_AUTOBUILD"] = "1"
    if args.toolchain and args.toolchain.is_dir():
        os.environ["CCCP_LAUNCHER_ROOT"] = str(args.toolchain.resolve().parent)

    from cccp import cpuext

    extension = cpuext._build(verbose=True)  # 构建工具专用入口
    if extension is None:
        raise RuntimeError(cpuext.last_error())
    built = Path(extension.__file__).resolve()
    target = native_dir / built.name
    if not target.exists() or not built.samefile(target):
        shutil.copy2(built, target)

    copied_dlls: list[str] = []
    if args.toolchain and args.toolchain.is_dir():
        for name, source in choose_runtime_dlls(args.toolchain).items():
            dll_target = native_dir / name
            # 扩展已加载时 Windows 会锁定其依赖 DLL；现有随包 DLL 无需覆盖。
            if not dll_target.exists():
                shutil.copy2(source, dll_target)
            copied_dlls.append(name)

    check_code = (
        "import json,sys;sys.path[:0]=[r'" + str(engine) + "',r'" + str(vendor) + "'];"
        "from cccp.cpuext import extension_status,route_topk_sigmoid_cpu;"
        "import torch;"
        "r=route_topk_sigmoid_cpu(torch.tensor([[1.,3.,2.]]),torch.zeros(3),"
        "torch.ones(3,dtype=torch.bool),2,True,1.0);"
        "s=extension_status();s['route_indices']=r[1].tolist() if r else None;"
        "print(json.dumps(s,ensure_ascii=False))"
    )
    env = dict(os.environ)
    env["CCCP_CPU_AUTOBUILD"] = "0"
    env.pop("TORCH_EXTENSIONS_DIR", None)
    checked = subprocess.run(
        [sys.executable, "-c", check_code], env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    print(checked.stdout, end="")
    if checked.returncode != 0 or '"available": true' not in checked.stdout.lower():
        raise RuntimeError("预编译算子离线加载自检失败")
    print(json.dumps({
        "artifact": str(target),
        "bytes": target.stat().st_size,
        "runtime_dlls": sorted(copied_dlls),
        "offline_selftest": "passed",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
