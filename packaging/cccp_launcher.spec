# PyInstaller spec — CCCP Launcher 单文件桌面应用
# 用法: pyinstaller packaging/cccp_launcher.spec（产出 dist/CCCP-Launcher.exe）
from PyInstaller.utils.hooks import collect_all
from pathlib import Path

block_cipher = None
ROOT = Path(SPECPATH).resolve().parent

# 打入只读资源(冻结后位于 sys._MEIPASS,见 launcher/resources.py)
datas = [
    (str(ROOT / "webui"), "webui"),
    (str(ROOT / "docs" / "中文使用手册.md"), "docs"),
    (str(ROOT / "docs" / "依赖与离线环境说明.md"), "docs"),
    (str(ROOT / "docs" / "AMD核显兼容性说明.md"), "docs"),
]
binaries = []
hiddenimports = [
    "launcher.shell",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
]

# pywebview 按需收集；正式离线构建会在构建前强制校验该依赖。
try:
    d, b, h = collect_all("webview")
    datas += d
    binaries += b
    hiddenimports += h
except Exception:  # noqa: BLE001 —— 无 pywebview 的构建环境
    pass

a = Analysis(
    [str(ROOT / "app_entry.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch", "pytest", "numpy", "pandas", "matplotlib",
        "modelscope", "transformers", "tensorflow", "jax", "flax",
        "scipy", "sklearn", "huggingface_hub", "tokenizers", "tiktoken",
        "PyQt5", "PyQt6", "PySide2", "PySide6",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="CCCP-Launcher",
    icon=str(ROOT / "webui" / "images" / "icon.ico"),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # 桌面应用:不弹控制台
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
