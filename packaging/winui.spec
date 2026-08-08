# PyInstaller spec — WINUI-EXE 单文件桌面应用
# 用法: pyinstaller packaging/winui.spec(产出 dist/TPQ-WinUI.exe)
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# 打入只读资源(冻结后位于 sys._MEIPASS,见 launcher/resources.py)
datas = [
    ("webui", "webui"),
    ("profiles/builtin", "profiles/builtin"),
    ("docs", "docs"),
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

# pywebview 按需收集(构建机已安装则一并打入;缺失则运行时降级浏览器)
try:
    d, b, h = collect_all("webview")
    datas += d
    binaries += b
    hiddenimports += h
except Exception:  # noqa: BLE001 —— 无 pywebview 的构建环境
    pass

a = Analysis(
    ["app_entry.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch", "pytest", "numpy", "pandas", "matplotlib",
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
    name="TPQ-WinUI",
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
