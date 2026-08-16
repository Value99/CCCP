#!/usr/bin/env bash
# 构建 dist/CCCP-Launcher.exe(Windows Git Bash)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}

echo "[1/3] 安装构建依赖…"
"$PY" -m pip install --quiet -r requirements.txt pyinstaller

echo "[2/3] PyInstaller 打包…"
"$PY" -m PyInstaller --clean --noconfirm packaging/winui.spec

echo "[3/3] 完成: dist/CCCP-Launcher.exe"
ls -lh dist/CCCP-Launcher.exe
echo "双击即启动原生窗口（不会自动打开外部浏览器）。"
