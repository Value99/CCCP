# 构建 dist\TPQ-WinUI.exe(PowerShell)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "[1/3] 安装构建依赖…"
python -m pip install --quiet -r requirements.txt pyinstaller

Write-Host "[2/3] PyInstaller 打包…"
python -m PyInstaller --clean --noconfirm packaging/winui.spec

Write-Host "[3/3] 完成: dist\TPQ-WinUI.exe"
Get-Item dist\TPQ-WinUI.exe | Format-List Name, Length
Write-Host "双击即启动(原生窗口;WebView2 缺失时自动降级浏览器)。"
