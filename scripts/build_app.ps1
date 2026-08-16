# Build dist\CCCP-Launcher.exe with the bundled Miniconda environment.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$runtime = Join-Path (Get-Location) "runtime\cpu\env\python.exe"
if (-not (Test-Path -LiteralPath $runtime)) {
    throw "Bundled Miniconda environment is missing: runtime\cpu\env\python.exe"
}
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONUTF8 = "1"
$env:PATH = "$(Split-Path $runtime);$(Split-Path $runtime)\Scripts;$(Split-Path $runtime)\Library\bin;$env:PATH"

Write-Host "[1/3] Validate bundled Miniconda build environment..."
& $runtime -c "import PyInstaller, fastapi, webview; print('PyInstaller', PyInstaller.__version__)"

Write-Host "[2/3] Build with PyInstaller..."
& $runtime -m PyInstaller --clean --noconfirm packaging/cccp_launcher.spec

Write-Host "[3/3] Complete: dist\CCCP-Launcher.exe"
Copy-Item -LiteralPath "dist\CCCP-Launcher.exe" -Destination "CCCP-Launcher.exe" -Force
Get-Item dist\CCCP-Launcher.exe | Format-List Name, Length
Write-Host "Root CCCP-Launcher.exe synchronized and ready to launch."
