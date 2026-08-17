# Build the exact CUDA binaries shipped with the offline launcher.
param()

$ErrorActionPreference = "Stop"
$script = Join-Path $PSScriptRoot "prebuild_gpu_ops.ps1"
foreach ($architecture in @("7.5", "8.6", "8.9", "9.0", "12.0")) {
    Write-Host "[CUDA $architecture] 构建并写入随包算子..."
    & $script -Backend cuda -Architecture $architecture -InstallPackaged
    if ($LASTEXITCODE) { throw "CUDA $architecture 随包算子构建失败" }
}
