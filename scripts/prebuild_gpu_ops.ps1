# Prebuild CCCP fused operators without requiring a GPU on the packaging machine.
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("cuda", "amd")]
    [string]$Backend,
    [string]$Architecture = "",
    [switch]$InstallPackaged
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$env:PYTHONPATH = Join-Path $root "engine\CCCP-Engine"
$env:PYTHONUTF8 = "1"
$env:CCCP_REQUIRE_FUSED = "1"

if ($Backend -eq "cuda") {
    $python = Join-Path $root "runtime\cuda\env\python.exe"
    if ($Architecture) {
        $env:CCCP_FORCE_GPU_BUILD = "1"
        $env:CCCP_CUDA_ARCH = $Architecture
    } else {
        Remove-Item Env:CCCP_FORCE_GPU_BUILD -ErrorAction SilentlyContinue
        Remove-Item Env:CCCP_CUDA_ARCH -ErrorAction SilentlyContinue
    }
} else {
    $python = Join-Path $root "runtime\amd\env\python.exe"
    if ($Architecture) {
        $env:CCCP_FORCE_GPU_BUILD = "1"
        $env:CCCP_ROCM_ARCH = $Architecture
    } else {
        Remove-Item Env:CCCP_FORCE_GPU_BUILD -ErrorAction SilentlyContinue
        Remove-Item Env:CCCP_ROCM_ARCH -ErrorAction SilentlyContinue
    }
}
if (-not (Test-Path -LiteralPath $python)) { throw "缺少 $Backend 环境：$python" }
& $python -c "from cccp import fusedext; raise SystemExit(0 if fusedext.prebuild() else 1)"
if ($LASTEXITCODE) { throw "$Backend 融合算子预编译失败" }
if ($InstallPackaged) {
    & $python -c "from cccp import fusedext; print('packaged GPU operator', fusedext.install_prebuilt())"
    if ($LASTEXITCODE) { throw "$Backend 融合算子写入发行树失败" }
}
