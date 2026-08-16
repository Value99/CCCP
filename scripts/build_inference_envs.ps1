# Build isolated, portable inference environments for CPU, NVIDIA CUDA and AMD ROCm.
param(
    [ValidateSet("all", "cuda", "amd")]
    [string]$Backend = "all",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$conda = Join-Path $root "runtime\miniconda\Scripts\conda.exe"
$cpu = Join-Path $root "runtime\cpu\env"
if (-not (Test-Path -LiteralPath $conda)) { throw "缺少随包 Miniconda：$conda" }
if (-not (Test-Path -LiteralPath (Join-Path $cpu "python.exe"))) { throw "缺少 CPU 基础环境：$cpu" }

function Remove-Environment([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return }
    if (-not $Force) { throw "环境已存在：$Path（确认后使用 -Force 重建）" }
    $runtimeRoot = (Resolve-Path -LiteralPath (Join-Path $root "runtime")).Path
    $parent = (Resolve-Path -LiteralPath (Split-Path -Parent $Path)).Path
    if (-not $parent.StartsWith($runtimeRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "拒绝删除 runtime 之外的环境：$Path"
    }
    Remove-Item -LiteralPath $Path -Recurse -Force
}

function Assert-Command([int]$Code, [string]$Message) {
    if ($Code -ne 0) { throw $Message }
}

if ($Backend -in @("all", "cuda")) {
    $cuda = Join-Path $root "runtime\cuda\env"
    Remove-Environment $cuda
    if (-not (Test-Path -LiteralPath $cuda)) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $cuda) | Out-Null
        Write-Host "[CUDA 1/3] 克隆公共依赖环境..."
        # Clone only from the local package cache. Explicitly overriding the
        # channel avoids requiring end users to accept Anaconda channel terms.
        & $conda create --yes --offline --override-channels --channel conda-forge `
            --prefix $cuda --clone $cpu
        Assert-Command $LASTEXITCODE "CUDA 环境克隆失败"
        Write-Host "[CUDA 2/3] 安装官方 PyTorch CUDA 13.0..."
        & (Join-Path $cuda "python.exe") -m pip install --no-cache-dir --force-reinstall `
            "torch==2.13.0" --index-url "https://download.pytorch.org/whl/cu130"
        Assert-Command $LASTEXITCODE "CUDA PyTorch 安装失败"
        Write-Host "[CUDA 3/3] 安装离线首次编译所需 CUDA 13.0.2 组件..."
        # 与 NVIDIA 官方 cuda-toolkit 13.0.2 元包中的 Windows 版本锁一致。
        # 这些组件同时提供 NVCC、CCCL/CUB、cuBLAS、cuSOLVER、cuSPARSE
        # 与链接器，发行包离线运行时无需再下载 CUDA Toolkit。
        & (Join-Path $cuda "python.exe") -m pip install --no-cache-dir --force-reinstall `
            "nvidia-cuda-runtime==13.0.96" `
            "nvidia-cuda-crt==13.0.88" `
            "nvidia-nvvm==13.0.88" `
            "nvidia-cuda-nvcc==13.0.88" `
            "nvidia-cuda-cccl==13.0.85" `
            "nvidia-cublas==13.1.0.3" `
            "nvidia-cusolver==12.0.4.66" `
            "nvidia-cusparse==12.6.3.3" `
            "nvidia-nvjitlink==13.0.88"
        Assert-Command $LASTEXITCODE "CUDA 13.0.2 编译组件安装失败"
    }
    & (Join-Path $cuda "python.exe") -c "import torch; print(torch.__version__,torch.version.cuda,torch.version.hip,torch.cuda.is_available())"
    Assert-Command $LASTEXITCODE "CUDA 环境导入失败"
    Write-Host "[CUDA 验证] 生成并验证离线 cuBLAS 导入库..."
    $env:CCCP_LAUNCHER_ROOT = $root
    $env:CCCP_FUSED = "0"
    $env:PYTHONPATH = Join-Path $root "engine\CCCP-Engine"
    & (Join-Path $cuda "python.exe") -c "from cccp import fusedext; from cccp.cpuext import _configure_bundled_windows_toolchain; fusedext._configure_packaged_gpu_toolchain(); assert _configure_bundled_windows_toolchain() or __import__('shutil').which('lib.exe'); print('cuBLAS import library', fusedext._ensure_windows_cublas_import_library())"
    Assert-Command $LASTEXITCODE "CUDA 离线 cuBLAS/MSVC 工具链验证失败"
}

if ($Backend -in @("all", "amd")) {
    $amd = Join-Path $root "runtime\amd\env"
    Remove-Environment $amd
    if (-not (Test-Path -LiteralPath $amd)) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $amd) | Out-Null
        Write-Host "[AMD 1/5] 建立 Python 3.12 独立环境..."
        & $conda create --yes --override-channels --channel conda-forge `
            --prefix $amd "python=3.12" pip
        Assert-Command $LASTEXITCODE "AMD Python 环境创建失败"
        Write-Host "[AMD 2/5] 安装 CCCP/API 公共依赖..."
        & (Join-Path $amd "python.exe") -m pip install --no-cache-dir `
            "tokenizers>=0.22,<0.24" "tiktoken>=0.11,<1" "psutil>=7,<8" `
            "ninja>=1.13,<2" "numpy>=2,<3" "fastapi>=0.116,<1" `
            "uvicorn>=0.35,<1" "pydantic>=2,<3" "starlette>=0.47,<1" `
            "httpx>=0.27" "PyYAML>=6" "Pillow>=10,<13" "av>=14,<16"
        Assert-Command $LASTEXITCODE "AMD 公共依赖安装失败"
        Write-Host "[AMD 3/5] 安装 AMD 官方 ROCm 7.2.1 Core SDK..."
        & (Join-Path $amd "python.exe") -m pip install --no-cache-dir `
            "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl" `
            "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl" `
            "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl" `
            "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz"
        Assert-Command $LASTEXITCODE "AMD ROCm SDK 安装失败"
        Write-Host "[AMD 4/5] 安装 AMD 官方 PyTorch ROCm 7.2.1..."
        & (Join-Path $amd "python.exe") -m pip install --no-cache-dir `
            "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
        Assert-Command $LASTEXITCODE "AMD PyTorch 安装失败"
        Write-Host "[AMD 5/5] 提取融合算子所需的紧凑 ROCm 开发头文件..."
        $amdHeaders = Join-Path $root "runtime\amd\devinclude"
        & (Join-Path $amd "python.exe") (Join-Path $PSScriptRoot "extract_rocm_devheaders.py") `
            --prefix $amd --output $amdHeaders
        Assert-Command $LASTEXITCODE "AMD ROCm 开发头文件提取失败"
    }
    & (Join-Path $amd "python.exe") -c "import torch; print(torch.__version__,torch.version.cuda,torch.version.hip,torch.cuda.is_available())"
    Assert-Command $LASTEXITCODE "AMD 环境导入失败"
}

Write-Host "多环境构建完成。CPU 保持 runtime/cpu/env；新环境位于 runtime/cuda/env 与 runtime/amd/env。"
