# Create a clean, versioned end-user directory without launcher source/tests/build cache.
param(
    [string]$Version = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$versionText = Get-Content -LiteralPath (Join-Path $root "launcher\__init__.py") -Raw -Encoding UTF8
$match = [regex]::Match($versionText, '__version__\s*=\s*"([^"]+)"')
if (-not $match.Success) { throw "无法读取 launcher/__init__.py 版本号" }
$sourceVersion = $match.Groups[1].Value
$rootVersion = [System.IO.File]::ReadAllText((Join-Path $root "VERSION"), [System.Text.Encoding]::UTF8).Trim()
if ($rootVersion -ne $sourceVersion) {
    throw "VERSION ($rootVersion) 与 launcher/__init__.py ($sourceVersion) 不一致"
}
if (-not $Version) { $Version = $rootVersion }
if ($Version -ne $sourceVersion) {
    throw "请求版本 $Version 与源码版本 $sourceVersion 不一致，请先统一版本号"
}

$python = Join-Path $root "runtime\cpu\env\python.exe"
$cudaPython = Join-Path $root "runtime\cuda\env\python.exe"
$amdPython = Join-Path $root "runtime\amd\env\python.exe"
$exe = Join-Path $root "CCCP-Launcher.exe"
foreach ($required in @($python, $cudaPython, $amdPython, $exe, (Join-Path $root "engine\CCCP-Engine\cccp\__main__.py"))) {
    if (-not (Test-Path -LiteralPath $required)) { throw "缺少封装必需文件：$required" }
}

$packageParent = Join-Path $root "封装"
New-Item -ItemType Directory -Force -Path $packageParent | Out-Null
$packageParent = (Resolve-Path -LiteralPath $packageParent).Path
$releaseName = "CCCP-Launcher-v$Version-win-x64-offline"
$release = Join-Path $packageParent $releaseName
if (Test-Path -LiteralPath $release) {
    if (-not $Force) { throw "封装目录已存在：$release（确认后加 -Force 重建）" }
    if ((Split-Path -Parent $release) -ne $packageParent) { throw "拒绝清理封装根目录之外的路径" }
    # ROCm/Tensile contains paths longer than the legacy Windows MAX_PATH limit.
    # Use the Win32 extended-length prefix so repeated -Force packaging remains reliable.
    $longRelease = if ($release.StartsWith("\\")) {
        "\\?\UNC\" + $release.Substring(2)
    } else {
        "\\?\" + $release
    }
    [System.IO.Directory]::Delete($longRelease, $true)
}
New-Item -ItemType Directory -Force -Path $release | Out-Null

function Copy-Tree([string]$Source, [string]$Destination) {
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    $arguments = @(
        $Source, $Destination, "/E", "/COPY:DAT", "/DCOPY:DAT", "/R:1", "/W:1",
        "/NFL", "/NDL", "/NJH", "/NJS", "/NP",
        "/XD", "__pycache__", ".pytest_cache", ".git", "build", "dist", "tests",
        "/XF", "*.pyc", "*.pyo", "*.log"
    )
    & robocopy @arguments | Out-Null
    if ($LASTEXITCODE -gt 7) { throw "复制失败（robocopy=$LASTEXITCODE）：$Source" }
}

Write-Host "[1/6] 复制最终 EXE 入口..."
Copy-Item -LiteralPath $exe -Destination (Join-Path $release "CCCP-Launcher.exe")
Copy-Item -LiteralPath (Join-Path $root "VERSION") -Destination (Join-Path $release "VERSION")

Write-Host "[2/6] 复制 CPU/CUDA/AMD 独立环境、Miniconda、CCCP 引擎与编译工具链..."
Copy-Tree (Join-Path $root "runtime\cpu\env") (Join-Path $release "runtime\cpu\env")
Copy-Tree (Join-Path $root "runtime\cuda\env") (Join-Path $release "runtime\cuda\env")
Copy-Tree (Join-Path $root "runtime\amd\env") (Join-Path $release "runtime\amd\env")
Copy-Tree (Join-Path $root "runtime\amd\devinclude") (Join-Path $release "runtime\amd\devinclude")
Copy-Tree (Join-Path $root "runtime\miniconda") (Join-Path $release "runtime\miniconda")
$engineSource = Join-Path $root "engine\CCCP-Engine"
$engineRelease = Join-Path $release "engine\CCCP-Engine"
$resolvedEngineSource = (Resolve-Path -LiteralPath $engineSource).Path
$expectedEngineSource = [System.IO.Path]::GetFullPath((Join-Path $root "engine\CCCP-Engine"))
if (-not [string]::Equals($resolvedEngineSource, $expectedEngineSource, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "推理引擎来源不匹配，拒绝封装：$resolvedEngineSource"
}
$quantFramework = [System.IO.Path]::GetFullPath((Join-Path (Split-Path $root -Parent) "CCCP"))
if ([string]::Equals($resolvedEngineSource, $quantFramework, [System.StringComparison]::OrdinalIgnoreCase) -or
    $resolvedEngineSource.StartsWith($quantFramework + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "检测到保密 CCCP 量化框架路径，拒绝封装：$resolvedEngineSource"
}
Copy-Tree (Join-Path $engineSource "cccp") (Join-Path $engineRelease "cccp")
Copy-Tree (Join-Path $engineSource "_vendor") (Join-Path $engineRelease "_vendor")
Copy-Item -LiteralPath (Join-Path $engineSource "LICENSE") -Destination (Join-Path $engineRelease "LICENSE")
# The offline product contains only the inference runtime.  Standalone
# developer benchmark/API-client modules are not needed by the launcher.
foreach ($relative in @("cccp\benchmark.py", "cccp\api_cli_chat.py")) {
    $developerModule = Join-Path $engineRelease $relative
    if (Test-Path -LiteralPath $developerModule) { [System.IO.File]::Delete($developerModule) }
}
Copy-Item -LiteralPath (Join-Path $root "packaging\engine_runtime_main.py") -Destination (Join-Path $engineRelease "cccp\__main__.py") -Force
# Runtime source is required for offline JIT, but internal engine READMEs are
# not end-user launcher documentation.  Keep third-party licenses, remove only
# CCCP developer-facing markdown from the runtime package.
foreach ($relative in @(
    "cccp\README.md",
    "cccp\chat_adapters\README.md",
    "cccp\configs\README.md",
    "cccp\csrc\README.md",
    "cccp\ops\README.md"
)) {
    $developerDoc = Join-Path $engineRelease $relative
    if (Test-Path -LiteralPath $developerDoc) { [System.IO.File]::Delete($developerDoc) }
}
Copy-Tree (Join-Path $root "toolchain") (Join-Path $release "toolchain")
# Conda's downloaded/extracted package cache is needed to create environments,
# but the three finished environments above no longer read it.  Excluding this
# duplicate cache saves roughly 0.6 GiB while retaining conda.exe and the full
# CPU/CUDA/AMD runtime plus offline compiler toolchain.
$condaPackageCache = Join-Path $release "runtime\miniconda\pkgs"
if (Test-Path -LiteralPath $condaPackageCache) {
    $resolvedCache = (Resolve-Path -LiteralPath $condaPackageCache).Path
    $resolvedRuntime = (Resolve-Path -LiteralPath (Join-Path $release "runtime\miniconda")).Path
    if (-not $resolvedCache.StartsWith($resolvedRuntime + [System.IO.Path]::DirectorySeparatorChar)) {
        throw "拒绝清理 Miniconda 目录之外的包缓存：$resolvedCache"
    }
    $longCache = if ($resolvedCache.StartsWith("\\")) {
        "\\?\UNC\" + $resolvedCache.Substring(2)
    } else {
        "\\?\" + $resolvedCache
    }
    [System.IO.Directory]::Delete($longCache, $true)
}

Write-Host "[3/6] 创建空配置库并复制中文文档..."
New-Item -ItemType Directory -Force -Path (Join-Path $release "profiles\user") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $release "models") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $release "data") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $release "docs") | Out-Null
foreach ($name in @(
    "中文使用手册.md", "依赖与离线环境说明.md", "AMD核显兼容性说明.md"
)) {
    $source = Join-Path $root "docs\$name"
    if (Test-Path -LiteralPath $source) { Copy-Item -LiteralPath $source -Destination (Join-Path $release "docs\$name") }
}
Copy-Item -LiteralPath (Join-Path $root "docs\中文使用手册.md") -Destination (Join-Path $release "使用手册.md")

Write-Host "[4/6] 验证发行目录不含启动器工程源码..."
$forbidden = @("launcher", "webui", "tests", "scripts", "packaging", "build", "dist", "app_entry.py", "pytest.ini")
foreach ($name in $forbidden) {
    if (Test-Path -LiteralPath (Join-Path $release $name)) { throw "发行目录误含工程项：$name" }
}
& $python (Join-Path $root "scripts\audit_offline_release.py") $release
if ($LASTEXITCODE) { throw "发行目录用户文档边界审计失败" }

Write-Host "[5/6] 验证离线依赖..."
$releasePython = Join-Path $release "runtime\cpu\env\python.exe"
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
& $releasePython -c "import torch,fastapi,webview,modelscope,huggingface_hub; from modelscope import snapshot_download; print('offline dependencies ok', torch.__version__)"
if ($LASTEXITCODE) { throw "发行目录 Miniconda 依赖自检失败" }
& (Join-Path $release "runtime\cuda\env\python.exe") -c "import torch; assert torch.version.cuda and not torch.version.hip; print('CUDA environment ok',torch.__version__,torch.version.cuda)"
if ($LASTEXITCODE) { throw "发行目录 CUDA 环境自检失败" }
$env:CCCP_LAUNCHER_ROOT = $release
$env:CCCP_FUSED = "0"
$env:PYTHONPATH = $engineRelease
& (Join-Path $release "runtime\cuda\env\python.exe") -c "import shutil; from cccp import fusedext; from cccp.cpuext import _configure_bundled_windows_toolchain; fusedext._configure_packaged_gpu_toolchain(); assert _configure_bundled_windows_toolchain() or shutil.which('lib.exe'); print('CUDA offline link assets ok', fusedext._ensure_windows_cublas_import_library())"
if ($LASTEXITCODE) { throw "发行目录 CUDA cuBLAS/MSVC 离线链接自检失败" }
& (Join-Path $release "runtime\cuda\env\python.exe") -c "from cccp.kimi_experts import _cudart_library; print('CUDA runtime discovery ok', _cudart_library()._name)"
if ($LASTEXITCODE) { throw "发行目录 CUDA Runtime 动态发现自检失败" }
& (Join-Path $release "runtime\amd\env\python.exe") -c "import torch; assert torch.version.hip; print('AMD environment ok',torch.__version__,torch.version.hip)"
if ($LASTEXITCODE) { throw "发行目录 AMD ROCm/HIP 环境自检失败" }

Write-Host "[6/6] 生成版本信息和 SHA-256 清单..."
$info = @{
    name=$releaseName; version=$Version; model_weights_included=$false;
    model_profiles_included=$false;
    launcher_source_included=$false; quantization_framework_included=$false;
    engine_scope="inference-runtime-only"; inference_backends=@("cpu","cuda","amd");
    driver_included=$false; driver_note="GPU 驱动由操作系统提供"
} | ConvertTo-Json
[System.IO.File]::WriteAllText(
    (Join-Path $release "封装信息.json"),
    $info,
    (New-Object System.Text.UTF8Encoding($false))
)
& $python (Join-Path $root "scripts\generate_release_manifest.py") $release --version $Version
if ($LASTEXITCODE) { throw "SHA-256 清单生成失败" }

Write-Host "完成：$release"
Get-ChildItem -LiteralPath $release | Select-Object Name, Length
