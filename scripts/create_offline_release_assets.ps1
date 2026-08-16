param(
    [Parameter(Mandatory=$true)][string]$ReleaseDirectory,
    [string]$Version = "",
    [string]$OutputDirectory = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Get-CccpFileSha256([string]$Path) {
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $algorithm = [System.Security.Cryptography.SHA256]::Create()
        try {
            $digest = $algorithm.ComputeHash($stream)
            return ([System.BitConverter]::ToString($digest)).Replace("-", "").ToLowerInvariant()
        }
        finally {
            $algorithm.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }
}

$root = Split-Path $PSScriptRoot -Parent
if (-not $Version) {
    $Version = [System.IO.File]::ReadAllText((Join-Path $root "VERSION"), [System.Text.Encoding]::UTF8).Trim()
}
$release = (Resolve-Path -LiteralPath $ReleaseDirectory).Path
$releaseName = Split-Path -Leaf $release
$expectedName = "CCCP-Launcher-v$Version-win-x64-offline"
if ($releaseName -ne $expectedName) { throw "Release directory name mismatch: expected $expectedName" }
foreach ($required in @("CCCP-Launcher.exe", "VERSION", "runtime", "engine", "toolchain", "SHA256SUMS.txt")) {
    if (-not (Test-Path -LiteralPath (Join-Path $release $required))) { throw "Missing release item: $required" }
}
$python = Join-Path $root "runtime\cpu\env\python.exe"
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $root "release-assets\v$Version"
}
$output = [System.IO.Path]::GetFullPath($OutputDirectory)
$assetParent = [System.IO.Path]::GetFullPath((Join-Path $root "release-assets"))
if (-not $output.StartsWith($assetParent + [System.IO.Path]::DirectorySeparatorChar)) {
    throw "OutputDirectory must stay inside $assetParent"
}
if (Test-Path -LiteralPath $output) {
    if (-not $Force) { throw "Asset directory already exists; use -Force: $output" }
    $longOutput = if ($output.StartsWith("\\")) { "\\?\UNC\" + $output.Substring(2) } else { "\\?\" + $output }
    [System.IO.Directory]::Delete($longOutput, $true)
}
New-Item -ItemType Directory -Force -Path $output | Out-Null

Write-Host "[1/6] Verify every release file against SHA256SUMS.txt..."
& $python (Join-Path $PSScriptRoot "verify_release_manifest.py") $release
if ($LASTEXITCODE) { throw "Release SHA-256 manifest verification failed" }

Write-Host "[2/6] Build the double-click offline bootstrap..."
$setup = Join-Path $output "CCCP-Launcher-$Version-Offline-Setup.exe"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "build_offline_bootstrap.ps1") -Output $setup
if ($LASTEXITCODE) { throw "Offline bootstrap build failed" }

Write-Host "[3/6] Create one ZIP64 archive (temporary)..."
$archiveName = "$expectedName.zip"
$archive = Join-Path $output $archiveName
$releaseParent = Split-Path -Parent $release
& tar.exe -a -cf $archive -C $releaseParent $releaseName
if ($LASTEXITCODE -or -not (Test-Path -LiteralPath $archive)) { throw "ZIP creation failed" }

Write-Host "[4/6] Validate and split into GitHub assets below 2 GiB..."
& $python (Join-Path $PSScriptRoot "split_offline_archive.py") $archive $output --version $Version --root-directory $releaseName --part-mib 1900
if ($LASTEXITCODE) { throw "Archive validation/split failed" }

$resolvedArchive = (Resolve-Path -LiteralPath $archive).Path
if ((Split-Path -Parent $resolvedArchive) -ne $output) { throw "Refusing to remove archive outside asset directory" }
[System.IO.File]::Delete($resolvedArchive)

Write-Host "[5/6] Add standalone updater and install instructions..."
Copy-Item -LiteralPath (Join-Path $release "CCCP-Launcher.exe") -Destination (Join-Path $output "CCCP-Launcher.exe")
$installTemplate = [System.IO.File]::ReadAllText(
    (Join-Path $root "packaging\OFFLINE_INSTALL.txt"),
    [System.Text.Encoding]::UTF8
)
if (-not $installTemplate.Contains("{version}")) {
    throw "OFFLINE_INSTALL.txt is missing the {version} placeholder"
}
[System.IO.File]::WriteAllText(
    (Join-Path $output "INSTALL.txt"),
    $installTemplate.Replace("{version}", $Version),
    (New-Object System.Text.UTF8Encoding($false))
)

Write-Host "[6/6] Generate final asset checksums..."
$checksumPath = Join-Path $output "SHA256SUMS.txt"
$lines = @()
foreach ($file in (Get-ChildItem -LiteralPath $output -File | Where-Object Name -ne "SHA256SUMS.txt" | Sort-Object Name)) {
    $hash = Get-CccpFileSha256 $file.FullName
    $lines += "$hash  $($file.Name)"
}
[System.IO.File]::WriteAllLines($checksumPath, $lines, (New-Object System.Text.UTF8Encoding($false)))
Get-ChildItem -LiteralPath $output -File | Sort-Object Name | Select-Object Name, Length
