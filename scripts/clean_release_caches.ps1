param([Parameter(Mandatory = $true)][string]$ReleaseDir)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath (Split-Path $PSScriptRoot -Parent)).Path
$release = (Resolve-Path -LiteralPath $ReleaseDir).Path
$packageRoot = Split-Path -Parent $release
$releaseName = Split-Path -Leaf $release
if (((Split-Path -Parent $packageRoot) -ne $root) -or ($releaseName -notlike "CCCP-Launcher-v*-win-x64-offline")) {
    throw "Refusing to clean outside the versioned release directory: $release"
}
foreach ($marker in @("VERSION", "CCCP-Launcher.exe")) {
    if (-not (Test-Path -LiteralPath (Join-Path $release $marker))) {
        throw "Release marker is missing: $marker"
    }
}

$longRelease = if ($release.StartsWith("\\")) {
    "\\?\UNC\" + $release.Substring(2)
} else {
    "\\?\" + $release
}
$caches = @(
    [System.IO.Directory]::EnumerateDirectories(
        $longRelease,
        "__pycache__",
        [System.IO.SearchOption]::AllDirectories
    )
)
$removedFiles = 0
$removedBytes = 0L
foreach ($cache in $caches | Sort-Object Length -Descending) {
    foreach ($file in [System.IO.Directory]::EnumerateFiles(
        $cache, "*", [System.IO.SearchOption]::AllDirectories
    )) {
        $info = [System.IO.FileInfo]::new($file)
        $removedFiles += 1
        $removedBytes += $info.Length
    }
    if ([System.IO.Directory]::Exists($cache)) {
        [System.IO.Directory]::Delete($cache, $true)
    }
}

[pscustomobject]@{
    Release = $release
    RemovedDirectories = $caches.Count
    RemovedFiles = $removedFiles
    RemovedBytes = $removedBytes
}
