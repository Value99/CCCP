param(
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
if (-not $Output) {
    $version = [System.IO.File]::ReadAllText((Join-Path $root "VERSION"), [System.Text.Encoding]::UTF8).Trim()
    $Output = Join-Path $root "release-assets\CCCP-Launcher-$version-Offline-Setup.exe"
}
$outputDirectory = Split-Path -Parent $Output
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

$compiler = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if (-not (Test-Path -LiteralPath $compiler)) {
    $compiler = Join-Path $env:WINDIR "Microsoft.NET\Framework\v4.0.30319\csc.exe"
}
if (-not (Test-Path -LiteralPath $compiler)) { throw "Missing .NET Framework C# compiler" }

$source = Join-Path $root "packaging\offline_bootstrap.cs"
$icon = Join-Path $root "webui\images\icon.ico"
$arguments = @(
    "/nologo", "/target:winexe", "/optimize+", "/platform:anycpu",
    "/out:$Output", "/win32icon:$icon",
    "/reference:System.dll", "/reference:System.Core.dll",
    "/reference:System.Drawing.dll", "/reference:System.Windows.Forms.dll",
    "/reference:System.IO.Compression.dll", "/reference:System.IO.Compression.FileSystem.dll",
    "/reference:System.Runtime.Serialization.dll", $source
)
& $compiler @arguments
if ($LASTEXITCODE -or -not (Test-Path -LiteralPath $Output)) {
    throw "Offline bootstrap build failed"
}
Get-Item -LiteralPath $Output | Select-Object FullName, Length
