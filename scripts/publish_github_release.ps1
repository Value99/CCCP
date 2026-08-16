param(
    [Parameter(Mandatory=$true)][string]$AssetsDirectory,
    [string]$Repository = "Value99/CCCP",
    [string]$Version = "",
    [string]$NotesFile = "",
    [string]$GitRepository = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$root = Split-Path $PSScriptRoot -Parent
if (-not $Version) {
    $Version = [System.IO.File]::ReadAllText((Join-Path $root "VERSION"), [System.Text.Encoding]::UTF8).Trim()
}
$assets = (Resolve-Path -LiteralPath $AssetsDirectory).Path
if (-not $NotesFile) { $NotesFile = Join-Path $root "packaging\GITHUB_RELEASE_$Version.md" }
if (-not $GitRepository) { $GitRepository = Join-Path $root "github-release-repo" }
if (-not (Test-Path -LiteralPath (Join-Path $assets "CCCP-Launcher-v$Version-offline.parts.json"))) {
    throw "Offline parts manifest is missing"
}
if (Get-ChildItem -LiteralPath $assets -Recurse -File | Where-Object { $_.Extension -in @(".safetensors", ".gguf", ".ckpt", ".onnx") }) {
    throw "Refusing to upload model weight files"
}

$credentialRequest = "protocol=https`nhost=github.com`n`n"
$rawCredential = $credentialRequest | git -C $GitRepository credential fill 2>$null
$credential = @{}
foreach ($line in $rawCredential) {
    $separator = $line.IndexOf("=")
    if ($separator -gt 0) { $credential[$line.Substring(0, $separator)] = $line.Substring($separator + 1) }
}
$secret = $credential["password"]
if (-not $secret) { throw "GitHub credential not found; run one authenticated git command first" }
$headers = @{
    "Authorization" = "Bearer $secret"
    "Accept" = "application/vnd.github+json"
    "X-GitHub-Api-Version" = "2022-11-28"
    "User-Agent" = "CCCP-Launcher-Release"
}
$tag = "v$Version"
$api = "https://api.github.com/repos/$Repository"
$notesText = [System.IO.File]::ReadAllText(
    (Resolve-Path -LiteralPath $NotesFile).Path,
    [System.Text.Encoding]::UTF8
)
try {
    $release = Invoke-RestMethod -Uri "$api/releases/tags/$tag" -Headers $headers -TimeoutSec 30
} catch {
    if ($_.Exception.Response -and [int]$_.Exception.Response.StatusCode -eq 404) {
        # Windows PowerShell 5.1 attaches PSPath/PSDrive note properties to the
        # String returned by Get-Content. ConvertTo-Json would serialize that
        # enriched object instead of a JSON string, which GitHub rejects.
        $body = @{
            tag_name = $tag
            target_commitish = "main"
            name = "CCCP Launcher $Version Full Offline"
            body = $notesText
            draft = $false
            prerelease = $false
        } | ConvertTo-Json
        $release = Invoke-RestMethod -Uri "$api/releases" -Headers $headers -Method Post -ContentType "application/json; charset=utf-8" -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 60
    } else { throw }
}

$metadataBody = @{
    name = "CCCP Launcher $Version Full Offline"
    body = $notesText
    draft = $false
    prerelease = $false
} | ConvertTo-Json
$release = Invoke-RestMethod -Uri "$api/releases/$($release.id)" -Headers $headers -Method Patch -ContentType "application/json; charset=utf-8" -Body ([System.Text.Encoding]::UTF8.GetBytes($metadataBody)) -TimeoutSec 60

$localFiles = Get-ChildItem -LiteralPath $assets -File | Sort-Object Name
foreach ($file in $localFiles) {
    if ($file.Length -ge 2GB) { throw "GitHub asset must be under 2 GiB: $($file.Name)" }
    $existing = @($release.assets | Where-Object name -eq $file.Name)
    if ($existing.Count -eq 1 -and [long]$existing[0].size -eq $file.Length -and $existing[0].state -eq "uploaded") {
        Write-Host "Verified existing $($file.Name) ($([math]::Round($file.Length / 1MB, 1)) MiB); skipping upload."
        continue
    }
    foreach ($old in $existing) {
        Invoke-RestMethod -Uri "$api/releases/assets/$($old.id)" -Headers $headers -Method Delete -TimeoutSec 30 | Out-Null
    }
    $encoded = [System.Uri]::EscapeDataString($file.Name)
    $uploadUrl = "https://uploads.github.com/repos/$Repository/releases/$($release.id)/assets?name=$encoded"
    Write-Host "Uploading $($file.Name) ($([math]::Round($file.Length / 1MB, 1)) MiB)..."
    $uploaded = $null
    for ($attempt = 1; $attempt -le 3 -and -not $uploaded; $attempt++) {
        try {
            $uploaded = Invoke-RestMethod -Uri $uploadUrl -Headers $headers -Method Post -InFile $file.FullName -ContentType "application/octet-stream" -TimeoutSec 0
        } catch {
            if ($attempt -eq 3) { throw }
            Start-Sleep -Seconds (5 * $attempt)
            $fresh = Invoke-RestMethod -Uri "$api/releases/$($release.id)" -Headers $headers -TimeoutSec 30
            foreach ($partial in @($fresh.assets | Where-Object name -eq $file.Name)) {
                Invoke-RestMethod -Uri "$api/releases/assets/$($partial.id)" -Headers $headers -Method Delete -TimeoutSec 30 | Out-Null
            }
        }
    }
    if ($uploaded.state -ne "uploaded" -or [long]$uploaded.size -ne $file.Length) {
        throw "Remote asset verification failed: $($file.Name)"
    }
}

$final = $null
for ($attempt = 1; $attempt -le 3 -and -not $final; $attempt++) {
    try {
        $final = Invoke-RestMethod -Uri "$api/releases/$($release.id)" -Headers $headers -TimeoutSec 30
    } catch {
        if ($attempt -eq 3) { throw }
        Start-Sleep -Seconds (3 * $attempt)
    }
}
foreach ($file in $localFiles) {
    $remote = @($final.assets | Where-Object name -eq $file.Name)
    if ($remote.Count -ne 1 -or [long]$remote[0].size -ne $file.Length -or $remote[0].state -ne "uploaded") {
        throw "Final remote audit failed: $($file.Name)"
    }
}
[pscustomobject]@{ReleaseUrl=$final.html_url;Tag=$final.tag_name;Assets=$final.assets.Count} | ConvertTo-Json
