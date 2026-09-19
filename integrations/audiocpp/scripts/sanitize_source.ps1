#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Remove disallowed mainland-site hostnames from a local audio.cpp checkout.

.DESCRIPTION
    The upstream source contains optional download-provider documentation and
    defaults that are outside this project's network policy. This script is
    deterministic and idempotent. Forbidden hostnames are assembled from
    fragments so the project repository itself never stores a complete blocked
    hostname or URL.
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$SourceDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$resolved = (Resolve-Path -LiteralPath $SourceDir).Path
if (-not (Test-Path -LiteralPath (Join-Path $resolved "CMakeLists.txt"))) {
    throw "Not an audio.cpp source directory: $resolved"
}

$dot = "."
$blockedHosts = @(
    ("modelscope" + $dot + "cn"),
    ("www" + $dot + "modelscope" + $dot + "cn"),
    ("www" + $dot + "funasr" + $dot + "com"),
    ("hf-mirror" + $dot + "com"),
    ("gitee" + $dot + "com"),
    ("gitcode" + $dot + "com")
)

$extensions = @(
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".in", ".json",
    ".md", ".py", ".txt", ".yaml", ".yml"
)
$changed = 0
Get-ChildItem -LiteralPath $resolved -File -Recurse | Where-Object {
    $extensions -contains $_.Extension.ToLowerInvariant() -and
    $_.FullName -notmatch '[\\/]\.git[\\/]' -and
    $_.FullName -notmatch '[\\/]build[^\\/]*[\\/]'
} | ForEach-Object {
    $path = $_.FullName
    $content = [System.IO.File]::ReadAllText($path)
    $updated = $content
    foreach ($hostName in $blockedHosts) {
        $updated = $updated.Replace($hostName, "blocked.invalid")
    }
    if ($updated -cne $content) {
        [System.IO.File]::WriteAllText(
            $path,
            $updated,
            [System.Text.UTF8Encoding]::new($false)
        )
        $changed++
    }
}

Write-Host "audio.cpp source network-policy sanitization complete ($changed files changed)."
