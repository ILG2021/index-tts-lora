#!/usr/bin/env pwsh
<#
.SYNOPSIS
    下载 audio.cpp 预编译 Windows CUDA 二进制文件。

.DESCRIPTION
    从 audio.cpp GitHub Releases 下载适合 Windows CUDA 环境的预编译包，
    解压到 integrations\audiocpp\bin\。

    参考 MOSS-TTS openmoss 的部署模式——此脚本负责准备 C++ 推理二进制，
    无需本地编译 CMake 工程。

.PARAMETER Version
    audio.cpp Release 版本号（如 v0.8.1），默认自动获取最新版。

.PARAMETER OutputDir
    二进制解压目录，默认 integrations\audiocpp\bin。

.EXAMPLE
    .\integrations\audiocpp\scripts\download_audiocpp.ps1

.EXAMPLE
    .\integrations\audiocpp\scripts\download_audiocpp.ps1 -Version v0.8.1
#>

param(
    [string]$Version = "",
    [string]$OutputDir = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot ".." ".." "..")
if (-not $OutputDir) {
    $OutputDir = Join-Path $ProjectRoot "integrations" "audiocpp" "bin"
}

Write-Host "=== audio.cpp 预编译二进制下载 ===" -ForegroundColor Cyan
Write-Host "目标目录：$OutputDir"

# ── 1. 确定版本 ─────────────────────────────────────────────────────────────
if (-not $Version) {
    Write-Host "正在查询最新 Release 版本..." -ForegroundColor Yellow
    try {
        $latest = Invoke-RestMethod -Uri "https://api.github.com/repos/0xShug0/audio.cpp/releases/latest" -UseBasicParsing
        $Version = $latest.tag_name
        Write-Host "最新版本：$Version"
    } catch {
        Write-Error "无法获取最新版本：$_`n请手动指定 -Version 参数（例如 -Version v0.8.1）"
    }
}

# ── 2. 构造下载 URL ──────────────────────────────────────────────────────────
# audio.cpp Release 资产命名约定（以 Windows CUDA 包为主）：
#   audio.cpp-<version>-win-cuda-<arch>.zip
# 注意：实际资产名请在 GitHub Releases 页面确认：
#   https://github.com/0xShug0/audio.cpp/releases
$BaseUrl = "https://github.com/0xShug0/audio.cpp/releases/download/$Version"

# 尝试常见的 Windows CUDA 包名（按优先级排列）
$CandidateNames = @(
    "audio.cpp-$Version-win-cuda.zip",
    "audio.cpp-$Version-windows-cuda.zip",
    "audio.cpp-win-cuda-$Version.zip",
    "audiocpp-$Version-windows-cuda.zip"
)

Write-Host ""
Write-Host "正在从 GitHub Releases $Version 查找 Windows CUDA 包..." -ForegroundColor Yellow

# 获取 Release 资产列表以确认实际文件名
try {
    $releases = Invoke-RestMethod `
        -Uri "https://api.github.com/repos/0xShug0/audio.cpp/releases/tags/$Version" `
        -UseBasicParsing
    $assets = $releases.assets | Where-Object { $_.name -match "win" -and $_.name -match "\.zip$" }
    if ($assets) {
        Write-Host "找到以下 Windows 资产："
        $assets | ForEach-Object { Write-Host "  - $($_.name)" }
        # 新版 Release 将 CUDA 拆成 runtime 与 profile 两个包，两者都需要。
        $runtimeAsset = $assets | Where-Object { $_.name -match "cuda-runtime" } | Select-Object -First 1
        $profileAsset = $assets | Where-Object { $_.name -match "cuda-fast" } | Select-Object -First 1
        if ($runtimeAsset -and $profileAsset) {
            $SelectedAssets = @($runtimeAsset, $profileAsset)
        } else {
            $cudaAsset = $assets | Where-Object { $_.name -match "cuda" } | Select-Object -First 1
            if (-not $cudaAsset) { $cudaAsset = $assets | Select-Object -First 1 }
            $SelectedAssets = @($cudaAsset)
        }
    } else {
        Write-Warning "未在 $Version 的 Release 中找到 Windows zip 资产。"
        Write-Host "请访问以下地址手动下载并解压到 $OutputDir ："
        Write-Host "  https://github.com/0xShug0/audio.cpp/releases/tag/$Version"
        exit 1
    }
} catch {
    Write-Warning "无法列出 Release 资产（$_ ）。使用备选 URL 格式。"
    $SelectedAssets = @([pscustomobject]@{
        browser_download_url = "$BaseUrl/$($CandidateNames[0])"
        name = $CandidateNames[0]
    })
}

# ── 3. 下载并解压 ────────────────────────────────────────────────────────────
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
foreach ($asset in $SelectedAssets) {
    $TempZip = Join-Path $env:TEMP $asset.name
    Write-Host ""
    Write-Host "下载：$($asset.browser_download_url)" -ForegroundColor Yellow
    Write-Host "→ $TempZip"
    try {
        Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $TempZip -UseBasicParsing
    } catch {
        Write-Error "下载失败：$_`n请检查网络连接或手动下载：$($asset.browser_download_url)"
    }
    Write-Host "解压到：$OutputDir" -ForegroundColor Yellow
    Expand-Archive -Path $TempZip -DestinationPath $OutputDir -Force
    Remove-Item $TempZip -Force
}

# ── 4. 验证关键二进制 ────────────────────────────────────────────────────────
Write-Host ""
Write-Host "验证关键二进制文件..." -ForegroundColor Yellow

$RequiredBinaries = @("audiocpp_cli.exe", "audiocpp_server.exe", "audiocpp_gguf.exe")
$Missing = @()
foreach ($bin in $RequiredBinaries) {
    $candidates = Get-ChildItem -Path $OutputDir -Filter $bin -Recurse -ErrorAction SilentlyContinue
    if ($candidates) {
        $binPath = $candidates[0].FullName
        # 如果二进制在子目录中，将其移到 OutputDir 顶层
        if ($candidates[0].DirectoryName -ne $OutputDir) {
            Move-Item $binPath $OutputDir -Force
            Write-Host "  ✓ $bin（从子目录移至顶层）"
        } else {
            Write-Host "  ✓ $bin"
        }
    } else {
        $Missing += $bin
        Write-Warning "  ✗ $bin 未找到"
    }
}

if ($Missing.Count -gt 0) {
    Write-Warning "以下二进制未找到：$($Missing -join ', ')"
    Write-Host "请检查解压内容或参考 audio.cpp 文档手动安装。"
    Write-Host "解压目录：$OutputDir"
} else {
    Write-Host ""
    Write-Host "✅ audio.cpp 预编译二进制已就绪！" -ForegroundColor Green
    Write-Host ""
    Write-Host "后续步骤："
    Write-Host "  1. 转换模型权重："
    Write-Host "       python scripts\convert_to_gguf.py --model-dir checkpoints --audiocpp-dir $OutputDir"
    Write-Host "  2. 启动 WebUI（audio.cpp 后端）："
    Write-Host "       .\scripts\windows\webui_audiocpp.ps1"
}
