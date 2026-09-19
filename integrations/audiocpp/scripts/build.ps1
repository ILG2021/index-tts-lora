#!/usr/bin/env pwsh
<#
.SYNOPSIS
    从源码编译 audio.cpp（进阶，需要 CMake 3.20+ 和 CUDA Toolkit 12+）。

.DESCRIPTION
    克隆 audio.cpp 仓库并在 CUDA 模式下编译。
    推荐先尝试 download_audiocpp.ps1（预编译二进制）。

.PARAMETER AudioCppDir
    audio.cpp 源码克隆目录，默认 integrations\audiocpp\src\audio.cpp

.PARAMETER BuildDir
    CMake 构建目录，默认 integrations\audiocpp\src\audio.cpp\build-cuda

.PARAMETER OutputDir
    编译产物复制目标（覆盖 bin\），默认 integrations\audiocpp\bin

.PARAMETER CudaArch
    CUDA 架构（native 表示仅为当前 GPU 编译），默认 native

.EXAMPLE
    .\integrations\audiocpp\scripts\build.ps1
#>

param(
    [string]$AudioCppDir = "",
    [string]$BuildDir = "",
    [string]$OutputDir = "",
    [string]$CudaArch = "native"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot ".." ".." "..")
if (-not $AudioCppDir) { $AudioCppDir = Join-Path $ProjectRoot "integrations" "audiocpp" "src" "audio.cpp" }
if (-not $BuildDir)    { $BuildDir    = Join-Path $AudioCppDir "build-cuda" }
if (-not $OutputDir)   { $OutputDir   = Join-Path $ProjectRoot "integrations" "audiocpp" "bin" }

Write-Host "=== audio.cpp 源码编译（Windows CUDA） ===" -ForegroundColor Cyan
Write-Host "源码目录：$AudioCppDir"
Write-Host "构建目录：$BuildDir"
Write-Host "输出目录：$OutputDir"
Write-Host "CUDA 架构：$CudaArch"
Write-Host ""

# ── 1. 检查依赖 ───────────────────────────────────────────────────────────────
function Test-Command { param($cmd); return (Get-Command $cmd -ErrorAction SilentlyContinue) -ne $null }

$missing = @()
if (-not (Test-Command "cmake"))   { $missing += "CMake 3.20+" }
if (-not (Test-Command "git"))     { $missing += "Git" }
if (-not (Test-Command "nvcc"))    { $missing += "CUDA Toolkit 12+" }
if (-not (Test-Command "cl.exe") -and -not (Test-Command "clang-cl.exe")) {
    $missing += "MSVC (Visual Studio 2022)"
}
if ($missing.Count -gt 0) {
    Write-Error "缺少以下依赖：`n  $($missing -join "`n  ")`n请安装后重试。"
}

# ── 2. 克隆（若不存在） ───────────────────────────────────────────────────────
if (-not (Test-Path $AudioCppDir)) {
    Write-Host "克隆 audio.cpp v0.8.1 仓库..." -ForegroundColor Yellow
    git clone --depth 1 --branch v0.8.1 https://github.com/0xShug0/audio.cpp.git $AudioCppDir
    Write-Host "克隆完成"
} else {
    Write-Host "源码目录已存在，跳过克隆（如需更新请手动 git pull）"
}

# ── 2.1 应用 IndexTTS2 runtime LoRA 扩展 ─────────────────────────────────────
$LoraPatch = Join-Path $ProjectRoot "integrations" "audiocpp" "patches" "0001-index-tts2-runtime-lora.patch"
if (-not (Test-Path $LoraPatch)) { Write-Error "缺少 runtime LoRA patch：$LoraPatch" }
& git -C $AudioCppDir apply --check $LoraPatch 2>$null
if ($LASTEXITCODE -eq 0) {
    & git -C $AudioCppDir apply $LoraPatch
    if ($LASTEXITCODE -ne 0) { Write-Error "应用 IndexTTS2 runtime LoRA patch 失败" }
    Write-Host "已应用 IndexTTS2 runtime LoRA patch" -ForegroundColor Green
} else {
    & git -C $AudioCppDir apply --reverse --check $LoraPatch 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "audio.cpp 源码既不能应用也不包含 runtime LoRA patch；请确认使用 v0.8.1 干净源码"
    }
    Write-Host "IndexTTS2 runtime LoRA patch 已存在，跳过" -ForegroundColor Gray
}

# ── 3. CMake 配置 ─────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "CMake 配置..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

$cmakeArgs = @(
    "-S", $AudioCppDir,
    "-B", $BuildDir,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DENGINE_ENABLE_CUDA=ON",
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch",
    # 只编译 core + index_tts2，减少编译时间
    "-DAUDIOCPP_MODEL_SET=custom",
    "-DAUDIOCPP_MODELS=index_tts2",
    # 构建产物输出到统一目录
    "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$BuildDir\bin"
)

Write-Host "cmake $($cmakeArgs -join ' ')"
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { Write-Error "CMake 配置失败（退出码 $LASTEXITCODE）" }

# ── 4. 编译 ──────────────────────────────────────────────────────────────────
Write-Host ""
$Cpus = (Get-CimInstance -ClassName Win32_ComputerSystem).NumberOfLogicalProcessors
Write-Host "编译（-j $Cpus）..." -ForegroundColor Yellow
cmake --build $BuildDir --config Release --parallel $Cpus
if ($LASTEXITCODE -ne 0) { Write-Error "编译失败（退出码 $LASTEXITCODE）" }

# ── 5. 复制产物 ───────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "复制产物到 $OutputDir..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$BuiltBinDir = Join-Path $BuildDir "bin"
@("audiocpp_cli.exe", "audiocpp_server.exe", "audiocpp_gguf.exe") | ForEach-Object {
    $src = Join-Path $BuiltBinDir $_
    if (Test-Path $src) {
        Copy-Item $src $OutputDir -Force
        Write-Host "  ✓ $_"
    } else {
        Write-Warning "  ✗ $_ 未生成（编译可能跳过了此目标）"
    }
}

# 复制 GGML 后端 DLL（CUDA 等）
Get-ChildItem $BuiltBinDir -Filter "*.dll" -ErrorAction SilentlyContinue |
    ForEach-Object { Copy-Item $_.FullName $OutputDir -Force }

Write-Host ""
Write-Host "✅ 编译完成！" -ForegroundColor Green
Write-Host "二进制位于：$OutputDir"
Write-Host ""
Write-Host "后续步骤："
Write-Host "  python scripts\convert_to_gguf.py --model-dir checkpoints --audiocpp-dir $OutputDir"
