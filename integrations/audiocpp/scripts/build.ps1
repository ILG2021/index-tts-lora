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

.PARAMETER CudaToolkitRoot
    CUDA Toolkit 根目录（例如 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4）。
    默认从 CUDA_PATH 或 nvcc.exe 自动推导，并作为 CMake `-T cuda=...` 传入。

.EXAMPLE
    .\integrations\audiocpp\scripts\build.ps1
#>

param(
    [string]$AudioCppDir = "",
    [string]$BuildDir = "",
    [string]$OutputDir = "",
    [string]$CudaArch = "native",
    [string]$CudaToolkitRoot = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot ".." ".." "..")
if (-not $AudioCppDir) { $AudioCppDir = Join-Path $ProjectRoot "integrations" "audiocpp" "src" "audio.cpp" }
if (-not $BuildDir)    { $BuildDir    = Join-Path $AudioCppDir "build-cuda" }
if (-not $OutputDir)   { $OutputDir   = Join-Path $ProjectRoot "integrations" "audiocpp" "bin" }

if (-not $CudaToolkitRoot -and $env:CUDA_PATH) {
    $CudaToolkitRoot = $env:CUDA_PATH
}
if (-not $CudaToolkitRoot) {
    $cuda124 = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA\v12.4"
    if (Test-Path -LiteralPath (Join-Path $cuda124 "bin\nvcc.exe")) {
        $CudaToolkitRoot = $cuda124
    }
}
if (-not $CudaToolkitRoot) {
    $nvccCommand = Get-Command "nvcc.exe" -ErrorAction SilentlyContinue
    if ($nvccCommand) {
        $CudaToolkitRoot = Split-Path (Split-Path $nvccCommand.Source -Parent) -Parent
    }
}
if ($CudaToolkitRoot) {
    $CudaToolkitRoot = [System.IO.Path]::GetFullPath($CudaToolkitRoot).TrimEnd('\')
}

Write-Host "=== audio.cpp 源码编译（Windows CUDA） ===" -ForegroundColor Cyan
Write-Host "源码目录：$AudioCppDir"
Write-Host "构建目录：$BuildDir"
Write-Host "输出目录：$OutputDir"
Write-Host "CUDA 架构：$CudaArch"
Write-Host "CUDA Toolkit：$($CudaToolkitRoot ? $CudaToolkitRoot : '未找到')"
Write-Host ""

# ── 1. 检查依赖 ───────────────────────────────────────────────────────────────
function Test-Command { param($cmd); return (Get-Command $cmd -ErrorAction SilentlyContinue) -ne $null }

# 普通 PowerShell 不会自动加载 MSVC 环境。先尝试通过 vswhere/VsDevCmd
# 定位已安装的 VS 2022 C++ 工具链，避免要求用户必须从
# "Developer PowerShell for VS 2022" 启动。
function Initialize-MsvcEnvironment {
    if ((Test-Command "cl.exe") -or (Test-Command "clang-cl.exe")) { return $true }

    $vswhereCandidates = @(
        (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"),
        (Join-Path $env:ProgramFiles "Microsoft Visual Studio\Installer\vswhere.exe")
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }

    $installPath = $null
    foreach ($vswhere in $vswhereCandidates) {
        $candidate = & $vswhere -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath 2>$null
        if ($LASTEXITCODE -eq 0 -and $candidate) {
            $installPath = ($candidate | Select-Object -First 1).Trim()
            break
        }
    }

    if (-not $installPath) {
        $editionRoots = @("BuildTools", "Community", "Professional", "Enterprise")
        foreach ($edition in $editionRoots) {
            $candidate = Join-Path $env:ProgramFiles "Microsoft Visual Studio\2022\$edition"
            if (Test-Path -LiteralPath (Join-Path $candidate "Common7\Tools\VsDevCmd.bat")) {
                $installPath = $candidate
                break
            }
        }
    }
    if (-not $installPath) { return $false }

    $devCmd = Join-Path $installPath "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path -LiteralPath $devCmd)) { return $false }

    Write-Host "加载 Visual Studio C++ 开发环境：$installPath" -ForegroundColor Gray
    $environmentLines = & $env:ComSpec /d /s /c `
        "`"$devCmd`" -no_logo -arch=x64 -host_arch=x64 && set" 2>$null
    if ($LASTEXITCODE -ne 0) { return $false }
    foreach ($line in $environmentLines) {
        if ($line -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
    return (Test-Command "cl.exe") -or (Test-Command "clang-cl.exe")
}

$msvcAvailable = Initialize-MsvcEnvironment
$missing = @()
if (-not (Test-Command "cmake"))   { $missing += "CMake 3.20+" }
if (-not (Test-Command "git"))     { $missing += "Git" }
if (-not (Test-Command "nvcc"))    { $missing += "CUDA Toolkit 12+ (nvcc.exe)" }
if (-not $CudaToolkitRoot -or -not (Test-Path -LiteralPath (Join-Path $CudaToolkitRoot "bin\nvcc.exe"))) {
    $missing += "有效的 CUDA Toolkit 根目录（可用 -CudaToolkitRoot 指定）"
}
if (-not $msvcAvailable) {
    $missing += "MSVC v143 x64 C++ tools (Visual Studio Build Tools 2022)"
}
if ($missing.Count -gt 0) {
    $detail = $missing -join "`n - "
    throw @"
缺少以下构建依赖：
 - $detail

如果缺少 MSVC，请在 Visual Studio Installer 中安装“使用 C++ 的桌面开发”，
并确保包含 MSVC v143 x64/x86、Windows 10/11 SDK 和 C++ CMake tools。
安装后可直接重跑本脚本，也可从“Developer PowerShell for VS 2022”运行。
"@
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
$PatchDir = Join-Path $ProjectRoot "integrations" "audiocpp" "patches"
$Patches = @(
    (Join-Path $PatchDir "0001-index-tts2-runtime-lora.patch"),
    (Join-Path $PatchDir "0002-msvc-utf8.patch")
)
foreach ($Patch in $Patches) {
    if (-not (Test-Path $Patch)) { Write-Error "缺少 audio.cpp patch：$Patch" }
    & git -C $AudioCppDir apply --check $Patch 2>$null
    if ($LASTEXITCODE -eq 0) {
        & git -C $AudioCppDir apply $Patch
        if ($LASTEXITCODE -ne 0) { Write-Error "应用 audio.cpp patch 失败：$Patch" }
        Write-Host "已应用 $([System.IO.Path]::GetFileName($Patch))" -ForegroundColor Green
    } else {
        & git -C $AudioCppDir apply --reverse --check $Patch 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Error "audio.cpp 源码既不能应用也不包含 patch：$Patch`n请确认使用 v0.8.1 干净源码"
        }
        Write-Host "$([System.IO.Path]::GetFileName($Patch)) 已存在，跳过" -ForegroundColor Gray
    }
}

# ── 2.2 清理上游源码中不符合项目网络策略的链接 ───────────────────
$SanitizeScript = Join-Path $ProjectRoot "integrations" "audiocpp" "scripts" "sanitize_source.ps1"
if (-not (Test-Path $SanitizeScript)) { Write-Error "缺少源码网络策略清理脚本：$SanitizeScript" }
& $SanitizeScript -SourceDir $AudioCppDir
if ($LASTEXITCODE -ne 0) { Write-Error "清理 audio.cpp 源码链接失败" }

# ── 3. CMake 配置 ─────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "CMake 配置..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

$cmakeArgs = @(
    "-S", $AudioCppDir,
    "-B", $BuildDir,
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-T", "cuda=$CudaToolkitRoot",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DENGINE_ENABLE_CUDA=ON",
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch",
    # 禁止编译上游模型管理/网络下载能力；本项目只加载本地 GGUF/adapter
    "-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF",
    # 只编译 core + index_tts2，减少编译时间
    "-DAUDIOCPP_MODEL_SET=custom",
    "-DAUDIOCPP_MODELS=index_tts2",
    # 构建产物输出到统一目录
    "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$BuildDir\bin"
)

$cachePath = Join-Path $BuildDir "CMakeCache.txt"
if (Test-Path -LiteralPath $cachePath) {
    $expectedToolset = "CMAKE_GENERATOR_TOOLSET:INTERNAL=cuda=$CudaToolkitRoot"
    $cacheContent = Get-Content -LiteralPath $cachePath -Raw
    if ($cacheContent -notmatch [regex]::Escape($expectedToolset)) {
        throw @"
现有 CMake cache 是用其他 CUDA toolset/生成器创建的：
  $cachePath
请删除构建目录后重试（不会删除源码、GGUF 或 LoRA）：
  Remove-Item -LiteralPath `"$BuildDir`" -Recurse -Force
"@
    }
}

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
