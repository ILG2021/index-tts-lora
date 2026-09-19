#!/usr/bin/env pwsh
<#
.SYNOPSIS
    初始化 audio.cpp 推理环境（下载二进制、转换 GGUF、可选量化）。

.DESCRIPTION
    一键完成 audio.cpp 推理所需的全部准备工作：
      1. 下载 audio.cpp 预编译 Windows CUDA 二进制
      2. 将 IndexTTS2 checkpoints 转换为 GGUF 格式
      3. 可选：量化为 Q8_0（推荐，约减少 50% 体积）
      4. 可选：将 LoRA checkpoint 转换为运行时 adapter

    参考架构：MOSS-TTS openmoss 部署模式。

.PARAMETER ModelDir
    IndexTTS2 checkpoints 目录，默认 checkpoints

.PARAMETER AudiocppDir
    audio.cpp 二进制目标目录，默认 integrations\audiocpp\bin

.PARAMETER Dtype
    GGUF 转换精度：f16 | bf16 | q8_0，默认 f16

.PARAMETER Quantize
    是否额外生成 Q8_0 量化版本（推荐），默认 $true

.PARAMETER LoraCheckpoint
    可选：LoRA checkpoint .pth 文件路径。指定后转换为运行时 safetensors adapter

.PARAMETER LoraName
    可选：adapter 名称，默认为 checkpoint 文件名

.PARAMETER AudiocppVersion
    audio.cpp Release 版本，留空则自动获取最新版

.EXAMPLE
    .\scripts\windows\setup_audiocpp.ps1

.EXAMPLE
    .\scripts\windows\setup_audiocpp.ps1 `
        -Dtype f16 `
        -Quantize `
        -LoraCheckpoint trained_ckcts\model_epoch5.pth

.EXAMPLE
    .\scripts\windows\setup_audiocpp.ps1 -AudiocppVersion v0.8.1
#>

param(
    [string]$ModelDir = "checkpoints",
    [string]$AudiocppDir = "",
    [string]$Dtype = "f16",
    [switch]$Quantize = $true,
    [string]$LoraCheckpoint = "",
    [string]$LoraName = "",
    [string]$AudiocppVersion = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot ".." "..")
if (-not $AudiocppDir) {
    $AudiocppDir = Join-Path $ProjectRoot "integrations" "audiocpp" "bin"
}

$VenvPython = Join-Path $ProjectRoot ".venv" "Scripts" "python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Error "未找到 .venv\Scripts\python.exe。请先运行 scripts\windows\setup.ps1 安装环境。"
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " IndexTTS2 + audio.cpp 推理环境初始化" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "模型目录：  $ModelDir"
Write-Host "二进制目录：$AudiocppDir"
Write-Host "转换精度：  $Dtype"
Write-Host "Q8_0 量化：$(if ($Quantize) { '是' } else { '否' })"
if ($LoraCheckpoint) { Write-Host "LoRA：      $LoraCheckpoint" }
Write-Host ""

# ── 步骤 1: 下载 audio.cpp 二进制 ────────────────────────────────────────────
Write-Host "[1/4] 下载 audio.cpp 预编译二进制..." -ForegroundColor Yellow

$DownloadScript = Join-Path $ProjectRoot "integrations" "audiocpp" "scripts" "download_audiocpp.ps1"
$downloadArgs = @("-OutputDir", $AudiocppDir)
if ($AudiocppVersion) { $downloadArgs += @("-Version", $AudiocppVersion) }
& $DownloadScript @downloadArgs

# ── 步骤 2: 基础模型 → GGUF 转换 ─────────────────────────────────────────────
$WeightsDir = Join-Path $ProjectRoot "integrations" "audiocpp" "weights"
$BaseGguf   = Join-Path $WeightsDir "base.gguf"
$BaseQ8Gguf = Join-Path $WeightsDir "base-q8_0.gguf"

Write-Host ""
Write-Host "[2/4] 转换 IndexTTS2 基础权重 → GGUF ($Dtype)..." -ForegroundColor Yellow

New-Item -ItemType Directory -Force -Path $WeightsDir | Out-Null

& $VenvPython (Join-Path $ProjectRoot "scripts" "convert_to_gguf.py") `
    --model-dir $ModelDir `
    --audiocpp-dir $AudiocppDir `
    --output $BaseGguf `
    --dtype $Dtype

if ($LASTEXITCODE -ne 0) {
    Write-Error "GGUF 转换失败（退出码 $LASTEXITCODE）"
}
Write-Host "  ✓ 基础 GGUF 已生成：$BaseGguf" -ForegroundColor Green

# ── 步骤 3: Q8_0 量化（可选）────────────────────────────────────────────────
if ($Quantize -and $Dtype -ne "q8_0") {
    Write-Host ""
    Write-Host "[3/4] 量化为 Q8_0（约减少 50% 体积）..." -ForegroundColor Yellow

    $GgufExe = Join-Path $AudiocppDir "audiocpp_gguf.exe"
    if (Test-Path $GgufExe) {
        & $VenvPython (Join-Path $ProjectRoot "scripts" "convert_to_gguf.py") `
            --model-dir $ModelDir `
            --audiocpp-dir $AudiocppDir `
            --output $BaseQ8Gguf `
            --dtype q8_0
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Q8_0 量化失败（退出码 $LASTEXITCODE），将使用 f16 版本。"
        } else {
            Write-Host "  ✓ Q8_0 GGUF 已生成：$BaseQ8Gguf" -ForegroundColor Green
        }
    } else {
        Write-Warning "  未找到 audiocpp_gguf.exe，跳过量化步骤。"
    }
} else {
    Write-Host ""
    Write-Host "[3/4] 跳过 Q8_0 量化（$( if ($Dtype -eq 'q8_0') { '已在转换时指定 q8_0' } else { '未启用' } )）" -ForegroundColor Gray
}

# ── 步骤 4: LoRA 转换（可选）────────────────────────────────────────────────
if ($LoraCheckpoint) {
    Write-Host ""
    Write-Host "[4/4] 生成运行时 LoRA adapter..." -ForegroundColor Yellow

    if (-not (Test-Path $LoraCheckpoint)) {
        Write-Warning "LoRA checkpoint 不存在：$LoraCheckpoint，跳过 LoRA 转换"
    } else {
        $VoicesDir = Join-Path $WeightsDir "voices"
        New-Item -ItemType Directory -Force -Path $VoicesDir | Out-Null

        $LoraFileName = if ($LoraName) { $LoraName } else {
            [System.IO.Path]::GetFileNameWithoutExtension($LoraCheckpoint)
        }
        $LoraAdapter = Join-Path $VoicesDir "$LoraFileName.safetensors"
        & $VenvPython (Join-Path $ProjectRoot "scripts" "convert_lora_adapter.py") `
            --checkpoint $LoraCheckpoint `
            --output $LoraAdapter

        if ($LASTEXITCODE -ne 0) {
            Write-Error "LoRA adapter 转换失败，请检查 checkpoint 与基础模型是否匹配。"
        } else {
            Write-Host "  ✓ 运行时 LoRA adapter：$LoraAdapter" -ForegroundColor Green
        }
    }
} else {
    Write-Host ""
    Write-Host "[4/4] 未指定 LoRA checkpoint，跳过 LoRA 转换。" -ForegroundColor Gray
}

# ── 汇总 ─────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " 初始化完成！" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "已生成的文件："
Get-ChildItem -Path $WeightsDir -Recurse -Filter "*.gguf" -ErrorAction SilentlyContinue |
    ForEach-Object { Write-Host "  $($_.FullName)  ($([math]::Round($_.Length/1GB, 2)) GB)" }
Write-Host ""
Write-Host "后续步骤："
$MainGguf = if (Test-Path $BaseQ8Gguf) { $BaseQ8Gguf } else { $BaseGguf }
Write-Host "  1. 快速验证（CLI）："
Write-Host "     .\integrations\audiocpp\bin\audiocpp_cli.exe ``"
Write-Host "         --task clon --family index_tts2 ``"
Write-Host "         --model $MainGguf ``"
Write-Host "         --voice-ref 参考音频.wav --text '你好' --out test.wav"
Write-Host ""
Write-Host "  2. 启动 WebUI（audio.cpp 后端）："
Write-Host "     .\scripts\windows\webui_audiocpp.ps1 -Model $MainGguf"
