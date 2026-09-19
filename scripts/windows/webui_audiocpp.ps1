#!/usr/bin/env pwsh
<#
.SYNOPSIS
    使用 audio.cpp 后端启动 IndexTTS2 LoRA WebUI。

.DESCRIPTION
    启动 audiocpp_server（C++ 推理进程）并挂载 Gradio WebUI 前端。
    参考 MOSS-TTS openmoss 部署模式。

    典型用法：
      - 基础模型：.\scripts\windows\webui_audiocpp.ps1
      - 带运行时 LoRA：  .\scripts\windows\webui_audiocpp.ps1 -Lora "speaker-a=weights/voices/speaker-a.safetensors"
      - 多 LoRA：  .\scripts\windows\webui_audiocpp.ps1 -Lora @("a=pa.gguf", "b=pb.gguf")

.PARAMETER Model
    GGUF 权重文件路径，默认自动选取 integrations\audiocpp\weights\ 下
    的 Q8_0 版本（若存在），否则用 f16 版本

.PARAMETER Lora
    运行时 LoRA safetensors，格式 "name=path"，可传多个（数组）

.PARAMETER AudiocppExe
    audiocpp_server 可执行文件路径

.PARAMETER AudiocppBackend
    ggml 计算后端：cuda | cpu | vulkan | metal | hip，默认 cuda

.PARAMETER AudiocppPort
    audiocpp_server 监听端口，默认 8080

.PARAMETER Host
    Gradio WebUI 监听地址，默认 127.0.0.1

.PARAMETER Port
    Gradio WebUI 监听端口，默认 7860

.PARAMETER RootPath
    反向代理子路径（Gradio root_path 参数）

.PARAMETER Preload
    是否在 WebUI 启动时立即加载模型（默认首次推理时懒加载）

.EXAMPLE
    .\scripts\windows\webui_audiocpp.ps1

.EXAMPLE
    .\scripts\windows\webui_audiocpp.ps1 `
        -Model integrations\audiocpp\weights\base-q8_0.gguf `
        -Lora "speaker-a=integrations\audiocpp\weights\voices\speaker-a.safetensors"

.EXAMPLE
    .\scripts\windows\webui_audiocpp.ps1 -RootPath "/indextts2"
#>

param(
    [string]$Model = "",
    [string[]]$Lora = @(),
    [string]$AudiocppExe = "",
    [string]$AudiocppBackend = "cuda",
    [int]$AudiocppPort = 8080,
    [string]$Host = "127.0.0.1",
    [int]$Port = 7860,
    [string]$RootPath = "",
    [switch]$Preload
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot ".." "..")
$VenvPython  = Join-Path $ProjectRoot ".venv" "Scripts" "python.exe"
$WeightsDir  = Join-Path $ProjectRoot "integrations" "audiocpp" "weights"
$BinDir      = Join-Path $ProjectRoot "integrations" "audiocpp" "bin"

# ── 自动选择可执行文件 ────────────────────────────────────────────────────────
if (-not $AudiocppExe) {
    $AudiocppExe = Join-Path $BinDir "audiocpp_server.exe"
}

# ── 自动选择模型权重（Q8_0 优先）────────────────────────────────────────────
if (-not $Model) {
    $q8 = Join-Path $WeightsDir "base-q8_0.gguf"
    $f16 = Join-Path $WeightsDir "base.gguf"
    if (Test-Path $q8) {
        $Model = $q8
        Write-Host "自动选择 Q8_0 权重：$q8" -ForegroundColor Gray
    } elseif (Test-Path $f16) {
        $Model = $f16
        Write-Host "自动选择 f16 权重：$f16" -ForegroundColor Gray
    } else {
        Write-Error (
            "未找到 GGUF 权重文件。请先运行：`n" +
            "  .\scripts\windows\setup_audiocpp.ps1"
        )
    }
}

# ── 校验 ─────────────────────────────────────────────────────────────────────
if (-not (Test-Path $VenvPython)) {
    Write-Error "未找到 .venv\Scripts\python.exe。请先运行 scripts\windows\setup.ps1"
}
if (-not (Test-Path $AudiocppExe)) {
    Write-Error (
        "未找到 audiocpp_server.exe：$AudiocppExe`n" +
        "请先运行 .\scripts\windows\setup_audiocpp.ps1"
    )
}
if (-not (Test-Path $Model)) {
    Write-Error "GGUF 权重不存在：$Model"
}

# ── 显示启动信息 ──────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " IndexTTS2 WebUI — audio.cpp 后端" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  模型       ：$Model"
Write-Host "  C++ 后端   ：$AudiocppBackend"
Write-Host "  Server 端口：$AudiocppPort"
Write-Host "  WebUI 地址 ：http://${Host}:${Port}"
if ($Lora) {
    Write-Host "  LoRA       ："
    $Lora | ForEach-Object { Write-Host "    - $_" }
}
Write-Host ""

# ── 构造 python webui.py 参数 ─────────────────────────────────────────────────
$pyArgs = @(
    (Join-Path $ProjectRoot "webui.py"),
    "--backend", "audiocpp",
    "--audiocpp-exe", $AudiocppExe,
    "--model", $Model,
    "--audiocpp-backend", $AudiocppBackend,
    "--audiocpp-port", $AudiocppPort,
    "--host", $Host,
    "--port", $Port
)

foreach ($l in $Lora) {
    $pyArgs += @("--lora", $l)
}

if ($RootPath) {
    $pyArgs += @("--root-path", $RootPath)
}

if ($Preload) {
    $pyArgs += "--preload"
}

Write-Host "启动命令：" -ForegroundColor Gray
Write-Host "  $VenvPython $($pyArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

# ── 启动 ─────────────────────────────────────────────────────────────────────
& $VenvPython @pyArgs
