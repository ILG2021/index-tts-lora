$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
if (-not (Test-Path ".venv\Scripts\indextts2.exe")) { throw "Run scripts\windows\setup.ps1 first." }
& .\.venv\Scripts\indextts2.exe download --source huggingface --model-dir checkpoints
if ($LASTEXITCODE -ne 0) { throw "IndexTTS2 model download failed." }
$required = @("config.yaml", "bpe.model", "gpt.pth", "s2mel.pth", "wav2vec2bert_stats.pt", "feat1.pt", "feat2.pt")
$missing = @($required | Where-Object { -not (Test-Path (Join-Path "checkpoints" $_)) })
if ($missing.Count -gt 0) { throw "Model download incomplete. Missing: $($missing -join ', ')" }
Write-Host "IndexTTS2 models are ready in checkpoints\."
