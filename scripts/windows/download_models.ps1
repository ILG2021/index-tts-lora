$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
if (-not (Test-Path ".venv\Scripts\indextts2.exe")) { throw "Run scripts\windows\setup.ps1 first." }
& .\.venv\Scripts\indextts2.exe download --source huggingface --model-dir checkpoints
if ($LASTEXITCODE -ne 0) { throw "IndexTTS2 model download failed." }
$W2vDir = "checkpoints\hf_cache\w2v-bert-2.0"
$W2vWeightNames = @(
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json"
)
$W2vComplete = (Test-Path (Join-Path $W2vDir "config.json")) -and
    (@($W2vWeightNames | Where-Object { Test-Path (Join-Path $W2vDir $_) }).Count -gt 0)
if (-not $W2vComplete) {
    Write-Host "Wav2Vec2-BERT cache is incomplete; downloading missing files..."
    & .\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='facebook/w2v-bert-2.0', local_dir=r'checkpoints\hf_cache\w2v-bert-2.0')"
    if ($LASTEXITCODE -ne 0) { throw "Wav2Vec2-BERT download failed." }
}
$required = @("config.yaml", "bpe.model", "gpt.pth", "s2mel.pth", "wav2vec2bert_stats.pt", "feat1.pt", "feat2.pt")
$missing = @($required | Where-Object { -not (Test-Path (Join-Path "checkpoints" $_)) })
if ($missing.Count -gt 0) { throw "Model download incomplete. Missing: $($missing -join ', ')" }
$W2vComplete = (Test-Path (Join-Path $W2vDir "config.json")) -and
    (@($W2vWeightNames | Where-Object { Test-Path (Join-Path $W2vDir $_) }).Count -gt 0)
if (-not $W2vComplete) { throw "Wav2Vec2-BERT download is incomplete: $W2vDir" }
Write-Host "IndexTTS2 models are ready in checkpoints\."
