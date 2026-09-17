param(
    [string]$ProcessedDir = "processed_data/ljspeech",
    [int]$BatchSize = 2,
    [int]$GradAccumulation = 8,
    [int]$Epochs = 10
)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
& .\.venv\Scripts\python.exe trainers\train_gpt_v2_lora.py `
    --train-manifest "$ProcessedDir\gpt_pairs_train.jsonl" `
    --val-manifest "$ProcessedDir\gpt_pairs_val.jsonl" `
    --tokenizer checkpoints\bpe.model `
    --config checkpoints\config.yaml `
    --base-checkpoint checkpoints\gpt.pth `
    --output-dir trained_ckpts `
    --batch-size $BatchSize `
    --grad-accumulation $GradAccumulation `
    --epochs $Epochs `
    --learning-rate 1e-4 `
    --num-workers 0 `
    --amp
if ($LASTEXITCODE -ne 0) { throw "Training failed." }
