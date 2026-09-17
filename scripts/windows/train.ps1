param(
    [string]$ProcessedDir = "processed_data/ljspeech_zh",
    [int]$BatchSize = 1,
    [int]$GradAccumulation = 16,
    [int]$Epochs = 10
)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
& .\.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; p=torch.cuda.get_device_properties(0); print(f'GPU: {p.name}; VRAM: {p.total_memory / 2**30:.1f} GiB; CUDA: {torch.version.cuda}')"
if ($LASTEXITCODE -ne 0) { throw "CUDA preflight failed. Check the NVIDIA driver and cu128 PyTorch installation." }
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
