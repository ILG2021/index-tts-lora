param(
    [Parameter(Mandatory=$true)][string]$DatasetDir,
    [string]$Output = "datasets/ljspeech.jsonl",
    [string]$ProcessedDir = "processed_data/ljspeech"
)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
& .\.venv\Scripts\python.exe tools\prepare_ljspeech.py --dataset-dir $DatasetDir --output $Output
if ($LASTEXITCODE -ne 0) { throw "LJSpeech conversion failed; extraction was not started." }
& .\.venv\Scripts\python.exe tools\preprocess_data_v2.py --manifest $Output --output-dir $ProcessedDir --tokenizer checkpoints\bpe.model --config checkpoints\config.yaml --gpt-checkpoint checkpoints\gpt.pth --language en --device cuda --workers 0
if ($LASTEXITCODE -ne 0) { throw "IndexTTS2 feature extraction failed." }
& .\.venv\Scripts\python.exe tools\build_gpt_prompt_pairs.py --manifest "$ProcessedDir\train_manifest.jsonl" --output "$ProcessedDir\gpt_pairs_train.jsonl"
if ($LASTEXITCODE -ne 0) { throw "Training pair generation failed." }
& .\.venv\Scripts\python.exe tools\build_gpt_prompt_pairs.py --manifest "$ProcessedDir\val_manifest.jsonl" --output "$ProcessedDir\gpt_pairs_val.jsonl"
if ($LASTEXITCODE -ne 0) { throw "Validation pair generation failed." }
