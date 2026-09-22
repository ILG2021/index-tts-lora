param(
    [Parameter(Mandatory=$true)][string]$DatasetDir,
    [string]$Output = "datasets/ljspeech.jsonl",
    [string]$ProcessedDir = "processed_data/ljspeech_zh",
    [ValidateSet("zh", "en", "ja")][string]$Language = "zh",
    [switch]$SpeakerFromFolder
)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
Write-Host "Checking IndexTTS2 base and auxiliary models..."
& (Join-Path $PSScriptRoot "download_models.ps1")
if ($LASTEXITCODE -ne 0) { throw "Model check/download failed; preprocessing was not started." }
$PrepareArgs = @(
    "tools\prepare_ljspeech.py",
    "--dataset-dir", $DatasetDir,
    "--output", $Output,
    "--language", $Language
)
if ($SpeakerFromFolder) {
    $PrepareArgs += "--speaker-from-folder"
}
& .\.venv\Scripts\python.exe @PrepareArgs
if ($LASTEXITCODE -ne 0) { throw "LJSpeech conversion failed; extraction was not started." }
& .\.venv\Scripts\python.exe tools\preprocess_data_v2.py --manifest $Output --output-dir $ProcessedDir --tokenizer checkpoints\bpe.model --config checkpoints\config.yaml --gpt-checkpoint checkpoints\gpt.pth --language $Language --device cuda --workers 0
if ($LASTEXITCODE -ne 0) { throw "IndexTTS2 feature extraction failed." }
& .\.venv\Scripts\python.exe tools\build_gpt_prompt_pairs.py --manifest "$ProcessedDir\train_manifest.jsonl" --output "$ProcessedDir\gpt_pairs_train.jsonl"
if ($LASTEXITCODE -ne 0) { throw "Training pair generation failed." }
& .\.venv\Scripts\python.exe tools\build_gpt_prompt_pairs.py --manifest "$ProcessedDir\val_manifest.jsonl" --output "$ProcessedDir\gpt_pairs_val.jsonl"
if ($LASTEXITCODE -ne 0) { throw "Validation pair generation failed." }
