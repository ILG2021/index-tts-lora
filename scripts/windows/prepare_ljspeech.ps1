param(
    [Parameter(Mandatory=$true)][string]$DatasetDir,
    [string]$Output = "finetune_data/ljspeech.lst"
)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
& .\.venv\Scripts\python.exe tools\prepare_ljspeech.py --dataset-dir $DatasetDir --output $Output
if ($LASTEXITCODE -ne 0) { throw "LJSpeech conversion failed; extraction was not started." }
& .\.venv\Scripts\python.exe tools\extract_codec.py --audio_list $Output --extract_condition
if ($LASTEXITCODE -ne 0) { throw "Feature extraction failed." }
