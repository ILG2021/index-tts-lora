$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
if (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonCommand = "py"
    $PythonArgs = @("-3.10")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonCommand = "python"
    $PythonArgs = @()
} else {
    throw "Python 3.10 was not found. Install it and enable the Python launcher or PATH entry."
}
if (-not (Test-Path ".venv")) { & $PythonCommand @PythonArgs -m venv .venv }
if (-not (Test-Path ".venv\Scripts\python.exe")) { throw "Virtual environment creation failed." }
& .\.venv\Scripts\python.exe -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 12) else 1)"
if ($LASTEXITCODE -ne 0) { throw "IndexTTS2 requires Python 3.10 or 3.11." }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed." }
& .\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchaudio==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "PyTorch 2.8.0 CUDA 12.8 installation failed." }
New-Item -ItemType Directory -Force vendor | Out-Null
$IndexTtsCommit = "ee40fa7d6c6b8a2c7f06105f9f1e65775b74868c"
if (-not (Test-Path "vendor\index-tts\.git")) {
    git clone https://github.com/index-tts/index-tts.git vendor\index-tts
    if ($LASTEXITCODE -ne 0) { throw "Official IndexTTS2 source checkout failed." }
    git -C vendor\index-tts checkout $IndexTtsCommit
    if ($LASTEXITCODE -ne 0) { throw "Pinned IndexTTS2 revision checkout failed." }
}
$ActualIndexTtsCommit = git -C vendor\index-tts rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $ActualIndexTtsCommit.Trim() -ne $IndexTtsCommit) {
    throw "vendor\index-tts is not at the required revision $IndexTtsCommit."
}
& .\.venv\Scripts\python.exe -m pip install -e vendor\index-tts
if ($LASTEXITCODE -ne 0) { throw "IndexTTS2 installation failed." }
& .\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
if ($LASTEXITCODE -ne 0) { throw "LoRA and WebUI dependency installation failed." }
& .\.venv\Scripts\python.exe -m pip check
if ($LASTEXITCODE -ne 0) { throw "Installed dependencies are inconsistent." }
& .\.venv\Scripts\python.exe -c "import torch, torchaudio; assert torch.__version__.split('+')[0] == '2.8.0'; assert torchaudio.__version__.split('+')[0] == '2.8.0'; assert torch.version.cuda == '12.8'"
if ($LASTEXITCODE -ne 0) { throw "Expected PyTorch/torchaudio 2.8.0 with CUDA 12.8." }
Write-Host "Environment ready. Activate with: .\.venv\Scripts\Activate.ps1"
