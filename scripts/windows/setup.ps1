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
& .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed." }
& .\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchaudio==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "PyTorch 2.8.0 CUDA 12.8 installation failed." }
& .\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
& .\.venv\Scripts\python.exe -m pip install -e . --no-build-isolation
if ($LASTEXITCODE -ne 0) { throw "Project installation failed." }
Write-Host "Environment ready. Activate with: .\.venv\Scripts\Activate.ps1"
