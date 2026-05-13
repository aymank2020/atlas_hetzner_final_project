# run-local.ps1 - Single-episode dry-run on local Windows machine.
# Usage: .\scripts\run-local.ps1
# Requires: Python 3.10+, Playwright browsers installed, .env populated.

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

Write-Host "[run-local] Starting dry-run solver from: $ProjectRoot" -ForegroundColor Cyan

# Activate venv if available
$VenvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    & $VenvActivate
    Write-Host "[run-local] Activated .venv" -ForegroundColor Green
}

# Load .env if present
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), "Process")
        }
    }
    Write-Host "[run-local] Loaded .env" -ForegroundColor Green
}

# Unbuffered mode for live log output
python -u atlas_web_auto_solver.py --config configs/config_local_dev.yaml @args

$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Host "[run-local] Solver exited with code $exitCode" -ForegroundColor Yellow
}
exit $exitCode
