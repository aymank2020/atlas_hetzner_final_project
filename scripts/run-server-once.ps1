# run-server-once.ps1 - Single production run on server (headless).
# Usage: .\scripts\run-server-once.ps1
# Optional: .\scripts\run-server-once.ps1 --config configs/config_hetzner_production.yaml

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$Timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
Write-Host "[run-server] Starting production solver at $Timestamp" -ForegroundColor Cyan

# Activate venv if available
$VenvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    & $VenvActivate
    Write-Host "[run-server] Activated .venv" -ForegroundColor Green
}

# Load .env if present
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), "Process")
        }
    }
    Write-Host "[run-server] Loaded .env" -ForegroundColor Green
}

# Default to server config unless overridden
$ConfigPath = "configs/config_hetzner_production.yaml"
if ($args -contains "--config") {
    # Let the user override; pass all args through
} else {
    $args = @("--config", $ConfigPath) + $args
}

# Ensure logs directory exists
if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }

$LogFile = "logs/solver_${Timestamp}.log"
Write-Host "[run-server] Logging to $LogFile" -ForegroundColor Green

# Run with unbuffered output, tee to log file
python -u atlas_web_auto_solver.py @args 2>&1 | Tee-Object -FilePath $LogFile

$exitCode = $LASTEXITCODE
Write-Host "[run-server] Solver exited with code $exitCode at $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor $(if ($exitCode -eq 0) {"Green"} else {"Yellow"})
exit $exitCode
