# run-server-scheduled.ps1 - Continuous loop: run solver, pause, repeat.
# Usage: .\scripts\run-server-scheduled.ps1
# Ctrl+C to stop gracefully between cycles.

param(
    [int]$PauseBetweenRunsSec = 120,
    [int]$MaxRuns = 0,           # 0 = unlimited
    [string]$Config = "configs/config_hetzner_production.yaml"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

# Activate venv if available
$VenvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    & $VenvActivate
}

# Load .env if present
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), "Process")
        }
    }
}

if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }

$runCount = 0
$consecutiveFailures = 0
$maxConsecutiveFailures = 5

Write-Host "[scheduler] Starting continuous solver loop" -ForegroundColor Cyan
Write-Host "[scheduler] Config: $Config | Pause: ${PauseBetweenRunsSec}s | MaxRuns: $(if ($MaxRuns -eq 0) {'unlimited'} else {$MaxRuns})" -ForegroundColor Cyan

while ($true) {
    $runCount++
    $Timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
    $LogFile = "logs/solver_${Timestamp}.log"

    Write-Host "`n[scheduler] === Run #$runCount at $Timestamp ===" -ForegroundColor Cyan

    python -u atlas_web_auto_solver.py --config $Config 2>&1 | Tee-Object -FilePath $LogFile
    $exitCode = $LASTEXITCODE

    if ($exitCode -eq 0) {
        $consecutiveFailures = 0
        Write-Host "[scheduler] Run #$runCount completed successfully" -ForegroundColor Green
    } else {
        $consecutiveFailures++
        Write-Host "[scheduler] Run #$runCount failed (exit $exitCode, consecutive failures: $consecutiveFailures)" -ForegroundColor Yellow
        if ($consecutiveFailures -ge $maxConsecutiveFailures) {
            Write-Host "[scheduler] Too many consecutive failures ($maxConsecutiveFailures). Stopping." -ForegroundColor Red
            exit 1
        }
    }

    if ($MaxRuns -gt 0 -and $runCount -ge $MaxRuns) {
        Write-Host "[scheduler] Reached max runs ($MaxRuns). Stopping." -ForegroundColor Green
        exit 0
    }

    # Backoff on failure
    $pause = if ($consecutiveFailures -gt 0) {
        [math]::Min($PauseBetweenRunsSec * [math]::Pow(1.5, $consecutiveFailures), 600)
    } else {
        $PauseBetweenRunsSec
    }

    Write-Host "[scheduler] Pausing ${pause}s before next run..." -ForegroundColor DarkGray
    Start-Sleep -Seconds $pause
}
