# setup-local.ps1 - One-time setup for a fresh Windows machine.
# Usage: Run in PowerShell: .\scripts\setup-local.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

Write-Host "=== Atlas Solver Windows Setup ===" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot" -ForegroundColor Cyan

# -------------------------------------------------------------------
# 1. Check Python
# -------------------------------------------------------------------
Write-Host "`n[1/6] Checking Python..." -ForegroundColor Yellow
try {
    $pyVersion = python --version 2>&1
    Write-Host "  Found: $pyVersion" -ForegroundColor Green
    $major, $minor = ($pyVersion -replace 'Python ', '').Split('.')
    if ([int]$major -lt 3 -or ([int]$major -eq 3 -and [int]$minor -lt 10)) {
        Write-Host "  WARNING: Python 3.10+ recommended. Found $pyVersion" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ERROR: Python not found. Install Python 3.10+ from https://python.org" -ForegroundColor Red
    exit 1
}

# -------------------------------------------------------------------
# 2. Create virtual environment
# -------------------------------------------------------------------
Write-Host "`n[2/6] Setting up virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "  Created .venv" -ForegroundColor Green
} else {
    Write-Host "  .venv already exists" -ForegroundColor DarkGray
}
# Activate venv
& ".\.venv\Scripts\Activate.ps1"
Write-Host "  Virtual environment activated" -ForegroundColor Green

# -------------------------------------------------------------------
# 3. Install Python packages
# -------------------------------------------------------------------
Write-Host "`n[3/6] Installing Python packages..." -ForegroundColor Yellow
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Write-Host "  Python packages installed." -ForegroundColor Green

# -------------------------------------------------------------------
# 4. Install Playwright browsers
# -------------------------------------------------------------------
Write-Host "`n[4/6] Installing Playwright browsers (Chromium)..." -ForegroundColor Yellow
python -m playwright install chromium
Write-Host "  Playwright Chromium installed." -ForegroundColor Green

# -------------------------------------------------------------------
# 5. Check ffmpeg
# -------------------------------------------------------------------
Write-Host "`n[5/6] Checking ffmpeg..." -ForegroundColor Yellow
try {
    $ffVersion = ffmpeg -version 2>&1 | Select-Object -First 1
    Write-Host "  Found: $ffVersion" -ForegroundColor Green
} catch {
    Write-Host "  WARNING: ffmpeg not found on PATH." -ForegroundColor Yellow
    Write-Host "  Video optimization will fall back to OpenCV (lower quality)." -ForegroundColor Yellow
    Write-Host "  Install from: https://ffmpeg.org/download.html" -ForegroundColor Yellow
    Write-Host "  Or via winget:  winget install Gyan.FFmpeg" -ForegroundColor Yellow
}

# -------------------------------------------------------------------
# 6. Create runtime directories + .env
# -------------------------------------------------------------------
Write-Host "`n[6/6] Creating runtime directories..." -ForegroundColor Yellow
$dirs = @(".state", "outputs", "logs")
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Path $d | Out-Null
        Write-Host "  Created: $d" -ForegroundColor Green
    } else {
        Write-Host "  Exists:  $d" -ForegroundColor DarkGray
    }
}

if (Test-Path ".env") {
    Write-Host "  .env found." -ForegroundColor Green
} else {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "  Created .env from .env.example. Edit it with your credentials." -ForegroundColor Yellow
    } else {
        Write-Host "  WARNING: No .env or .env.example found. Create .env with credentials." -ForegroundColor Yellow
    }
}

Write-Host "`n=== Setup Complete ===" -ForegroundColor Cyan
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Edit .env with your Atlas/Gemini/Gmail credentials" -ForegroundColor White
Write-Host "  2. Activate venv:   .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  3. Test dry-run:    .\scripts\run-local.ps1" -ForegroundColor White
Write-Host "  4. Production:      .\scripts\run-server-once.ps1" -ForegroundColor White
Write-Host "  5. Continuous:      .\scripts\run-server-scheduled.ps1" -ForegroundColor White
