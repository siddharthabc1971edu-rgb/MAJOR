$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root

try {
    Write-Host "Checking dependency consistency..."
    python -m pip check

    Write-Host "Compiling sources..."
    python -B -m compileall -q app.py run_experiment.py src

    Write-Host "Running fast tests..."
    python -B -m pytest -p no:cacheprovider -m "not slow"

    Write-Host "Running slow integration test..."
    python -B -m pytest -p no:cacheprovider -m slow

    Write-Host "All checks passed."
}
finally {
    Pop-Location
}
