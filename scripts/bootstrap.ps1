param([string]$Python = '')
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectPath
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if ($Python) { & $Python -m venv .venv }
    else {
        $hasPython311 = $false
        try {
            & py -3.11 -c 'import sys; print(sys.version)' 2>$null
            $hasPython311 = ($LASTEXITCODE -eq 0)
        } catch { $hasPython311 = $false }
        if ($hasPython311) { & py -3.11 -m venv .venv }
        else { Write-Host 'Python 3.11 unavailable; using compatible 3.12'; & py -3.12 -m venv .venv }
    }
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed; install Python 3.11 or pass -Python path' }
}
& .\.venv\Scripts\python.exe -m pip install -r requirements/dev.lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& .\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'Package installation failed' }
& .\.venv\Scripts\python.exe -m alignlab --help
if ($LASTEXITCODE -ne 0) { throw 'CLI validation failed' }
& .\.venv\Scripts\python.exe scripts/bootstrap_sources.py
if ($LASTEXITCODE -ne 0) { throw 'Pinned upstream source bootstrap failed' }
