$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
& .\.venv\Scripts\python.exe -m alignlab smoke
if ($LASTEXITCODE -ne 0) { throw 'Smoke pipeline failed' }
& .\.venv\Scripts\python.exe -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
