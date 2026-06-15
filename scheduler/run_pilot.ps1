# Portfolio Pilot scheduled runner.
# Snapshots the portfolio to SQLite history and writes a dated AI analyst note.
# Self-locating: derives the project root from this script's own folder, so it
# works regardless of where the repo lives. Registered via register_task.ps1.

$ErrorActionPreference = "Stop"
$proj = Split-Path -Parent $PSScriptRoot     # parent of scheduler\ = project root
Set-Location $proj

$python = Join-Path $proj ".venv\Scripts\python.exe"
$holdings = Join-Path $proj "data\holdings.csv"
$notesDir = Join-Path $proj "notes"
New-Item -ItemType Directory -Force -Path $notesDir | Out-Null

$stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
$log = Join-Path $notesDir "note_$stamp.txt"

# UTF-8 so the note/symbols render in the log file.
$env:PYTHONIOENCODING = "utf-8"

& $python -m src.cli --holdings $holdings --snapshot --note *>&1 |
    Tee-Object -FilePath $log

Write-Output "Wrote $log"
