# Registers (or updates) the Windows Scheduled Task for Portfolio Pilot.
# Default: every Monday 08:00. Edit $Schedule below or pass schtasks flags to change.
# Run this once from PowerShell:  powershell -ExecutionPolicy Bypass -File scheduler\register_task.ps1
# Remove later with:              schtasks /Delete /TN "PortfolioPilot" /F

$runner = Join-Path $PSScriptRoot "run_pilot.ps1"
$action = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$runner`""

schtasks /Create /TN "PortfolioPilot" /TR $action /SC WEEKLY /D MON /ST 08:00 /F
Write-Output "Task 'PortfolioPilot' registered: weekly, Monday 08:00."
Write-Output "Change cadence e.g. daily:  schtasks /Create /TN PortfolioPilot /TR '$action' /SC DAILY /ST 08:00 /F"
