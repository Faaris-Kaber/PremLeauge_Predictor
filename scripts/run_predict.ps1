param(
  # NOTE: PowerShell variables are case-insensitive; $Home conflicts with the built-in $HOME (read-only).
  [Parameter(Mandatory=$true)][Alias("Home")][string]$HomeTeam,
  [Parameter(Mandatory=$true)][Alias("Away")][string]$AwayTeam,
  [string]$RunId = "run"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
  throw "Venv python not found at $py. Create it with: python -m venv .venv"
}

$env:EPL_RUN_ID = $RunId

Write-Host "Using interpreter: $py"
Write-Host "RunId: $env:EPL_RUN_ID"

& $py -m epl_predictor.predict --home $HomeTeam --away $AwayTeam


