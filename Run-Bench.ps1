# Score the three demo tickets on each catalog model and write results\bench-*.jsonl.
# Close Cursor first so the checkpoints have the RAM. Then, from this folder:
#   .\Run-Bench.ps1
#   .\Run-Bench.ps1 -Models 0.8b,2b
param(
    [string]$Models
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "The Python environment is missing. Run .\Start-Demo.ps1 once, then run this script."
}

$Arguments = @("bench\run_bench.py")
if ($Models) {
    $Arguments += @("--models", $Models)
}
& $Python @Arguments
exit $LASTEXITCODE
