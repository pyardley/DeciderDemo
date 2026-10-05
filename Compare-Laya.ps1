# Score Laya's 50 labelled tickets with each Decider checkpoint.
# Close Cursor first. The 4B model can take a couple of hours for all 50.
#   .\Compare-Laya.ps1 -Laya C:\Users\PaulYardley\PycharmProjects\laya-ollama-demo
#   .\Compare-Laya.ps1 -Laya C:\Users\PaulYardley\PycharmProjects\laya-ollama-demo -Models 0.8b,2b
param(
    [Parameter(Mandatory = $true)]
    [string]$Laya,
    [string]$Models
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "The Python environment is missing. Run .\Start-Demo.ps1 once, then run this script."
}
if (-not (Test-Path $Laya)) {
    throw "Laya project not found: $Laya"
}

$Arguments = @("bench\compare_laya.py", "--laya", $Laya)
if ($Models) {
    $Arguments += @("--models", $Models)
}
& $Python @Arguments
exit $LASTEXITCODE
