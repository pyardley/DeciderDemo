# Download the active Decider checkpoint and open the local demo.
# .\Start-Demo.ps1
# .\Start-Demo.ps1 -Model 2b
param(
    [string]$Model
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$CatalogPath = Join-Path $Root "demo\models.json"
$Catalog = Get-Content -Raw $CatalogPath | ConvertFrom-Json
$Known = @($Catalog.models.PSObject.Properties.Name)
if ($Model) {
    if ($Known -notcontains $Model) {
        throw "Unknown model '$Model'. Choose one of: $($Known -join ', ')"
    }
    $Updated = [System.IO.File]::ReadAllText($CatalogPath) -replace '"active"\s*:\s*"[^"]*"', ('"active": "' + $Model + '"')
    [System.IO.File]::WriteAllText($CatalogPath, $Updated)
    $Catalog = $Updated | ConvertFrom-Json
}

$ActiveId = $Catalog.active
$Pin = $Catalog.models.$ActiveId
if (-not $Pin) {
    throw "demo/models.json has no entry for active model '$ActiveId'."
}
$ModelDir = Join-Path $Root ("models\" + $Pin.directory)
$Python = Join-Path $Root ".venv\Scripts\python.exe"
Write-Host "Model: $($Pin.name) ($ActiveId)"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not on PATH. Install it from https://docs.astral.sh/uv/ and run this script again."
}

if (-not (Test-Path $Python)) {
    Write-Host "Creating a Python 3.12 environment..."
    uv venv --python 3.12 .venv
}

Write-Host "Installing the CPU build of PyTorch, then Decider..."
uv pip install --python $Python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python $Python "decider-ai[serve]"

if (-not (Test-Path (Join-Path $ModelDir "model.safetensors"))) {
    Write-Host "Downloading $($Pin.repo) @ $($Pin.revision)..."
    & $Python -c @"
from huggingface_hub import snapshot_download
snapshot_download('$($Pin.repo)', revision='$($Pin.revision)', local_dir=r'$ModelDir')
print('downloaded')
"@
}

$Busy = $false
try {
    $Probe = New-Object System.Net.Sockets.TcpClient
    $Probe.Connect("127.0.0.1", 8787)
    $Probe.Close()
    $Busy = $true
} catch {
    $Busy = $false
}
if ($Busy) {
    throw "Port 8787 is already in use. Stop the running demo with Ctrl+C in its window, then run this script again."
}

$Server = Start-Process -FilePath $Python -ArgumentList (Join-Path $Root "demo\server.py") -WorkingDirectory $Root -PassThru -NoNewWindow
$Deadline = (Get-Date).AddSeconds(30)
$Listening = $false
while ((Get-Date) -lt $Deadline) {
    try {
        $Client = New-Object System.Net.Sockets.TcpClient
        $Client.Connect("127.0.0.1", 8787)
        $Client.Close()
        $Listening = $true
        break
    } catch {
        Start-Sleep -Milliseconds 250
    }
}
if ($Listening) {
    Write-Host "Opening http://127.0.0.1:8787"
    Start-Process "http://127.0.0.1:8787"
} else {
    Write-Host "Demo server did not open port 8787. Check the log above."
}
if ($Server) { Wait-Process -Id $Server.Id }
