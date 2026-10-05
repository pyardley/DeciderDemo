# Download the Jev Arena pin of Decider 4B · v2 and open the local demo.
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$Pin = Get-Content -Raw (Join-Path $Root "demo\model.json") | ConvertFrom-Json
$ModelDir = Join-Path $Root "models\decider-4b-v2"
$Python = Join-Path $Root ".venv\Scripts\python.exe"

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
    Write-Host "Downloading $($Pin.repo) @ $($Pin.revision) (about 8.4 GB)..."
    & $Python -c @"
from huggingface_hub import snapshot_download
snapshot_download('$($Pin.repo)', revision='$($Pin.revision)', local_dir=r'$ModelDir')
print('downloaded')
"@
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
