# Decider 4B · v2 demo

A one-page desk for [Mapika's Decider 4B](https://huggingface.co/Mapika/decider-4b), revision `v2`. You paste a support ticket. The model returns a probability for every option on three questions: which team, whether a refund was asked for, and how urgent it is. It does not write a reply.

[Jev Arena](https://github.com/theaiautomators/jev-arena) evaluates this checkpoint under the name **Decider 4B · v2**. The pin is `49564ddcfccafb6db563eb757c1d41e6c78dcb56` in `arena/registry.py`. This demo loads that same commit. Decider is a one-pass decision model. Its weights are not in the Ollama library, so this demo serves them with Decider's own Python runtime.

## Requirements

- Windows with PowerShell
- [uv](https://docs.astral.sh/uv/) on `PATH`
- About 12 GB of free disk space (8.4 GB of weights, plus the Python environment)
- 32 GB of RAM if you want the model, Windows, and a browser resident together

The published weights are 8.4 GB in bf16. The demo process holds about 9.4 GB once they are loaded. A 4 GB GPU cannot hold them, so the demo runs on CPU in bf16. The library's CPU default is float32, about 17 GB, which does not fit in 16 GB of RAM. On a 16 GB machine the weights spill into the page file and a decision can take a few minutes.

## Install uv

In PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen the terminal so `uv` is on `PATH`. Check with:

```powershell
uv --version
```

## Install and start

From this folder:

```powershell
.\Start-Demo.ps1
```

The script does four things:

1. Creates `.venv` with Python 3.12, downloading that interpreter through uv if it is not already installed.
2. Installs the CPU build of PyTorch from `https://download.pytorch.org/whl/cpu`, then `decider-ai[serve]`.
3. Downloads `Mapika/decider-4b` at revision `49564ddcfccafb6db563eb757c1d41e6c78dcb56` into `models/decider-4b-v2` (about 8.4 GB). A later run skips this when `model.safetensors` is already there.
4. Starts the demo on http://127.0.0.1:8787 and opens that address. The first start spends a minute or two loading the weights. The page stays on **Loading weights…** until the model is ready.

Leave that PowerShell window open. Closing it stops the server.

## Install by hand

Use this when you want to run the same steps yourself.

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv\Scripts\python.exe "decider-ai[serve]"

.\.venv\Scripts\python.exe -c @"
from huggingface_hub import snapshot_download
snapshot_download(
    'Mapika/decider-4b',
    revision='49564ddcfccafb6db563eb757c1d41e6c78dcb56',
    local_dir=r'models\decider-4b-v2',
)
"@

.\.venv\Scripts\python.exe demo\server.py
```

Then open http://127.0.0.1:8787.

## Use the demo

Pick **Duplicate charge**, **App crash**, or **Locked out**, or edit the ticket text. **Ask Decider** sends the ticket and the refund policy as the state, and asks three questions:

| Question | Type | Options |
| --- | --- | --- |
| Which team should handle this? | choice | billing, technical, account |
| Is the customer asking for a refund? | yes/no | yes, no |
| How urgently does a person need to step in? | score | can wait, reply today, drop everything |

Each bar is the probability of an option you defined. Switching tickets clears the previous answers.
