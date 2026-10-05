# Decider demo

A one-page desk for [Mapika's Decider](https://huggingface.co/Mapika) models. You paste a support ticket. The model returns a probability for every option on three questions: which team, whether a refund was asked for, and how urgent it is. It does not write a reply.

The demo starts on **Decider 4B · v2**, the checkpoint [Jev Arena](https://github.com/theaiautomators/jev-arena) evaluates under that name (`49564ddcfccafb6db563eb757c1d41e6c78dcb56` in `arena/registry.py`). Decider is a one-pass decision model. Its weights are not in the Ollama library, so this demo serves them with Decider's own Python runtime.

## Requirements

- Windows with PowerShell
- [uv](https://docs.astral.sh/uv/) on `PATH`
- Free disk for the checkpoint you select: 8.4 GB for 4B, 3.8 GB for 2B, or 1.4 GB for 0.8B, plus about 2 GB for the Python environment
- For the 4B checkpoint in float32, 32 GB of RAM. This demo loads that checkpoint in bf16 instead, about 9.4 GB, which still pages on a 16 GB machine

A 4 GB GPU cannot hold the 4B or 2B weights. The 0.8B file is 1.4 GB. This install uses the CPU build of PyTorch, so every model here runs on CPU. The library's fast CPU dtype is float32. The 2B and 0.8B entries use it. The 4B entry uses bf16 because the float32 copy is about 17 GB and does not fit in 16 GB of RAM.

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
3. Downloads the active checkpoint from `demo/models.json` into `models/`. A later run skips this when that model's `model.safetensors` is already there.
4. Starts the demo on http://127.0.0.1:8787 and opens that address. The page stays on **Loading weights…** until the model is ready.

Leave that PowerShell window open. Closing it stops the server.

## Change the model

The catalog is `demo/models.json`. `active` selects the checkpoint. The ids are `4b-v2`, `2b`, and `0.8b`.

Stop the running demo with Ctrl+C, then start the one you want:

```powershell
.\Start-Demo.ps1 -Model 2b
.\Start-Demo.ps1 -Model 0.8b
.\Start-Demo.ps1 -Model 4b-v2
```

`-Model` writes that id into `active` and downloads the weights if they are not already in `models/`. You can edit `active` yourself instead; the next `.\Start-Demo.ps1` uses whatever is there.

Each entry sets `repo`, `revision`, `directory`, `device`, and `dtype`. `device` stays `cpu` with the CPU build of PyTorch installed by this demo. `dtype` is `float32` for 2B and 0.8B, and `bfloat16` for 4B.

## Install by hand

Use this when you want to run the same steps yourself. The revision and folder below are the default 4B pin. For another model, copy `repo`, `revision`, and `directory` from `demo/models.json`.

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

## Models

Figures are from the publishers' model cards, measured on their hardware. GPU times are on an NVIDIA B300. They are not times for this laptop.

[Decider 2B v11](https://huggingface.co/Mapika/decider-2b) and [Decider 4B v2](https://huggingface.co/Mapika/decider-4b) share one rebuilt regression set: 67 in-task tasks and 28 held-out tasks. [Decider 0.8B](https://huggingface.co/Mapika/decider-0.8b) was scored on an earlier 93-task protocol (69 in-task, 24 held-out). On that older protocol the 2B scored 0.809 in-task and 0.739 held-out, against the 0.8B's 0.776 and 0.707.

| | Decider 0.8B | Decider 2B v11 | Decider 4B · v2 |
| --- | --- | --- | --- |
| Catalog id | `0.8b` | `2b` | `4b-v2` |
| Weights | 1.4 GB bf16 | 3.8 GB bf16 | 8.4 GB bf16 |
| Dtype in this demo | float32, about 3 GB | float32, about 8 GB | bf16, about 9.4 GB |
| Regression accuracy, in-task / held-out | 0.776 / 0.707 | 0.802 / 0.752 | 0.824 / 0.779 |
| Regression ECE, in-task / held-out | 0.032 / 0.096 | 0.038 / 0.083 | 0.041 / 0.080 |
| JevBench public, easy / standard / hard | — | 1.000 / 0.889 / 0.577 | 1.000 / 0.986 / 0.676 |
| JevBench hard-tier calibration error | — | 0.175 | 0.071 |
| Bespoke public suite, macro / micro | — | 0.706 / 0.711 | 0.773 / 0.781 |
| Live MiniWoB++, sampled, 22 tasks | — | 90.3% | 88.1% |
| Publisher GPU time | about 1.5× the 2B on its regression run | 4 ms with CUDA graphs | 5.2 ms with CUDA graphs, 35 ms eager |

The 0.8B card does not publish JevBench, Bespoke, or MiniWoB numbers. On the tasks it does share with the 2B, short routing, yes/no, and JSON lookups stay within about one to four points, and the larger gaps are knowledge questions such as ARC, OpenBookQA, and HellaSwag.

On this 16 GB laptop the 4B checkpoint in bf16 does not stay fully resident, and one ticket took about three minutes. The 2B and 0.8B entries are the ones that fit in RAM in float32, which is the library's fast CPU path.

## Benchmark the demo tickets

`Run-Bench.ps1` loads each catalog model in its own process, asks the same three tickets the page asks, and appends one JSON object per line to `results/bench-<timestamp>.jsonl`. Close Cursor and other large apps first so the process can keep the weights in RAM. The 4B checkpoint can take several minutes per ticket on this machine. The 0.8B and 2B models run first. Ctrl+C keeps every row already written.

Download any missing checkpoint before the run (`.\Start-Demo.ps1 -Model 2b`, and the same for `0.8b` or `4b-v2`). Then:

```powershell
.\Run-Bench.ps1
.\Run-Bench.ps1 -Models 0.8b,2b
```

The file has no gold labels. Each `decision` row stores the choice, the probabilities, `usage.input_tokens`, `request_ms`, and the process memory. A `model` row stores load time, dtype, and temperature. Bring `results/bench-*.jsonl` back and ask for a report from that file.

## Use the demo

Pick **Duplicate charge**, **App crash**, or **Locked out**, or edit the ticket text. **Ask Decider** sends the ticket and the refund policy as the state, and asks three questions:

| Question | Type | Options |
| --- | --- | --- |
| Which team should handle this? | choice | billing, technical, account |
| Is the customer asking for a refund? | yes/no | yes, no |
| How urgently does a person need to step in? | score | can wait, reply today, drop everything |

Each bar is the probability of an option you defined. Switching tickets clears the previous answers.
