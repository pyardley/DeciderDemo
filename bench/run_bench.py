"""Run the demo tickets on every catalog model and append a JSONL record file.

One process loads one checkpoint, scores the tickets, then exits so the next
model gets the RAM back. The file is flushed after every record, so a stopped
run still leaves the rows that finished.

    .\\.venv\\Scripts\\python.exe bench\\run_bench.py
    .\\.venv\\Scripts\\python.exe bench\\run_bench.py --models 0.8b,2b
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "demo" / "models.json"
RESULTS = ROOT / "results"

POLICY = (
    "Duplicate charges are eligible for a refund. Delivery delays are not refundable; "
    "send a status update. Login lockouts go to account access, not billing."
)

SCENARIOS = [
    {
        "id": "duplicate",
        "label": "Duplicate charge",
        "ticket": (
            "I was charged twice for order A-104. Please refund the duplicate. "
            "The second charge posted this morning and I need it back before rent is due."
        ),
    },
    {
        "id": "crash",
        "label": "App crash",
        "ticket": (
            "The iOS app crashes every time I open a PDF attachment. This started after "
            "yesterday's update. Android still works. I do not need a refund."
        ),
    },
    {
        "id": "lockout",
        "label": "Locked out",
        "ticket": (
            "I cannot sign in. It says the account is locked after too many attempts. "
            "I have a meeting in an hour and need the shared drive. Please do not refund anything."
        ),
    },
]

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "Charges, invoices, refunds and duplicate payments",
            "technical": "Bugs, crashes and outages",
            "account": "Login, lockouts and access",
        },
    },
    "refund": {
        "type": "noul",
        "instructions": "Is the customer asking for a refund?",
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgently does a person need to step in?",
        "criteria": ["can wait", "reply today", "drop everything"],
    },
}

# Smallest first, so a stopped run still has the models that fit in 16 GB.
DEFAULT_ORDER = ("0.8b", "2b", "4b-v2")


def catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def memory_mb() -> dict:
    """Working set and private bytes for this process. Windows only; empty elsewhere."""
    if sys.platform != "win32":
        return {}
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel.K32GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    query.restype = wintypes.BOOL
    if not query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return {}
    return {
        "working_set_mb": round(counters.WorkingSetSize / (1024 * 1024), 1),
        "private_mb": round(counters.PrivateUsage / (1024 * 1024), 1),
    }


def available_mb() -> float | None:
    if sys.platform != "win32":
        return None
    import ctypes

    class Status(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = Status()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return round(status.ullAvailPhys / (1024 * 1024), 1)


def machine() -> dict:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "available_mb": available_mb(),
    }


def summarize(answers: dict) -> dict:
    summary: dict = {}
    department = answers.get("department") or {}
    if "choice" in department:
        summary["department"] = {
            "choice": department.get("choice"),
            "confidence": department.get("confidence"),
            "probabilities": department.get("probabilities"),
        }
    refund = answers.get("refund") or {}
    if isinstance(refund.get("noul"), (int, float)):
        yes = float(refund["noul"])
        summary["refund"] = {"yes": yes, "choice": "yes" if yes >= 0.5 else "no"}
    urgency = answers.get("urgency") or {}
    levels = QUESTIONS["urgency"]["criteria"]
    probabilities = urgency.get("probabilities") or {}
    if probabilities:
        best = max(probabilities, key=probabilities.get)
        index = int(best) if str(best).isdigit() else None
        named = {}
        for key, value in probabilities.items():
            label = levels[int(key)] if str(key).isdigit() and int(key) < len(levels) else str(key)
            named[label] = value
        summary["urgency"] = {
            "score": urgency.get("score"),
            "level": levels[index] if index is not None and index < len(levels) else str(best),
            "probabilities": named,
        }
    return summary


def ensure_weights(spec: dict) -> tuple[Path, float | None]:
    """Return the safetensors path, downloading the pinned revision when it is absent."""
    folder = ROOT / "models" / spec["directory"]
    weights = folder / "model.safetensors"
    if weights.is_file():
        return weights, None
    print(f"Downloading {spec['repo']} @ {spec['revision'][:12]}...", flush=True)
    started = time.perf_counter()
    from huggingface_hub import snapshot_download

    snapshot_download(spec["repo"], revision=spec["revision"], local_dir=str(folder))
    if not weights.is_file():
        raise FileNotFoundError(f"Download of {spec['repo']} finished without {weights.name}")
    return weights, round(time.perf_counter() - started, 3)


def score_model(model_id: str, spec: dict, out: Path, repeat: int) -> int:
    import torch
    from decider.infer import Decider

    header = {
        "type": "model",
        "at": now(),
        "id": model_id,
        "name": spec["name"],
        "repo": spec["repo"],
        "revision": spec["revision"],
        "device": spec.get("device", "cpu"),
        "dtype": spec.get("dtype", "float32"),
    }
    try:
        weights, download_s = ensure_weights(spec)
    except Exception:
        header["error"] = traceback.format_exc()
        append(out, header)
        print(header["error"], flush=True)
        return 1
    header["weights"] = str(weights)
    if download_s is not None:
        header["download_s"] = download_s

    print(f"Loading {spec['name']} ({spec.get('dtype')}, {spec.get('device', 'cpu')})...", flush=True)
    started = time.perf_counter()
    try:
        torch.set_num_threads(6)
        decider = Decider(
            str(weights.parent),
            device=spec.get("device", "cpu"),
            dtype=getattr(torch, spec.get("dtype", "float32")),
            use_graphs=False,
        )
    except Exception:
        header["error"] = traceback.format_exc()
        header["load_s"] = round(time.perf_counter() - started, 3)
        append(out, header)
        print(header["error"], flush=True)
        return 1

    header["load_s"] = round(time.perf_counter() - started, 3)
    header["runtime_name"] = getattr(decider, "name", spec["name"])
    header["temperature"] = getattr(decider, "T", None)
    by_type = getattr(decider, "T_by_type", None)
    if isinstance(by_type, dict):
        header["temperature_by_type"] = by_type
    header["memory"] = memory_mb()
    header["available_mb"] = available_mb()
    append(out, header)
    print(f"Loaded in {header['load_s']} s. Private {header['memory'].get('private_mb')} MB.", flush=True)

    failures = 0
    for scenario in SCENARIOS:
        state = {"ticket": scenario["ticket"], "refund_policy": POLICY}
        for run in range(1, repeat + 1):
            print(f"  {scenario['label']} run {run}/{repeat}...", flush=True)
            row = {
                "type": "decision",
                "at": now(),
                "id": model_id,
                "name": spec["name"],
                "scenario": scenario["id"],
                "label": scenario["label"],
                "run": run,
                "state": state,
            }
            t0 = time.perf_counter()
            try:
                result = decider.system_one(state, QUESTIONS)
            except Exception:
                row["request_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                row["error"] = traceback.format_exc()
                failures += 1
            else:
                row["request_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                row["usage"] = result.get("usage")
                row["model_name"] = result.get("model")
                row["answers"] = result.get("answers")
                row["summary"] = summarize(result.get("answers") or {})
            row["memory"] = memory_mb()
            row["available_mb"] = available_mb()
            append(out, row)
            if row.get("error"):
                print(f"    failed after {row['request_ms']} ms", flush=True)
            else:
                summary = row["summary"]
                print(
                    f"    {row['request_ms']} ms"
                    f"  team={summary.get('department', {}).get('choice')}"
                    f"  refund={summary.get('refund', {}).get('choice')}"
                    f"  urgency={summary.get('urgency', {}).get('level')}"
                    f"  tokens={row.get('usage', {}).get('input_tokens')}",
                    flush=True,
                )
    return 1 if failures else 0


def parse_models(raw: str | None) -> list[str]:
    known = catalog()["models"]
    if not raw:
        chosen = [model_id for model_id in DEFAULT_ORDER if model_id in known]
        chosen += [model_id for model_id in known if model_id not in chosen]
        return chosen
    chosen = [part.strip() for part in raw.split(",") if part.strip()]
    missing = [model_id for model_id in chosen if model_id not in known]
    if missing:
        raise SystemExit(f"Unknown model id(s): {', '.join(missing)}. Known: {', '.join(known)}")
    return chosen


def parent(models: list[str], out: Path, repeat: int) -> int:
    append(out, {
        "type": "run",
        "at": now(),
        "machine": machine(),
        "models": models,
        "repeat": repeat,
        "policy": POLICY,
        "scenarios": SCENARIOS,
        "questions": QUESTIONS,
        "note": (
            "Answers on the three demo tickets. There is no gold label, so this file "
            "records choices, probabilities, tokens, and timings rather than accuracy."
        ),
    })
    print(f"Writing {out}", flush=True)
    status = 0
    for model_id in models:
        spec = catalog()["models"][model_id]
        print(f"\n=== {spec['name']} ===", flush=True)
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", model_id, "--out", str(out), "--repeat", str(repeat)],
            cwd=ROOT,
        )
        if completed.returncode != 0:
            status = completed.returncode
            append(out, {
                "type": "model_failed",
                "at": now(),
                "id": model_id,
                "exit_code": completed.returncode,
            })
    append(out, {"type": "done", "at": now(), "ok": status == 0})
    print(f"\nFinished. Records are in {out}", flush=True)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the demo tickets with each Decider checkpoint.")
    parser.add_argument("--models", help="Comma-separated catalog ids. Default: 0.8b, 2b, 4b-v2.")
    parser.add_argument("--out", type=Path, help="JSONL path. Default: results/bench-<timestamp>.jsonl")
    parser.add_argument("--repeat", type=int, default=1, help="How many times to score each ticket.")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")

    if args.worker:
        spec = catalog()["models"][args.worker]
        return score_model(args.worker, spec, args.out, args.repeat)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or (RESULTS / f"bench-{stamp}.jsonl")
    return parent(parse_models(args.models), out, args.repeat)


if __name__ == "__main__":
    raise SystemExit(main())
