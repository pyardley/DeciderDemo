"""Score Laya's 50 labelled tickets with each Decider checkpoint.

The question is the three-way wording that won the Laya criteria search
(auto_reply / escalate_to_human / ignore). Laya's own choices are read from
that project's baseline_argmax result, which is that wording. This script does
not call Laya. It asks Decider the same question and records whether the choice
matches Laya and whether it matches the label.

One process loads one checkpoint, then exits so the next model gets the RAM back.
The file is flushed after every record.

    .\\.venv\\Scripts\\python.exe bench\\compare_laya.py --laya C:\\path\\to\\laya-ollama-demo
    .\\.venv\\Scripts\\python.exe bench\\compare_laya.py --laya C:\\path\\to\\laya-ollama-demo --models 0.8b,2b
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from run_bench import RESULTS, append, available_mb, catalog, ensure_weights, memory_mb, now, parse_models

LABELS = ("auto_reply", "escalate_to_human", "ignore")

# The best three-way wording from the Laya criteria search (a1 / e3 / i9).
ACTION = {
    "action": {
        "type": "choice",
        "instructions": "What action should be taken for this customer support ticket?",
        "criteria": {
            "auto_reply": "A how-to or FAQ: password reset, Bluetooth pairing, order tracking, firmware, or account settings.",
            "escalate_to_human": "Broken hardware, a refund, a safety issue, or an account takeover.",
            "ignore": "Not a customer support request.",
        },
    }
}


def load_messages(laya: Path, limit: int | None) -> list[dict]:
    messages_path = laya / "data" / "customer_messages.json"
    baseline_path = laya / "experiments" / "results" / "baseline_argmax.json"
    if not messages_path.is_file() or not baseline_path.is_file():
        raise SystemExit(
            f"{laya} is missing data/customer_messages.json or experiments/results/baseline_argmax.json."
        )
    messages = json.loads(messages_path.read_text(encoding="utf-8"))["messages"]
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in baseline["per_message"]}
    missing = [row["id"] for row in messages if row["id"] not in by_id]
    if missing:
        raise SystemExit(f"Laya baseline has no prediction for: {', '.join(missing)}")
    chosen = messages[:limit] if limit else messages
    return [
        {
            "id": row["id"],
            "label": row["label"],
            "text": row["text"],
            "laya": by_id[row["id"]]["prediction"],
        }
        for row in chosen
    ]


def confusion(rows: list[dict], prediction_key: str) -> dict:
    table = {label: {guess: 0 for guess in LABELS} for label in LABELS}
    for row in rows:
        table[row["label"]][row[prediction_key]] += 1
    return table


def print_table(title: str, table: dict) -> None:
    width = max(len(label) for label in LABELS)
    print(f"  {title}")
    print(f"  {'actual':<{width}} " + " ".join(f"{label:>{width}}" for label in LABELS))
    for label in LABELS:
        counts = " ".join(f"{table[label][guess]:>{width}}" for guess in LABELS)
        print(f"  {label:<{width}} {counts}")


def score_model(model_id: str, spec: dict, messages: list[dict], out: Path) -> int:
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
        "messages": len(messages),
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
    header["memory"] = memory_mb()
    header["available_mb"] = available_mb()
    append(out, header)
    print(f"Loaded in {header['load_s']} s. Private {header['memory'].get('private_mb')} MB.", flush=True)

    rows = []
    failures = 0
    for message in messages:
        print(f"  {message['id']}...", flush=True)
        row = {
            "type": "decision",
            "at": now(),
            "id": model_id,
            "name": spec["name"],
            "message_id": message["id"],
            "label": message["label"],
            "text": message["text"],
            "laya": message["laya"],
        }
        t0 = time.perf_counter()
        try:
            result = decider.system_one(message["text"], ACTION)
        except Exception:
            row["request_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            row["error"] = traceback.format_exc()
            failures += 1
        else:
            answer = (result.get("answers") or {}).get("action") or {}
            choice = answer.get("choice")
            row["request_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            row["usage"] = result.get("usage")
            row["choice"] = choice
            row["confidence"] = answer.get("confidence")
            row["probabilities"] = answer.get("probabilities")
            row["agrees_with_label"] = choice == message["label"]
            row["agrees_with_laya"] = choice == message["laya"]
            rows.append(row)
        row["memory"] = memory_mb()
        row["available_mb"] = available_mb()
        append(out, row)
        if row.get("error"):
            print(f"    failed after {row['request_ms']} ms", flush=True)
        else:
            mark = "label" if row["agrees_with_label"] else "miss"
            same = "same" if row["agrees_with_laya"] else "differs"
            print(
                f"    {row['request_ms']} ms  {row['choice']}  {mark}  Laya {same}",
                flush=True,
            )

    correct = sum(row["agrees_with_label"] for row in rows)
    same = sum(row["agrees_with_laya"] for row in rows)
    laya_correct = sum(message["laya"] == message["label"] for message in messages if message["id"] in {row["message_id"] for row in rows})
    summary = {
        "type": "summary",
        "at": now(),
        "id": model_id,
        "name": spec["name"],
        "scored": len(rows),
        "correct": correct,
        "agrees_with_laya": same,
        "laya_correct": laya_correct,
        "confusion": confusion(rows, "choice") if rows else {},
        "request_ms_total": round(sum(row["request_ms"] for row in rows), 1),
    }
    append(out, summary)
    print(
        f"{spec['name']}: {correct}/{len(rows)} match the label, {same}/{len(rows)} match Laya. "
        f"Laya matches the label on {laya_correct}/{len(messages)}.",
        flush=True,
    )
    if rows:
        print_table("Decider, rows are the label", summary["confusion"])
    return 1 if failures else 0


def parent(models: list[str], messages: list[dict], laya: Path, out: Path) -> int:
    laya_correct = sum(row["laya"] == row["label"] for row in messages)
    append(out, {
        "type": "run",
        "at": now(),
        "laya": str(laya),
        "models": models,
        "messages": len(messages),
        "laya_correct": laya_correct,
        "question": ACTION,
        "note": (
            "Same 50 labelled tickets and the winning three-way wording. "
            "Laya choices come from experiments/results/baseline_argmax.json. "
            "There is one Decider choice per message, the highest probability."
        ),
    })
    print(f"Writing {out}", flush=True)
    print(f"Laya matches the label on {laya_correct}/{len(messages)} of the messages in this run.", flush=True)
    status = 0
    for model_id in models:
        spec = catalog()["models"][model_id]
        print(f"\n=== {spec['name']} ===", flush=True)
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                model_id,
                "--laya",
                str(laya),
                "--out",
                str(out),
                "--limit",
                str(len(messages)),
            ],
            cwd=Path(__file__).resolve().parent,
        )
        if completed.returncode != 0:
            status = completed.returncode
            append(out, {"type": "model_failed", "at": now(), "id": model_id, "exit_code": completed.returncode})
    append(out, {"type": "done", "at": now(), "ok": status == 0})
    print(f"\nFinished. Records are in {out}", flush=True)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Decider with Laya on the 50 labelled tickets.")
    parser.add_argument("--laya", type=Path, required=True, help="Path to the laya-ollama-demo project.")
    parser.add_argument("--models", help="Comma-separated catalog ids. Default: 0.8b, 2b, 4b-v2.")
    parser.add_argument("--out", type=Path, help="JSONL path. Default: results/laya-compare-<timestamp>.jsonl")
    parser.add_argument("--limit", type=int, help="Score only the first N messages.")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    laya = args.laya.expanduser().resolve()
    messages = load_messages(laya, args.limit)
    if args.worker:
        spec = catalog()["models"][args.worker]
        return score_model(args.worker, spec, messages, args.out)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or (RESULTS / f"laya-compare-{stamp}.jsonl")
    return parent(parse_models(args.models), messages, laya, out)


if __name__ == "__main__":
    raise SystemExit(main())
