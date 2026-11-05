"""Write a run directory: config.json, summary.json/csv, comparisons.json/csv, per_query.jsonl."""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np


def git_rev() -> str:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "src", "experiments", "configs"],
                                stderr=subprocess.DEVNULL) != 0
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def environment() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        env["torch"] = torch.__version__
        env["device"] = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    except ImportError:
        env["torch"] = None
    return env


def summarise(per_query: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    out = {}
    for name, rows in per_query.items():
        keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float)) and not isinstance(rows[0][k], bool)]
        out[name] = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    return out


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_run(out_dir: str | Path, config: dict, per_query: dict[str, list[dict]], comparisons: list[dict],
              extra: dict | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarise(per_query)
    cfg = {**config, "git": git_rev(), "environment": environment(),
           "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **(extra or {})}
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "comparisons.json").write_text(json.dumps(comparisons, indent=2))
    _write_csv(out / "summary.csv", [{"system": n, **{k: round(v, 4) for k, v in s.items()}} for n, s in summary.items()])
    _write_csv(out / "comparisons.csv", comparisons)
    with open(out / "per_query.jsonl", "w") as f:
        for n, rows in per_query.items():
            for r in rows:
                f.write(json.dumps({"system": n, **r}) + "\n")
    return out
