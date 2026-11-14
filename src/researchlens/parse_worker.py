"""Parse one PDF in a separate process: path in, Document JSON out (file).

Isolation reasons (docs/architecture/LOCAL_ARCHITECTURE.md §12, R4): untrusted PDFs are parsed by PyMuPDF, which can hang, use
unbounded memory or crash. The child is a pure function: it never touches the database or storage (only the API process
writes), imports only the ingestion code (no torch), and is killed by the parent on timeout, cancellation, or when its resident
memory exceeds the limit (`RLIMIT_AS` is not enforced on macOS, so memory is polled).

The child is started with `python -m researchlens.parse_worker`, NOT multiprocessing: that works whatever the parent's
`__main__` is (a script, `python -c`, stdin, a notebook, Streamlit, Uvicorn), whereas `spawn` must re-import the parent's main module.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


def main(argv: list[str]) -> int:                          # runs in the child process
    pdf, out = argv[0], argv[1]
    max_pages = int(argv[2]) if len(argv) > 2 and argv[2] != "-" else None
    try:
        from researchlens.ingestion.pymupdf_ingestor import PyMuPDFIngestor
        doc = PyMuPDFIngestor().ingest(pdf, max_pages=max_pages)
        payload = {"ok": True, "document": doc.model_dump_json()}
    except Exception as e:                                  # parse failures travel back as text
        payload = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    Path(out).write_text(json.dumps(payload))
    return 0


@dataclass
class ParseResult:
    ok: bool
    document_json: str | None = None
    error: str | None = None


def rss_mb(pid: int) -> float | None:
    """Resident memory of a process in MB via `ps` (works on macOS and Linux)."""
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=2).stdout.strip()
        return int(out) / 1024 if out else None
    except Exception:
        return None


def parse_pdf(path: str, max_pages: int | None = None, timeout_s: float = 60.0, max_rss_mb: float = 1500.0,
              should_cancel=lambda: False, poll_s: float = 0.1) -> ParseResult:
    with tempfile.TemporaryDirectory(prefix="rlx-parse-out-") as td:
        out = Path(td) / "result.json"
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)       # the child sees the same modules (editable installs, src/)
        env["OMP_NUM_THREADS"] = "1"
        proc = subprocess.Popen([sys.executable, "-m", "researchlens.parse_worker", path, str(out), "-" if max_pages is None else str(max_pages)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
        start, last_rss_check = time.monotonic(), 0.0
        try:
            while True:
                code = proc.poll()
                if code is not None:
                    if out.exists():
                        data = json.loads(out.read_text())
                        return ParseResult(True, document_json=data["document"]) if data.get("ok") else ParseResult(False, error=data["error"])
                    err = (proc.stderr.read() or b"").decode("utf-8", "replace").strip().splitlines()
                    return ParseResult(False, error=f"the parser process crashed (exit code {code})" + (f": {err[-1][:200]}" if err else ""))
                if should_cancel():
                    return ParseResult(False, error="cancelled")
                now = time.monotonic()
                if now - start > timeout_s:
                    return ParseResult(False, error=f"parsing exceeded the time limit of {timeout_s:g} s and was stopped")
                if now - last_rss_check > 0.5:
                    last_rss_check = now
                    m = rss_mb(proc.pid)
                    if m is not None and m > max_rss_mb:
                        return ParseResult(False, error=f"parsing used more than {max_rss_mb:g} MB of memory and was stopped")
                time.sleep(poll_s)
        finally:
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:                                  # pragma: no cover
                pass
            if proc.stderr:
                proc.stderr.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
