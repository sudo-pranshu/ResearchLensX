"""Parse one PDF in a separate (spawned) process: bytes/path in, Document JSON out.

Isolation reasons (docs/architecture/LOCAL_ARCHITECTURE.md §12, R4): untrusted PDFs are parsed by PyMuPDF, which can hang, use
unbounded memory or crash. The child is a pure function: it never touches the database or storage (only the API process
writes), imports only the ingestion code (no torch), and is killed by the parent on timeout or when its memory exceeds the limit.
`spawn` is used on every platform: forking a process that has loaded torch is unsafe on macOS.
"""
from __future__ import annotations

import multiprocessing as mp
import subprocess
import time
from dataclasses import dataclass


def _entry(path: str, max_pages: int | None, conn) -> None:                  # runs in the child process
    try:
        from researchlens.ingestion.pymupdf_ingestor import PyMuPDFIngestor
        doc = PyMuPDFIngestor().ingest(path, max_pages=max_pages)
        conn.send(("ok", doc.model_dump_json()))
    except Exception as e:                                                   # parse failures travel back as text
        conn.send(("err", f"{type(e).__name__}: {e}"))
    finally:
        conn.close()


@dataclass
class ParseResult:
    ok: bool
    document_json: str | None = None
    error: str | None = None


def rss_mb(pid: int) -> float | None:
    """Resident memory of a process in MB via `ps` (works on macOS and Linux; RLIMIT_AS is not enforced on macOS)."""
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=2).stdout.strip()
        return int(out) / 1024 if out else None
    except Exception:
        return None


def parse_pdf(path: str, max_pages: int | None = None, timeout_s: float = 60.0, max_rss_mb: float = 1500.0,
              should_cancel=lambda: False, poll_s: float = 0.1) -> ParseResult:
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_entry, args=(path, max_pages, child), daemon=True)
    proc.start()
    child.close()
    start, last_rss_check = time.monotonic(), 0.0
    try:
        while True:
            if parent.poll(poll_s):
                try:
                    kind, payload = parent.recv()
                except EOFError:
                    return ParseResult(False, error="the parser process ended without a result")
                return ParseResult(True, document_json=payload) if kind == "ok" else ParseResult(False, error=payload)
            if should_cancel():
                return ParseResult(False, error="cancelled")
            now = time.monotonic()
            if now - start > timeout_s:
                return ParseResult(False, error=f"parsing exceeded the time limit of {timeout_s:g} s and was stopped")
            if not proc.is_alive() and not parent.poll(0):
                return ParseResult(False, error=f"the parser process crashed (exit code {proc.exitcode})")
            if now - last_rss_check > 0.5:
                last_rss_check = now
                m = rss_mb(proc.pid)
                if m is not None and m > max_rss_mb:
                    return ParseResult(False, error=f"parsing used more than {max_rss_mb:g} MB of memory and was stopped")
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(timeout=5)
        parent.close()
