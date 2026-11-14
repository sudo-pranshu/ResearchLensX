#!/usr/bin/env python3
"""Phase 4 measurements on real PDFs and real models (reproducible; prints a table, writes nothing to the repository).

  .venv/bin/python scripts/bench_phase4.py data/raw/pdfs/*.pdf

1. Upload through the job queue (spawned parser, original stored, index rebuilt): wall time for all files.
2. Adding the last paper to a library that already has the others, with e5 dense retrieval:
     "full re-embed"  = a fresh library over ALL papers with an empty embedding store (what the old corpus-keyed cache cost on any change)
     "incremental"    = the same library after adding only the last paper to an already-embedded library
3. Peak resident memory after using e5, the reranker and NLI in one process across several scopes.
Needs the dense extra and the cached models (first run downloads them).
"""
import resource
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import researchlens._runtime  # noqa: F401,E402
from researchlens.models import ModelManager  # noqa: E402
from researchlens.workspace import Settings, Workspace  # noqa: E402


def timed(fn):
    t = time.perf_counter(); out = fn(); return out, time.perf_counter() - t


def main(paths):
    files = [(Path(p).name, Path(p).read_bytes()) for p in paths]
    assert len(files) >= 3, "pass at least three PDFs"
    td = Path(tempfile.mkdtemp(prefix="rlx-bench-"))
    models = ModelManager()                       # ONE manager shared by every workspace below, so model load is paid once and not timed
    ws = Workspace(td / "a" / "lib.sqlite", models=models)
    rep, t_up = timed(lambda: ws.upload(files))
    print(f"[1] upload via jobs: {len(rep.added)} added, {len(rep.failed)} failed, {t_up:.1f} s total ({t_up / len(files):.1f} s per file, includes parser start-up)")

    ws.search("warm up the model", Settings(retriever="e5"))      # load e5 and run one batch: later timings exclude cold start
    first, last = files[:-1], files[-1:]
    ws1 = Workspace(td / "b" / "lib.sqlite", models=models)
    ws1.upload(first)
    _, t_base = timed(lambda: ws1.search("datasets used for evaluation", Settings(retriever="e5")))          # embeds the first N-1 papers
    n_before = ws1.cache.count()
    ws1.upload(last)
    _, t_inc = timed(lambda: ws1.search("datasets used for evaluation", Settings(retriever="e5")))           # embeds only the new paper
    n_after = ws1.cache.count()
    ws2 = Workspace(td / "c" / "lib.sqlite", models=models)
    ws2.upload(files)
    _, t_full = timed(lambda: ws2.search("datasets used for evaluation", Settings(retriever="e5")))          # fresh: embeds everything
    print(f"[2] e5 index after adding the last paper: incremental {t_inc:.1f} s ({n_after - n_before} new chunk vectors) "
          f"vs full re-embed {t_full:.1f} s ({ws2.cache.count()} vectors); both with the model already loaded "
          f"(building the first N-1 papers took {t_base:.1f} s, {n_before} vectors)")

    s = Settings(retriever="rrf:bm25+e5", rerank=True, verify=True, k=4)
    ids = [d.doc_id for d in ws2.library()]
    for scope in (None, ids[:2], ids[2:4], [ids[0]]):
        ws2.ask("Which datasets are used?", s, scope)
    ws2.verify_claim("Dense retrieval is evaluated on several datasets.", s, None)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    st = ws2.models.status()
    print(f"[3] peak RSS after e5 + reranker + NLI across 4 scopes: {rss:.0f} MB; resident models {st['resident']} loads {st['loads']}")


if __name__ == "__main__":
    main(sys.argv[1:])
