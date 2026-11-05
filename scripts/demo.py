"""End-to-end CPU demo: retrieve -> extractive answer -> (mock) verification, with citations.

  python scripts/demo.py "Which datasets are used?"                       # synthetic fixtures
  python scripts/demo.py --qasper data/raw/qasper/qasper-dev-v0.3.json --papers 20 "What baselines do they compare to?"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from researchlens.mocks import MockVerifier, load_sample_documents  # noqa: E402
from researchlens.pipeline import build_default_pipeline            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("question")
ap.add_argument("--qasper", default=None)
ap.add_argument("--papers", type=int, default=20)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--no-verify", action="store_true")
args = ap.parse_args()

if args.qasper:
    from researchlens.data.qasper import load_qasper_documents
    docs = load_qasper_documents(args.qasper, limit=args.papers)
else:
    docs = load_sample_documents()
pipe = build_default_pipeline(docs, verifier=None if args.no_verify else MockVerifier())
ans = pipe.ask(args.question, k=args.k, verify=not args.no_verify)
titles = {d.doc_id: d.title for d in docs}

if ans.is_mock:
    print("*** MOCK COMPONENT IN USE - verification labels are NOT model predictions ***\n")
print(f"Q: {ans.question}\n")
for s in ans.sentences:
    v = f"  [{s.verification.label} {s.verification.confidence:.2f} via {s.verification.model}]" if s.verification else ""
    print(f"- {s.text} " + "".join(f"[{c + 1}]" for c in s.citations) + v)
print("\nSources:")
for i, p in enumerate(ans.passages, 1):
    print(f" [{i}] {titles[p.chunk.doc_id][:70]} :: {p.chunk.section} ({p.retriever} {p.score:.2f})")
print("\ntimings_ms:", {k: round(v, 1) for k, v in ans.timings_ms.items()})
