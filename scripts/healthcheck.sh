#!/usr/bin/env bash
# Health checks. Exit code 0 only if every check passes.
#   scripts/healthcheck.sh            environment + pipeline smoke test (+ running app, if one is up)
#   scripts/healthcheck.sh --app      also require the app to be running and answering
# Checks: (1) packages import, (2) a tiny generated PDF is ingested, searched and answered offline (BM25, no model download),
#         (3) the Streamlit health endpoint of the running app.
set -uo pipefail
. "$(cd "$(dirname "$0")" && pwd)/_common.sh"
cd "$RLX_ROOT"
NEED_APP=0; [ "${1:-}" = "--app" ] && NEED_APP=1
FAIL=0
[ -x "$RLX_PY" ] || die "no virtual environment: run ./run.sh (or scripts/setup.sh) first"

if "$RLX_PY" - <<'PY' 2>/tmp/rlx_import.err; then ok "packages import (researchlens, PyMuPDF, Streamlit)"; else cat /tmp/rlx_import.err >&2; warn "package import failed: run scripts/setup.sh --force"; FAIL=1; fi
import researchlens, fitz, streamlit  # noqa: F401
PY

SMOKE="$("$RLX_PY" - <<'PY' 2>&1
import tempfile, warnings, os
warnings.filterwarnings("ignore")
import pymupdf
from researchlens.ingestion.pymupdf_ingestor import PyMuPDFIngestor
from researchlens.pipeline import build_default_pipeline
from researchlens.qa.engine import MultiDocQA
from researchlens.generation.extractive import ExtractiveGenerator

d = tempfile.mkdtemp(); path = os.path.join(d, "smoke.pdf")
doc = pymupdf.open(); page = doc.new_page()
page.insert_text((72, 100), "A Smoke Test Paper", fontsize=20, fontname="hebo")
page.insert_text((72, 150), "1 Introduction", fontsize=12, fontname="hebo")
page.insert_textbox(pymupdf.Rect(72, 160, 520, 400),
    "We evaluate retrieval on the Smokebench dataset using recall. The Smokebench dataset covers scientific papers. "
    "Our method improves recall over the baseline in every experiment we ran.", fontsize=10)
doc.save(path); doc.close()
parsed = PyMuPDFIngestor().ingest(path)
pipe = build_default_pipeline([parsed])
qa = MultiDocQA(pipe.retriever, ExtractiveGenerator(), [parsed])
res = qa.answer("Which dataset is used for evaluation?", verify=False)
assert not res.abstained and res.references and res.references[0].page == 1, "QA smoke test failed"
print("ingested 1 page, answered with a page-cited reference")
PY
)"
if [ $? -eq 0 ] && printf '%s' "$SMOKE" | grep -q "page-cited"; then ok "pipeline smoke test: $SMOKE"; else printf '%s\n' "$SMOKE" | tail -n 8 >&2; warn "pipeline smoke test failed"; FAIL=1; fi

PORT="$(cat "$RLX_PORT_FILE" 2>/dev/null || true)"; PID="$(cat "$RLX_PID_FILE" 2>/dev/null || true)"
if [ -n "$PORT" ] && pid_alive "$PID"; then
  if http_ok "http://127.0.0.1:$PORT/_stcore/health"; then ok "app is up: http://localhost:$PORT"; else warn "app process $PID is running but port $PORT does not answer"; FAIL=1; fi
elif [ "$NEED_APP" = 1 ]; then
  warn "the app is not running (start it with ./run.sh)"; FAIL=1
else
  ok "app not running (environment checks only)"
fi
exit "$FAIL"
