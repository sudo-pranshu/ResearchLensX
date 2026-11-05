# Pawan: Claim Verification, Generation, H2

**You may edit:** `src/researchlens/verification/**`, `generation/**`, `data/scifact.py`, `evaluation/attribution.py`, `experiments/h2_*`, `configs/h2_*`, `results/h2/**`, `tests/test_verification*.py`, `tests/test_generation*.py`, `tests/test_attribution*.py`, `tests/test_scifact*.py`, `docs/human_eval_guidelines.md`.
**Do not edit:** `contracts.py` (request changes via an issue labelled `interface-change`), `pipeline.py`, `retrieval/**`, Gaurav's paths.
**Validate:** `python -m pytest -q`. Every model-loading class gets a test that is skipped when the model isn't cached (`pytest.importorskip` + an env flag `RLX_RUN_MODEL_TESTS=1`), so CI stays offline.

**Contracts you implement** (`src/researchlens/contracts.py`):
```python
class NLIVerifier:            # verification/nli.py
    name: str; is_mock = False
    def verify(self, claim: Claim, passages: list[RetrievedPassage]) -> VerificationResult: ...
class LLMRAGGenerator:        # generation/llm_rag.py
    name: str; is_mock = False
    def generate(self, question: str, passages: list[RetrievedPassage]) -> Answer: ...   # mode="rag", citations 0-based
```
Develop against `build_default_pipeline(load_sample_documents())` or QASPER documents. You don't need anyone else's code to start.

## Day 1–2 (milestone "Day 2: setup & baselines")
| # | Task | Done when |
|---|---|---|
| W1 | `scripts/download_scifact.sh` + `data/scifact.py`: claims (id, claim, evidence {doc_id: [{sentences, label}]}, cited_doc_ids) and corpus (doc_id, title, abstract sentences). Map SUPPORT→SUPPORTS, CONTRADICT→CONTRADICTS, no evidence→INSUFFICIENT. Add a row to DATASET_REGISTER | Loader test on a 3-claim fixture; audit counts (train 809 / dev 300 claims, 5,183 docs) printed and recorded |
| W2 | Classical baseline: TF-IDF(claim ⊕ sentence) features + `LogisticRegression` 3-way, rationale-sentence level → abstract-level label | SciFact dev macro-F1 committed in `results/h2/scifact_lr/` |
| W3 | `NLIVerifier` with a pretrained MNLI-family DeBERTa-v3 checkpoint (pick on Day 1, record the HF id + revision). entailment→SUPPORTS, contradiction→CONTRADICTS, neutral→INSUFFICIENT; aggregate over passages = max-confidence non-neutral, else INSUFFICIENT | Protocol test passes; SciFact dev macro-F1 (zero-shot, gold abstracts) on CPU committed |

## Day 3–4
- W4 `verification/claims.py`: split answers into claims (sentence split + strip citation markers; optional LR "is-claim" filter).
- W5 `generation/llm_rag.py` + `experiments/h2_generate.py`: prompt with numbered passages, parse `[n]` markers into `AnswerSentence.citations`. Must run with `--limit 3` on CPU using a tiny model for testing, and unchanged on Colab with the real model. Writes JSONL.
- W6 `evaluation/attribution.py`: ALCE-style citation recall/precision with a pluggable judge (protocol in EXPERIMENT_PROTOCOL §3). Hand-made unit cases.

## Day 5–8
- W7 Verified-RAG revision (`generation/verified.py`): re-attribute or drop unsupported sentences. Ablation flag `--verify-only`.
- W8 Calibration: temperature scaling on SciFact held-out train; report ECE before/after.
- W9 H2 runs on cached generations; judge validation; human-eval sheet + guidelines; McNemar.
- GPU jobs: give Pranshu a single command + config. Never ask for an ad-hoc run without a committed config.

## Claude session rules
Start each session with: "Read CLAUDE.md and docs/development/PAWAN_WORKFLOW.md. I'm working on issue #N." If Claude proposes editing `contracts.py` or files outside your paths, stop and open an `interface-change` issue instead.
