import json
import math

import numpy as np
import pytest

from researchlens.data.qasper import load_qasper
from researchlens.evaluation import metrics as M
from researchlens.evaluation import stats as S
from researchlens.processing.chunking import Section, section_aware_chunks
from researchlens.retrieval.dense import DenseRetriever, LSAEncoder
from researchlens.retrieval.fusion import LinearFusionRetriever, RRFRetriever, reciprocal_rank_fusion
from researchlens.retrieval.lexical import BM25Retriever, TfidfRetriever

DOCS = [
    "BERT is fine-tuned on SQuAD for extractive question answering.",
    "We evaluate on the QASPER dataset of NLP papers with evidence annotations.",
    "Topic models such as LDA are trained with expectation maximisation.",
    "Dense passage retrieval encodes questions and passages with dual encoders.",
    "BM25 is a strong lexical baseline for ad-hoc retrieval.",
]


def test_bm25_matches_reference_formula():
    r = BM25Retriever(k1=1.2, b=0.75, remove_stopwords=False).fit(DOCS)
    q = "bm25 lexical retrieval"
    scores = r.score(q)
    # Hand-computed reference
    from researchlens.retrieval.lexical import tokenize
    toks = [tokenize(d, False) for d in DOCS]
    N, avgdl = len(toks), np.mean([len(t) for t in toks])
    ref = []
    for t in toks:
        s = 0.0
        for term in tokenize(q, False):
            df = sum(term in d for d in toks)
            tf = t.count(term)
            if tf == 0:
                continue
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            s += idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * len(t) / avgdl))
        ref.append(s)
    assert np.allclose(scores, ref)
    assert r.search(q, 1)[0][0] == 4


def test_candidates_restrict_search():
    r = BM25Retriever().fit(DOCS)
    res = r.search("retrieval", k=5, candidates=np.array([0, 1, 3]))
    assert {d for d, _ in res} <= {0, 1, 3}
    assert res[0][0] == 3


def test_tfidf_and_lsa_rank_obvious_match_first():
    for r in (TfidfRetriever(), DenseRetriever(LSAEncoder(n_components=4), name="lsa")):  # full rank: no near-ties across BLAS builds
        r.fit(DOCS)
        assert r.search("LDA topic models expectation maximisation", 1)[0][0] == 2


def test_rrf_formula():
    fused = dict(reciprocal_rank_fusion([[1, 2, 3], [3, 1]], k=60))
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[2] == pytest.approx(1 / 62)


def test_hybrid_retrievers_run():
    bm, lsa = BM25Retriever().fit(DOCS), DenseRetriever(LSAEncoder(3), name="lsa").fit(DOCS)
    for h in (RRFRetriever([bm, lsa], fit_components=False), LinearFusionRetriever(bm, lsa, fit_components=False)):
        h.fit(DOCS)
        res = h.search("lexical baseline BM25", k=3, candidates=np.array([1, 3, 4]))
        assert res[0][0] == 4 and len(res) == 3


def test_metrics():
    ranked, gold = [5, 2, 9, 1], {2, 1}
    assert M.recall_at_k(ranked, gold, 2) == 0.5
    assert M.mrr_at_k(ranked, gold) == 0.5
    ideal = 1 + 1 / math.log2(3)
    assert M.ndcg_at_k(ranked, gold, 10) == pytest.approx((1 / math.log2(3) + 1 / math.log2(5)) / ideal)
    assert M.set_f1([1, 2], [2, 3]) == 0.5


def test_stats_detects_real_difference_and_not_null():
    rng = np.random.default_rng(1)
    a = rng.random(300)
    b = np.clip(a + 0.05 + rng.normal(0, 0.05, 300), 0, 1)
    c = S.compare("a", a, "b", b, "m")
    assert c.delta > 0 and c.ci_low > 0 and c.p_randomization < 0.01
    null = S.compare("a", a, "a2", a + rng.normal(0, 1e-3, 300) * rng.choice([-1, 1], 300), "m")
    assert null.p_randomization > 0.05
    assert S.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert S.mcnemar([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0


def test_chunking_respects_sections():
    secs = [Section("Intro", ["a " * 10]), Section("Method", [" ".join(f"w{i}" for i in range(600))])]
    ch = section_aware_chunks(secs, max_tokens=256, overlap=32)
    assert ch[0].section == "Intro"
    method = [c for c in ch if c.section == "Method"]
    assert len(method) == 3 and all(len(c.text.split()) <= 256 for c in method)
    assert method[1].text.split()[0] == "w224"  # overlap of 32


@pytest.fixture
def tiny_qasper(tmp_path):
    data = {
        "p1": {"title": "Paper One", "abstract": "", "figures_and_tables": [{"caption": "Table 1: Results on SQuAD.", "file": "t1.png"}],
               "full_text": [{"section_name": "Introduction", "paragraphs": ["We study BM25.", "Dense retrieval helps."]},
                             {"section_name": "Experiments", "paragraphs": ["We use the SQuAD dataset."]}],
               "qas": [{"question": "What dataset do they use?", "question_id": "q1",
                        "answers": [{"answer": {"unanswerable": False, "extractive_spans": ["SQuAD"], "yes_no": None,
                                                "free_form_answer": "", "evidence": ["We use the SQuAD dataset.",
                                                "FLOAT SELECTED: Table 1: Results on SQuAD."], "highlighted_evidence": []},
                                     "annotation_id": "a", "worker_id": "w"}]},
                       {"question": "Unanswerable?", "question_id": "q2",
                        "answers": [{"answer": {"unanswerable": True, "extractive_spans": [], "yes_no": None,
                                                "free_form_answer": "", "evidence": [], "highlighted_evidence": []},
                                     "annotation_id": "b", "worker_id": "w"}]}]},
    }
    p = tmp_path / "qasper-tiny.json"
    p.write_text(json.dumps(data))
    return p


def test_qasper_loader(tiny_qasper):
    s = load_qasper(tiny_qasper)
    assert s.stats["paragraphs"] == 4 and s.stats["questions"] == 2
    assert s.stats["evidence_unmatched"] == 0 and s.stats["questions_without_evidence"] == 1
    q1 = s.questions[0]
    assert {s.paragraphs[i].text for i in q1.gold_union} == {"We use the SQuAD dataset.", "Table 1: Results on SQuAD."}
    assert s.paragraphs[2].indexed_text("Paper One").startswith("Paper One | Experiments")


def test_h1_script_end_to_end(tiny_qasper, tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import h1_retrieval
    summary, comps = h1_retrieval.main(["--qasper", str(tiny_qasper), "--systems", "bm25", "tfidf", "lsa",
                                        "rrf:bm25+lsa", "linear:bm25+lsa", "--lsa-dim", "2", "--out", str(tmp_path)])
    assert set(summary) == {"bm25", "tfidf", "lsa", "rrf:bm25+lsa", "linear:bm25+lsa"}
    assert len(comps) == 6


def test_build_windows_maps_back_to_paragraphs(tiny_qasper):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from h1_retrieval import build_windows
    s = load_qasper(tiny_qasper)
    texts, owners, paper_units = build_windows(s, w=4)
    assert paper_units["p1"] == list(range(len(texts)))
    float_pids = {p.pid for p in s.paragraphs if p.is_float}
    covered = {pid for o in owners for pid in o}
    assert covered == {p.pid for p in s.paragraphs} - float_pids     # every text paragraph reachable, no captions
    assert all(len(t.split("\n", 1)[1].split()) <= 4 for t in texts)
