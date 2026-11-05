"""Verification components tested offline (fake NLI scorer; no model downloads)."""
from pathlib import Path

import numpy as np
import pytest

from researchlens.contracts import Chunk, Claim, RetrievedPassage, Verifier
from researchlens.data.scifact import audit, load_claims, load_corpus
from researchlens.evaluation.classification import classification_report, confusion
from researchlens.verification.classical import LRVerifierBaseline
from researchlens.verification.nli import NLIVerifier, decide

FIX = Path(__file__).parent / "fixtures" / "scifact"


def fake_scorer(premises, hypotheses):
    """SUPPORTS if premise contains 'reduces', CONTRADICTS if 'does not', else neutral."""
    rows = []
    for p in premises:
        if "does not" in p:
            rows.append([0.05, 0.9, 0.05])
        elif "reduces" in p:
            rows.append([0.9, 0.05, 0.05])
        else:
            rows.append([0.1, 0.1, 0.8])
    return np.array(rows)


def passage(i, text):
    return RetrievedPassage(chunk=Chunk(chunk_id=f"c{i}", doc_id="d", section="s", text=text, order=i),
                            score=1.0, rank=i + 1, retriever="t")


def test_decide_thresholds():
    assert decide(np.array([0.6, 0.3, 0.1]))[0] == "SUPPORTS"
    assert decide(np.array([0.2, 0.7, 0.1]))[0] == "CONTRADICTS"
    assert decide(np.array([0.45, 0.1, 0.45]))[0] == "INSUFFICIENT"
    assert decide(np.array([0.6, 0.3, 0.1]), support_threshold=0.7)[0] == "INSUFFICIENT"


def test_nli_verifier_contract_sentence_granularity():
    v = NLIVerifier(fake_scorer, granularity="sentence")
    v.scorer.name = "fake"
    assert isinstance(v, Verifier) and v.is_mock is False
    ps = [passage(0, "Background text. Drug A reduces blood pressure."), passage(1, "Nothing relevant here.")]
    r = v.verify(Claim(claim_id="1", text="Drug A reduces blood pressure", source="user"), ps)
    assert r.label == "SUPPORTS" and r.is_mock is False
    assert r.evidence[0].label == "SUPPORTS" and r.evidence[0].rationale_sentences == [1]
    assert r.evidence[1].label == "INSUFFICIENT" and r.evidence[1].rationale_sentences == []
    assert abs(sum(r.evidence[0].probs.values()) - 1) < 1e-6


def test_aggregation_prefers_stronger_and_handles_empty():
    v = NLIVerifier(fake_scorer, granularity="passage")
    r = v.verify(Claim(claim_id="1", text="x", source="user"), [passage(0, "It does not work.")])
    assert r.label == "CONTRADICTS"
    assert v.verify(Claim(claim_id="2", text="x", source="user"), []).label == "INSUFFICIENT"


def test_scifact_loader_and_labels():
    corpus, claims = load_corpus(FIX), load_claims("dev", FIX)
    c1, c2 = claims
    assert c1.label_for("11") == "SUPPORTS" and c1.label_for("22") == "CONTRADICTS"
    assert c2.label_for("33") == "INSUFFICIENT" and c1.rationale_sentences("11") == [1]
    a = audit(claims, corpus)
    assert a["cited_pairs_by_label"] == {"SUPPORTS": 1, "CONTRADICTS": 1, "INSUFFICIENT": 1}


def test_lr_baseline_runs_and_predicts_valid_labels():
    corpus, claims = load_corpus(FIX), load_claims("dev", FIX)
    lr = LRVerifierBaseline().fit_vocab([d.text for d in corpus.values()])
    items = [(c, d, c.label_for(d)) for c in claims for d in c.cited_doc_ids]
    X = np.array([lr.features(c.claim, corpus[d].sentences) for c, d, _ in items])
    lr.fit(X, [y for *_, y in items])
    assert set(lr.predict(X)) <= {"SUPPORTS", "CONTRADICTS", "INSUFFICIENT"}
    assert np.allclose(lr.predict_proba(X).sum(1), 1)


def test_classification_report():
    gold = ["SUPPORTS", "SUPPORTS", "CONTRADICTS", "INSUFFICIENT"]
    pred = ["SUPPORTS", "INSUFFICIENT", "CONTRADICTS", "INSUFFICIENT"]
    M = confusion(gold, pred)
    assert M.tolist() == [[1, 0, 1], [0, 1, 0], [0, 0, 1]]
    rep = classification_report(gold, pred, n_resamples=200)
    assert rep["SUPPORTS"]["recall"] == 0.5 and rep["INSUFFICIENT"]["precision"] == 0.5
    assert rep["macro_f1"] == pytest.approx((2/3 + 1 + 2/3) / 3)
    assert rep["macro_f1_ci95"][0] <= rep["macro_f1"] <= rep["macro_f1_ci95"][1]


def test_lr_baseline_with_missing_class():
    corpus, claims = load_corpus(FIX), load_claims("dev", FIX)
    lr = LRVerifierBaseline().fit_vocab([d.text for d in corpus.values()])
    c = claims[0]
    X = np.array([lr.features(c.claim, corpus[d].sentences) for d in c.cited_doc_ids])
    lr.fit(X, ["SUPPORTS", "CONTRADICTS"])                 # no INSUFFICIENT in training
    P = lr.predict_proba(X)
    assert P.shape == (2, 3) and np.all(P[:, 2] == 0)
