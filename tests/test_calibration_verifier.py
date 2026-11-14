"""Calibration maths and the multi-evidence / explanation logic of NLIVerifier (fake scorer, offline)."""
import numpy as np

from researchlens.contracts import Chunk, Claim, RetrievedPassage
from researchlens.evaluation.calibration import apply_temperature, ece, fit_temperature, nll
from researchlens.verification.nli import NLIVerifier


def test_temperature_identity_and_monotone_sharpening():
    P = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    assert np.allclose(apply_temperature(P, 1.0), P, atol=1e-6)
    soft, sharp = apply_temperature(P, 3.0), apply_temperature(P, 0.5)
    assert soft.max() < P.max() < sharp.max() and np.allclose(soft.sum(1), 1)


def test_fit_temperature_reduces_nll_on_overconfident_model():
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 3, n)
    right = rng.random(n) < 0.6                                   # only 60% accurate but very confident
    pred = np.where(right, y, (y + 1) % 3)
    P = np.full((n, 3), 0.01); P[np.arange(n), pred] = 0.98
    T = fit_temperature(P, y)
    assert T > 1.5
    assert nll(apply_temperature(P, T), y) < nll(P, y) and ece(apply_temperature(P, T), y) < ece(P, y)


def test_ece_perfectly_calibrated_is_small():
    P = np.array([[0.5, 0.5, 0.0]] * 4); y = np.array([0, 1, 0, 1])
    assert ece(P, y) < 1e-9


def passages(texts):
    return [RetrievedPassage(chunk=Chunk(chunk_id=f"c{i}", doc_id="d", section="Results", text=t, order=i, page=i + 3),
                             score=1.0, rank=i + 1, retriever="t") for i, t in enumerate(texts)]


def keyword_scorer(premises, hyps):
    """Premise entails the hypothesis if it contains ALL hypothesis keywords 'A' and 'B'; contradicts if it says 'NOT'."""
    out = []
    for p, h in zip(premises, hyps):
        if "NOT" in p:
            out.append([0.05, 0.9, 0.05])
        elif all(k in p for k in ("A", "B")):
            out.append([0.9, 0.05, 0.05])
        else:
            out.append([0.1, 0.1, 0.8])
    return np.array(out)


def test_direct_support_and_explanation_cites_page():
    v = NLIVerifier(keyword_scorer, granularity="sentence", combine=True)
    r = v.verify(Claim(claim_id="1", text="A and B", source="user"), passages(["Filler text here.", "We find A and B together."]))
    assert r.label == "SUPPORTS" and r.support == "direct"
    assert "[2] Results, p. 4" in r.explanation and "not proof" in r.explanation


def test_combined_support_needs_two_passages_and_is_labelled_inference():
    def scorer(prem, hyp):
        # entails only when the premise mentions BOTH facts; each passage alone mentions only one
        return np.array([[0.9, 0.05, 0.05] if ("fact1" in p and "fact2" in p) else [0.1, 0.1, 0.8] for p in prem])
    ps = passages(["We show fact1 clearly.", "Separately, fact2 holds."])
    claim = Claim(claim_id="1", text="fact1 and fact2", source="user")
    off = NLIVerifier(scorer, granularity="sentence", combine=False).verify(claim, ps)
    on = NLIVerifier(scorer, granularity="sentence", combine=True).verify(claim, ps)
    assert off.label == "INSUFFICIENT" and off.support is None
    assert on.label == "SUPPORTS" and on.support == "combined"
    assert "combining" in on.explanation and "reasonable inference" in on.explanation
    one = NLIVerifier(scorer, granularity="sentence", combine=True).verify(claim, ps[:1])
    assert one.label == "INSUFFICIENT"                           # a single passage can never be 'combined'


def test_contradiction_conflict_and_insufficient_explanations():
    v = NLIVerifier(keyword_scorer, granularity="sentence")
    r = v.verify(Claim(claim_id="1", text="A and B", source="user"), passages(["We do NOT observe it.", "We find A and B."]))
    assert r.label in {"SUPPORTS", "CONTRADICTS"}
    assert "conflicts" in r.explanation or "Entailed" in r.explanation
    n = v.verify(Claim(claim_id="1", text="A and B", source="user"), passages(["Nothing relevant."]))
    assert n.label == "INSUFFICIENT" and "not the same as the claim being false" in n.explanation
    e = v.verify(Claim(claim_id="1", text="A and B", source="user"), [])
    assert e.label == "INSUFFICIENT" and "No evidence passages" in e.explanation


def test_temperature_changes_reported_confidence_not_the_label():
    ps = passages(["We find A and B."])
    claim = Claim(claim_id="1", text="A and B", source="user")
    raw = NLIVerifier(keyword_scorer, granularity="sentence").verify(claim, ps)
    cal = NLIVerifier(keyword_scorer, granularity="sentence", temperature=2.0).verify(claim, ps)
    assert raw.label == cal.label == "SUPPORTS"
    assert cal.confidence < raw.confidence                       # overconfident scores are pulled down
    assert abs(sum(cal.evidence[0].probs.values()) - 1) < 1e-6
