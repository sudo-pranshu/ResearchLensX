"""Multi-document QA engine: balancing, traceability, abstention, claim status (offline, CPU)."""
from researchlens.contracts import Document, Section
from researchlens.generation.extractive import ExtractiveGenerator
from researchlens.mocks import MockVerifier
from researchlens.pipeline import build_default_pipeline
from researchlens.qa.engine import ABSTAIN_TEXT, MultiDocQA, balance_across_documents, evidence_coverage, format_reference


def doc(doc_id, title, paras, pages=None):
    return Document(doc_id=doc_id, title=title, abstract="", source="fixture",
                    sections=[Section(title="Methods", paragraphs=paras, pages=pages or [])])


A = doc("A", "Dense retrieval paper", ["We evaluate dense retrieval on the SciFact dataset with recall metrics.",
                                       "Dense retrieval uses the SciFact dataset and the QASPER dataset for evaluation.",
                                       "Dense retrieval models are trained with contrastive learning and evaluated on datasets."],
        pages=[1, 2, 3])
B = doc("B", "Verification paper", ["Claim verification is evaluated on the SciFact dataset using macro F1 scores."], pages=[5])
C = doc("C", "Cooking paper", ["Pasta is boiled in salted water for ten minutes."])


def make(verifier=None, min_cov=0.4):
    pipe = build_default_pipeline([A, B, C])
    return MultiDocQA(pipe.retriever, ExtractiveGenerator(max_sentences=3), [A, B, C], verifier, min_cov)


def test_balancing_caps_per_doc_and_includes_other_papers():
    r = make().index.search("which datasets evaluate SciFact", k=20)
    out = balance_across_documents(r, k=3, max_per_doc=2)
    docs = [p.chunk.doc_id for p in out]
    assert len(out) == 3 and docs.count("A") <= 2 and "B" in docs
    assert [p.rank for p in out] == [1, 2, 3]


def test_answer_cites_pages_and_titles_without_inventing():
    res = make().answer("which datasets are used for evaluation on SciFact", k=3, max_per_doc=2, verify=False)
    assert not res.abstained and res.references
    by_chunk = {r.chunk_id: r for r in res.references}
    for s, st in zip(res.answer.sentences, res.sentences):
        for rid in st.ref_ids:
            ref = res.references[rid - 1]
            assert s.text in ref.text                         # extractive: every sentence is copied from its cited passage
            assert ref.title in {"Dense retrieval paper", "Verification paper", "Cooking paper"}
    pages = {r.doc_id: r.page for r in res.references}
    assert pages.get("B") == 5                                # page comes from the chunk, nothing made up
    assert all(st.status == "unchecked" for st in res.sentences)
    assert {"A", "B"} <= set(res.docs_used)


def test_page_unknown_is_not_fabricated():
    res = make().answer("how long should pasta be boiled in salted water", k=2, verify=False)
    ref = next(r for r in res.references if r.doc_id == "C")
    assert ref.page is None and "page unknown" in format_reference(ref)


def test_abstains_when_evidence_does_not_cover_question():
    res = make().answer("what is the capital of France and its population in 2020", k=3)
    assert res.abstained and res.answer.sentences[0].text == ABSTAIN_TEXT
    assert res.sentences[0].status == "abstained" and res.sentences[0].ref_ids == [] and res.abstain_reason


def test_claim_status_and_mock_flag_propagate():
    res = make(verifier=MockVerifier()).answer("which datasets are used for evaluation on SciFact", k=3)
    assert res.is_mock is True                                # mock verifier => visible mock flag
    assert {s.status for s in res.sentences} <= {"supported", "unsupported", "contradicted", "conflicting", "inferred"}


def test_doc_filter_and_coverage():
    res = make().answer("which datasets are used for evaluation", k=3, doc_ids=["B"], verify=False)
    assert {r.doc_id for r in res.references} == {"B"}
    assert evidence_coverage("", []) == 0.0


def test_conflicting_and_inferred_statuses():
    import numpy as np
    from researchlens.verification.nli import NLIVerifier

    def scorer(prem, hyp):
        return np.array([[0.9, 0.05, 0.05] if "supports" in p else [0.05, 0.9, 0.05] if "refutes" in p else [0.1, 0.1, 0.8] for p in prem])
    d1 = doc("X", "X paper", ["This result supports the claim about dense retrieval datasets."], pages=[1])
    d2 = doc("Y", "Y paper", ["This result refutes the claim about dense retrieval datasets."], pages=[2])
    from researchlens.contracts import Answer, AnswerSentence

    class BothCited:
        name, is_mock = "stub", False
        def generate(self, question, passages):
            return Answer(question=question, passages=passages, mode="rag", generator="stub",
                          sentences=[AnswerSentence(text="The papers disagree about the claim.", citations=[0, 1])])
    pipe = build_default_pipeline([d1, d2])
    qa = MultiDocQA(pipe.retriever, BothCited(), [d1, d2], NLIVerifier(scorer, granularity="sentence"))
    res = qa.answer("claim about dense retrieval datasets", k=2, max_per_doc=1)
    assert [s.status for s in res.sentences] == ["conflicting"] and res.sentences[0].ref_ids == [1, 2]
    assert not res.is_mock
