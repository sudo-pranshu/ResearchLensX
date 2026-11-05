import numpy as np
import pytest

from researchlens.contracts import (Answer, Document, Generator, KnowledgeGraph, PassageRetriever, Verifier,
                                    VerificationResult)
from researchlens.evaluation import stats as S
from researchlens.generation.extractive import ExtractiveGenerator, split_sentences
from researchlens.mocks import MockVerifier, load_sample_documents
from researchlens.pipeline import build_default_pipeline
from researchlens.processing.chunking import chunk_document


@pytest.fixture
def docs():
    return load_sample_documents()


def test_sample_documents_validate(docs):
    assert len(docs) == 2 and all(isinstance(d, Document) for d in docs)
    assert all(d.title.startswith("SYNTHETIC FIXTURE") for d in docs)


def test_chunk_document_ids_and_abstract(docs):
    chunks = chunk_document(docs[0])
    assert chunks[0].section == "Abstract"
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert [c.order for c in chunks] == list(range(len(chunks)))
    abstract_only = Document(doc_id="x", title="t", abstract="Only an abstract.", sections=[], source="fixture")
    assert [c.text for c in chunk_document(abstract_only)] == ["Only an abstract."]


def test_pipeline_end_to_end_and_mock_flag(docs):
    pipe = build_default_pipeline(docs, verifier=MockVerifier())
    assert isinstance(pipe.retriever, PassageRetriever)
    assert isinstance(pipe.generator, Generator) and isinstance(pipe.verifier, Verifier)
    ans = pipe.ask("Which datasets are used for evaluation?", k=3)
    assert isinstance(ans, Answer) and ans.passages and ans.sentences
    assert ans.is_mock is True                     # mock verifier must propagate
    for s in ans.sentences:
        assert isinstance(s.verification, VerificationResult)
        assert all(0 <= c < len(ans.passages) for c in s.citations)
        # extractive: every sentence is copied from its cited passage
        assert s.text in ans.passages[s.citations[0]].chunk.text
    Answer.model_validate_json(ans.model_dump_json())  # JSON round trip


def test_pipeline_without_verifier_is_not_mock(docs):
    ans = build_default_pipeline(docs).ask("reciprocal rank fusion", k=2, verify=False)
    assert ans.is_mock is False and ans.passages[0].chunk.doc_id == "sample-hybrid-retrieval"


def test_doc_filter_and_compare(docs):
    pipe = build_default_pipeline(docs)
    ans = pipe.ask("SciFact dataset", k=5, doc_ids=["sample-claim-verification"], verify=False)
    assert {p.chunk.doc_id for p in ans.passages} == {"sample-claim-verification"}
    comp = pipe.compare("Which dataset is used?", [d.doc_id for d in docs], verify=False)
    assert set(comp) == {d.doc_id for d in docs}


def test_split_sentences():
    assert split_sentences("We use BERT. It works well! Is F1 high? Yes.") == ["We use BERT.", "It works well!", "Is F1 high?", "Yes."]


def test_cluster_bootstrap_is_wider_under_clustering():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 10)
    cluster_effect = rng.normal(0, 0.3, 40)[groups]          # strong within-paper correlation
    a = rng.random(400)
    b = a + 0.02 + cluster_effect + rng.normal(0, 0.01, 400)
    _, lo_iid, hi_iid = S.paired_bootstrap(a, b)
    _, lo_cl, hi_cl = S.paired_bootstrap(a, b, groups=groups)
    assert (hi_cl - lo_cl) > 2 * (hi_iid - lo_iid)
    c = S.compare("a", a, "b", b, "m", groups=groups)
    assert c.ci_method == "cluster_bootstrap" and c.p_randomization > 0.05


def test_kg_contract_roundtrip():
    g = KnowledgeGraph.model_validate({"nodes": [{"id": "p1", "type": "paper", "label": "P"}],
                                       "edges": [{"source": "p1", "target": "p1", "type": "cites"}]})
    assert g.nodes[0].type == "paper"
