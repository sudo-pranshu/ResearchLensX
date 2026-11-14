"""Acceptance test 1 (architecture review): an incrementally maintained library gives exactly the results of a fresh build,
and adding a paper embeds only that paper's new chunks. Fully offline: a deterministic hashing encoder replaces e5."""
import hashlib
import re

import numpy as np
import pytest

from researchlens.contracts import Document, Section
from researchlens.models import ModelManager
from researchlens.workspace import Settings, Workspace

THEMES = {
    "A": "dense retrieval encoders contrastive pretraining passage ranking recall benchmark queries",
    "B": "claim verification entailment contradiction evidence abstracts annotators scientific claims",
    "C": "knowledge graph entities relations provenance extraction methods datasets citations papers",
    "D": "protein folding structure prediction sequence alignment biology accuracy residues models",
}
QUERIES = ["dense retrieval recall", "claim verification entailment", "knowledge graph provenance", "protein structure prediction",
           "scientific papers evidence", "datasets and methods", "ranking queries benchmark", "contradiction annotators",
           "citations extraction", "sequence alignment accuracy"] * 2


def make_doc(tag: str) -> Document:
    words = THEMES[tag].split()
    paras = [" ".join(np.random.default_rng(i + ord(tag)).permutation(words).tolist()) + f" paper {tag} paragraph {i}."
             for i in range(6)]
    return Document(doc_id=f"doc-{tag}", title=f"Paper {tag} about {words[0]} {words[1]}", abstract=THEMES[tag], source="fixture",
                    sections=[Section(title="Methods", paragraphs=paras, pages=list(range(1, 7)))])


class HashingEncoder:
    """Bag-of-hashed-words encoder: deterministic, similar texts get similar vectors; records every doc text it embeds."""
    cache_key = "hashing|v1"
    meta = {"model": "hashing"}

    def __init__(self):
        self.doc_texts: list[str] = []

    @staticmethod
    def _vec(text):
        v = np.zeros(64, dtype=np.float32)
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        return v

    def encode_docs(self, texts):
        self.doc_texts += list(texts); return np.stack([self._vec(t) for t in texts])

    def encode_queries(self, texts):
        return np.stack([self._vec(t) for t in texts])


def new_ws(tmp_path, name):
    ws = Workspace(tmp_path / f"{name}.sqlite", models=ModelManager())
    enc = HashingEncoder()
    ws.models.register("enc:e5", enc)                                  # the "e5" retriever now uses the offline encoder
    return ws, enc


def snapshot(ws, spec, scope=None):
    out = []
    for q in QUERIES:
        ps, _ = ws.search(q, Settings(retriever=spec, k=10, dedup=None, exclude_sections=()), scope)
        out.append([(p.chunk.chunk_id, round(p.score, 6)) for p in ps])
    return out


@pytest.mark.parametrize("spec", ["bm25", "tfidf", "lsa", "e5", "rrf:bm25+e5", "rrf:bm25+lsa"])
def test_add_delete_add_equals_fresh_build(tmp_path, spec):
    inc, enc_inc = new_ws(tmp_path, "inc")
    inc.add_documents([make_doc("A"), make_doc("B"), make_doc("C")])
    snapshot(inc, spec)                                                # warm everything for the 3-paper library
    assert inc.remove("doc-B")
    inc.add_documents([make_doc("D")])
    fresh, _ = new_ws(tmp_path, "fresh")
    fresh.add_documents([make_doc("A"), make_doc("C"), make_doc("D")])
    for scope in (None, ["doc-C"], ["doc-A", "doc-D"]):
        assert snapshot(inc, spec, scope) == snapshot(fresh, spec, scope), (spec, scope)


def test_adding_a_paper_embeds_only_its_chunks_and_shared_vectors_survive_deletion(tmp_path):
    ws, enc = new_ws(tmp_path, "emb")
    ws.add_documents([make_doc("A"), make_doc("B"), make_doc("C")])
    ws.search("dense retrieval", Settings(retriever="e5"))
    n_abc = len(enc.doc_texts)
    assert n_abc == len(set(enc.doc_texts)) > 0                        # every chunk embedded exactly once
    ws.add_documents([make_doc("D")])
    ws.search("protein structure", Settings(retriever="e5"))
    d_chunks = len(enc.doc_texts) - n_abc
    assert 0 < d_chunks < n_abc                                         # only D's chunks were embedded, not the whole library
    before = ws.cache.count("hashing|v1|doc")
    ws.remove("doc-B")                                                  # B's vectors are swept, A/C/D's are kept
    after = ws.cache.count("hashing|v1|doc")
    assert 0 < after < before
    ws.search("knowledge graph", Settings(retriever="e5"))
    assert len(enc.doc_texts) == n_abc + d_chunks                       # nothing re-embedded after the deletion
