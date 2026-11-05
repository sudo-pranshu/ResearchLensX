"""H2 generation / revision / attribution logic tested offline with fake LLM + fake NLI."""
import math

import numpy as np

from researchlens.contracts import Chunk, Generator, RetrievedPassage
from researchlens.evaluation.attribution import CitationJudge, answer_f1, token_f1
from researchlens.generation.llm_rag import ABSTAIN, LLMRAGGenerator, build_messages, is_abstention, parse_answer
from researchlens.generation.verified import VerifiedReviser
from researchlens.verification.claims import extract_claims
from researchlens.verification.nli import NLIVerifier

P = ["We evaluate on SQuAD and report F1.", "Our model uses BERT-base.", "The weather was sunny.",
     "We do not evaluate on GLUE."]


def passages():
    return [RetrievedPassage(chunk=Chunk(chunk_id=f"c{i}", doc_id="d", section="Exp", text=t, order=i),
                             score=1.0, rank=i + 1, retriever="t") for i, t in enumerate(P)]


def keyword_nli(premises, hyps):
    """Entail if every content word (len>3) of the hypothesis appears in the premise; contradict on 'do not'."""
    out = []
    for p, h in zip(premises, hyps):
        words = [w.strip(".,").lower() for w in h.split() if len(w.strip(".,")) > 3]
        if "do not" in p and any(w in p.lower() for w in words):
            out.append([0.05, 0.9, 0.05])
        elif words and all(w in p.lower() for w in words):
            out.append([0.9, 0.05, 0.05])
        else:
            out.append([0.1, 0.1, 0.8])
    return np.array(out)


def test_parse_answer_citations_and_markers():
    s = parse_answer("They use SQuAD [1]. The encoder is BERT-base [2][3]. Extra [9].", 4)
    assert [x.text for x in s] == ["They use SQuAD.", "The encoder is BERT-base.", "Extra."]
    assert [x.citations for x in s] == [[0], [1, 2], []]          # [9] out of range is dropped
    assert parse_answer("A claim [1, 2].", 4)[0].citations == [0, 1]


def test_generator_contract_with_fake_llm():
    gen = LLMRAGGenerator(lambda msgs: "They evaluate on SQuAD [1].", name="fake")
    assert isinstance(gen, Generator)
    a = gen.generate("Which dataset?", passages())
    assert a.mode == "rag" and a.sentences[0].citations == [0]
    msgs = build_messages("Which dataset?", passages())
    assert "[1] (Exp) We evaluate on SQuAD" in msgs[1]["content"]
    assert is_abstention(ABSTAIN) and not is_abstention("They use SQuAD.")


def test_reviser_reassigns_prunes_and_drops():
    gen = LLMRAGGenerator(lambda m: "", name="fake")
    raw = "We evaluate on SQuAD [3]. Our model uses BERT-base [2][3]. The weather was rainy [3]. We evaluate on GLUE [1]."
    a = gen.generate("q", passages(), raw=raw)
    v = NLIVerifier(keyword_nli, granularity="passage")
    out, log = VerifiedReviser(v).revise(a)
    acts = {x["sentence"]: x["action"] for x in log}
    assert acts == {0: "reassigned", 1: "pruned", 2: "dropped_unsupported", 3: "dropped_contradicted"}
    assert [s.citations for s in out.sentences] == [[0], [1]]
    assert all(s.verification is not None and not s.verification.is_mock for s in out.sentences)
    vo, _ = VerifiedReviser(v, mode="verify_only").revise(a)        # ablation: unchanged text/citations
    assert [s.citations for s in vo.sentences] == [s.citations for s in a.sentences]


def test_reviser_all_dropped_becomes_abstention():
    a = LLMRAGGenerator(lambda m: "", name="f").generate("q", passages(), raw="The weather was rainy today [3].")
    out, _ = VerifiedReviser(NLIVerifier(keyword_nli, granularity="passage")).revise(a)
    assert out.sentences[0].text == ABSTAIN and extract_claims(out) == []


def test_citation_judge_alce_semantics():
    gen = LLMRAGGenerator(lambda m: "", name="f")
    a = gen.generate("q", passages(), raw="We evaluate on SQuAD [1][3]. Our model uses BERT-base [3].")
    m = CitationJudge(keyword_nli).evaluate(a)
    assert m["citation_recall"] == 0.5                # sentence 1 supported by {1,3}; sentence 2 not by {3}
    assert m["citation_precision"] == 1 / 3           # [1] needed, [3] irrelevant twice
    ab = gen.generate("q", passages(), raw=ABSTAIN)
    assert CitationJudge(keyword_nli).evaluate(ab)["abstained"] and math.isnan(CitationJudge(keyword_nli).evaluate(ab)["citation_recall"])


def test_answer_f1():
    assert token_f1("The SQuAD dataset", "SQuAD dataset") == 1.0
    assert answer_f1("BERT", ["RoBERTa", "BERT"]) == 1.0
