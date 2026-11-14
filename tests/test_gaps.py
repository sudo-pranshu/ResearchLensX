from researchlens.analysis.gaps import find_gaps, gaps_markdown
from researchlens.contracts import Document, Section
from researchlens.knowledge_graph.builder import GraphBuilder
from tests.test_knowledge_graph import A, B


def mk(i, title, sections, abstract="We study retrieval for science."):
    return Document(doc_id=i, title=title, abstract=abstract, source="fixture",
                    sections=[Section(title=s, paragraphs=ps, pages=list(range(1, len(ps) + 1))) for s, ps in sections.items()])


P1 = mk("P1", "Alpha: Dense Retrieval", {"Limitations": [
    "However, our approach fails to generalize to out-of-domain scientific corpora because the encoder overfits to training vocabulary.",
    "In future work we plan to study multilingual scientific retrieval across several languages."]})
P2 = mk("P2", "Beta: Hybrid Search", {"Discussion": [
    "A limitation of this work is that the model fails to generalize to out-of-domain corpora since the encoder overfits training vocabulary.",
    "Claim 2: The coronavirus cannot thrive in warmer climates and it has failed in many studies."]})


def test_repeated_limitation_across_papers_is_clustered_with_evidence():
    gaps = find_gaps([P1, P2])
    rep = [g for g in gaps if g.kind == "repeated_limitation"]
    assert len(rep) == 1 and set(rep[0].papers) == {P1.title, P2.title}
    assert all(e.quote and e.page is not None for e in rep[0].evidence)
    assert rep[0].basis == "collection-pattern"


def test_dataset_examples_and_boilerplate_are_not_limitations():
    quotes = [e.quote for g in find_gaps([P1, P2]) for e in g.evidence]
    assert not any("coronavirus" in q for q in quotes)


def test_open_question_from_future_work_cue():
    g = [g for g in find_gaps([P1, P2]) if g.kind == "open_question"]
    assert g and "multilingual" in g[0].evidence[0].quote and g[0].basis == "author-stated"


def test_every_gap_has_all_required_fields_and_scope_caveat():
    gaps = find_gaps([A, B, P1, P2], GraphBuilder().build([A, B, P1, P2]))
    assert gaps
    for g in gaps:
        assert g.evidence and g.why_unaddressed and g.research_question and g.methodology and g.limitations
        assert g.confidence in ("low", "medium")                    # never "high": no expert review
        assert "NOT been checked" in g.scope                        # no novelty claim beyond the collection
        assert g.methodology.startswith("(Template)")               # suggestions are labelled as templates


def test_missing_comparison_is_collection_scoped():
    a = mk("X1", "Mxo: A Method", {"Methods": ["The Mxo model is described in detail here. The Mxo model is trained on large corpora. The Mxo model works well in practice. The Mxo model scales to many documents."]})
    b = mk("X2", "Study of Data", {"Experiments": ["We evaluate on the Zedd dataset in our experiments. The Zedd dataset is large and diverse. Results on the Zedd dataset improve steadily."]})
    kg = GraphBuilder(min_mentions=2).build([a, b])
    cmp_ = [g for g in find_gaps([a, b], kg) if g.kind == "missing_comparison"]
    assert cmp_ and "no paper was detected reporting Mxo on Zedd" in cmp_[0].statement and cmp_[0].basis == "absence-in-collection"
    assert "not evidence that the comparison is missing from the literature" in cmp_[0].limitations


def test_markdown_lists_evidence_and_caveats():
    md = gaps_markdown(find_gaps([P1, P2]))
    assert "Proposed research question" in md and "Scope:" in md and "p. " in md


def test_related_work_and_cited_statements_are_not_attributed_to_the_paper():
    rw = mk("R", "Gamma: Survey Paper", {"Related Work": ["Prior retrievers are limited to English and our survey finds they fail to generalize widely."],
                                         "Discussion": ["Smith et al. [3] note the encoder fails to generalize to new domains and we agree with this view fully.",
                                                        "A limitation of this work is that our evaluation covers only scientific abstracts, not full text."]})
    quotes = [e.quote for g in find_gaps([rw]) for e in g.evidence]
    assert not any("Prior retrievers" in q for q in quotes)          # related work excluded
    assert not any("Smith et al." in q for q in quotes)              # citation marker => statement about someone else's work
    assert any("our evaluation covers only scientific abstracts" in q for q in quotes)


def test_one_sentence_is_used_by_only_one_gap():
    quotes = [e.quote for g in find_gaps([P1, P2]) for e in g.evidence]
    assert len(quotes) == len(set(quotes))


def test_gaps_are_interleaved_across_papers_not_first_upload_first():
    many = mk("M", "Many: Limitations Paper", {"Limitations": [f"A limitation of this work is aspect number {i} of the evaluation protocol design here." for i in range(8)]})
    one = mk("O", "One: Other Paper", {"Limitations": ["A limitation of this work is the single narrow domain of the training corpus used."]})
    gaps = find_gaps([many, one], max_per_kind=3)
    assert {"Many: Limitations Paper", "One: Other Paper"} <= {p for g in gaps for p in g.papers}


def test_missing_comparison_not_claimed_when_pair_is_mentioned_once_together():
    a = mk("X1", "Mxo: A Method", {"Methods": ["The Mxo model is described in detail here. The Mxo model is trained on large corpora. The Mxo model works well in practice. The Mxo model scales to many documents.",
                                               "We also evaluate the Mxo model on the Zedd dataset once in a small side experiment here."]})
    b = mk("X2", "Study of Data", {"Experiments": ["We evaluate on the Zedd dataset in our experiments. The Zedd dataset is large and diverse. Results on the Zedd dataset improve steadily."]})
    kg = GraphBuilder(min_mentions=2).build([a, b])
    assert not [g for g in find_gaps([a, b], kg) if g.kind == "missing_comparison"]


def test_confidence_note_and_template_prefix():
    from researchlens.analysis.gaps import CONFIDENCE_NOTE
    assert "not a probability" in CONFIDENCE_NOTE
    assert all(g.research_question.startswith("(Template)") for g in find_gaps([P1, P2]))
