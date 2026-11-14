from researchlens.analysis.compare import comparison_table, conflicting_pairs, entity_overlap, table_markdown, table_rows_flat
from researchlens.knowledge_graph.builder import GraphBuilder
from tests.test_knowledge_graph import A, B


def test_table_cells_trace_to_pages_and_quotes():
    kg = GraphBuilder().build([A, B])
    t = comparison_table(kg, [A, B])
    a, b = t
    assert any(i.text == "ColBERTish" for i in a.cells["method"])
    assert any(i.text == "MS MARCO" and i.page is not None and i.quote for i in a.cells["dataset"])
    assert a.cells["limitation"] and all(i.quote for i in a.cells["limitation"])
    assert not b.cells["limitation"]                                   # nothing extracted -> empty, rendered as "not detected"
    flat = table_rows_flat(t)
    assert flat[1]["Limitations"] == "not detected" and "(p. " in flat[0]["Datasets"]


def test_entity_overlap_shared_vs_unique():
    kg = GraphBuilder().build([A, B])
    ov = entity_overlap(kg)
    assert "MS MARCO" in ov["dataset"]["shared"]
    assert "MRR" in ov["metric"]["unique"] or "MRR" in ov["metric"]["shared"]


def test_markdown_export_contains_every_paper_and_page_refs():
    md = table_markdown(comparison_table(GraphBuilder().build([A, B]), [A, B]))
    assert A.title in md and B.title in md and "(p. " in md and "not detected" in md


def test_conflicting_pairs_empty_without_verifier():
    assert conflicting_pairs(GraphBuilder().build([A, B])) == []
