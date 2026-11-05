"""QASPER v0.3 loader (raw JSON release from AllenAI).

Raw format (one JSON object per split, keyed by paper id):
  {paper_id: {"title", "abstract",
              "full_text": [{"section_name", "paragraphs": [str, ...]}, ...],
              "figures_and_tables": [{"caption", "file"}, ...],
              "qas": [{"question", "question_id",
                       "answers": [{"answer": {"unanswerable", "extractive_spans", "yes_no",
                                               "free_form_answer", "evidence", "highlighted_evidence"},
                                    "annotation_id", "worker_id"}, ...]}, ...]}}

Evidence strings are copies of paragraph text. Evidence pointing at a figure/table
is written as "FLOAT SELECTED: <caption>", so captions are indexed as pseudo-paragraphs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

FLOAT_PREFIX = "FLOAT SELECTED: "


@dataclass
class Paragraph:
    pid: int            # global index into the corpus
    paper_id: str
    section: str
    text: str
    is_float: bool = False

    def indexed_text(self, title: str = "", with_section: bool = True) -> str:
        """Text given to retrievers. Section prefixing is an H1 ablation switch."""
        if not with_section:
            return self.text
        head = " | ".join(x for x in (title, self.section) if x)
        return f"{head}\n{self.text}" if head else self.text


@dataclass
class Question:
    qid: str
    paper_id: str
    question: str
    gold_union: set[int]                                # union of evidence pids over annotators
    gold_per_annotator: list[set[int]] = field(default_factory=list)
    answer_types: list[str] = field(default_factory=list)
    unmatched_evidence: int = 0                         # evidence strings we failed to align
    answers: list[str] = field(default_factory=list)    # gold answer string per annotator (for Answer-F1)

    @property
    def majority_type(self) -> str:
        from collections import Counter
        return Counter(self.answer_types).most_common(1)[0][0] if self.answer_types else "none"


@dataclass
class QasperSplit:
    paragraphs: list[Paragraph]
    paper_pids: dict[str, list[int]]
    titles: dict[str, str]
    questions: list[Question]
    stats: dict


def _answer_type(a: dict) -> str:
    if a.get("unanswerable"):
        return "none"
    if a.get("extractive_spans"):
        return "extractive"
    if a.get("yes_no") is not None:
        return "boolean"
    if a.get("free_form_answer"):
        return "abstractive"
    return "none"


def _answer_string(a: dict) -> str:
    """Official QASPER convention: spans joined, else free-form, else Yes/No, else 'Unanswerable'."""
    if a.get("unanswerable"):
        return "Unanswerable"
    if a.get("extractive_spans"):
        return ", ".join(a["extractive_spans"])
    if a.get("free_form_answer"):
        return a["free_form_answer"]
    if a.get("yes_no") is not None:
        return "Yes" if a["yes_no"] else "No"
    return ""


def _norm(s: str) -> str:
    return " ".join(s.split())


def load_qasper(path: str | Path, include_floats: bool = True) -> QasperSplit:
    data = json.loads(Path(path).read_text())
    paragraphs: list[Paragraph] = []
    paper_pids: dict[str, list[int]] = {}
    titles: dict[str, str] = {}
    questions: list[Question] = []
    n_unmatched = n_evidence = n_no_evidence = 0

    for paper_id, paper in data.items():
        titles[paper_id] = paper.get("title", "")
        lookup: dict[str, int] = {}
        pids: list[int] = []

        def add(section, text, is_float=False):
            text = (text or "").strip()
            if not text:
                return
            key = _norm(text)
            if key in lookup:          # duplicate paragraph text inside one paper
                return
            p = Paragraph(len(paragraphs), paper_id, section or "", text, is_float)
            paragraphs.append(p)
            lookup[key] = p.pid
            pids.append(p.pid)

        for sec in paper.get("full_text", []) or []:
            for para in sec.get("paragraphs", []) or []:
                add(sec.get("section_name"), para)
        if include_floats:
            for ft in paper.get("figures_and_tables", []) or []:
                add("Figure/Table", ft.get("caption"), is_float=True)
        paper_pids[paper_id] = pids

        for qa in paper.get("qas", []):
            per_ann, types, unmatched, strings = [], [], 0, []
            for ans in qa.get("answers", []):
                a = ans["answer"]
                types.append(_answer_type(a))
                strings.append(_answer_string(a))
                ev = set()
                for e in a.get("evidence", []) or []:
                    n_evidence += 1
                    key = _norm(e[len(FLOAT_PREFIX):] if e.startswith(FLOAT_PREFIX) else e)
                    if key in lookup:
                        ev.add(lookup[key])
                    else:
                        unmatched += 1
                per_ann.append(ev)
            union = set().union(*per_ann) if per_ann else set()
            if not union:
                n_no_evidence += 1
            n_unmatched += unmatched
            questions.append(Question(qa["question_id"], paper_id, qa["question"], union, per_ann, types, unmatched, strings))

    stats = {
        "papers": len(paper_pids), "paragraphs": len(paragraphs), "questions": len(questions),
        "questions_without_evidence": n_no_evidence,
        "evidence_strings": n_evidence, "evidence_unmatched": n_unmatched,
    }
    return QasperSplit(paragraphs, paper_pids, titles, questions, stats)


# ---------------------------------------------------------------- contract adapter
def load_qasper_documents(path: str | Path, limit: int | None = None, paper_ids: list[str] | None = None):
    """QASPER raw JSON -> list[contracts.Document] (source="qasper"). Used by the demo app
    so the full pipeline runs on real papers without depending on PDF ingestion."""
    from researchlens.contracts import Document, Section as CSection

    data = json.loads(Path(path).read_text())
    ids = paper_ids or list(data)[:limit]
    docs = []
    for pid in ids:
        p = data[pid]
        sections = [CSection(title=(s.get("section_name") or "").split(":::")[-1].strip() or "Untitled",
                             paragraphs=[x for x in (s.get("paragraphs") or []) if x and x.strip()])
                    for s in p.get("full_text", []) or []]
        sections = [s for s in sections if s.paragraphs]
        docs.append(Document(doc_id=pid, title=p.get("title", ""), abstract=p.get("abstract", ""),
                             sections=sections, source="qasper", source_path=str(path)))
    return docs
