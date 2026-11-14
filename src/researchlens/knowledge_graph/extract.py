"""Rule-based scientific information extraction with provenance.

Nothing here is generated: every method / dataset / metric mention, finding and limitation is a verbatim
span or sentence from a chunk, returned with (doc_id, chunk_id, section, page, quote) so it can be audited.
Precision is favoured over recall; the patterns are deliberately conservative. Known weaknesses: names that
are not capitalised, multi-word dataset names, and sentences whose cue words are phrased unusually are missed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from researchlens.contracts import Chunk
from researchlens.generation.extractive import split_sentences

NAME = r"[A-Z][A-Za-z0-9\-\.]*(?:\s+[A-Z][A-Za-z0-9\-\.]*){0,2}"
DATASET_PATTERNS = [
    re.compile(rf"\b({NAME})\s+(?:dataset|benchmark|corpus|collection|test set)s?\b"),
    re.compile(rf"\b(?:evaluate[sd]?|evaluation|trained|tested|experiments?|results?)\s+(?:is |are )?(?:on|using)\s+(?:the\s+)?({NAME})\b"),
]
METHOD_PATTERNS = [
    re.compile(rf"\b(?:we|authors?)\s+(?:propose|introduce|present|develop|design)\s+(?:a |an |the |new )*({NAME})\b"),
    re.compile(rf"\b({NAME})\s+(?:model|method|approach|architecture|encoder|retriever|framework|algorithm)s?\b"),
    re.compile(rf"\b(?:based on|built on|using|uses|use|fine-tun\w+|apply|applies)\s+(?:the\s+)?({NAME})\b"),
]
METRICS = {  # surface (case-sensitive where short) -> canonical
    "F1": "F1", "F-score": "F1", "accuracy": "accuracy", "BLEU": "BLEU", "ROUGE": "ROUGE", "METEOR": "METEOR",
    "perplexity": "perplexity", "AUROC": "AUROC", "AUC": "AUC", "MRR": "MRR", "nDCG": "nDCG", "NDCG": "nDCG",
    "Recall@": "Recall@k", "recall@": "Recall@k", "Precision@": "Precision@k", "precision@": "Precision@k",
    "exact match": "Exact Match", "EM score": "Exact Match", "macro-F1": "macro-F1", "MAP": "MAP",
}
_METRIC_RE = re.compile("|".join(re.escape(k) + (r"\b" if not k.endswith("@") else "") for k in sorted(METRICS, key=len, reverse=True)))
STOP_FIRST = {"The", "This", "That", "These", "Those", "Our", "We", "In", "On", "For", "A", "An", "It", "Its", "Each", "All",
              "Table", "Figure", "Section", "Appendix", "However", "Moreover", "Such", "Both", "One", "Two", "Then", "When",
              "While", "Since", "Because", "Although", "Here", "There", "They", "Their", "Using", "Based", "With", "Without"}
GENERIC = {"Model", "Method", "Approach", "Dataset", "Benchmark", "Neural", "Language", "Training", "Test", "Task", "Data",
           "Pre", "Fine", "Existing", "Previous", "Recent", "Standard", "Baseline", "Proposed", "Deep", "Large", "Small",
           "NLP", "Python", "Dense", "Sentence", "Wiki", "Decoding", "Fast", "Thorough", "Sparse", "Neural Network", "Learning"}
FINDING_CUES = re.compile(r"\b(outperform\w*|achieve[sd]?|improv\w+|obtain\w*|surpass\w*|we (?:find|show|observe|found|demonstrate)\b|"
                          r"results (?:show|indicate|suggest)|state-of-the-art|significantly)\b", re.I)
LIMITATION_CUES = re.compile(r"\b(limitation\w*|fails? to|fail(?:ed|ure)s?|cannot|can not|does not (?:generali[sz]e|scale|handle)|"
                             r"only (?:evaluate[sd]?|tested|covers?|considers?)|we leave\b|future work|remains? (?:an )?open|"
                             r"left (?:for|to) future|not (?:able|robust)|struggle\w*)\b", re.I)
FINDING_SECTIONS = ("abstract", "result", "experiment", "conclusion", "discussion", "evaluation", "analysis", "introduction")
LIMIT_SECTIONS = ("limitation", "discussion", "conclusion", "future", "analysis", "error")


@dataclass
class Mention:
    kind: str                    # method | dataset | metric | finding | limitation
    text: str                    # surface form (entities) or the full sentence (findings/limitations)
    chunk_id: str
    doc_id: str
    section: str
    page: int | None
    quote: str                   # the sentence it came from

    def evidence(self) -> dict:
        return {"doc_id": self.doc_id, "chunk_id": self.chunk_id, "section": self.section, "page": self.page, "quote": self.quote}


def _clean_name(s: str) -> str | None:
    s = s.strip(" .,-;:()")
    words = s.split()
    while words and words[0] in STOP_FIRST:
        words = words[1:]
    while words and (words[-1] in STOP_FIRST or words[-1] in GENERIC):
        words = words[:-1]
    if not words:
        return None
    s = " ".join(words)
    if len(s) < 3 or len(s) > 40 or s in GENERIC or s.isdigit() or not re.search(r"[A-Za-z]", s):
        return None
    if all(w in GENERIC or w in STOP_FIRST for w in words):
        return None
    return s


def extract_from_chunk(chunk: Chunk) -> list[Mention]:
    out: list[Mention] = []
    sec = chunk.section.lower()
    for sent in split_sentences(chunk.text):
        if len(sent) < 25 or len(sent) > 600:
            continue
        mk = lambda kind, text: Mention(kind, text, chunk.chunk_id, chunk.doc_id, chunk.section, chunk.page, sent)
        seen: set[tuple[str, str]] = set()          # one mention per (kind, name) per sentence, however many patterns hit
        for pats, kind in ((DATASET_PATTERNS, "dataset"), (METHOD_PATTERNS, "method")):
            for pat in pats:
                for m in pat.finditer(sent):
                    name = _clean_name(m.group(1))
                    if name and (kind, name.lower()) not in seen:
                        seen.add((kind, name.lower()))
                        out.append(mk(kind, name))
        for m in _METRIC_RE.finditer(sent):
            if ("metric", METRICS[m.group(0)]) not in seen:
                seen.add(("metric", METRICS[m.group(0)]))
                out.append(mk("metric", METRICS[m.group(0)]))
        if any(s in sec for s in FINDING_SECTIONS) and FINDING_CUES.search(sent) and re.search(r"\d|than|state-of-the-art", sent):
            out.append(mk("finding", sent))
        if any(s in sec for s in LIMIT_SECTIONS) and LIMITATION_CUES.search(sent):
            out.append(mk("limitation", sent))
    return out


def title_method(title: str) -> str | None:
    """'BERT: Pre-training of ...' -> 'BERT' (a short name before a colon in the title)."""
    head = title.split(":")[0].strip() if ":" in title else ""
    if head and len(head.split()) <= 3 and len(head) <= 24 and re.match(r"^[A-Z][A-Za-z0-9\-]*(?:\s+[A-Z0-9][A-Za-z0-9\-]*)*$", head):
        return head
    return None
