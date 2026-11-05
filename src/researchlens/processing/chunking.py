"""Section-aware chunking for parsed papers (GROBID/PyMuPDF output).

Rules: chunks never cross a section boundary; paragraphs are the default unit; a
paragraph longer than `max_tokens` is split into overlapping windows; each chunk
carries its section title for prefixing at index time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class Section:
    title: str
    paragraphs: list[str]


@dataclass
class Chunk:
    section: str
    text: str
    para_index: int     # index of the source paragraph within its section
    window: int         # window number inside that paragraph (0 if unsplit)


def section_aware_chunks(sections: Iterable[Section], max_tokens: int = 256, overlap: int = 32) -> list[Chunk]:
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")
    chunks: list[Chunk] = []
    step = max_tokens - overlap
    for sec in sections:
        for pi, para in enumerate(sec.paragraphs):
            words = para.split()
            if not words:
                continue
            if len(words) <= max_tokens:
                chunks.append(Chunk(sec.title, para.strip(), pi, 0))
                continue
            for w, start in enumerate(range(0, len(words), step)):
                piece = words[start:start + max_tokens]
                chunks.append(Chunk(sec.title, " ".join(piece), pi, w))
                if start + max_tokens >= len(words):
                    break
    return chunks


# ---------------------------------------------------------------- contract adapter
def chunk_document(doc, max_tokens: int = 256, overlap: int = 32, include_abstract: bool = True):
    """Document (contracts) -> list[Chunk] (contracts). Chunk ids are deterministic."""
    from researchlens.contracts import Chunk as CChunk
    from researchlens.contracts import Section as CSection

    sections = list(doc.sections)
    if include_abstract and doc.abstract:
        sections = [CSection(title="Abstract", paragraphs=[doc.abstract])] + sections
    out = []
    for si, sec in enumerate(sections):
        for c in section_aware_chunks([Section(sec.title, sec.paragraphs)], max_tokens, overlap):
            out.append(CChunk(chunk_id=f"{doc.doc_id}::s{si}::p{c.para_index}::w{c.window}",
                              doc_id=doc.doc_id, section=sec.title, text=c.text, order=len(out)))
    return out
