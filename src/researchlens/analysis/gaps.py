"""Research-gap discovery over an uploaded collection.

Every gap is grounded: it carries verbatim evidence (paper, section, page, quote) and states its BASIS:
  author-stated        the papers themselves say it (limitation / future-work / open-question cue sentences)
  collection-pattern   several papers state related limitations (TF-IDF similarity across papers)
  absence-in-collection  a method and a dataset both occur in the collection but are never combined in one paper

Scope discipline: "absence-in-collection" gaps only say something about THESE papers. No gap is ever called novel with
respect to the literature at large; that would need an external literature search, which this module does not do.
Proposed research questions and methodologies are TEMPLATES filled with the extracted names/quotes (suggestions for a
human to evaluate), not findings. Confidence is capped at "medium": no gap here has been reviewed by an expert.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

from researchlens.contracts import Document, KnowledgeGraph
from researchlens.generation.extractive import split_sentences
from researchlens.knowledge_graph.extract import BACKGROUND_SECTIONS, EXPLICIT_LIMIT, is_limitation_sentence
from researchlens.processing.chunking import chunk_document

FUTURE_CUES = re.compile(r"\b(future work|future research|future studies|we leave\b|left for future|remains? (?:an )?open|"
                         r"open (?:question|problem|challenge)s?|it would be (?:interesting|valuable|useful|fruitful)|"
                         r"further (?:research|work|investigation|study) is (?:needed|required)|has yet to|not yet been)\b", re.I)
DATASET_LIMIT = re.compile(r"\b(limited (?:to|by)|we limited|biased?|lack\w*|only (?:covers?|contains?|includes?|in English)|restricted to|"
                           r"no benchmark is perfect|small (?:size|number|scale)|English[- ]only|not (?:cover|include)\w*)\b", re.I)
CONFIDENCE_NOTE = "Confidence is an ordinal flag, not a probability: medium = three or more papers share the pattern; low = everything else. No gap has had expert review."
SCOPE_NOTE = ("Scope: this concerns only the uploaded papers. Whether it is open in the wider literature has NOT been checked.")


@dataclass
class Evidence:
    doc_id: str
    title: str
    page: int | None
    section: str
    quote: str


@dataclass
class Gap:
    id: str
    kind: str                       # repeated_limitation | stated_limitation | open_question | dataset_limitation | missing_comparison
    basis: str                      # author-stated | collection-pattern | absence-in-collection
    statement: str
    papers: list[str]               # titles
    evidence: list[Evidence]
    why_unaddressed: str
    research_question: str
    methodology: str
    limitations: str
    confidence: str                 # "low" | "medium" (never higher: no expert review)
    scope: str = SCOPE_NOTE


def _ev(d: Document, c, sent: str) -> Evidence:
    return Evidence(d.doc_id, d.title, c.page, c.section, sent)


def _sentences(documents: Sequence[Document]):
    for d in documents:
        for c in chunk_document(d):
            if c.section.lower() in ("references", "bibliography") or any(b in c.section.lower() for b in BACKGROUND_SECTIONS):
                continue
            for s in split_sentences(c.text):
                if 30 <= len(s) <= 500:
                    yield d, c, s


def _cluster(texts: list[str], owners: list[str], threshold: float) -> list[tuple[list[int], float]]:
    """Complete-linkage clusters (every pair from different papers must reach `threshold`, so chains A~B~C do not merge).
    Returns [(member indices, minimum pairwise cosine across papers)] for clusters spanning >= 2 papers."""
    if len(texts) < 2:
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts)
    S = (X @ X.T).toarray()
    pairs = sorted(((S[i, j], i, j) for i, j in combinations(range(len(texts)), 2) if owners[i] != owners[j] and S[i, j] >= threshold),
                   reverse=True)
    clusters: list[list[int]] = []
    where: dict[int, int] = {}
    for _, i, j in pairs:
        ci, cj = where.get(i), where.get(j)
        if ci is None and cj is None:
            clusters.append([i, j]); where[i] = where[j] = len(clusters) - 1
        elif ci is not None and cj is None:
            if all(S[j, m] >= threshold or owners[j] == owners[m] for m in clusters[ci]):
                clusters[ci].append(j); where[j] = ci
        elif cj is not None and ci is None:
            if all(S[i, m] >= threshold or owners[i] == owners[m] for m in clusters[cj]):
                clusters[cj].append(i); where[i] = cj
    out = []
    for c in clusters:
        if len({owners[m] for m in c}) >= 2:
            mc = min(S[a_, b_] for a_, b_ in combinations(c, 2) if owners[a_] != owners[b_])
            out.append((c, float(mc)))
    return out


def _round_robin(items: list, owner_of, limit: int) -> list:
    """Interleave items across papers so output is not dominated by whichever paper was uploaded first."""
    buckets: dict[str, list] = {}
    for it in items:
        buckets.setdefault(owner_of(it), []).append(it)
    out, depth = [], 0
    while len(out) < limit and any(depth < len(v) for v in buckets.values()):
        for v in buckets.values():
            if depth < len(v) and len(out) < limit:
                out.append(v[depth])
        depth += 1
    return out


def find_gaps(documents: Sequence[Document], kg: KnowledgeGraph | None = None, similarity: float = 0.22,
              max_per_kind: int = 6) -> list[Gap]:
    gaps: list[Gap] = []
    lim, fut, dsl = [], [], []
    seen: set[str] = set()
    for d, c, s in _sentences(documents):
        if s in seen:
            continue
        seen.add(s)
        if is_limitation_sentence(s, c.section) and not (FUTURE_CUES.search(s) and not EXPLICIT_LIMIT.search(s)) and c.section.lower() not in ("abstract",) and not re.match(r"^\W*(appendix|figure|table)", s, re.I):
            lim.append((d, c, s))
            if DATASET_LIMIT.search(s) and re.search(r"\b(dataset|corpus|benchmark|annotat\w+|data|domain)\b", s, re.I):
                dsl.append((d, c, s))
        if FUTURE_CUES.search(s):
            fut.append((d, c, s))

    # ---- repeated limitations across papers (collection-pattern)
    clustered: set[int] = set()
    used_quotes: set[str] = set()                       # one sentence is used by at most one gap, whatever its kind
    for g, mincos in sorted(_cluster([s for _, _, s in lim], [d.doc_id for d, _, _ in lim], similarity), key=lambda x: -x[1])[:max_per_kind]:
        clustered.update(g)
        ev = [_ev(*lim[i]) for i in g[:4]]
        titles = sorted({lim[i][0].title for i in g})
        gaps.append(Gap(
            id=f"gap-rep-{len(gaps) + 1}", kind="repeated_limitation", basis="collection-pattern",
            statement=f"{len(titles)} papers contain lexically similar limitation sentences (min pairwise cosine {mincos:.2f}; unverified as the same issue; compare the quotes).",
            papers=titles, evidence=ev,
            why_unaddressed="Each paper reports the limitation about its own work. Not checked whether other uploaded papers address it.",
            research_question="(Template) Can a method be designed or evaluated that removes the limitation stated above and is tested against the "
                              f"approaches of {', '.join(titles[:3])}?",
            methodology="(Template) Reproduce the affected setting from each paper, isolate the limiting factor by ablation, "
                        "then evaluate a modification on the same benchmarks with paired significance testing.",
            limitations="Similarity is lexical (TF-IDF), so two sentences can share vocabulary without describing the same limitation; "
                        "inspect the quotes.", confidence="medium" if len(titles) >= 3 else "low"))
        used_quotes.update(lim[i][2] for i in g)

    # ---- single-paper stated limitations not in a cluster (author-stated), interleaved across papers
    singles = [i for i in range(len(lim)) if i not in clustered and lim[i][2] not in used_quotes]
    for i in _round_robin(singles, lambda i: lim[i][0].doc_id, max_per_kind):
        d, c, s = lim[i]
        used_quotes.add(s)
        gaps.append(Gap(
            id=f"gap-lim-{len(gaps) + 1}", kind="stated_limitation", basis="author-stated",
            statement=f"{d.title.split(':')[0][:40]} states: “{s[:200]}{'…' if len(s) > 200 else ''}”", papers=[d.title], evidence=[_ev(d, c, s)],
            why_unaddressed="The authors present it as a limitation of their own work. Not checked whether other uploaded papers address it.",
            research_question="(Template) How can the stated limitation be overcome, and does the fix hold on benchmarks beyond the original paper's?",
            methodology="(Template) Re-run the original experiment, vary the factor named in the quote, report effect sizes with confidence intervals.",
            limitations="Based on one sentence; context may qualify it.", confidence="low"))

    # ---- open questions / future work (author-stated)
    fut = [x for x in fut if x[2] not in used_quotes]
    per_paper: dict[str, int] = {}
    for d, c, s in _round_robin(fut, lambda x: x[0].doc_id, max_per_kind):
        if per_paper.get(d.doc_id, 0) >= 2:
            continue
        per_paper[d.doc_id] = per_paper.get(d.doc_id, 0) + 1
        used_quotes.add(s)
        gaps.append(Gap(
            id=f"gap-open-{len(gaps) + 1}", kind="open_question", basis="author-stated",
            statement=f"Open or left for future work in {d.title.split(':')[0][:40]}: “{s[:200]}{'…' if len(s) > 200 else ''}”", papers=[d.title], evidence=[_ev(d, c, s)],
            why_unaddressed="Stated by the authors as not yet done. Not checked whether other uploaded papers address it.",
            research_question=f"(Template) Investigate the question left open in '{d.title[:70]}' (quoted evidence), "
                              "using the paper's own data and baselines as the starting point.",
            methodology="(Template) Replicate the baseline, implement the future-work direction, compare with identical splits and metrics.",
            limitations="Future-work sentences are sometimes boilerplate; the direction may already be addressed elsewhere in the literature.",
            confidence="low"))

    # ---- dataset limitations (subset of limitation sentences that talk about data)
    n_ds = 0
    for d, c, s in _round_robin([x for x in dsl if x[2] not in used_quotes], lambda x: x[0].doc_id, max_per_kind):
        n_ds += 1
        used_quotes.add(s)
        gaps.append(Gap(
            id=f"gap-data-{len(gaps) + 1}", kind="dataset_limitation", basis="author-stated",
            statement=f"Data limitation stated in {d.title.split(':')[0][:40]}: “{s[:200]}{'…' if len(s) > 200 else ''}”", papers=[d.title], evidence=[_ev(d, c, s)],
            why_unaddressed="Reported as a property of the dataset/setting used in the paper. Not checked whether other uploaded papers address it.",
            research_question="(Template) Does the reported result hold on data with different domain, size or annotation scheme?",
            methodology="(Template) Select an alternative dataset covering the missing condition and repeat the evaluation with the same protocol.",
            limitations="Whether the data issue materially affects conclusions is not established by the quote alone.", confidence="low"))

    # ---- missing comparisons: method x dataset never co-occurring in one paper (absence-in-collection)
    if kg is not None:
        by_id = {n.id: n for n in kg.nodes}
        paper_ents: dict[str, dict[str, set[str]]] = {}
        counts: dict[str, int] = {}
        for e in kg.edges:
            if e.type in ("uses_method", "evaluates_on"):
                kind = "method" if e.type == "uses_method" else "dataset"
                paper_ents.setdefault(e.source, {"method": set(), "dataset": set()})[kind].add(e.target)
                counts[e.target] = counts.get(e.target, 0) + int(e.attrs.get("mentions", 1))
        generic = {"transformer", "faiss", "wordpiece", "s2orc", "mlm", "ltr", "rtl"}      # architectures/tools/tasks, not comparable methods
        methods = {m for p in paper_ents.values() for m in p["method"]
                   if counts.get(m, 0) >= 4 and by_id[m].label.lower() not in generic}
        datasets = {x for p in paper_ents.values() for x in p["dataset"] if counts.get(x, 0) >= 2}
        # a pairing counts as "done" if ANY paper mentions both at all (raw mentions, no threshold): conservative on purpose
        raw = {n.attrs["doc_id"]: (set(n.attrs.get("raw_methods", [])), set(n.attrs.get("raw_datasets", [])))
               for n in kg.nodes if n.type == "paper"}
        norm = lambda nid: nid.split(":")[-1]
        done = {(m, x) for ms, xs in raw.values() for m in ms for x in xs}
        done = {(f"entity:method:{m}", f"entity:dataset:{x}") for m, x in done}
        cand = sorted(((m, x) for m in methods for x in datasets if (m, x) not in done),
                      key=lambda mx: -(counts[mx[0]] + counts[mx[1]]))
        titles = {n.attrs["doc_id"]: n.label for n in kg.nodes if n.type == "paper"}
        for m, x in cand[:max_per_kind]:
            mn, xn = by_id[m], by_id[x]
            mp = [titles[p] for p in mn.attrs.get("papers", []) if p in titles]
            xp = [titles[p] for p in xn.attrs.get("papers", []) if p in titles]
            ev = []
            for e in kg.edges:
                if e.target in (m, x) and e.type in ("uses_method", "evaluates_on") and e.attrs.get("evidence"):
                    q = e.attrs["evidence"][0]
                    if q.get("quote"):
                        ev.append(Evidence(q["doc_id"], titles.get(q["doc_id"].split("paper:")[-1], q["doc_id"]), q.get("page"), q.get("section", ""), q["quote"]))
                        if len(ev) >= 2:
                            break
            gaps.append(Gap(
                id=f"gap-cmp-{len(gaps) + 1}", kind="missing_comparison", basis="absence-in-collection",
                statement=f"{mn.label} and {xn.label} both appear in the collection, but no paper was detected reporting {mn.label} on {xn.label}.",
                papers=sorted(set(mp + xp)), evidence=ev,
                why_unaddressed=f"Within the uploaded papers, {mn.label} is used in {len(mp)} paper(s) and {xn.label} in {len(xp)}, not detected together.",
                research_question=f"(Template) How does {mn.label} perform on {xn.label} compared with the methods already evaluated there?",
                methodology=f"(Template) Run {mn.label} on {xn.label} with the dataset's standard protocol and metrics; compare with the baselines "
                            f"reported in the papers that use {xn.label}, using paired bootstrap tests.",
                limitations="Based on automatic entity extraction (recall is limited, so a comparison may exist but not be detected) and on a small collection; "
                            "this is not evidence that the comparison is missing from the literature.", confidence="low"))
    return gaps


def gaps_markdown(gaps: Sequence[Gap]) -> str:
    """Markdown listing; note CONFIDENCE_NOTE applies to every gap."""
    out = []
    for g in gaps:
        out.append(f"#### {g.id} · {g.kind.replace('_', ' ')} · basis: {g.basis} · confidence: {g.confidence}\n{g.statement}\n")
        out.append("**Evidence**\n" + "\n".join(f"- “{e.quote}” — {e.title[:70]}, §{e.section}, "
                                                  f"{'p. ' + str(e.page) if e.page is not None else 'page unknown'}" for e in g.evidence))
        out.append(f"\n**Status in the collection (not checked against the other papers):** {g.why_unaddressed}\n"
                   f"**Proposed research question:** {g.research_question}\n**Potential methodology:** {g.methodology}\n"
                   f"**Limitations:** {g.limitations}\n*{g.scope}*\n")
    return "\n".join(out)
