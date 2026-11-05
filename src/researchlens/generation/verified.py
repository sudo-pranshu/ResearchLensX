"""Verified RAG: claim-level verification + citation reassignment + unsupported-claim handling.
Owner: Pawan (initial: Pranshu, sprint).

For each claim (answer sentence), the verifier judges it against EVERY retrieved passage:
  * cited passage(s) SUPPORT     -> keep; prune citations that do not support it        ("kept"/"pruned")
  * only uncited passages SUPPORT-> reassign citations to the best supporting passages ("reassigned")
  * a passage CONTRADICTS, none SUPPORT -> drop the sentence                            ("dropped_contradicted")
  * no passage SUPPORTS          -> drop the sentence                                   ("dropped_unsupported")
If every sentence is dropped, the answer becomes an explicit abstention.
mode="verify_only" annotates labels but leaves text and citations unchanged (ablation).
"""
from __future__ import annotations

from researchlens.contracts import VERDICTS, Answer, AnswerSentence, EvidenceJudgement, VerificationResult
from researchlens.generation.llm_rag import ABSTAIN
from researchlens.verification.claims import extract_claims


class VerifiedReviser:
    def __init__(self, verifier, mode: str = "revise", reassign: bool = True, drop_unsupported: bool = True,
                 max_citations: int = 2):
        if mode not in ("revise", "verify_only"):
            raise ValueError(mode)
        self.verifier, self.mode = verifier, mode
        self.reassign, self.drop_unsupported, self.max_citations = reassign, drop_unsupported, max_citations

    def revise(self, answer: Answer) -> tuple[Answer, list[dict]]:
        texts = [p.chunk.text for p in answer.passages]
        claims = dict(extract_claims(answer))
        new_sents, log = [], []
        for i, s in enumerate(answer.sentences):
            if i not in claims:                                        # abstention / fragment: keep as is
                new_sents.append(s.model_copy())
                continue
            judged = self.verifier.judge_texts(claims[i].text, texts) if texts else []
            evidence = [EvidenceJudgement(chunk_id=p.chunk.chunk_id, label=l,
                                          probs={v: float(x) for v, x in zip(VERDICTS, pr)}, rationale_sentences=r)
                        for p, (l, pr, r) in zip(answer.passages, judged)]
            label, conf = self.verifier.aggregate(judged) if judged else ("INSUFFICIENT", 1.0)
            vr = VerificationResult(claim=claims[i], label=label, confidence=conf, evidence=evidence,
                                    model=getattr(self.verifier.scorer, "name", "scorer"), is_mock=False)
            supporting = sorted((k for k, (l, _, _) in enumerate(judged) if l == "SUPPORTS"),
                                key=lambda k: -judged[k][1][0])
            contradicting = [k for k, (l, _, _) in enumerate(judged) if l == "CONTRADICTS"]
            cited_support = [k for k in s.citations if k in supporting]
            if cited_support:
                action, cites = ("kept" if cited_support == s.citations else "pruned"), cited_support
            elif supporting and self.reassign:
                action, cites = "reassigned", supporting[: self.max_citations]
            elif supporting:
                action, cites = "unsupported_by_citation", s.citations
            elif contradicting:
                action, cites = "dropped_contradicted", None
            else:
                action, cites = "dropped_unsupported", None
            log.append({"sentence": i, "label": label, "action": action, "old": s.citations, "new": cites})
            if self.mode == "verify_only":
                new_sents.append(AnswerSentence(text=s.text, citations=s.citations, verification=vr))
            elif cites is None and self.drop_unsupported:
                continue
            else:
                new_sents.append(AnswerSentence(text=s.text, citations=cites if cites is not None else s.citations,
                                                verification=vr))
        if not new_sents:
            new_sents = [AnswerSentence(text=ABSTAIN, citations=[])]
        out = answer.model_copy(update={"sentences": new_sents, "mode": "verified_rag",
                                        "generator": answer.generator + f"+verified({self.mode})"})
        return out, log
