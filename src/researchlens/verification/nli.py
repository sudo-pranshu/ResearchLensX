"""NLI-based claim verification.

Labels follow contracts.Verdict: SUPPORTS / CONTRADICTS / INSUFFICIENT
(INSUFFICIENT is displayed as "NOT ENOUGH INFO", matching SciFact's NEI).

  scorer   = HFNLIScorer("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")      # verifier V
  verifier = NLIVerifier(scorer)
  result   = verifier.verify(claim, passages)                                  # -> VerificationResult

Premise granularity:
  * "passage": the whole passage is the premise (truncated at max_length tokens);
  * "sentence": each sentence of the passage is scored; the passage judgement takes the
    sentence with the strongest SUPPORTS/CONTRADICTS evidence (rationale = that sentence).
The scorer is injected so the decision logic is unit-testable without downloading a model.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from researchlens.contracts import VERDICTS, Claim, EvidenceJudgement, RetrievedPassage, VerificationResult

# scorer(premises, hypotheses) -> array (n, 3), columns in VERDICTS order, rows sum to 1
Scorer = Callable[[Sequence[str], Sequence[str]], np.ndarray]


class HFNLIScorer:
    """Any HF sequence-classification NLI model. Column order is derived from config.id2label."""

    def __init__(self, model_name: str, batch_size: int = 16, max_length: int = 512, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from researchlens.retrieval.dense import pick_device
        self.torch = torch
        self.device = device or pick_device()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        self.bs, self.max_length = batch_size, max_length
        id2label = {int(i): l.lower() for i, l in self.model.config.id2label.items()}
        want = {"SUPPORTS": "entail", "CONTRADICTS": "contra", "INSUFFICIENT": "neutral"}
        self.cols = []
        for v in VERDICTS:
            match = [i for i, l in id2label.items() if l.startswith(want[v])]
            if len(match) != 1:
                raise ValueError(f"{model_name}: cannot map {v} from id2label={id2label}")
            self.cols.append(match[0])
        self.meta = {"model": model_name, "revision": getattr(self.model.config, "_commit_hash", None) or "unknown",
                     "id2label": id2label, "max_length": max_length}
        self.name = model_name

    def __call__(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(premises), 3), dtype=np.float32)
        order = np.argsort([-len(p) for p in premises], kind="stable")
        for s in range(0, len(premises), self.bs):
            idx = order[s:s + self.bs]
            enc = self.tok([premises[i] for i in idx], [hypotheses[i] for i in idx], truncation="only_first",
                           max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                logits = self.model(**enc).logits.float()
            probs = self.torch.softmax(logits, dim=-1).cpu().numpy()
            out[idx] = probs[:, self.cols]
        return out


def decide(probs: np.ndarray, support_threshold: float = 0.5, contradict_threshold: float = 0.5) -> tuple[str, float]:
    """One (3,) probability row -> (label, confidence). Non-neutral labels need their threshold."""
    s, c, n = float(probs[0]), float(probs[1]), float(probs[2])
    if s >= support_threshold and s >= c:
        return "SUPPORTS", s
    if c >= contradict_threshold and c > s:
        return "CONTRADICTS", c
    return "INSUFFICIENT", max(n, 1.0 - max(s, c))


class NLIVerifier:
    is_mock = False

    def __init__(self, scorer: Scorer, granularity: str = "sentence", support_threshold: float = 0.5,
                 contradict_threshold: float = 0.5, sentence_splitter=None, name: str | None = None,
                 combine: bool = False, combine_top: int = 3):
        if granularity not in ("passage", "sentence"):
            raise ValueError("granularity must be 'passage' or 'sentence'")
        from researchlens.generation.extractive import split_sentences
        self.scorer, self.granularity = scorer, granularity
        self.ts, self.tc = support_threshold, contradict_threshold
        self.split = sentence_splitter or split_sentences
        self.combine, self.combine_top = combine, combine_top      # multi-evidence support (sentence granularity only)
        self.name = name or f"nli:{getattr(scorer, 'name', 'scorer')}:{granularity}"

    # -- core: claim vs a list of premise texts --------------------------------------------
    def _score(self, claim: str, texts: Sequence[str]) -> list[list[tuple[str, np.ndarray]]]:
        """Per text: [(unit_text, probs(3,)), ...]; units are sentences, or the whole text for 'passage'."""
        if self.granularity == "passage":
            P = self.scorer(list(texts), [claim] * len(texts))
            return [[(t, p)] for t, p in zip(texts, P)]
        units, owner = [], []
        for ti, t in enumerate(texts):
            for si, s in enumerate(self.split(t) or [t]):
                units.append(s); owner.append(ti)
        P = self.scorer(units, [claim] * len(units)) if units else np.zeros((0, 3))
        rows: list[list[tuple[str, np.ndarray]]] = [[] for _ in texts]
        for u, ti, p in zip(units, owner, P):
            rows[ti].append((u, p))
        return rows

    def _judge_rows(self, rows):
        out = []
        for unit_rows in rows:
            si, (_, p) = max(enumerate(unit_rows), key=lambda r: max(r[1][1][0], r[1][1][1]))
            label, _ = decide(p, self.ts, self.tc)
            out.append((label, p, [si] if label != "INSUFFICIENT" and self.granularity == "sentence" else []))
        return out

    def judge_texts(self, claim: str, texts: Sequence[str]) -> list[tuple[str, np.ndarray, list[int]]]:
        """For each text: (label, probs(3,), rationale sentence indices)."""
        return self._judge_rows(self._score(claim, texts))

    def aggregate(self, judged: list[tuple[str, np.ndarray, list[int]]]) -> tuple[str, float]:
        """Claim-level label over all passages: strongest SUPPORTS vs strongest CONTRADICTS."""
        best_s = max((p[0] for l, p, _ in judged if l == "SUPPORTS"), default=None)
        best_c = max((p[1] for l, p, _ in judged if l == "CONTRADICTS"), default=None)
        if best_s is None and best_c is None:
            conf = min((max(float(p[2]), 1 - max(float(p[0]), float(p[1]))) for _, p, _ in judged), default=1.0)
            return "INSUFFICIENT", conf
        if best_c is None or (best_s is not None and best_s >= best_c):
            return "SUPPORTS", float(best_s)
        return "CONTRADICTS", float(best_c)

    # -- contract ----------------------------------------------------------------------------
    def verify(self, claim: Claim, passages: list[RetrievedPassage]) -> VerificationResult:
        rows = self._score(claim.text, [p.chunk.text for p in passages]) if passages else []
        judged = self._judge_rows(rows)
        evidence = [EvidenceJudgement(chunk_id=p.chunk.chunk_id, label=l,
                                      probs={v: float(x) for v, x in zip(VERDICTS, probs)}, rationale_sentences=r)
                    for p, (l, probs, r) in zip(passages, judged)]
        label, conf = self.aggregate(judged)
        support, combined_ids = None, []
        if label == "SUPPORTS":
            support = "direct"
        elif label == "INSUFFICIENT" and self.combine and self.granularity == "sentence" and len(passages) >= 2:
            # Multi-evidence: no single passage entails the claim. Join the best-supporting sentence of the top
            # `combine_top` passages and test the claim against that short joint premise.
            best = []
            for pi, unit_rows in enumerate(rows):
                si, (s_text, p) = max(enumerate(unit_rows), key=lambda r: r[1][1][0])
                best.append((float(p[0]), pi, si, s_text))
            best = sorted(best, reverse=True)[: self.combine_top]
            premise = " ".join(b[3] for b in best)
            pj = self.scorer([premise], [claim.text])[0]
            lj, cj = decide(pj, self.ts, self.tc)
            if lj == "SUPPORTS":
                label, conf, support = "SUPPORTS", cj, "combined"
                combined_ids = [b[1] for b in best]
                for pi, si in ((b[1], b[2]) for b in best):
                    evidence[pi].rationale_sentences = [si]
        res = VerificationResult(claim=claim, label=label, confidence=conf, evidence=evidence,
                                 model=getattr(self.scorer, "name", "scorer"), is_mock=False, support=support)
        res.explanation = explain(res, passages, combined_ids)
        return res


def explain(res: VerificationResult, passages: Sequence[RetrievedPassage], combined_ids: Sequence[int] = ()) -> str:
    """Deterministic explanation assembled from the judgements (no generated text)."""
    def where(i: int) -> str:
        c = passages[i].chunk
        return f"[{i + 1}] {c.section}" + (f", p. {c.page}" if c.page is not None else "")

    n = len(passages)
    note = " Model confidence is not proof that the claim is scientifically true."
    if n == 0:
        return "No evidence passages were available, so the claim cannot be assessed." + note
    ev = res.evidence
    if res.label == "SUPPORTS" and res.support == "combined":
        return (f"Supported only by combining {', '.join(where(i) for i in combined_ids)}; no single passage entails the claim "
                f"(joint entailment {res.confidence:.2f}). Treat as a reasonable inference, not a direct statement." + note)
    if res.label == "SUPPORTS":
        i = max((k for k, e in enumerate(ev) if e.label == "SUPPORTS"), key=lambda k: ev[k].probs["SUPPORTS"])
        others = sum(1 for e in ev if e.label == "SUPPORTS") - 1
        extra = f" {others} more passage(s) also entail it." if others > 0 else ""
        return f"Entailed by {where(i)} (entailment {ev[i].probs['SUPPORTS']:.2f}).{extra}" + note
    if res.label == "CONTRADICTS":
        i = max((k for k, e in enumerate(ev) if e.label == "CONTRADICTS"), key=lambda k: ev[k].probs["CONTRADICTS"])
        sup = any(e.label == "SUPPORTS" for e in ev)
        extra = " Other passages entail it, so the evidence conflicts." if sup else ""
        return f"Contradicted by {where(i)} (contradiction {ev[i].probs['CONTRADICTS']:.2f}).{extra}" + note
    return (f"None of the {n} passage(s) entails or contradicts the claim (insufficient evidence in the retrieved text, "
            f"which is not the same as the claim being false)." + note)
