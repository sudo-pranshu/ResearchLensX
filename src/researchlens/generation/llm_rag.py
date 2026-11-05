"""Conventional RAG generator with an open-weight instruct LLM. Owner: Pawan (initial: Pranshu, sprint).

  llm = HFChatLLM("Qwen/Qwen2.5-1.5B-Instruct")        # greedy decoding, pinned revision recorded
  gen = LLMRAGGenerator(llm)
  ans = gen.generate(question, passages)               # Answer(mode="rag"), citations 0-based

The LLM is injected (callable: messages -> text) so prompt building and citation parsing are
unit-tested offline. The SAME generator output is used by baseline and verified conditions.
"""
from __future__ import annotations

import hashlib
import re
import time
from typing import Callable

from researchlens.contracts import Answer, AnswerSentence, RetrievedPassage
from researchlens.generation.extractive import split_sentences

SYSTEM = ("You answer questions about a scientific paper using ONLY the numbered passages provided. "
          "Write at most 3 short sentences. End every sentence with the number(s) of the passage(s) that "
          "support it in square brackets, e.g. [2] or [1][3]. If the passages do not contain the answer, reply "
          "exactly: The passages do not contain enough information.")
ABSTAIN = "The passages do not contain enough information."
_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_ABSTAIN_RE = re.compile(r"(do(es)? not|don't) (contain|provide|mention|include)|not enough information|cannot be determined|no information", re.I)


def build_messages(question: str, passages: list[RetrievedPassage], max_chars: int = 1200) -> list[dict]:
    blocks = [f"[{i}] ({p.chunk.section}) {p.chunk.text[:max_chars]}" for i, p in enumerate(passages, 1)]
    user = "Passages:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer:"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def prompt_hash(messages: list[dict]) -> str:
    return hashlib.sha1(repr(messages).encode()).hexdigest()[:12]


def is_abstention(text: str) -> bool:
    return bool(_ABSTAIN_RE.search(text))


def parse_answer(raw: str, n_passages: int) -> list[AnswerSentence]:
    """Split into sentences; collect [n] markers (1-based in text) -> 0-based citations; strip markers."""
    raw = raw.strip()
    # keep markers attached to the sentence they close: "... method [1]. Next" -> "... method [1]." "Next"
    raw = re.sub(r"\.\s*((?:\[\d+(?:\s*,\s*\d+)*\]\s*)+)", lambda m: " " + m.group(1).strip() + ". ", raw)
    out = []
    for s in split_sentences(raw):
        cites = []
        for m in _CITE.finditer(s):
            for n in m.group(1).split(","):
                k = int(n) - 1
                if 0 <= k < n_passages and k not in cites:
                    cites.append(k)
        text = re.sub(r"\s+([.,;:])", r"\1", _CITE.sub("", s)).strip()
        if len(text.strip(" .")) > 0:
            out.append(AnswerSentence(text=text, citations=cites))
    return out


class HFChatLLM:
    def __init__(self, model_name: str, max_new_tokens: int = 160, device: str | None = None, dtype: str = "float16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from researchlens.retrieval.dense import pick_device
        self.torch = torch
        self.device = device or pick_device()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        td = getattr(torch, dtype) if self.device != "cpu" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=td).to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self.name = model_name
        self.meta = {"model": model_name, "revision": getattr(self.model.config, "_commit_hash", None) or "unknown",
                     "decoding": "greedy", "max_new_tokens": max_new_tokens, "dtype": str(td)}

    def __call__(self, messages: list[dict]) -> str:
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


class LLMRAGGenerator:
    is_mock = False

    def __init__(self, llm: Callable[[list[dict]], str], name: str | None = None):
        self.llm = llm
        self.name = name or f"rag:{getattr(llm, 'name', 'llm')}"

    def generate(self, question: str, passages: list[RetrievedPassage], raw: str | None = None) -> Answer:
        """`raw` lets experiments replay a cached generation instead of calling the LLM."""
        t = time.perf_counter()
        if raw is None:
            raw = self.llm(build_messages(question, passages))
        sentences = parse_answer(raw, len(passages))
        return Answer(question=question, sentences=sentences, passages=passages, mode="rag",
                      generator=self.name, timings_ms={"generate": (time.perf_counter() - t) * 1e3})
