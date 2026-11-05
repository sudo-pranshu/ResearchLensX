# Experiment Protocol

The hypotheses and decision rule are defined in [PROJECT_BIBLE.md §B](PROJECT_BIBLE.md#b-research-specification). This file explains *how* experiments are run and logged.

## 1. Run discipline
1. Every run is launched from a **committed** state (`git status` clean) with a YAML file in `configs/`. The run writes `results/<run_id>/config.json` with the git hash.
2. **Dev for everything** (prompts, thresholds, RRF k, model choice). **Test once** per final system: the run id goes into the experiment register, and test runs are never repeated after looking at results. If a bug forces a re-run, document it in the register.
3. Outputs: `config.json`, `summary.json`, `comparisons.json` (committed); `per_query.jsonl` (committed if < 5 MB, otherwise on Drive with its path recorded).
4. Seeds: 0 everywhere. Neural encoders run in eval mode; LLM decoding is greedy (temperature 0); the prompt text is stored in the config.
5. Register: append a row to PROJECT_BIBLE §G in the same PR as the results.

## 2. H1 retrieval
```bash
python experiments/h1_retrieval.py --config configs/h1_classical_dev.yaml
python experiments/h1_retrieval.py --config configs/h1_dense_dev.yaml          # Colab
python experiments/h1_retrieval.py --config configs/h1_final_test.yaml         # once, Day 7
```
- Corpus: every paragraph + figure/table caption of every paper in the split, indexed as `"{title} | {section}\n{text}"`.
- Within-paper: candidates = that paper's paragraphs. Gold = union of annotators' evidence, with headings excluded and logged as `evidence_unmatched`.
- Questions with no gold evidence are excluded (dev: 82/1,005), and the count is logged.
- Statistics: `compare_many(..., groups=paper_ids)` gives the paper-level cluster bootstrap (10k) CI and cluster sign-flip p-values, Holm-adjusted across the baselines of one hybrid.
- SciFact retrieval: queries = claims, corpus = 5,183 abstracts (title + abstract), gold = abstracts with evidence labels; dev split; claim-level statistics.

## 3. H2 verification
1. **Verifier component:** SciFact dev, given gold abstracts → label macro-F1. Report the LR baseline, the zero-shot NLI model, and the optional fine-tuned model. Model selection uses an 80/20 split of train.
2. **Generations:** `experiments/h2_generate.py --config configs/h2_generate_test.yaml` on Colab writes `results/h2/generations_<split>_<model>.jsonl`, one record per question with the retrieved passages, prompt hash and raw output. Committed.
3. **Verified RAG** runs on CPU from cached generations, so the GPU is not needed again.
4. **Citation P/R (ALCE-style)** uses judge model J ≠ verifier V:
   - *Citation recall* (per sentence) = 1 if J(concat(cited passages) ⇒ sentence) = entailment.
   - *Citation precision* (per citation) = 1 if the citation is needed: it entails the sentence alone, or removing it breaks entailment.
   - Each is averaged per question, then across questions.
5. **Human evaluation:** 50 questions sampled with seed 0 from the H2 test set, two systems each, shown in random order with the system identity hidden. Per sentence: supported / partially / unsupported; per citation: correct / incorrect. 15 questions are annotated by all three members (Fleiss' κ); judge–human agreement (Cohen's κ) is computed on these items. Sheet template: `docs/human_eval_guidelines.md` (Pawan, Day 7).

## 4. Statistical reporting template
`metric: baseline X.XXX → treatment Y.YYY, Δ = +Z.ZZZ [95% CI lo, hi], p_holm = P (cluster bootstrap, n_q questions / n_p papers)`

## 5. What may not be reported as a result
Dev numbers presented as final; any number from a run without a committed config; mock outputs; cherry-picked case studies presented as typical. Case studies (H3) are labelled "illustrative".
