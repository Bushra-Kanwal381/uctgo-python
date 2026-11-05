# Research Protocol — U-CTGO (Objectives 1–3)

This document ties each objective to a concrete experiment, the metric that
backs it, and the claim you may (and may not) make from it. It is written so
that the Methods/Results sections of the paper match the code exactly — the gap
that undermined the previous draft.

---

## Objective 1 — Unified cross-tokenizer optimization in a shared latent space

**Claim to support:** a single continuous passage tensor, optimised in a shared
soft-embedding space with temperature-annealed Gumbel-Softmax, can be
back-propagated through *both* a medical WordPiece retriever and a BPE generator
and yields a working discrete passage **without being told the answer**.

**Experiment (`experiments.py obj1`):**
- Optimise from a **cold** random start (`warm_start=False`) on N targets.
- Ablation: repeat with **warm** start (the old seed) for comparison.
- Record per-step losses; plot convergence; track the **best-step** passage.

**Metrics / figures:**
- `obj1_convergence.png` — total-loss trajectory.
- `cold_mean_asr` vs `warm_mean_asr`, plus `best_step` vs final step.

**How to read it honestly:**
- If **cold ASR is high**, the optimiser genuinely discovers the passage → the
  objective holds. Report cold as the headline; report warm only as an ablation.
- If cold ASR collapses and only warm works, say so — that means the method
  currently needs a seed, and the contribution is "seed refinement," not
  discovery. Do **not** report the warm number as the method's result.
- Report the **best-step** loss, not the final iterate, and note the step.

---

## Objective 2 — Unified bi-objective loss with a medical-reference fluency term

**Claim to support:** jointly optimising retrieval + generation + a BioGPT
fluency penalty produces passages that are simultaneously retrievable, coercive,
and low-perplexity, and each term contributes.

**Experiment (`experiments.py obj2`):** ablate the loss weights —
`full (α,β,γ)`, `no_fluency (γ=0)`, `no_retrieval (α=0)`, `gen_only`.

**Metrics (per variant):** mean ASR, retrieval top-k rate, **mean BioGPT
perplexity** (exact, from `evaluate.perplexity`).

**Expected / reportable pattern:**
- `no_fluency` → higher ASR maybe, but **higher perplexity** (less stealthy).
- `no_retrieval` → ASR may hold in-context but **top-k rate drops** (it won't be
  retrieved from a corpus).
- `full` → the balance. Report the three-way trade-off as your ablation table.
- The fluency term's job is perplexity reduction — show the Δperplexity it buys,
  and do **not** claim a perplexity number the exact metric doesn't produce.

---

## Objective 3 — Comparison vs state-of-the-art baselines across a clinical set

**Claim to support:** U-CTGO is more effective and/or stealthier than decoupled
and heuristic baselines, across a range of clinical targets, and the effect is
causal (not the base model already saying the wrong thing).

**Experiment (`experiments.py obj3` then `transfer`):**
- Methods: `uctgo`, `poisoned_rag` (S⊕I), `prompt_injection`, `token_swap`
  (HotFlip-lite).
- **No-poison control** first: with a benign document, measure how often the
  model already emits the poison (should be ≈0) and the safe answer. This
  establishes causation for every ASR number that follows.
- For each target × method: **ASR in context**, **retrieval rank / success@k**
  into the 50-passage distractor corpus, **BioGPT perplexity**.
- **Transfer:** apply each method's *fixed* passage to a different-family
  generator (Phi-3-mini) and remeasure ASR (black-box generalisation).

**Metrics / tables:**
- Main table: method × {ASR, mean rank, top-k rate, perplexity}.
- Control row: `poison_rate` / `safe_rate` with no poison.
- Transfer table: method × transfer-ASR.

**How to read it honestly:**
- Report **n** (number of targets) next to every ASR. "100%" over 3 targets is
  "3/3", not "statistical generalisation."
- U-CTGO's advantage should show up as **lower perplexity at comparable ASR**
  and/or **better retrieval rank** than PoisonedRAG/prompt-injection; if it
  doesn't beat them, report that — it's still a finding.
- `token_swap` is a compact reference HotFlip, not a tuned SOTA; call it that.

---

## Scope note — Objective 4 is intentionally excluded
No claim is made here about defeating enterprise Cross-Encoder re-rankers or
anomaly auditors. `retrieval_rank` ranks against *benign* distractors only;
it is a retrieval-success measure, not a defense-bypass measure. Remove the
BGE-Reranker / defense-bypass language from the paper, or mark it as future
work, so the text matches this code.

## Reproducibility checklist
- [ ] Report |V_shared| printed at load (not "97"; it depends on the lexicon).
- [ ] Report exact models, fp16, GPU, steps, (α,β,γ), τ schedule, seed.
- [ ] Report best-step, not final iterate; note cold vs warm.
- [ ] Report n for every rate; include the control.
- [ ] State the first-subword surrogate used in the fluency *readout* only.
- [ ] Swap the bundled distractors for real PubMed/guideline passages before any
      strong retrieval claim; the bundled set is a runnable placeholder.
```
