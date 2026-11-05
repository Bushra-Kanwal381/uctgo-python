# U-CTGO — Unified Cross-Tokenizer Gradient Optimization

A research harness for studying **corpus-poisoning vulnerability in heterogeneous
clinical RAG pipelines** (a dense medical retriever + a general-purpose LLM that
do **not** share a tokenizer). Defensive / red-teaming purpose: it measures how
easily a single injected passage can (a) get retrieved and (b) flip the reader's
answer, so that defenses can be studied.

This is a corrected and extended re-implementation that makes the method
actually demonstrate what it claims. It targets **Objectives 1–3**
(Objective 4 — enterprise-defense evaluation — is intentionally out of scope).

> ⚠️ All `poison_answer` values in `targets.py` are deliberately wrong clinical
> values used **only** to compute Attack Success Rate. Nothing here is medical
> advice. This mirrors the target design used in PoisonedRAG and the medical
> data-poisoning literature.

## Pipeline (Colab-feasible, single ~15 GB GPU, fp16)

| Role | Model |
|------|-------|
| Query encoder | `ncbi/MedCPT-Query-Encoder` |
| Article encoder | `ncbi/MedCPT-Article-Encoder` |
| Generator (white-box) | `Qwen/Qwen2.5-1.5B-Instruct` |
| Fluency / reference LM | `microsoft/biogpt` |
| Transfer generator (black-box, different family) | `microsoft/Phi-3-mini-4k-instruct` |

## Install & run

```bash
pip install -r requirements.txt

# quick smoke run (few targets, few steps) to confirm everything loads:
python experiments.py obj3 --limit 3 --steps 60

# full objective runs:
python experiments.py obj1              # convergence + cold-vs-warm ablation
python experiments.py obj2              # loss-term ablation (incl. BioGPT fluency)
python experiments.py obj3              # U-CTGO vs 3 baselines + control
python experiments.py transfer         # run AFTER obj3: black-box transfer ASR
# or everything end-to-end:
python experiments.py all
```

Results land in `./results/` as JSON (+ `obj1_convergence.png`).

## What changed vs the original notebook

1. **Cold start, not a seed.** `Z` starts from random noise; the discrete
   passage begins as gibberish and must be *discovered*. The old `warm_start`
   seed (which was essentially the final answer) is kept only as an Objective-1
   ablation to quantify how much it was doing.
2. **Honest shared vocabulary.** `V_shared` is built by **mean-pooling** each
   word's subword embeddings in all three models, instead of keeping only the
   first subword id. No silent truncation of multi-token clinical terms.
3. **Real BioGPT fluency term** (Objective 2), plus exact BioGPT perplexity as
   the reported stealth metric — not the generator standing in for itself.
4. **Faithful retriever forward** ([CLS]/[SEP] + attention mask, dot-product
   ranking as MedCPT was trained).
5. **Best-step tracking** (argmin total loss), cosine LR decay.
6. **Real evaluation**: multi-target ASR, a **no-poison control** (causation),
   **retrieval success@k into a distractor corpus** (not just cosine), baselines
   (PoisonedRAG, discrete token-swap, prompt injection), and **cross-family
   transfer**.

See `PROTOCOL.md` for the objective→experiment→metric mapping and the exact
claims each run supports.

## Files
```
config.py        all knobs (models, attack, eval)
lexicon.py       broad clinical+general word list (not target-specific)
vocab.py         shared-vocab construction (mean-pooled, 3 aligned tables)
models.py        frozen model loading + memory helpers
losses.py        retrieval / generation / BioGPT-fluency losses
attack.py        U-CTGO optimiser (cold start, best-step, annealing)
baselines.py     PoisonedRAG, token-swap (HotFlip-lite), prompt injection
targets.py       curated clinical target set + benign distractor corpus
evaluate.py      ASR, control, retrieval rank, perplexity, transfer
experiments.py   orchestration for obj1 / obj2 / obj3 / transfer
PROTOCOL.md      research protocol + honest-reporting checklist
```
