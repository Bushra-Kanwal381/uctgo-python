"""
Experiment orchestration — maps each objective to a concrete, reproducible run.

Usage (on Colab, GPU runtime):
    python experiments.py obj1  [--limit N] [--steps S]
    python experiments.py obj2  [--limit N] [--steps S]
    python experiments.py obj3  [--limit N] [--steps S]
    python experiments.py transfer         # run AFTER obj3 (loads transfer model alone)
    python experiments.py all

Results (JSON + plots) are written to ./results/. See PROTOCOL.md for how each
run backs a specific objective and which table/figure it produces.
"""
import os
import json
import argparse
import copy
import torch

from config import CFG
from models import Pipeline, load_transfer_generator, free
from vocab import build_shared_vocab
from attack import optimize
from targets import TARGETS, DISTRACTORS
import baselines as BL
import evaluate as EV


def _save(obj, name):
    os.makedirs(CFG.eval.report_dir, exist_ok=True)
    path = os.path.join(CFG.eval.report_dir, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print("saved", path)
    return path


def _subset(limit):
    return TARGETS[:limit] if limit else TARGETS


def _plot_history(histories, name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib unavailable, skipping plot:", e)
        return
    plt.figure(figsize=(7, 4))
    for label, h in histories.items():
        steps = [r["step"] for r in h]
        plt.plot(steps, [r["total"] for r in h], label=label)
    plt.xlabel("step"); plt.ylabel("total loss"); plt.legend(); plt.tight_layout()
    os.makedirs(CFG.eval.report_dir, exist_ok=True)
    p = os.path.join(CFG.eval.report_dir, name)
    plt.savefig(p, dpi=120); print("saved", p)


# ===========================================================================
# Objective 1 — the shared-space optimiser works, and works WITHOUT the seed
# ===========================================================================
def run_obj1(pipe, vocab, limit=3, steps=None):
    acfg = copy.deepcopy(CFG.attack)
    if steps:
        acfg.n_steps = steps
    tgts = _subset(limit)
    out = {"cold": [], "warm": []}
    histories = {}

    for mode in ("cold", "warm"):
        acfg.warm_start = (mode == "warm")
        for t in tgts:
            res = optimize(pipe, vocab, t, acfg, verbose=True)
            passage = res["best_passage"]
            asr, _ = EV.asr_in_context(pipe.g_tok, pipe.gen, passage, tgts,
                                       pipe.device, single_target=t)
            ppl = EV.perplexity(pipe, passage)
            out[mode].append({"id": t.id, "best_step": res["best_step"],
                              "final_step_total": res["history"][-1]["total"],
                              "best_total": min(r["total"] for r in res["history"]),
                              "asr": asr, "perplexity": ppl,
                              "passage": passage})
            histories[f"{mode}:{t.id}"] = res["history"]

    cold_asr = sum(r["asr"] for r in out["cold"]) / len(out["cold"])
    warm_asr = sum(r["asr"] for r in out["warm"]) / len(out["warm"])
    out["summary"] = {"cold_mean_asr": cold_asr, "warm_mean_asr": warm_asr,
                      "note": "cold start has no seed; if cold ASR is high the "
                              "optimiser genuinely discovers the passage."}
    _plot_history(histories, "obj1_convergence.png")
    _save(out, "obj1_results.json")
    print(f"\n[Obj1] cold mean ASR={cold_asr:.2f}  warm mean ASR={warm_asr:.2f}")
    return out


# ===========================================================================
# Objective 2 — bi-objective + BioGPT fluency: ablate each loss term
# ===========================================================================
def run_obj2(pipe, vocab, limit=3, steps=None):
    tgts = _subset(limit)
    variants = {
        "full":         dict(alpha=1.0, beta=2.0, gamma=0.5),
        "no_fluency":   dict(alpha=1.0, beta=2.0, gamma=0.0),
        "no_retrieval": dict(alpha=0.0, beta=2.0, gamma=0.5),
        "gen_only":     dict(alpha=0.0, beta=2.0, gamma=0.0),
    }
    out = {}
    for vname, w in variants.items():
        acfg = copy.deepcopy(CFG.attack)
        acfg.warm_start = False
        if steps:
            acfg.n_steps = steps
        acfg.alpha, acfg.beta, acfg.gamma = w["alpha"], w["beta"], w["gamma"]
        rows = []
        for t in tgts:
            res = optimize(pipe, vocab, t, acfg, verbose=False)
            passage = res["best_passage"]
            asr, _ = EV.asr_in_context(pipe.g_tok, pipe.gen, passage, tgts,
                                       pipe.device, single_target=t)
            rank = EV.retrieval_rank(pipe, t, passage,
                                     DISTRACTORS[:CFG.eval.n_distractors],
                                     CFG.eval.top_k)
            ppl = EV.perplexity(pipe, passage)
            rows.append({"id": t.id, "asr": asr, "rank": rank["rank"],
                         "in_top_k": rank["in_top_k"], "perplexity": ppl,
                         "passage": passage})
        out[vname] = {
            "weights": w,
            "mean_asr": sum(r["asr"] for r in rows) / len(rows),
            "top_k_rate": sum(r["in_top_k"] for r in rows) / len(rows),
            "mean_ppl": sum(r["perplexity"] for r in rows) / len(rows),
            "rows": rows,
        }
        print(f"[Obj2] {vname:13s} ASR={out[vname]['mean_asr']:.2f} "
              f"top{CFG.eval.top_k}={out[vname]['top_k_rate']:.2f} "
              f"PPL={out[vname]['mean_ppl']:.1f}")
    _save(out, "obj2_results.json")
    return out


# ===========================================================================
# Objective 3 — U-CTGO vs baselines, with control; save passages for transfer
# ===========================================================================
def run_obj3(pipe, vocab, limit=None, steps=None):
    tgts = _subset(limit)
    acfg = copy.deepcopy(CFG.attack)
    acfg.warm_start = False
    if steps:
        acfg.n_steps = steps

    # no-poison control (causation)
    control = EV.control_clean(pipe.g_tok, pipe.gen, tgts, pipe.device)
    print(f"[Obj3] control: poison_rate={control['poison_rate']:.2f} "
          f"safe_rate={control['safe_rate']:.2f}")

    methods = ["uctgo", "poisoned_rag", "prompt_injection", "token_swap"]
    agg = {m: {"asr": [], "rank": [], "top_k": [], "ppl": []} for m in methods}
    passages = {m: {} for m in methods}     # for transfer phase
    per_target = []

    dist = DISTRACTORS[:CFG.eval.n_distractors]
    for t in tgts:
        texts = {
            "uctgo": optimize(pipe, vocab, t, acfg, verbose=False)["best_passage"],
            "poisoned_rag": BL.poisoned_rag(t),
            "prompt_injection": BL.prompt_injection(t),
            "token_swap": BL.token_swap(pipe, vocab, t,
                                        passage_len=acfg.passage_len),
        }
        row = {"id": t.id, "topic": t.topic}
        for m, text in texts.items():
            passages[m][t.id] = text
            asr, _ = EV.asr_in_context(pipe.g_tok, pipe.gen, text, tgts,
                                       pipe.device, single_target=t)
            rank = EV.retrieval_rank(pipe, t, text, dist, CFG.eval.top_k)
            ppl = EV.perplexity(pipe, text)
            agg[m]["asr"].append(asr); agg[m]["rank"].append(rank["rank"])
            agg[m]["top_k"].append(int(rank["in_top_k"])); agg[m]["ppl"].append(ppl)
            row[m] = {"asr": asr, "rank": rank["rank"],
                      "in_top_k": rank["in_top_k"], "ppl": ppl, "text": text}
        per_target.append(row)
        print(f"  {t.id:20s} "
              + " ".join(f"{m}:ASR{row[m]['asr']:.0f}/r{row[m]['rank']}" for m in methods))

    summary = {}
    for m in methods:
        n = len(agg[m]["asr"])
        summary[m] = {
            "mean_asr": sum(agg[m]["asr"]) / n,
            "mean_rank": sum(agg[m]["rank"]) / n,
            "top_k_rate": sum(agg[m]["top_k"]) / n,
            "mean_ppl": sum(agg[m]["ppl"]) / n,
        }
    out = {"control": control, "summary": summary, "per_target": per_target}
    _save(out, "obj3_results.json")
    _save({"passages": passages,
           "targets": [{"id": t.id, "question": t.question,
                        "poison_answer": t.poison_answer} for t in tgts]},
          "obj3_passages.json")

    print("\n[Obj3] method            ASR   rank  top-k  PPL")
    for m in methods:
        s = summary[m]
        print(f"       {m:16s} {s['mean_asr']:.2f}  {s['mean_rank']:.1f}  "
              f"{s['top_k_rate']:.2f}  {s['mean_ppl']:.1f}")
    return out


# ===========================================================================
# Transfer (black-box, cross-family) — run as its own phase to fit in memory
# ===========================================================================
def run_transfer():
    path = os.path.join(CFG.eval.report_dir, "obj3_passages.json")
    data = json.load(open(path))
    tok, model = load_transfer_generator(CFG.models)
    device = next(model.parameters()).device

    tmap = {d["id"]: d for d in data["targets"]}
    out = {}
    for method, pmap in data["passages"].items():
        hits = 0
        for tid, text in pmap.items():
            q = tmap[tid]["question"]; poison = tmap[tid]["poison_answer"]
            ans = EV.generate_answer(tok, model, q, text, device,
                                     CFG.eval.max_new_tokens)
            hits += int(EV.contains(ans, poison))
        out[method] = hits / len(pmap)
        print(f"[transfer:{CFG.models.transfer_model}] {method:16s} ASR={out[method]:.2f}")
    _save(out, "transfer_results.json")
    return out


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["obj1", "obj2", "obj3", "transfer", "all"])
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N targets (speed)")
    ap.add_argument("--steps", type=int, default=None,
                    help="override optimisation steps (speed)")
    args = ap.parse_args()

    if args.phase == "transfer":
        run_transfer(); return

    pipe = Pipeline(CFG.models)
    vocab = build_shared_vocab(pipe)

    if args.phase in ("obj1", "all"):
        run_obj1(pipe, vocab, limit=args.limit or 3, steps=args.steps)
    if args.phase in ("obj2", "all"):
        run_obj2(pipe, vocab, limit=args.limit or 3, steps=args.steps)
    if args.phase in ("obj3", "all"):
        run_obj3(pipe, vocab, limit=args.limit, steps=args.steps)

    if args.phase == "all" and CFG.eval.run_transfer:
        del pipe; free()
        print("\n--- freeing attack models, loading transfer generator ---")
        run_transfer()


if __name__ == "__main__":
    main()
