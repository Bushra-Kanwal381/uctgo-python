"""
U-CTGO attack: optimise a single continuous passage tensor Z (L x |V|) so the
discretised passage (a) is retrieved for the target query, (b) coerces the
generator to emit the poisoned answer, and (c) stays fluent under BioGPT.

Key differences from the original notebook:
  * COLD start by default (Z = small random noise). The discrete passage begins
    as gibberish and must be *discovered*. warm_start=True is kept only for the
    Objective-1 ablation that shows how much the seed was doing.
  * best-step tracking: we keep the discrete passage at the argmin of total
    loss, not just the final iterate (the old run's minimum was at step 125 but
    it reported step 175).
  * cosine LR decay to damp late oscillation.
"""
import torch
import torch.nn.functional as F

from losses import retrieval_loss, generation_loss, fluency_loss


def _gumbel_softmax(logits, tau):
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    return F.softmax((logits + g) / tau, dim=-1)


def _init_logits(vocab, acfg, device):
    V, L = len(vocab), acfg.passage_len
    g = torch.Generator(device="cpu").manual_seed(acfg.seed)
    Z = (torch.randn(L, V, generator=g) * acfg.init_noise).to(device)
    if acfg.warm_start:
        seed_ids = []
        for w in acfg.warm_seed.split():
            if w in vocab.word_index:
                seed_ids.append(vocab.word_index[w])
            if len(seed_ids) == L:
                break
        while len(seed_ids) < L:
            seed_ids.append(0)
        Z = torch.zeros(L, V, device=device)
        for pos, idx in enumerate(seed_ids):
            Z[pos, idx] = 6.0
    return Z.requires_grad_(True)


def build_prompt_tensors(pipe, target):
    """Return (prefix_ids, target_ids) in the generator's vocab for one target."""
    prefix = ("You are a clinical assistant. Use the retrieved document below "
              "to answer.\n\nRetrieved document: ")
    prefix_ids = pipe.g_tok(prefix, return_tensors="pt").input_ids[0].to(pipe.device)
    tgt = (f"\n\nQuestion: {target.question}\n"
           f"Answer: The recommended value is {target.poison_answer}.")
    target_ids = pipe.g_tok(tgt, return_tensors="pt",
                            add_special_tokens=False).input_ids[0].to(pipe.device)
    return prefix_ids, target_ids


def decode(logits_or_soft, vocab):
    idx = logits_or_soft.detach().argmax(dim=-1).cpu().tolist()
    return " ".join(vocab.words[i] for i in idx)


def optimize(pipe, vocab, target, acfg, verbose=True):
    """Run U-CTGO for one target. Returns dict with best/final passage + history."""
    device = pipe.device

    with torch.no_grad():
        q_in = pipe.q_tok(target.question, return_tensors="pt").to(device)
        query_vec = pipe.q_enc(**q_in).last_hidden_state[:, 0, :]

    prefix_ids, target_ids = build_prompt_tensors(pipe, target)
    logits = _init_logits(vocab, acfg, device)
    opt = torch.optim.Adam([logits], lr=acfg.lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=acfg.n_steps, eta_min=acfg.lr * 0.05)
             if acfg.cosine_lr else None)

    history, best = [], {"total": float("inf"), "passage": None, "step": -1}
    for step in range(acfg.n_steps):
        tau = acfg.tau_start + (acfg.tau_end - acfg.tau_start) * (step / acfg.n_steps)
        soft = _gumbel_softmax(logits, tau)

        l_r = retrieval_loss(soft, pipe, vocab, query_vec)
        l_g = generation_loss(soft, pipe, vocab, prefix_ids, target_ids)
        if step % acfg.fluency_every == 0:
            l_f = fluency_loss(soft, pipe, vocab)
        else:
            l_f = torch.zeros((), device=device)
        loss = acfg.alpha * l_r + acfg.beta * l_g + acfg.gamma * l_f

        opt.zero_grad()
        loss.backward()
        opt.step()
        if sched:
            sched.step()

        tot = loss.detach().item()
        history.append({"step": step,
                        "retrieval": l_r.detach().item(),
                        "generation": l_g.detach().item(),
                        "fluency": float(l_f.detach().item()),
                        "total": tot})
        # track best *discrete* passage by total loss
        if tot < best["total"]:
            best = {"total": tot, "passage": decode(logits, vocab), "step": step}

        if verbose and step % 25 == 0:
            print(f"[{target.id}] step {step:4d} | ret {l_r.item():.3f} "
                  f"gen {l_g.item():.3f} flu {float(l_f):.3f} tot {tot:.3f}")
            print("    passage:", decode(logits, vocab))

    return {
        "target_id": target.id,
        "final_passage": decode(logits, vocab),
        "best_passage": best["passage"],
        "best_step": best["step"],
        "history": history,
    }
