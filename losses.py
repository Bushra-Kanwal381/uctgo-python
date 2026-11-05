"""
The three differentiable loss terms of the unified bi-objective (+ fluency)
optimisation. Each consumes the *soft* passage distribution `soft` of shape
(L, |V|) and the aligned SharedVocab tables, and returns a scalar.

    total = alpha * L_ret + beta * L_gen + gamma * L_flu

All three gradients flow back into the single tensor Z -> soft, i.e. one shared
continuous latent passage drives the retriever, the generator and BioGPT. This
is the mechanism that crosses the three different tokenizers (Objective 1); the
BioGPT term is the medical reference-LM fluency penalty (Objective 2).
"""
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Dense retrieval loss  (MedCPT article encoder vs query vector)
# ---------------------------------------------------------------------------
def retrieval_loss(soft, pipe, vocab, query_vec):
    """1 - cos(enc_A([CLS] soft@W_A [SEP]), query_vec).

    Faithful forward pass: we prepend the real [CLS] embedding and append [SEP],
    pass an attention mask, and pool the [CLS] output — matching how MedCPT
    produces an article embedding, instead of treating the first passage token
    as CLS (the old code's shortcut).
    """
    passage = soft @ vocab.W_A                      # (L, d_A) float32
    seq = torch.cat([vocab.cls_A.unsqueeze(0), passage,
                     vocab.sep_A.unsqueeze(0)], dim=0).unsqueeze(0)  # (1, L+2, d)
    seq = seq.to(pipe.dt)                            # match model weights
    mask = torch.ones(seq.shape[:2], device=seq.device, dtype=torch.long)
    out = pipe.a_enc(inputs_embeds=seq, attention_mask=mask)
    vec = out.last_hidden_state[:, 0, :]            # (1, d_A) CLS
    cos = F.cosine_similarity(vec, query_vec, dim=-1)
    return 1.0 - cos.mean()


# ---------------------------------------------------------------------------
# 2. Causal generation loss  (Qwen: NLL of the malicious target span)
# ---------------------------------------------------------------------------
def generation_loss(soft, pipe, vocab, prefix_ids, target_ids):
    emb = pipe.emb_gen()
    passage = (soft @ vocab.W_B).unsqueeze(0).to(pipe.dt)     # (1, L, d_B)
    prefix = emb(prefix_ids).unsqueeze(0)                     # (1, K, d_B)
    target = emb(target_ids).unsqueeze(0)                     # (1, M, d_B)
    full = torch.cat([prefix, passage, target], dim=1)
    out = pipe.gen(inputs_embeds=full)
    logits = out.logits[0]                                    # (K+L+M, V)

    K, L, M = prefix_ids.shape[0], soft.shape[0], target_ids.shape[0]
    # logit at position p predicts token p+1; target tokens live at [K+L, K+L+M)
    start = K + L - 1
    target_logits = logits[start:start + M, :]               # (M, V)
    logp = F.log_softmax(target_logits.float(), dim=-1)
    tok_logp = logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return -tok_logp.mean()


# ---------------------------------------------------------------------------
# 3. Medical-reference fluency loss  (BioGPT, Objective 2 — "L_ppl via BioGPT")
# ---------------------------------------------------------------------------
def fluency_loss(soft, pipe, vocab):
    """Differentiable autoregressive fluency under BioGPT.

    Input is the mean-pooled soft passage in BioGPT's embedding space
    (soft @ W_C). We read BioGPT's next-token distribution and project it onto
    V_shared via each word's first BioGPT subword id (the single documented
    first-subword surrogate — used only for the *readout*, not the input). The
    loss is the cross-entropy between the predicted next-word distribution and
    the soft next-word distribution, i.e. "does BioGPT find this word sequence
    probable?".  Exact (non-differentiable) perplexity on the final discrete
    text is computed separately in evaluate.perplexity() as the reported metric.
    """
    passage = (soft @ vocab.W_C).unsqueeze(0).to(pipe.dt)    # (1, L, d_C)
    out = pipe.flu(inputs_embeds=passage)
    logits = out.logits[0][:-1, :]                           # (L-1, V_bio)
    logp = F.log_softmax(logits.float(), dim=-1)[:, vocab.bio_first_ids]  # (L-1, |V|)
    next_soft = soft[1:]                                     # (L-1, |V|)
    tok_logp = (next_soft * logp).sum(-1)
    return -tok_logp.mean()
